//! Online-neural arm (experiment): a **token-level** selective-SSM that learns
//! during compression and joins the mixer as one more [`TokenModel`].
//!
//! Unlike a byte-level LM, this arm operates on the **decoded u22 token sequence**
//! the Re-Pair stage emits — the same granularity the rest of this branch's models
//! were revamped to (see [`crate::models::ordern`]). Its recurrence steps once per
//! *complete token*, consuming a hashed embedding of the previous token's decoded
//! integer value; the residual output `h` summarizes the token history. Coding is
//! still byte-level (the entropy coder codes each byte as an 8-bit MSB-first tree),
//! so a small per-bit **head** turns `h` into a bit prediction, conditioned on the
//! in-progress ULEB128 decode state (`cur_payload`, `cur_bytes`, the bit-tree node
//! `c0`) — exactly the context [`crate::models::ordern::OrderN`] keys on, but with a
//! learned recurrent summary in place of a hashed table.
//!
//! Because each token-step's residual output feeds exactly one head (predicting the
//! *next* token's bits), the truncated-BPTT window layout is identical to a byte-LM:
//! slot `j` consumes token `t_j` → `xfinal[j]`, and its head predicts `t_{j+1}`.
//! Online learning (BPTT + Adam) runs at each window boundary, identically on encode
//! and decode, so the arm ships **no weights** — `L(D)` is just code.
//!
//! ## Candidate #3: replay + difficulty-routed compute
//!
//! Baseline online learning is single-pass. But at each window boundary both sides
//! already possess the window's tokens, so they can *deterministically re-train* on
//! them. [`Sel::learn_window`] does one base pass, then — gated by a deterministic
//! difficulty signal (the window's average bit-loss vs. an EMA of past windows) —
//! extra BPTT+Adam passes re-run from the window's start state. Configured via env
//! (encode+decode read the same): `LZR_SSM_REPLAY` (max extra passes; 0 = baseline),
//! `LZR_SSM_REPLAY_THRESH` (EMA multiplier gate; 0 = uniform, 1.0 = targeted).
//!
//! Compiled only under `#[cfg(test)]`: a measurement arm, not a shipped codec
//! feature (the f32 kernel is only bit-reproducible on one machine, so it must not
//! enter the on-disk format). Reproduce numbers via the `#[ignore]` `ssm_arm_e2e`.

use crate::mixer::stretch;
use crate::models::{Context, TokenModel};

/// Hashed token-embedding table rows (power of two): the arm indexes `emb` by a
/// multiplicative hash of the decoded token value, so it needs no knowledge of the
/// (data-dependent) vocabulary size and stays fixed-cost.
const EMB_ROWS: usize = 4096;
const EMB_SHIFT: u32 = 64 - 12; // log2(EMB_ROWS) = 12

/// Decode-feature width fed to the per-bit head alongside the hidden state:
/// 8 partial-byte bits + 8 bit-position one-hot + 3 `cur_bytes` one-hot + 14
/// `cur_payload` bits.
const F: usize = 33;

/// Max head bits collected per token: a u22 token is ≤ 3 ULEB128 bytes = ≤ 24 bits.
const MB: usize = 24;

/// Replay telemetry (test measurement only): how many windows were trained, and how
/// many of those earned extra replay passes — so the harness can report the replay
/// *fraction* (the selectivity of an absolute-loss gate).
static TOTAL_WINDOWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REPLAYED_WINDOWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Multiplicative hash of a decoded token value to an `emb` row.
fn emb_idx(v: u32) -> usize {
    ((u64::from(v)).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> EMB_SHIFT) as usize % EMB_ROWS
}

/// The per-bit decode features from the shared context: where we are in the current
/// token's ULEB128 varint and what's been decoded so far. Deterministic on both sides.
fn features(ctx: &Context) -> [f32; F] {
    let mut f = [0.0f32; F];
    let bpos = ctx.bpos as usize;
    // Partial current byte: the `bpos` bits coded so far (MSB first), as ±1.
    for i in 0..bpos {
        let bit = (ctx.c0 >> (bpos - 1 - i)) & 1;
        f[i] = bit as f32 * 2.0 - 1.0;
    }
    // Bit-position one-hot (0..=7).
    f[8 + bpos.min(7)] = 1.0;
    // In-progress token byte count one-hot (0..=2).
    f[16 + (ctx.cur_bytes.min(2) as usize)] = 1.0;
    // Accumulated decoded payload bits (≤ 14) of the in-progress token, as ±1.
    for i in 0..14 {
        f[19 + i] = ((ctx.cur_payload >> i) & 1) as f32 * 2.0 - 1.0;
    }
    f
}

/// `bias + Σ w[i]·x[i]` with 16 independent accumulators so the `mul_add` chain is
/// throughput-bound (auto-vectorizes to NEON `fmla`), not latency-bound.
fn dot(w: &[f32], x: &[f32], bias: f32) -> f32 {
    const LANES: usize = 16;
    let mut acc = [0.0f32; LANES];
    let chunks = x.len() / LANES;
    for c in 0..chunks {
        let base = c * LANES;
        for l in 0..LANES {
            acc[l] = w[base + l].mul_add(x[base + l], acc[l]);
        }
    }
    let mut s = bias + acc.iter().sum::<f32>();
    for i in (chunks * LANES)..x.len() {
        s = w[i].mul_add(x[i], s);
    }
    s
}

/// `dst[i] += k · src[i]`.
fn axpy(dst: &mut [f32], src: &[f32], k: f32) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d = s.mul_add(k, *d);
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// One Adam-trained parameter tensor (weights + first/second moments + grad).
#[derive(Debug)]
struct Tensor {
    w: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
}

impl Tensor {
    fn new(n: usize, rng: &mut dyn FnMut() -> f32) -> Self {
        Self { w: (0..n).map(|_| rng()).collect(), m: vec![0.0; n], v: vec![0.0; n], g: vec![0.0; n] }
    }
    fn zeros(n: usize) -> Self {
        Self { w: vec![0.0; n], m: vec![0.0; n], v: vec![0.0; n], g: vec![0.0; n] }
    }
    fn adam(&mut self, lr: f32, bc1: f32, bc2: f32) {
        const B1: f32 = 0.9;
        const B2: f32 = 0.999;
        const EPS: f32 = 1e-8;
        for j in 0..self.w.len() {
            let g = self.g[j];
            if g != 0.0 || self.m[j] != 0.0 {
                self.m[j] = B1 * self.m[j] + (1.0 - B1) * g;
                self.v[j] = B2 * self.v[j] + (1.0 - B2) * g * g;
                let mh = self.m[j] / bc1;
                let vh = self.v[j] / bc2;
                self.w[j] -= lr * mh / (vh.sqrt() + EPS);
                self.g[j] = 0.0;
            }
        }
    }
}

/// One selective-SSM layer's parameters (weights stored `[out, in]` row-major).
#[derive(Debug)]
struct Layer {
    win: Tensor,  // [d, d]   input u
    wa: Tensor,   // [d, d]   decay logits
    ba: Tensor,   // [d]
    wg: Tensor,   // [d, d]   output gate (no bias)
    wmix: Tensor, // [d, d]   channel mix
    bmix: Tensor, // [d]
}

/// The per-bit output head: a one-hidden-layer MLP from `[hidden state h ; decode
/// features φ]` to a single bit logit. The hidden layer is what lets `h` (constant
/// within a token) *interact* with the varint prefix carried in φ — a linear head
/// could only add a per-token constant bias, disconnecting the recurrence from the
/// per-bit prediction. `w2` is zero-initialized so the head starts neutral (½) and
/// the mixer down-weights it until it learns; it ships no weights.
#[derive(Debug)]
struct Head {
    hh: usize,   // hidden width H
    w1h: Tensor, // [H, d]  hidden <- hidden state
    w1f: Tensor, // [H, F]  hidden <- decode features
    b1: Tensor,  // [H]
    w2: Tensor,  // [H]     logit <- hidden
    b2: Tensor,  // [1]
}

/// Preallocated BPTT window: per-slot recurrence activations (flat
/// `(slot*layers + li)*d + k`) plus per-slot head data — the next token's bits,
/// their decode features, and the forward logits, sized `slot*MB + i`.
#[derive(Debug)]
struct Window {
    len: usize,
    tok: Vec<usize>, // emb row consumed at each slot
    xin: Vec<f32>,
    u: Vec<f32>,
    a: Vec<f32>,
    gate: Vec<f32>,
    sprev: Vec<f32>,
    snew: Vec<f32>,
    xfinal: Vec<f32>,
    // Head data: the bits of the token predicted from `xfinal[slot]`.
    hn: Vec<usize>,   // number of head bits at each slot
    hbit: Vec<u8>,    // target bit
    hphi: Vec<f32>,   // decode features [slot*MB + i][F]
    hz: Vec<f32>,     // head hidden activations (post-tanh) [slot*MB + i][H]
    hlogit: Vec<f32>, // forward logit
}

impl Window {
    fn new(wmax: usize, layers: usize, d: usize, hh: usize) -> Self {
        let zl = vec![0.0f32; wmax * layers * d];
        Self {
            len: 0,
            tok: vec![0; wmax],
            xin: zl.clone(),
            u: zl.clone(),
            a: zl.clone(),
            gate: zl.clone(),
            sprev: zl.clone(),
            snew: zl,
            xfinal: vec![0.0; wmax * d],
            hn: vec![0; wmax],
            hbit: vec![0; wmax * MB],
            hphi: vec![0.0; wmax * MB * F],
            hz: vec![0.0; wmax * MB * hh],
            hlogit: vec![0.0; wmax * MB],
        }
    }
}

/// Stacked selective-SSM token model with a per-bit head, truncated-BPTT online
/// learning, and difficulty-routed replay (candidate #3).
#[derive(Debug)]
struct Sel {
    d: usize,
    layers: usize,
    emb: Tensor, // [EMB_ROWS, d]
    lyr: Vec<Layer>,
    head: Head,
    s: Vec<Vec<f32>>,       // persistent per-layer state [layers][d]
    s_start: Vec<Vec<f32>>, // state at the pending window's start (for replay re-forward)
    win: Window,
    hcur: usize, // head bits collected for the current building token
    t: i32,      // Adam step count
    lr: f32,
    // Replay (candidate #3).
    replay_max: usize,
    replay_thresh: f32,
    replay_lr_scale: f32, // extra-pass lr multiplier (< 1 = gentler replay)
    replay_abs: f32,      // absolute loss gate (nats/bit): replay only windows above it; 0 = use EMA-relative
    replay_warmup: usize, // warmup-only gate: replay only the first N windows, then stop; 0 = disabled
    windows_seen: usize,  // windows trained so far (drives the warmup gate)
    ema: f32,
    ema_init: bool,
    // per-step / backward scratch (preallocated)
    x: Vec<f32>,
    u: Vec<f32>,
    a: Vec<f32>,
    g: Vec<f32>,
    y: Vec<f32>,
    dx: Vec<f32>,
    dxin: Vec<f32>,
    dyl: Vec<f32>,
    dg: Vec<f32>,
    ds: Vec<f32>,
    da: Vec<f32>,
    du: Vec<f32>,
    dsp: Vec<f32>,
    dza: Vec<f32>,
    dzg: Vec<f32>,
    ds_next: Vec<f32>,
}

impl Sel {
    #[expect(clippy::too_many_arguments, reason = "Test-only experimental knobs; kept flat to avoid a config struct.")]
    fn new(
        d: usize,
        layers: usize,
        hh: usize,
        wmax: usize,
        lr: f32,
        replay_max: usize,
        replay_thresh: f32,
        replay_lr_scale: f32,
        replay_abs: f32,
        replay_warmup: usize,
    ) -> Self {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut rng = || {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
        };
        let mk = |n: usize, rng: &mut dyn FnMut() -> f32| Tensor::new(n, rng);
        let lyr = (0..layers)
            .map(|_| Layer {
                win: mk(d * d, &mut rng),
                wa: mk(d * d, &mut rng),
                ba: Tensor::zeros(d),
                wg: mk(d * d, &mut rng),
                wmix: mk(d * d, &mut rng),
                bmix: Tensor::zeros(d),
            })
            .collect();
        // Head: hidden weights small-random (so tanh units diversify), output `w2`
        // zero (neutral start).
        let head = Head {
            hh,
            w1h: mk(hh * d, &mut rng),
            w1f: mk(hh * F, &mut rng),
            b1: Tensor::zeros(hh),
            w2: Tensor::zeros(hh),
            b2: Tensor::zeros(1),
        };
        Self {
            d,
            layers,
            emb: mk(EMB_ROWS * d, &mut rng),
            lyr,
            head,
            s: vec![vec![0.0; d]; layers],
            s_start: vec![vec![0.0; d]; layers],
            win: Window::new(wmax, layers, d, hh),
            hcur: 0,
            t: 0,
            lr,
            replay_max,
            replay_thresh,
            replay_lr_scale,
            replay_abs,
            replay_warmup,
            windows_seen: 0,
            ema: 0.0,
            ema_init: false,
            x: vec![0.0; d],
            u: vec![0.0; d],
            a: vec![0.0; d],
            g: vec![0.0; d],
            y: vec![0.0; d],
            dx: vec![0.0; d],
            dxin: vec![0.0; d],
            dyl: vec![0.0; d],
            dg: vec![0.0; d],
            ds: vec![0.0; d],
            da: vec![0.0; d],
            du: vec![0.0; d],
            dsp: vec![0.0; d],
            dza: vec![0.0; d],
            dzg: vec![0.0; d],
            ds_next: vec![0.0; layers * d],
        }
    }

    /// Consume one token (via its `emb` row `tok`) through all layers, writing
    /// recurrence activations into the pending slot (`win.len`) and the residual
    /// output into `xfinal`. Advances the recurrent state. Snapshots the window-start
    /// state on the window's first consume (for replay).
    fn step_forward(&mut self, tok: usize) {
        let (d, layers) = (self.d, self.layers);
        let s = self.win.len;
        if s == 0 {
            for li in 0..layers {
                self.s_start[li].copy_from_slice(&self.s[li]);
            }
        }
        self.win.tok[s] = tok;
        self.x.copy_from_slice(&self.emb.w[tok * d..tok * d + d]);
        for li in 0..layers {
            let base = (s * layers + li) * d;
            self.win.xin[base..base + d].copy_from_slice(&self.x);
            let l = &self.lyr[li];
            for j in 0..d {
                self.u[j] = dot(&l.win.w[j * d..j * d + d], &self.x, 0.0);
                self.a[j] = sigmoid(dot(&l.wa.w[j * d..j * d + d], &self.x, l.ba.w[j]));
                self.g[j] = sigmoid(dot(&l.wg.w[j * d..j * d + d], &self.x, 0.0));
            }
            let st = &mut self.s[li];
            for k in 0..d {
                let sprev = st[k];
                let sn = self.a[k] * sprev + (1.0 - self.a[k]) * self.u[k];
                st[k] = sn;
                self.y[k] = self.g[k] * sn;
                self.win.sprev[base + k] = sprev;
                self.win.snew[base + k] = sn;
                self.win.u[base + k] = self.u[k];
                self.win.a[base + k] = self.a[k];
                self.win.gate[base + k] = self.g[k];
            }
            for j in 0..d {
                self.x[j] += dot(&l.wmix.w[j * d..j * d + d], &self.y, l.bmix.w[j]);
            }
        }
        self.win.xfinal[s * d..s * d + d].copy_from_slice(&self.x);
    }

    /// Forward the per-bit head: logit for `P(next bit == 1)` from the current
    /// building slot's hidden state and decode features `phi`. Stores `phi`/logit
    /// into the pending head buffer for BPTT. Called once per coded bit.
    fn head_predict(&mut self, phi: &[f32]) -> f32 {
        let (d, s, i) = (self.d, self.win.len, self.hcur);
        let hh = self.head.hh;
        let pbase = (s * MB + i) * F;
        self.win.hphi[pbase..pbase + F].copy_from_slice(phi);
        let h = &self.win.xfinal[s * d..s * d + d];
        let zbase = (s * MB + i) * hh;
        for u in 0..hh {
            let pre = dot(&self.head.w1h.w[u * d..u * d + d], h, self.head.b1.w[u])
                + dot(&self.head.w1f.w[u * F..u * F + F], phi, 0.0);
            self.win.hz[zbase + u] = pre.tanh();
        }
        let logit = dot(&self.head.w2.w, &self.win.hz[zbase..zbase + hh], self.head.b2.w[0]);
        self.win.hlogit[s * MB + i] = logit;
        logit
    }

    /// Record the actual bit for the head prediction just made.
    fn head_bit(&mut self, bit: u8) {
        let s = self.win.len;
        self.win.hbit[s * MB + self.hcur] = bit;
        self.hcur += 1;
    }

    /// Recompute a slot's head logits from the current weights and re-forwarded
    /// `xfinal` (used by replay, where weights changed since the live forward).
    fn recompute_head_logits(&mut self, s: usize) {
        let (d, hh) = (self.d, self.head.hh);
        for i in 0..self.win.hn[s] {
            let pbase = (s * MB + i) * F;
            let zbase = (s * MB + i) * hh;
            let h = &self.win.xfinal[s * d..s * d + d];
            for u in 0..hh {
                let pre = dot(&self.head.w1h.w[u * d..u * d + d], h, self.head.b1.w[u])
                    + dot(&self.head.w1f.w[u * F..u * F + F], &self.win.hphi[pbase..pbase + F], 0.0);
                self.win.hz[zbase + u] = pre.tanh();
            }
            self.win.hlogit[s * MB + i] = dot(&self.head.w2.w, &self.win.hz[zbase..zbase + hh], self.head.b2.w[0]);
        }
    }

    /// Reverse-scan BPTT over the window: per slot, the head backward (into the
    /// hidden state and head params), then the recurrence reverse scan.
    fn backward(&mut self) {
        let (d, layers) = (self.d, self.layers);
        self.ds_next.iter_mut().for_each(|x| *x = 0.0);
        for s in (0..self.win.len).rev() {
            // Head (MLP) backward: dloss/dlogit = σ(logit) − bit; through w2, the tanh
            // hidden layer, and w1h/w1f — accumulating grad into the hidden state (dx)
            // and every head param.
            let hh = self.head.hh;
            self.dx.iter_mut().for_each(|x| *x = 0.0);
            for i in 0..self.win.hn[s] {
                let bit = f32::from(self.win.hbit[s * MB + i]);
                let dlogit = sigmoid(self.win.hlogit[s * MB + i]) - bit;
                self.head.b2.g[0] += dlogit;
                let pbase = (s * MB + i) * F;
                let zbase = (s * MB + i) * hh;
                let xf = &self.win.xfinal[s * d..s * d + d];
                let phi = &self.win.hphi[pbase..pbase + F];
                for u in 0..hh {
                    let zu = self.win.hz[zbase + u];
                    self.head.w2.g[u] += dlogit * zu;
                    let dzu = dlogit * self.head.w2.w[u] * (1.0 - zu * zu); // tanh′
                    self.head.b1.g[u] += dzu;
                    axpy(&mut self.head.w1h.g[u * d..u * d + d], xf, dzu);
                    axpy(&mut self.head.w1f.g[u * F..u * F + F], phi, dzu);
                    for k in 0..d {
                        self.dx[k] += dzu * self.head.w1h.w[u * d + k];
                    }
                }
            }
            // Reverse through layers; `dx` carries grad w.r.t. the layer's output.
            for li in (0..layers).rev() {
                let base = (s * layers + li) * d;
                let nbase = li * d;
                for k in 0..d {
                    self.y[k] = self.win.gate[base + k] * self.win.snew[base + k];
                }
                self.dxin.copy_from_slice(&self.dx);
                self.dyl.iter_mut().for_each(|x| *x = 0.0);
                let l = &mut self.lyr[li];
                for j in 0..d {
                    let dm = self.dx[j];
                    l.bmix.g[j] += dm;
                    axpy(&mut l.wmix.g[j * d..j * d + d], &self.y, dm);
                    axpy(&mut self.dyl, &l.wmix.w[j * d..j * d + d], dm);
                }
                for k in 0..d {
                    let sn = self.win.snew[base + k];
                    let gk = self.win.gate[base + k];
                    self.dg[k] = self.dyl[k] * sn;
                    self.ds[k] = self.dyl[k] * gk + self.ds_next[nbase + k];
                }
                for k in 0..d {
                    let ak = self.win.a[base + k];
                    let sp = self.win.sprev[base + k];
                    let uk = self.win.u[base + k];
                    self.da[k] = self.ds[k] * (sp - uk);
                    self.du[k] = self.ds[k] * (1.0 - ak);
                    self.dsp[k] = self.ds[k] * ak;
                    self.dza[k] = self.da[k] * ak * (1.0 - ak);
                    let gk = self.win.gate[base + k];
                    self.dzg[k] = self.dg[k] * gk * (1.0 - gk);
                }
                let xin = &self.win.xin[base..base + d];
                for j in 0..d {
                    l.ba.g[j] += self.dza[j];
                    axpy(&mut l.wa.g[j * d..j * d + d], xin, self.dza[j]);
                    axpy(&mut self.dxin, &l.wa.w[j * d..j * d + d], self.dza[j]);
                    axpy(&mut l.wg.g[j * d..j * d + d], xin, self.dzg[j]);
                    axpy(&mut self.dxin, &l.wg.w[j * d..j * d + d], self.dzg[j]);
                    axpy(&mut l.win.g[j * d..j * d + d], xin, self.du[j]);
                    axpy(&mut self.dxin, &l.win.w[j * d..j * d + d], self.du[j]);
                }
                self.ds_next[nbase..nbase + d].copy_from_slice(&self.dsp);
                self.dx.copy_from_slice(&self.dxin);
            }
            let tok = self.win.tok[s];
            axpy(&mut self.emb.g[tok * d..tok * d + d], &self.dx, 1.0);
        }
    }

    fn adam_step(&mut self, lr: f32) {
        self.t += 1;
        let bc1 = 1.0 - 0.9f32.powi(self.t);
        let bc2 = 1.0 - 0.999f32.powi(self.t);
        self.emb.adam(lr, bc1, bc2);
        self.head.w1h.adam(lr, bc1, bc2);
        self.head.w1f.adam(lr, bc1, bc2);
        self.head.b1.adam(lr, bc1, bc2);
        self.head.w2.adam(lr, bc1, bc2);
        self.head.b2.adam(lr, bc1, bc2);
        for l in &mut self.lyr {
            l.win.adam(lr, bc1, bc2);
            l.wa.adam(lr, bc1, bc2);
            l.ba.adam(lr, bc1, bc2);
            l.wg.adam(lr, bc1, bc2);
            l.wmix.adam(lr, bc1, bc2);
            l.bmix.adam(lr, bc1, bc2);
        }
        self.win.len = 0;
    }

    /// The pending window's average head loss in nats/bit (before `backward`).
    fn window_nats(&self) -> f32 {
        let mut nats = 0.0f32;
        let mut nbits = 0usize;
        for s in 0..self.win.len {
            for i in 0..self.win.hn[s] {
                let bit = f32::from(self.win.hbit[s * MB + i]);
                let p = sigmoid(self.win.hlogit[s * MB + i]);
                let pc = if bit > 0.5 { p } else { 1.0 - p };
                nats -= pc.max(1e-30).ln();
                nbits += 1;
            }
        }
        nats / nbits.max(1) as f32
    }

    /// How many EXTRA replay passes this window earns. Two gates (deterministic on
    /// both sides): if `replay_abs > 0`, an **absolute** loss gate — replay only
    /// windows whose per-bit loss exceeds it (Fable's "multi-pass on poorly-predicted
    /// data": fires during warmup *and* on hard regions, quiet on easy warm data).
    /// Otherwise the EMA-relative gate (replay windows above the running average).
    fn replay_passes(&mut self, avg_nats: f32) -> usize {
        if self.replay_max == 0 {
            return 0;
        }
        self.windows_seen += 1;
        let hard = if self.replay_warmup > 0 {
            // Warmup-only: replay the first N windows, then stop — the benefit is a
            // fixed warmup prefix whose fraction (and cost) vanishes as data grows.
            self.windows_seen <= self.replay_warmup
        } else if self.replay_abs > 0.0 {
            avg_nats > self.replay_abs
        } else {
            if !self.ema_init {
                self.ema = avg_nats;
                self.ema_init = true;
            }
            let h = avg_nats > self.ema * self.replay_thresh;
            self.ema = 0.99 * self.ema + 0.01 * avg_nats;
            h
        };
        if hard { self.replay_max } else { 0 }
    }

    /// Train the pending window: one base BPTT+Adam pass, then a difficulty-gated
    /// number of extra passes re-run from the window's start state (candidate #3).
    /// Leaves the recurrent state where the live forward left it.
    fn learn_window(&mut self) {
        if self.win.len == 0 {
            return;
        }
        let wl = self.win.len;
        let extra = self.replay_passes(self.window_nats());
        TOTAL_WINDOWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if extra > 0 {
            REPLAYED_WINDOWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        if extra == 0 {
            self.backward();
            self.adam_step(self.lr);
            return;
        }

        let s_live = self.s.clone();
        let toks: Vec<usize> = self.win.tok[..wl].to_vec();

        // Base pass at full lr; extra replay passes at a (typically gentler) lr to
        // counter Adam over-stepping on the repeated, correlated window gradients.
        self.backward();
        self.adam_step(self.lr);
        let replay_lr = self.lr * self.replay_lr_scale;

        for _ in 0..extra {
            for li in 0..self.layers {
                self.s[li].copy_from_slice(&self.s_start[li]);
            }
            for j in 0..wl {
                self.win.len = j;
                self.step_forward(toks[j]);
                self.recompute_head_logits(j);
            }
            self.win.len = wl;
            self.backward();
            self.adam_step(replay_lr);
        }

        self.s.clone_from(&s_live);
        self.win.len = 0;
    }

    /// Finalize the current building slot (its head data now holds the completed
    /// token's bits), flush if the window is full, then consume the completed token
    /// `v` into the next slot.
    fn finalize_and_consume(&mut self, v: u32, w_win: usize) {
        self.win.hn[self.win.len] = self.hcur;
        self.win.len += 1;
        if self.win.len >= w_win {
            self.learn_window();
        }
        self.step_forward(emb_idx(v));
        self.hcur = 0;
    }
}

/// The mixer-facing arm: owns a [`Sel`], mirrors the ULEB128 parser to detect token
/// boundaries, and drives the per-bit head.
#[derive(Debug)]
pub(crate) struct SsmModel {
    sel: Sel,
    // Mirror of the shared u22 parser (the model's `update` runs before the driver
    // advances `Context`'s parser, so the arm keeps its own to detect completion).
    mcur_payload: u32,
    mcur_bytes: u8,
    bitbuf: u32,
    nbits: u8,
    w_win: usize,
}

impl SsmModel {
    fn replay_cfg() -> (usize, f32, f32, f32, usize) {
        let max = std::env::var("LZR_SSM_REPLAY").ok().and_then(|x| x.parse().ok()).unwrap_or(0);
        let thresh = std::env::var("LZR_SSM_REPLAY_THRESH").ok().and_then(|x| x.parse().ok()).unwrap_or(1.0);
        let lr_scale = std::env::var("LZR_SSM_REPLAY_LR").ok().and_then(|x| x.parse().ok()).unwrap_or(1.0);
        let abs = std::env::var("LZR_SSM_REPLAY_ABS").ok().and_then(|x| x.parse().ok()).unwrap_or(0.0);
        let warmup = std::env::var("LZR_SSM_REPLAY_WARMUP").ok().and_then(|x| x.parse().ok()).unwrap_or(0);
        (max, thresh, lr_scale, abs, warmup)
    }

    /// Deterministic constructor (fixed-seed init). Dims/window/lr via env for sweeps.
    pub(crate) fn arm() -> Self {
        let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|x| x.parse().ok()).unwrap_or(d);
        let d = env("LZR_SSM_D", 112);
        let layers = env("LZR_SSM_L", 4);
        let hh = env("LZR_SSM_HEAD_H", 128); // 128 beat 64/192/256 at the 2MB warm operating point
        let w_win = env("LZR_SSM_WIN", 16);
        let lr = std::env::var("LZR_SSM_LR").ok().and_then(|x| x.parse().ok()).unwrap_or(1e-3);
        Self::with_dims(d, layers, hh, w_win, lr)
    }

    fn with_dims(d: usize, layers: usize, hh: usize, w_win: usize, lr: f32) -> Self {
        let (replay_max, replay_thresh, replay_lr_scale, replay_abs, replay_warmup) = Self::replay_cfg();
        Self::with_dims_cfg(d, layers, hh, w_win, lr, replay_max, replay_thresh, replay_lr_scale, replay_abs, replay_warmup)
    }

    /// Explicit constructor (replay config passed in) — used by tests.
    #[expect(clippy::too_many_arguments, reason = "Test-only experimental knobs; kept flat to avoid a config struct.")]
    fn with_dims_cfg(
        d: usize,
        layers: usize,
        hh: usize,
        w_win: usize,
        lr: f32,
        replay_max: usize,
        replay_thresh: f32,
        replay_lr_scale: f32,
        replay_abs: f32,
        replay_warmup: usize,
    ) -> Self {
        let mut sel =
            Sel::new(d, layers, hh, w_win, lr, replay_max, replay_thresh, replay_lr_scale, replay_abs, replay_warmup);
        sel.step_forward(emb_idx(0)); // BOS: prime `xfinal[0]` for predicting the first token
        Self { sel, mcur_payload: 0, mcur_bytes: 0, bitbuf: 0, nbits: 0, w_win }
    }

    /// Mirror one finalized byte through the ULEB128 parser; returns the decoded
    /// token value when a token completes.
    fn mirror_advance(&mut self, b: u8) -> Option<u32> {
        let done = match self.mcur_bytes {
            0 => {
                if b & 0x80 == 0 {
                    Some(u32::from(b))
                } else {
                    self.mcur_payload = u32::from(b & 0x7f);
                    self.mcur_bytes = 1;
                    None
                }
            }
            1 => {
                if b & 0x80 == 0 {
                    Some(self.mcur_payload | (u32::from(b) << 7))
                } else {
                    self.mcur_payload |= u32::from(b & 0x7f) << 7;
                    self.mcur_bytes = 2;
                    None
                }
            }
            _ => Some(self.mcur_payload | (u32::from(b) << 14)),
        };
        if done.is_some() {
            self.mcur_payload = 0;
            self.mcur_bytes = 0;
        }
        done
    }
}

impl TokenModel for SsmModel {
    fn predict(&mut self, ctx: &Context, _hist: &[u8]) -> i32 {
        let phi = features(ctx);
        let logit = self.sel.head_predict(&phi);
        let q = ((sigmoid(logit) * 4096.0) as i32).clamp(1, 4095);
        stretch(q)
    }

    fn update(&mut self, _ctx: &Context, _hist: &[u8], bit: u8) {
        self.sel.head_bit(bit);
        self.bitbuf = (self.bitbuf << 1) | u32::from(bit);
        self.nbits += 1;
        if self.nbits == 8 {
            let b = (self.bitbuf & 0xff) as u8;
            if let Some(v) = self.mirror_advance(b) {
                self.sel.finalize_and_consume(v, self.w_win);
            }
            self.bitbuf = 0;
            self.nbits = 0;
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    /// Finite-difference gradient check over the recurrence AND the per-bit head:
    /// build a short token sequence with synthetic per-slot head bits/features, run
    /// forward+backward, and compare analytic grads to central differences.
    #[test]
    fn ssm_gradient_check() {
        fn tensor(net: &mut Sel, li: usize, sel: usize) -> &mut Tensor {
            match (li, sel) {
                (usize::MAX, 0) => &mut net.emb,
                (usize::MAX, 1) => &mut net.head.w1h,
                (usize::MAX, 2) => &mut net.head.w1f,
                (usize::MAX, 3) => &mut net.head.b1,
                (usize::MAX, 4) => &mut net.head.w2,
                (usize::MAX, _) => &mut net.head.b2,
                (_, 0) => &mut net.lyr[li].win,
                (_, 1) => &mut net.lyr[li].wa,
                (_, 2) => &mut net.lyr[li].ba,
                (_, 3) => &mut net.lyr[li].wg,
                (_, 4) => &mut net.lyr[li].wmix,
                (_, _) => &mut net.lyr[li].bmix,
            }
        }
        let (d, layers, hh, w) = (5usize, 3usize, 4usize, 4usize);
        let mut net = Sel::new(d, layers, hh, w, 1e-3, 0, 1.0, 1.0, 0.0, 0);
        // `w2` is zero-initialized (neutral start), which would make the hidden-layer
        // grads (dz ∝ w2) vacuously zero. Perturb the head off its neutral point and
        // the biases so every backward path carries signal.
        for (i, wv) in net.head.w2.w.iter_mut().enumerate() {
            *wv = 0.1 * (i as f32 + 1.0);
        }
        for (i, bv) in net.head.b1.w.iter_mut().enumerate() {
            *bv = 0.05 * (i as f32) - 0.1;
        }
        net.head.b2.w[0] = 0.2;
        let toks = [1usize, 7, 3, 5];
        // Per-slot synthetic head: (feature vector, target bit). 2 bits/slot.
        let mkphi = |seed: usize| -> [f32; F] {
            let mut p = [0.0f32; F];
            for (k, pk) in p.iter_mut().enumerate() {
                *pk = (((seed * 31 + k * 7) % 13) as f32 / 6.0) - 1.0;
            }
            p
        };
        let heads: Vec<Vec<([f32; F], u8)>> =
            (0..w).map(|j| vec![(mkphi(j * 2), (j % 2) as u8), (mkphi(j * 2 + 1), ((j + 1) % 2) as u8)]).collect();
        let s0: Vec<Vec<f32>> = vec![vec![0.0; d]; layers];

        let run_loss = |net: &mut Sel| -> f32 {
            net.s.clone_from(&s0);
            net.win.len = 0;
            let mut nats = 0.0f32;
            for j in 0..w {
                net.win.len = j;
                net.step_forward(toks[j]);
                net.win.hn[j] = heads[j].len();
                for (i, (phi, bit)) in heads[j].iter().enumerate() {
                    net.hcur = i; // index this slot's i-th head bit
                    let logit = net.head_predict(phi);
                    net.win.hbit[j * MB + i] = *bit;
                    let p = sigmoid(logit);
                    let pc = if *bit == 1 { p } else { 1.0 - p };
                    nats -= pc.max(1e-30).ln();
                }
            }
            net.win.len = w;
            nats
        };

        run_loss(&mut net);
        net.backward();

        let eps = 1e-2f32;
        let mut max_rel = 0.0f32;
        let mut targets: Vec<(&str, usize, usize)> = vec![
            ("emb", usize::MAX, 0),
            ("head.w1h", usize::MAX, 1),
            ("head.w1f", usize::MAX, 2),
            ("head.b1", usize::MAX, 3),
            ("head.w2", usize::MAX, 4),
            ("head.b2", usize::MAX, 5),
        ];
        for li in 0..layers {
            for (i, name) in ["win", "wa", "ba", "wg", "wmix", "bmix"].iter().enumerate() {
                targets.push((name, li, i));
            }
        }

        for (name, li, sel) in targets {
            let len = tensor(&mut net, li, sel).w.len();
            for &j in &[0usize, len / 3, len / 2, len - 1] {
                let analytic = tensor(&mut net, li, sel).g[j];
                let orig = tensor(&mut net, li, sel).w[j];
                tensor(&mut net, li, sel).w[j] = orig + eps;
                let lp = run_loss(&mut net);
                tensor(&mut net, li, sel).w[j] = orig - eps;
                let lm = run_loss(&mut net);
                tensor(&mut net, li, sel).w[j] = orig;
                let fd = (lp - lm) / (2.0 * eps);
                let abs = (fd - analytic).abs();
                let rel = abs / (analytic.abs().max(fd.abs()) + 1e-4);
                max_rel = max_rel.max(rel);
                assert!(
                    rel < 3e-2 || abs < 5e-4,
                    "{name}[{j}] (layer {li}): analytic {analytic:.5} vs fd {fd:.5} (rel {rel:.4}, abs {abs:.5})"
                );
            }
        }
        println!("token-ssm gradient check ok, max relative error {max_rel:.5}");
    }

    /// The default profile's entropy models — the baseline the arm's marginal is over.
    fn baseline() -> (Vec<crate::entropy::ModelBuilder>, Vec<&'static str>) {
        use crate::models::match_model::MatchModel;
        use crate::models::ordern::OrderN;
        use crate::models::sparse::SparseModel;
        use crate::models::varint::VarintModel;
        let builders: Vec<crate::entropy::ModelBuilder> = vec![
            |c| Box::new(OrderN::new(0, c)),
            |c| Box::new(OrderN::new(1, c)),
            |c| Box::new(OrderN::new(2, c)),
            |c| Box::new(SparseModel::new(&[2], c)),
            |c| Box::new(SparseModel::new(&[2, 4], c)),
            |_| Box::new(VarintModel::new()),
            |c| Box::new(MatchModel::new(c)),
        ];
        let names = vec!["order0", "order1", "order2", "sparse2", "sparse24", "varint", "match"];
        (builders, names)
    }

    /// The pre-entropy byte stream (casefold → entities → repair, entropy off). A
    /// `LZR_NTOK` env caps the Re-Pair vocabulary (the small-vocab operating point the
    /// `--auto` search favours); unset uses the default cost-stop grammar.
    fn preprocess(slice: &[u8]) -> Vec<u8> {
        use crate::codec::{EncodeOptions, Profile};
        let mut prof = Profile::default();
        prof.disable("entropy").unwrap();
        let opts = std::env::var("LZR_NTOK")
            .ok()
            .and_then(|x| x.parse().ok())
            .and_then(|n| EncodeOptions::new(n).ok())
            .unwrap_or_default();
        prof.compressor_with(opts).encode(slice.to_vec())
    }

    /// Byte-exact round-trip through the entropy coder with the baseline models plus
    /// the token-level SSM arm — replay (#3) ON — on a small enwik8 slice.
    #[test]
    fn ssm_roundtrip() {
        use crate::entropy::{EntropyCoder, ModelBuilder};
        use crate::transform::Transform;
        let Ok(e8) = std::fs::read("corpora/hutter/enwik8") else {
            return;
        };
        let pre = preprocess(&e8[1_000_000..1_004_000]);
        fn small_arm(_: usize) -> Box<dyn TokenModel> {
            Box::new(SsmModel::with_dims_cfg(16, 2, 8, 8, 1e-3, 2, 1.0, 0.5, 0.0, 0))
        }
        let (mut b, mut n) = baseline();
        b.push(small_arm as ModelBuilder);
        n.push("ssm");
        let coder = EntropyCoder::new(b, n, true);
        let coded = coder.forward(pre.clone());
        assert_eq!(coder.inverse(coded).unwrap(), pre, "token-SSM arm (with replay) must round-trip");
    }

    /// e2e marginal bpb over the baseline stack. Slice via `LZR_LO`/`LZR_HI`
    /// (default `0..2_000_000`); dims/replay via `LZR_SSM_*`. Run:
    /// `cargo test --release ssm_arm_e2e -- --ignored --nocapture`
    #[test]
    #[ignore = "online token-SSM arm: e2e marginal bpb on enwik8"]
    fn ssm_arm_e2e() {
        use crate::entropy::{EntropyCoder, ModelBuilder};
        use crate::transform::Transform;
        use std::time::Instant;
        let env = |k: &str, d: usize| -> usize { std::env::var(k).ok().and_then(|x| x.parse().ok()).unwrap_or(d) };
        let Ok(e8) = std::fs::read("corpora/hutter/enwik8") else {
            println!("enwik8 not found; skipping");
            return;
        };
        let lo = env("LZR_LO", 0);
        let hi = env("LZR_HI", 2_000_000).min(e8.len());
        let orig = (hi - lo) as f64;
        let pre = preprocess(&e8[lo..hi]);

        let (bb, bn) = baseline();
        let base = EntropyCoder::new(bb, bn, true).forward(pre.clone()).len();
        let base_bpb = base as f64 * 8.0 / orig;

        fn arm(_: usize) -> Box<dyn TokenModel> {
            Box::new(SsmModel::arm())
        }
        let (mut ab, mut an) = baseline();
        ab.push(arm as ModelBuilder);
        an.push("ssm");
        use std::sync::atomic::Ordering;
        TOTAL_WINDOWS.store(0, Ordering::Relaxed);
        REPLAYED_WINDOWS.store(0, Ordering::Relaxed);
        let t1 = Instant::now();
        let armed = EntropyCoder::new(ab, an, true).forward(pre.clone()).len();
        let secs = t1.elapsed().as_secs_f64();
        let arm_bpb = armed as f64 * 8.0 / orig;
        let e9_eta = (1e9 / orig) * secs / 3600.0;
        let (tw, rw) = (TOTAL_WINDOWS.load(Ordering::Relaxed), REPLAYED_WINDOWS.load(Ordering::Relaxed));
        let replay_pct = if tw > 0 { 100.0 * rw as f64 / tw as f64 } else { 0.0 };
        println!(
            "baseline {base_bpb:.4} | +token-SSM arm {arm_bpb:.4} | marginal {:+.4} bpb  \
             ({secs:.0}s on {:.1} MB; enwik9 ETA ~{e9_eta:.1} h/dir; replayed {rw}/{tw} windows = {replay_pct:.1}%)",
            arm_bpb - base_bpb,
            orig / 1e6
        );
    }
}
