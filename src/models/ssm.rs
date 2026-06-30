//! Online-neural arm: a byte-level selective-SSM (SEL) that learns during
//! compression and joins the mixer as one more [`Model`], like the LSTM arm.
//!
//! Per finalized byte the net runs one step through `L` stacked selective-SSM
//! layers and emits a next-byte distribution; each bit is predicted by
//! marginalizing that distribution over the partial byte `c0` (the same bit-tree
//! contract the LSTM arm and the context models use). Online learning (truncated
//! BPTT + Adam) runs at each byte boundary, identically on encode and decode, so
//! the arm ships **no weights** — `L(D)` is just code.
//!
//! One layer (per-channel, input-dependent gated convex-combination recurrence):
//! `u = win·x`, `a = σ(wa·x + ba)`, `g = σ(wg·x)`, `s ← a⊙s + (1−a)⊙u`,
//! `y = g⊙s`, `x ← x + (wmix·y + bmix)` (residual). After `L` layers,
//! `logits = wo·x + bo` (vocab 256). The element-wise scan has a clean reverse-scan
//! backward (`ds_next = a⊙ds`), gradient-checked by finite differences
//! (`ssm_gradient_check`). The Stage-0 bake-off found this beats the LSTM arm
//! per-FLOP and grows with scale; the scalar forward was verified against burn.
//!
//! Opt-in behind the `ssm` feature — NOT shipped (same policy as the LSTM `arm`).

// Numeric kernel: pervasive index arithmetic and f32 casts; allowed module-wide
// to keep the SSM forward/backward math readable.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suboptimal_flops
)]

use crate::models::{Context, Model};

/// `bias + Σ w[i]·x[i]` with 16 independent accumulators so the `mul_add` chain is
/// throughput-bound (auto-vectorizes to NEON `fmla`), not latency-bound. Weights
/// are stored `[out, in]` row-major, so each output reads a contiguous row.
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
        Self {
            w: (0..n).map(|_| rng()).collect(),
            m: vec![0.0; n],
            v: vec![0.0; n],
            g: vec![0.0; n],
        }
    }
    fn zeros(n: usize) -> Self {
        Self {
            w: vec![0.0; n],
            m: vec![0.0; n],
            v: vec![0.0; n],
            g: vec![0.0; n],
        }
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

/// Preallocated BPTT window: forward activations for up to `wmax` steps. Per-slot
/// per-layer arrays are flat `(slot*layers + li)*d + k`; the saved softmax is
/// `slot*v + vi`. The reverse scan reads `xin, u, a, g, sprev, snew` per (slot,
/// layer), `xfinal` per slot (output-layer input), and reuses `p` as `dy`.
#[derive(Debug)]
struct Window {
    len: usize,
    tok: Vec<usize>,
    tgt: Vec<usize>,
    xin: Vec<f32>,
    u: Vec<f32>,
    a: Vec<f32>,
    gate: Vec<f32>,
    sprev: Vec<f32>,
    snew: Vec<f32>,
    xfinal: Vec<f32>,
    p: Vec<f32>,
}

impl Window {
    fn new(wmax: usize, layers: usize, d: usize, v: usize) -> Self {
        let zl = vec![0.0f32; wmax * layers * d];
        Self {
            len: 0,
            tok: vec![0; wmax],
            tgt: vec![0; wmax],
            xin: zl.clone(),
            u: zl.clone(),
            a: zl.clone(),
            gate: zl.clone(),
            sprev: zl.clone(),
            snew: zl,
            xfinal: vec![0.0; wmax * d],
            p: vec![0.0; wmax * v],
        }
    }
}

/// Stacked selective-SSM next-byte model with truncated-BPTT online learning.
#[derive(Debug)]
struct Sel {
    d: usize,
    layers: usize,
    v: usize,
    emb: Tensor, // [v, d]
    lyr: Vec<Layer>,
    wo: Tensor,       // [v, d]
    bo: Tensor,       // [v]
    s: Vec<Vec<f32>>, // persistent per-layer state [layers][d]
    win: Window,
    t: i32, // Adam step count
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
    ds_next: Vec<f32>, // [layers*d] recurrence grad carried across the reverse scan
}

impl Sel {
    fn new(d: usize, layers: usize, v: usize, wmax: usize) -> Self {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut rng = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
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
        Self {
            d,
            layers,
            v,
            emb: mk(v * d, &mut rng),
            lyr,
            wo: mk(v * d, &mut rng),
            bo: Tensor::zeros(v),
            s: vec![vec![0.0; d]; layers],
            win: Window::new(wmax, layers, d, v),
            t: 0,
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

    /// Forward one byte through all layers, writing activations into the pending
    /// window slot (`win.len`) and the next-byte softmax into `win.p`. Advances the
    /// recurrent state. Does NOT set a target or grow the window.
    fn step_forward(&mut self, tok: usize) {
        let (d, layers, v) = (self.d, self.layers, self.v);
        let s = self.win.len;
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
        let pslot = &mut self.win.p[s * v..s * v + v];
        for vi in 0..v {
            pslot[vi] = dot(&self.wo.w[vi * d..vi * d + d], &self.x, self.bo.w[vi]);
        }
        let max = pslot.iter().copied().fold(f32::MIN, f32::max);
        let mut sum = 0.0f32;
        for pv in pslot.iter_mut() {
            *pv = (*pv - max).exp();
            sum += *pv;
        }
        let inv = 1.0 / sum;
        for pv in pslot.iter_mut() {
            *pv *= inv;
        }
    }

    /// Reverse-scan BPTT over the window, accumulating grads into every tensor.
    fn backward(&mut self) {
        let (d, layers, v) = (self.d, self.layers, self.v);
        self.ds_next.iter_mut().for_each(|x| *x = 0.0);
        for s in (0..self.win.len).rev() {
            // Output layer: dy = p − onehot (in place), backprop to xfinal.
            let tgt = self.win.tgt[s];
            self.win.p[s * v + tgt] -= 1.0;
            self.dx.iter_mut().for_each(|x| *x = 0.0);
            let xfinal = &self.win.xfinal[s * d..s * d + d];
            for vi in 0..v {
                let dyv = self.win.p[s * v + vi];
                self.bo.g[vi] += dyv;
                axpy(&mut self.wo.g[vi * d..vi * d + d], xfinal, dyv);
                axpy(&mut self.dx, &self.wo.w[vi * d..vi * d + d], dyv);
            }
            // Reverse through layers; `dx` carries grad w.r.t. the layer's output.
            for li in (0..layers).rev() {
                let base = (s * layers + li) * d;
                let nbase = li * d;
                // Recompute y = g ⊙ snew (cheaper to recompute than tape).
                for k in 0..d {
                    self.y[k] = self.win.gate[base + k] * self.win.snew[base + k];
                }
                // mixed grads: dout == self.dx. dxin starts at the residual path.
                self.dxin.copy_from_slice(&self.dx);
                self.dyl.iter_mut().for_each(|x| *x = 0.0);
                let l = &mut self.lyr[li];
                for j in 0..d {
                    let dm = self.dx[j];
                    l.bmix.g[j] += dm;
                    axpy(&mut l.wmix.g[j * d..j * d + d], &self.y, dm);
                    axpy(&mut self.dyl, &l.wmix.w[j * d..j * d + d], dm);
                }
                // y = g ⊙ snew → dg, ds (plus the recurrence carry ds_next).
                for k in 0..d {
                    let sn = self.win.snew[base + k];
                    let gk = self.win.gate[base + k];
                    self.dg[k] = self.dyl[k] * sn;
                    self.ds[k] = self.dyl[k] * gk + self.ds_next[nbase + k];
                }
                // s = a⊙sprev + (1−a)⊙u → da, du, ds_prev (carry).
                for k in 0..d {
                    let ak = self.win.a[base + k];
                    let sp = self.win.sprev[base + k];
                    let uk = self.win.u[base + k];
                    self.da[k] = self.ds[k] * (sp - uk);
                    self.du[k] = self.ds[k] * (1.0 - ak);
                    self.dsp[k] = self.ds[k] * ak;
                    // gate pre-activation grads (σ′ = a(1−a); g′ = g(1−g)).
                    self.dza[k] = self.da[k] * ak * (1.0 - ak);
                    let gk = self.win.gate[base + k];
                    self.dzg[k] = self.dg[k] * gk * (1.0 - gk);
                }
                // weight grads + backprop into xin (over the saved layer input).
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
            // dx is now grad w.r.t. emb[tok]; scatter into the touched row.
            let tok = self.win.tok[s];
            axpy(&mut self.emb.g[tok * d..tok * d + d], &self.dx, 1.0);
        }
    }

    fn adam_step(&mut self, lr: f32) {
        self.t += 1;
        let bc1 = 1.0 - 0.9f32.powi(self.t);
        let bc2 = 1.0 - 0.999f32.powi(self.t);
        self.emb.adam(lr, bc1, bc2);
        self.wo.adam(lr, bc1, bc2);
        self.bo.adam(lr, bc1, bc2);
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

    fn learn_window(&mut self, lr: f32) {
        if self.win.len == 0 {
            return;
        }
        self.backward();
        self.adam_step(lr);
    }

    /// Online: forward one byte into the pending slot (no target yet).
    fn predict_dist(&mut self, tok: usize) {
        self.step_forward(tok);
    }

    /// Online: finalize the pending slot with its now-known target; flush (BPTT +
    /// Adam) once the window reaches `w_win`.
    fn observe(&mut self, target: usize, w_win: usize, lr: f32) {
        let s = self.win.len;
        self.win.tgt[s] = target;
        self.win.len += 1;
        if self.win.len >= w_win {
            self.learn_window(lr);
        }
    }

    /// Offline convenience for the gradient check: forward with a known target,
    /// finalizing the slot; returns the loss in nats.
    #[cfg(test)]
    fn forward(&mut self, tok: usize, target: usize) -> f32 {
        let s = self.win.len;
        self.step_forward(tok);
        let nats = -(self.win.p[s * self.v + target].max(1e-30)).ln();
        self.win.tgt[s] = target;
        self.win.len += 1;
        nats
    }
}

/// The mixer-facing arm: owns a [`Sel`], assembles coded bits into bytes, and
/// marginalizes the per-byte distribution to per-bit stretched logits.
#[derive(Debug)]
pub(crate) struct SsmModel {
    sel: Sel,
    cumsum: Vec<f32>, // [v+1] prefix sum of the current next-byte distribution
    bitbuf: u32,
    nbits: u8,
    w_win: usize,
    lr: f32,
}

impl SsmModel {
    const V: usize = 256;

    /// Deterministic constructor (fixed-seed init; both directions build identical
    /// state). Width/depth/window overridable via env for offline sweeps only.
    pub(crate) fn arm() -> Self {
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(d)
        };
        let d = env("LZR_SSM_D", 112);
        let layers = env("LZR_SSM_L", 4);
        let w_win = env("LZR_SSM_WIN", 16);
        let lr = std::env::var("LZR_SSM_LR")
            .ok()
            .and_then(|x| x.parse().ok())
            .unwrap_or(1e-3);
        Self::with_dims(d, layers, w_win, lr)
    }

    fn with_dims(d: usize, layers: usize, w_win: usize, lr: f32) -> Self {
        let mut sel = Sel::new(d, layers, Self::V, w_win);
        sel.predict_dist(0); // BOS: prime the first byte's distribution
        let mut m = Self {
            sel,
            cumsum: vec![0.0; Self::V + 1],
            bitbuf: 0,
            nbits: 0,
            w_win,
            lr,
        };
        m.refresh_cumsum();
        m
    }

    fn refresh_cumsum(&mut self) {
        let (v, s) = (self.sel.v, self.sel.win.len);
        let mut acc = 0.0f32;
        self.cumsum[0] = 0.0;
        for i in 0..Self::V {
            acc += self.sel.win.p[s * v + i];
            self.cumsum[i + 1] = acc;
        }
    }
}

impl Model for SsmModel {
    fn predict(&mut self, ctx: &Context) -> i32 {
        let bpos = ctx.bpos as usize;
        let prefix = (ctx.c0 - (1 << bpos)) as usize;
        let shift = 8 - bpos;
        let lo = prefix << shift;
        let half = 1usize << (shift - 1);
        let p0 = self.cumsum[lo + half] - self.cumsum[lo];
        let p1 = self.cumsum[lo + (1 << shift)] - self.cumsum[lo + half];
        let pr = p1 / (p0 + p1 + 1e-12);
        let q = ((pr * 4096.0) as i32).clamp(1, 4095);
        crate::mixer::stretch(q)
    }

    fn update(&mut self, _ctx: &Context, bit: u8) {
        self.bitbuf = (self.bitbuf << 1) | u32::from(bit);
        self.nbits += 1;
        if self.nbits == 8 {
            let byte = (self.bitbuf & 0xff) as usize;
            self.sel.observe(byte, self.w_win, self.lr); // finalize learning for this byte
            self.sel.predict_dist(byte); // produce the next byte's distribution
            self.refresh_cumsum();
            self.bitbuf = 0;
            self.nbits = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Finite-difference gradient check (the Stage-2 correctness gate): a
    /// wrong-but-consistent gradient still round-trips byte-exact, so this — not the
    /// round-trip test — verifies the backward. Multi-layer so the cross-layer
    /// residual and the inter-layer grad path are exercised.
    #[test]
    fn ssm_gradient_check() {
        // Borrow-friendly tensor selector: `li == usize::MAX` picks a global tensor,
        // else layer `li`'s `sel`-th tensor. Each call is a short, non-overlapping
        // borrow (no unsafe), so it composes with `run_loss(&mut net)`.
        fn tensor(net: &mut Sel, li: usize, sel: usize) -> &mut Tensor {
            match (li, sel) {
                (usize::MAX, 0) => &mut net.emb,
                (usize::MAX, 1) => &mut net.wo,
                (usize::MAX, _) => &mut net.bo,
                (_, 0) => &mut net.lyr[li].win,
                (_, 1) => &mut net.lyr[li].wa,
                (_, 2) => &mut net.lyr[li].ba,
                (_, 3) => &mut net.lyr[li].wg,
                (_, 4) => &mut net.lyr[li].wmix,
                (_, _) => &mut net.lyr[li].bmix,
            }
        }
        let (d, layers, v, w) = (5usize, 3usize, 6usize, 4usize);
        let mut net = Sel::new(d, layers, v, w);
        let inps = [1usize, 4, 0, 3];
        let tgts = [2usize, 5, 1, 4];
        let s0: Vec<Vec<f32>> = vec![vec![0.0; d]; layers];

        let run_loss = |net: &mut Sel| -> f32 {
            net.s.clone_from(&s0);
            net.win.len = 0;
            let mut nats = 0.0;
            for k in 0..w {
                nats += net.forward(inps[k], tgts[k]);
            }
            nats
        };

        run_loss(&mut net);
        net.backward();

        let eps = 1e-2f32;
        let mut max_rel = 0.0f32;
        let mut targets: Vec<(&str, usize, usize)> = vec![
            ("emb", usize::MAX, 0),
            ("wo", usize::MAX, 1),
            ("bo", usize::MAX, 2),
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
        run_loss(&mut net);
        net.backward();
        println!("ssm gradient check ok, max relative error {max_rel:.5}");
    }

    /// Byte-exact round-trip inside the real codec: encode an enwik8 slice with the
    /// baseline models + SEL arm, decode with a fresh identical set, assert equal.
    #[test]
    fn ssm_roundtrip() {
        use crate::codec::{baseline_models, code_stream_models, decode_stream_models};
        use crate::preprocessors::Pipeline;
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let data = Pipeline::default_pipeline().forward(&e8[1_000_000..1_004_000]);
        let mut enc = baseline_models(data.len());
        enc.push(SsmModel::with_dims(16, 2, 8, 1e-3).into());
        let coded = code_stream_models(enc, &data);
        let mut dec = baseline_models(data.len());
        dec.push(SsmModel::with_dims(16, 2, 8, 1e-3).into());
        assert_eq!(
            decode_stream_models(dec, &coded),
            data,
            "SEL arm must round-trip"
        );
    }
}
