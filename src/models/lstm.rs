//! Online-neural arm: a byte-level online LSTM that learns during compression
//! and joins the mixer as one more [`Model`].
//!
//! The LSTM steps once per finalized byte (vocab 256), producing a next-byte
//! distribution; each bit is predicted by marginalizing that distribution over
//! the partial byte `c0` via a prefix sum — the same bit-tree contract the
//! context models use. Online learning (truncated-BPTT window + Adam) runs at
//! each byte boundary, identically on encode and decode, so the arm ships **no
//! weights** — L(D) is just code. The dict-transformed stream already collapses
//! frequent words toward single code bytes, so a byte-level cell gets word-level
//! context for free without a large-vocabulary softmax.
//!
//! The backward pass is **gradient-checked** (finite differences,
//! `lstm_gradient_check`). The hot loop is allocation-free: all per-step scratch
//! and the BPTT window live in preallocated buffers reused across steps. The
//! offline `#[ignore]` tests (`lstm_arm_bpb`, `lstm_arm_e2e`) measure standalone
//! and e2e-marginal bpb on enwik8.

// Numeric kernel: pervasive index arithmetic and f32 casts; allowed module-wide
// to keep the LSTM forward/backward math readable.
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

fn dot(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 16;
    let mut acc = [0.0f32; LANES];
    let chunks = a.len() / LANES;
    for c in 0..chunks {
        let base = c * LANES;
        for (l, acc_l) in acc.iter_mut().enumerate() {
            *acc_l = a[base + l].mul_add(b[base + l], *acc_l);
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for i in (chunks * LANES)..a.len() {
        s = a[i].mul_add(b[i], s);
    }
    s
}

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

/// Preallocated BPTT window: forward activations for up to `wmax` steps, stored
/// in flat `slot*dim + k` arrays so the hot loop never allocates.
#[derive(Debug)]
struct Window {
    len: usize,
    tok: Vec<usize>,
    tgt: Vec<usize>,
    h_prev: Vec<f32>,
    c_prev: Vec<f32>,
    i: Vec<f32>,
    f: Vec<f32>,
    g: Vec<f32>,
    o: Vec<f32>,
    tc: Vec<f32>,
    h: Vec<f32>,
    p: Vec<f32>,
}

impl Window {
    fn new(wmax: usize, h: usize, v: usize) -> Self {
        let z = |n| vec![0.0f32; wmax * n];
        Self {
            len: 0,
            tok: vec![0; wmax],
            tgt: vec![0; wmax],
            h_prev: z(h),
            c_prev: z(h),
            i: z(h),
            f: z(h),
            g: z(h),
            o: z(h),
            tc: z(h),
            h: z(h),
            p: z(v),
        }
    }
}

/// Single-layer LSTM with a token embedding in and a full-vocabulary softmax
/// out. Gate layout in the `4h` vectors is `[i | f | g | o]`.
#[derive(Debug)]
struct Lstm {
    e: usize,
    h: usize,
    v: usize,
    emb: Tensor, // v * e (input lookup)
    wx: Tensor,  // 4h * e
    wh: Tensor,  // 4h * h
    b: Tensor,   // 4h
    wo: Tensor,  // v * h
    bo: Tensor,  // v
    h_prev: Vec<f32>,
    c_prev: Vec<f32>,
    win: Window,
    t: i32, // Adam step count
    // preallocated scratch
    a: Vec<f32>,
    c: Vec<f32>,
    dh: Vec<f32>,
    dc: Vec<f32>,
    dh_next: Vec<f32>,
    dc_next: Vec<f32>,
    da: Vec<f32>,
    dx: Vec<f32>,
    dh_prev: Vec<f32>,
}

impl Lstm {
    fn new(e: usize, h: usize, v: usize, wmax: usize) -> Self {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut rng = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
        };
        Self {
            e,
            h,
            v,
            emb: Tensor::new(v * e, &mut rng),
            wx: Tensor::new(4 * h * e, &mut rng),
            wh: Tensor::new(4 * h * h, &mut rng),
            b: Tensor::zeros(4 * h),
            wo: Tensor::new(v * h, &mut rng),
            bo: Tensor::zeros(v),
            h_prev: vec![0.0; h],
            c_prev: vec![0.0; h],
            win: Window::new(wmax, h, v),
            t: 0,
            a: vec![0.0; 4 * h],
            c: vec![0.0; h],
            dh: vec![0.0; h],
            dc: vec![0.0; h],
            dh_next: vec![0.0; h],
            dc_next: vec![0.0; h],
            da: vec![0.0; 4 * h],
            dx: vec![0.0; e],
            dh_prev: vec![0.0; h],
        }
    }

    /// Forward one token, returning the loss in **nats** (`-ln p[target]`).
    /// Writes activations into the next window slot and advances the state.
    /// Forward one step on `input`, writing activations into the pending window
    /// slot (`win.len`) and producing its next-token distribution into `win.p`.
    /// Advances the recurrent state but does NOT set a target or grow the window
    /// — `forward` (offline) and `observe` (online) finalize the slot.
    fn step_forward(&mut self, input: usize) {
        let (e, h, v) = (self.e, self.h, self.v);
        let s = self.win.len;
        let x = &self.emb.w[input * e..input * e + e];

        self.a.copy_from_slice(&self.b.w);
        for r in 0..4 * h {
            self.a[r] += dot(&self.wx.w[r * e..r * e + e], x)
                + dot(&self.wh.w[r * h..r * h + h], &self.h_prev);
        }
        let (hi, hh) = (h, 2 * h);
        for k in 0..h {
            let iv = sigmoid(self.a[k]);
            let fv = sigmoid(self.a[hi + k]);
            let gv = self.a[hh + k].tanh();
            let ov = sigmoid(self.a[3 * h + k]);
            let cv = fv * self.c_prev[k] + iv * gv;
            let tcv = cv.tanh();
            self.win.i[s * h + k] = iv;
            self.win.f[s * h + k] = fv;
            self.win.g[s * h + k] = gv;
            self.win.o[s * h + k] = ov;
            self.win.tc[s * h + k] = tcv;
            self.win.h[s * h + k] = ov * tcv;
            self.c[k] = cv;
        }

        let hslice = &self.win.h[s * h..s * h + h];
        let pslot = &mut self.win.p[s * v..s * v + v];
        pslot.copy_from_slice(&self.bo.w);
        for vi in 0..v {
            pslot[vi] += dot(&self.wo.w[vi * h..vi * h + h], hslice);
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
        self.win.tok[s] = input;
        self.win.h_prev[s * h..s * h + h].copy_from_slice(&self.h_prev);
        self.win.c_prev[s * h..s * h + h].copy_from_slice(&self.c_prev);
        self.h_prev.copy_from_slice(&self.win.h[s * h..s * h + h]);
        self.c_prev.copy_from_slice(&self.c);
    }

    /// Offline convenience: forward with a known target, finalizing the slot.
    /// Returns the loss in nats. Used by the offline measurement/gradient tests.
    #[cfg(test)]
    fn forward(&mut self, tok: usize, target: usize) -> f32 {
        let s = self.win.len;
        self.step_forward(tok);
        let nats = -(self.win.p[s * self.v + target].max(1e-30)).ln();
        self.win.tgt[s] = target;
        self.win.len += 1;
        nats
    }

    /// Online: write the prediction for the next token into the pending slot
    /// (no target yet, window not grown).
    fn predict_dist(&mut self, input: usize) {
        self.step_forward(input);
    }

    /// Online: finalize the pending slot with its now-known `target`; flush
    /// (truncated BPTT + Adam) once the window reaches `w_win`.
    fn observe(&mut self, target: usize, w_win: usize, lr: f32) {
        let s = self.win.len;
        self.win.tgt[s] = target;
        self.win.len += 1;
        if self.win.len >= w_win {
            self.backward();
            self.adam_step(lr);
        }
    }

    /// Truncated BPTT over the window → gradient accumulation (no update).
    fn backward(&mut self) {
        let (e, h, v) = (self.e, self.h, self.v);
        self.dh_next.iter_mut().for_each(|x| *x = 0.0);
        self.dc_next.iter_mut().for_each(|x| *x = 0.0);
        for s in (0..self.win.len).rev() {
            let hslice = &self.win.h[s * h..s * h + h];
            // output layer: reuse the saved p slot as dy = p - onehot
            let tgt = self.win.tgt[s];
            self.win.p[s * v + tgt] -= 1.0;
            self.dh.copy_from_slice(&self.dh_next);
            for vi in 0..v {
                let d = self.win.p[s * v + vi];
                axpy(&mut self.wo.g[vi * h..vi * h + h], hslice, d);
                axpy(&mut self.dh, &self.wo.w[vi * h..vi * h + h], d);
                self.bo.g[vi] += d;
            }
            // cell
            self.dc.copy_from_slice(&self.dc_next);
            for k in 0..h {
                let (iv, fv, gv, ov) = (
                    self.win.i[s * h + k],
                    self.win.f[s * h + k],
                    self.win.g[s * h + k],
                    self.win.o[s * h + k],
                );
                let tcv = self.win.tc[s * h + k];
                let do_ = self.dh[k] * tcv;
                let dck = self.dh[k] * ov * (1.0 - tcv * tcv) + self.dc[k];
                let df = dck * self.win.c_prev[s * h + k];
                let di = dck * gv;
                let dg = dck * iv;
                self.da[k] = di * iv * (1.0 - iv);
                self.da[h + k] = df * fv * (1.0 - fv);
                self.da[2 * h + k] = dg * (1.0 - gv * gv);
                self.da[3 * h + k] = do_ * ov * (1.0 - ov);
                self.dc[k] = dck;
            }
            // input / recurrent weights + back-prop to x, h_{t-1}, c_{t-1}
            let tok = self.win.tok[s];
            self.dx.iter_mut().for_each(|x| *x = 0.0);
            self.dh_prev.iter_mut().for_each(|x| *x = 0.0);
            for r in 0..4 * h {
                let d = self.da[r];
                axpy(
                    &mut self.wx.g[r * e..r * e + e],
                    &self.emb.w[tok * e..tok * e + e],
                    d,
                );
                axpy(&mut self.dx, &self.wx.w[r * e..r * e + e], d);
                axpy(
                    &mut self.wh.g[r * h..r * h + h],
                    &self.win.h_prev[s * h..s * h + h],
                    d,
                );
                axpy(&mut self.dh_prev, &self.wh.w[r * h..r * h + h], d);
                self.b.g[r] += d;
            }
            axpy(&mut self.emb.g[tok * e..tok * e + e], &self.dx, 1.0);
            for k in 0..h {
                self.dh_next[k] = self.dh_prev[k];
                self.dc_next[k] = self.dc[k] * self.win.f[s * h + k];
            }
        }
    }

    fn adam_step(&mut self, lr: f32) {
        self.t += 1;
        let bc1 = 1.0 - 0.9f32.powi(self.t);
        let bc2 = 1.0 - 0.999f32.powi(self.t);
        for t in [
            &mut self.emb,
            &mut self.wx,
            &mut self.wh,
            &mut self.b,
            &mut self.wo,
            &mut self.bo,
        ] {
            t.adam(lr, bc1, bc2);
        }
        self.win.len = 0;
    }

    #[cfg(test)]
    fn params(&self) -> usize {
        self.emb.w.len()
            + self.wx.w.len()
            + self.wh.w.len()
            + self.b.w.len()
            + self.wo.w.len()
            + self.bo.w.len()
    }
}

/// Byte-level online-LSTM arm as a mixer [`Model`]. The LSTM steps once per
/// finalized byte (vocab 256), producing a next-byte distribution; each bit is
/// then predicted by marginalizing that distribution over the partial byte
/// (`c0`) via a prefix sum — the same bit-tree contract the context models use.
/// Online learning (truncated BPTT + Adam) runs in `update` at each byte
/// boundary, identically on encode and decode, so it ships zero weights.
#[derive(Debug)]
pub(crate) struct ArmModel {
    lstm: Lstm,
    cumsum: Vec<f32>, // [257] prefix sums of the current next-byte distribution
    bitbuf: u32,
    nbits: u8,
    w_win: usize,
    lr: f32,
}

impl ArmModel {
    const E: usize = 32;
    const V: usize = 256;
    const WIN: usize = 16;
    /// Hidden width — the operating point. Trades bpb for throughput: the
    /// enwik8 e2e marginal/enwik9 ETA was h=64 −0.046/7.1h, h=96 −0.061/11.6h,
    /// h=128 −0.059(full)/16.9h. We ship h=128 for max gain and may decrease it
    /// for throughput headroom (h=96 keeps ~90% of the gain at 1.46×).
    pub(crate) const H: usize = 128;

    /// The shipped arm at the default hidden width [`Self::H`].
    pub(crate) fn arm() -> Self {
        Self::new(Self::H)
    }

    pub(crate) fn new(h: usize) -> Self {
        let mut lstm = Lstm::new(Self::E, h, Self::V, Self::WIN);
        lstm.predict_dist(0); // BOS → distribution for the first byte
        let mut m = Self {
            lstm,
            cumsum: vec![0.0; Self::V + 1],
            bitbuf: 0,
            nbits: 0,
            w_win: Self::WIN,
            lr: 1e-3,
        };
        m.refresh_cumsum();
        m
    }

    /// Prefix-sum the pending slot's next-byte distribution for O(1) bit ranges.
    fn refresh_cumsum(&mut self) {
        let (v, s) = (self.lstm.v, self.lstm.win.len);
        let mut acc = 0.0f32;
        self.cumsum[0] = 0.0;
        for i in 0..Self::V {
            acc += self.lstm.win.p[s * v + i];
            self.cumsum[i + 1] = acc;
        }
    }
}

impl Model for ArmModel {
    fn predict(&mut self, ctx: &Context) -> i32 {
        let bpos = ctx.bpos as usize;
        let prefix = (ctx.c0 - (1 << bpos)) as usize; // top `bpos` bits of the byte
        let shift = 8 - bpos;
        let lo = prefix << shift;
        let half = 1usize << (shift - 1);
        let p0 = self.cumsum[lo + half] - self.cumsum[lo];
        let p1 = self.cumsum[lo + (1 << shift)] - self.cumsum[lo + half];
        let pr = p1 / (p0 + p1 + 1e-12);
        #[allow(clippy::cast_possible_truncation)]
        let q = ((pr * 4096.0) as i32).clamp(1, 4095);
        crate::mixer::stretch(q)
    }

    fn update(&mut self, _ctx: &Context, bit: u8) {
        self.bitbuf = (self.bitbuf << 1) | u32::from(bit);
        self.nbits += 1;
        if self.nbits == 8 {
            let byte = (self.bitbuf & 0xff) as usize;
            self.lstm.observe(byte, self.w_win, self.lr); // finalize prediction of this byte
            self.lstm.predict_dist(byte); // predict the next byte
            self.refresh_cumsum();
            self.bitbuf = 0;
            self.nbits = 0;
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::items_after_statements,
    clippy::uninlined_format_args,
    clippy::doc_markdown
)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// Finite-difference gradient check on a tiny net — the reported bpb is only
    /// trustworthy if BPTT is correct. Perturbs sampled params of every tensor
    /// and compares the analytic gradient to `(L(+eps) - L(-eps)) / 2eps` of the
    /// window's total NLL (nats), holding the initial recurrent state fixed.
    #[test]
    fn lstm_gradient_check() {
        let (e, h, v, w) = (3usize, 4usize, 6usize, 4usize);
        let mut net = Lstm::new(e, h, v, w);
        let inps = [1usize, 4, 0, 3];
        let tgts = [2usize, 5, 1, 4];
        let h0 = vec![0.0f32; h];
        let c0 = vec![0.0f32; h];

        let run_loss = |net: &mut Lstm| -> f32 {
            net.h_prev.copy_from_slice(&h0);
            net.c_prev.copy_from_slice(&c0);
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
        let samples: [(&str, usize); 6] = [
            ("emb", net.emb.w.len()),
            ("wx", net.wx.w.len()),
            ("wh", net.wh.w.len()),
            ("b", net.b.w.len()),
            ("wo", net.wo.w.len()),
            ("bo", net.bo.w.len()),
        ];
        fn pick<'a>(net: &'a mut Lstm, name: &str) -> &'a mut Tensor {
            match name {
                "emb" => &mut net.emb,
                "wx" => &mut net.wx,
                "wh" => &mut net.wh,
                "b" => &mut net.b,
                "wo" => &mut net.wo,
                _ => &mut net.bo,
            }
        }
        for (name, len) in samples {
            for &j in &[0usize, len / 3, len / 2, len - 1] {
                let analytic = pick(&mut net, name).g[j];
                let orig = pick(&mut net, name).w[j];
                pick(&mut net, name).w[j] = orig + eps;
                let lp = run_loss(&mut net);
                pick(&mut net, name).w[j] = orig - eps;
                let lm = run_loss(&mut net);
                pick(&mut net, name).w[j] = orig;
                let fd = (lp - lm) / (2.0 * eps);
                let abs = (fd - analytic).abs();
                let rel = abs / (analytic.abs().max(fd.abs()) + 1e-4);
                max_rel = max_rel.max(rel);
                assert!(
                    rel < 3e-2 || abs < 5e-4,
                    "{name}[{j}]: analytic {analytic:.5} vs fd {fd:.5} (rel {rel:.4}, abs {abs:.5})"
                );
            }
        }
        // recompute grads after the FD perturbations disturbed the window
        run_loss(&mut net);
        net.backward();
        println!("gradient check ok, max relative error {max_rel:.5}");
    }

    /// Standalone online-LSTM next-token bpb on enwik8. Tokenizes the
    /// case-folded corpus with the shipped `words.dict` (top `LZR_N`, default
    /// all 2000) plus 256 byte-fallback tokens, runs the online learner, and
    /// reports bits-per-original-byte against v9's 1.6938. Slice via
    /// `LZR_LO`/`LZR_HI` (default `0..10_000_000`); `LZR_H` sets the hidden dim.
    /// Run: `cargo test --release lstm_arm_bpb -- --ignored --nocapture`
    #[test]
    #[ignore = "online-neural arm: standalone next-token bpb on enwik8"]
    fn lstm_arm_bpb() {
        use crate::preprocessors::Preprocessor;
        use crate::preprocessors::casefold::CaseFold;
        use std::collections::HashMap;

        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(d)
        };

        let Ok(words_txt) = std::fs::read_to_string("src/preprocessors/words.dict") else {
            return;
        };
        let n = env("LZR_N", 2000);
        let words: Vec<&[u8]> = words_txt
            .lines()
            .map(str::as_bytes)
            .filter(|w| !w.is_empty() && w.iter().all(u8::is_ascii_lowercase))
            .take(n)
            .collect();
        let word_id: HashMap<&[u8], usize> = words
            .iter()
            .enumerate()
            .map(|(i, w)| (*w, 256 + i))
            .collect();
        let v = 256 + words.len();

        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let lo = env("LZR_LO", 0);
        let hi = env("LZR_HI", 10_000_000).min(e8.len());
        let orig = (hi - lo) as f64;
        let folded = CaseFold.forward(&e8[lo..hi]);

        let mut seq: Vec<usize> = Vec::with_capacity(folded.len());
        let mut i = 0;
        while i < folded.len() {
            if folded[i].is_ascii_lowercase() {
                let s = i;
                while i < folded.len() && folded[i].is_ascii_lowercase() {
                    i += 1;
                }
                if let Some(&id) = word_id.get(&folded[s..i]) {
                    seq.push(id);
                } else {
                    seq.extend(folded[s..i].iter().map(|&b| b as usize));
                }
            } else {
                seq.push(folded[i] as usize);
                i += 1;
            }
        }

        let (e, h, w_win) = (64usize, env("LZR_H", 128), 16usize);
        let lr = 1e-3f32;
        let mut net = Lstm::new(e, h, v, w_win);
        println!(
            "online-LSTM arm: e={e} h={h} vocab={v} (N={}) params={} | {} tokens over {} bytes ({:.2} bytes/token)",
            words.len(),
            net.params(),
            seq.len(),
            hi - lo,
            orig / seq.len() as f64
        );

        let t0 = Instant::now();
        let mut total_nats = 0.0f64;
        for idx in 0..seq.len() {
            let inp = if idx == 0 { 0 } else { seq[idx - 1] };
            total_nats += f64::from(net.forward(inp, seq[idx]));
            if net.win.len >= w_win {
                net.backward();
                net.adam_step(lr);
            }
            if (idx + 1) % 1_000_000 == 0 {
                use std::io::Write as _;
                let secs = t0.elapsed().as_secs_f64();
                println!(
                    "  {}M tok | running bpb {:.4} | {:.0} tok/s",
                    (idx + 1) / 1_000_000,
                    total_nats
                        / std::f64::consts::LN_2
                        / (orig * (idx + 1) as f64 / seq.len() as f64),
                    (idx + 1) as f64 / secs
                );
                let _ = std::io::stdout().flush();
            }
        }
        if net.win.len > 0 {
            net.backward();
            net.adam_step(lr);
        }
        let secs = t0.elapsed().as_secs_f64();

        let bpb = total_nats / std::f64::consts::LN_2 / orig;
        let tps = seq.len() as f64 / secs;
        let e9_eta = (1e9 / orig) * seq.len() as f64 / tps / 3600.0;
        println!(
            "\n  bpb (per original byte) = {bpb:.4}   [v9 deterministic = 1.6938]\n  \
             {:.0} tok/s, {secs:.0}s for the slice; enwik9 ETA ~ {e9_eta:.1} h/direction\n  \
             NOTE: online bpb includes cold-start; a larger slice lowers it.",
            tps
        );
    }

    fn baseline_models() -> Vec<Box<dyn Model>> {
        use crate::models::context::ContextModel;
        use crate::models::match_model::MatchModel;
        let mut v: Vec<Box<dyn Model>> = (0..=6)
            .map(|n| Box::new(ContextModel::new(n)) as Box<dyn Model>)
            .collect();
        v.push(Box::new(ContextModel::word()));
        v.push(Box::new(MatchModel::new()));
        v
    }

    /// Round-trip correctness of the byte-level arm inside the real codec:
    /// encode an enwik8 slice (full pipeline) with the 9 baseline models + arm,
    /// decode with a fresh identical set, assert byte-exact. Online learning is
    /// deterministic (same scalar updates both directions), so this must hold.
    #[test]
    fn arm_roundtrip() {
        use crate::codec::{code_stream_models, decode_stream_models};
        use crate::preprocessors::Pipeline;

        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        // Small slice: keeps the debug coverage build fast while still
        // exercising encode→decode of the arm byte-for-byte.
        let data = Pipeline::default_pipeline().forward(&e8[1_000_000..1_004_000]);
        let mut enc_models = baseline_models();
        enc_models.push(Box::new(ArmModel::new(64)));
        let coded = code_stream_models(enc_models, &data);
        let mut dec_models = baseline_models();
        dec_models.push(Box::new(ArmModel::new(64)));
        let decoded = decode_stream_models(dec_models, &coded);
        assert_eq!(decoded, data, "arm codec must round-trip byte-exact");
    }

    /// e2e marginal value: encode a real enwik8 slice (full pipeline) with the 9
    /// baseline models, then with the byte-level arm added, and report both bpb
    /// and the delta. This is the number that decides the arm — its marginal
    /// contribution on top of the existing stack (decorrelation), not standalone.
    /// `LZR_LO`/`LZR_HI` (default `0..5_000_000`), `LZR_H` (hidden dim, default
    /// 128). Run: `cargo test --release lstm_arm_e2e -- --ignored --nocapture`
    #[test]
    #[ignore = "online-neural arm: e2e marginal bpb in the mixer on enwik8"]
    fn lstm_arm_e2e() {
        use crate::codec::code_stream_models;
        use crate::preprocessors::Pipeline;

        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(d)
        };
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let lo = env("LZR_LO", 0);
        let hi = env("LZR_HI", 5_000_000).min(e8.len());
        let orig = (hi - lo) as f64;
        let data = Pipeline::default_pipeline().forward(&e8[lo..hi]);
        let h = env("LZR_H", 128);

        let t0 = Instant::now();
        let base = code_stream_models(baseline_models(), &data).len();
        let base_secs = t0.elapsed().as_secs_f64();
        let base_bpb = base as f64 * 8.0 / orig;
        println!("baseline (9 models): {base_bpb:.4} bpb  ({base_secs:.0}s)");

        let mut withv = baseline_models();
        withv.push(Box::new(ArmModel::new(h)));
        let t1 = Instant::now();
        let arm = code_stream_models(withv, &data).len();
        let arm_secs = t1.elapsed().as_secs_f64();
        let arm_bpb = arm as f64 * 8.0 / orig;
        let e9_eta = (1e9 / orig) * arm_secs / 3600.0;
        println!(
            "+ byte-LSTM arm (h={h}): {arm_bpb:.4} bpb  ({arm_secs:.0}s)\n  \
             marginal delta = {:+.4} bpb   [baseline {base_bpb:.4} on this slice]\n  \
             arm-codec enwik9 ETA ~ {e9_eta:.1} h/direction (this slice scaled)",
            arm_bpb - base_bpb
        );
    }
}
