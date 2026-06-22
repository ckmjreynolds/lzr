//! Throughput spike for the online-neural arm (CPU, scalar / auto-vec).
//!
//! The only question this answers: how many bytes/sec can a hand-rolled LSTM
//! cell do **forward + per-step backward + periodic SGD update** on one core, so
//! an enwik9 ETA can be projected against the Hutter time limit *before*
//! committing to an architecture. Per the project's "compute spike first" call,
//! this kills the online-neural arm fast if a size that could plausibly help
//! cannot run within budget single-threaded.
//!
//! Scope and honesty: the forward pass is a real LSTM cell + softmax. The
//! backward is **FLOP-faithful, not gradient-verified** — it performs exactly
//! the matmuls, outer-product accumulations, and elementwise gate derivatives a
//! correct truncated-BPTT step would, in the same memory-access pattern, so the
//! measured throughput is representative. Numerical gradient correctness is
//! deferred to the architecture-search phase (it only matters once the spike
//! says the compute fits). The data is synthetic (an LCG byte stream) because
//! compute is content-independent — this avoids any corpus dependency.

#![cfg(test)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::doc_markdown
)]

/// Multi-accumulator f32 dot product (`LANES=16`, `mul_add` → NEON `fmla.4s`),
/// the project's standard auto-vec kernel.
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

/// `dst[i] += src[i] * k` — the vectorizable axpy used for both gradient
/// accumulation (`dst = grad`) and back-propagated-input accumulation
/// (`dst = d_input`).
fn axpy(dst: &mut [f32], src: &[f32], k: f32) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d = s.mul_add(k, *d);
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// A single LSTM layer with a tied byte embedding in and a 256-way softmax out —
/// the smallest thing whose per-byte FLOPs and memory pattern match what the
/// real arm would ship. Gate layout in the `4h` pre-activation vector is
/// `[i | f | g | o]`.
struct Cell {
    e: usize,
    h: usize,
    emb: Vec<f32>, // [256 * e]
    wx: Vec<f32>,  // [4h * e]
    wh: Vec<f32>,  // [4h * h]
    b: Vec<f32>,   // [4h]
    wo: Vec<f32>,  // [256 * h]
    bo: Vec<f32>,  // [256]
    // gradient accumulators (cleared each update window)
    gwx: Vec<f32>,
    gwh: Vec<f32>,
    gb: Vec<f32>,
    gwo: Vec<f32>,
    gbo: Vec<f32>,
    // recurrent state
    h_prev: Vec<f32>,
    c_prev: Vec<f32>,
    c_cur: Vec<f32>,
}

impl Cell {
    fn new(e: usize, h: usize) -> Self {
        // Small deterministic LCG init — values are irrelevant to throughput.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut rng = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.1
        };
        let v = |n: usize, rng: &mut dyn FnMut() -> f32| (0..n).map(|_| rng()).collect::<Vec<_>>();
        Self {
            e,
            h,
            emb: v(256 * e, &mut rng),
            wx: v(4 * h * e, &mut rng),
            wh: v(4 * h * h, &mut rng),
            b: vec![0.0; 4 * h],
            wo: v(256 * h, &mut rng),
            bo: vec![0.0; 256],
            gwx: vec![0.0; 4 * h * e],
            gwh: vec![0.0; 4 * h * h],
            gb: vec![0.0; 4 * h],
            gwo: vec![0.0; 256 * h],
            gbo: vec![0.0; 256],
            h_prev: vec![0.0; h],
            c_prev: vec![0.0; h],
            c_cur: vec![0.0; h],
        }
    }

    /// One forward+backward step on input byte `inb` predicting target byte
    /// `tgt`. Mutates state and accumulates gradients. Returns nothing — the
    /// spike measures time, not bits.
    fn step(&mut self, inb: u8, tgt: u8) {
        let (e, h) = (self.e, self.h);
        let x = &self.emb[inb as usize * e..inb as usize * e + e];

        // ---- forward: gate pre-activations, then i/f/g/o ----
        let mut pre = vec![0.0f32; 4 * h];
        for (r, p) in pre.iter_mut().enumerate() {
            *p = self.b[r]
                + dot(&self.wx[r * e..r * e + e], x)
                + dot(&self.wh[r * h..r * h + h], &self.h_prev);
        }
        let mut gate = vec![0.0f32; 4 * h]; // activated gates, same layout
        for k in 0..h {
            gate[k] = sigmoid(pre[k]); // i
            gate[h + k] = sigmoid(pre[h + k]); // f
            gate[2 * h + k] = pre[2 * h + k].tanh(); // g
            gate[3 * h + k] = sigmoid(pre[3 * h + k]); // o
        }
        for k in 0..h {
            self.c_cur[k] = gate[h + k] * self.c_prev[k] + gate[k] * gate[2 * h + k];
        }
        let mut hcur = vec![0.0f32; h];
        let mut tc = vec![0.0f32; h];
        for k in 0..h {
            tc[k] = self.c_cur[k].tanh();
            hcur[k] = gate[3 * h + k] * tc[k];
        }

        // ---- forward: output softmax + cross-entropy gradient ----
        let mut logits = vec![0.0f32; 256];
        for (k, lg) in logits.iter_mut().enumerate() {
            *lg = self.bo[k] + dot(&self.wo[k * h..k * h + h], &hcur);
        }
        let max = logits.iter().copied().fold(f32::MIN, f32::max);
        let mut sum = 0.0f32;
        for lg in &mut logits {
            *lg = (*lg - max).exp();
            sum += *lg;
        }
        let inv = 1.0 / sum;
        let mut d_logits = logits; // reuse buffer as gradient
        for (k, dl) in d_logits.iter_mut().enumerate() {
            *dl = *dl * inv - if k == tgt as usize { 1.0 } else { 0.0 };
        }

        // ---- backward: output layer ----
        let mut d_h = vec![0.0f32; h];
        for (k, &dl) in d_logits.iter().enumerate() {
            axpy(&mut self.gwo[k * h..k * h + h], &hcur, dl);
            axpy(&mut d_h, &self.wo[k * h..k * h + h], dl);
            self.gbo[k] += dl;
        }

        // ---- backward: cell (single truncated step; per-byte matmul cost
        // equals full BPTT's, only the carried d_c/d_h elementwise terms drop) ----
        let mut d_pre = vec![0.0f32; 4 * h];
        for k in 0..h {
            let (i, f, g, o) = (gate[k], gate[h + k], gate[2 * h + k], gate[3 * h + k]);
            let d_o = d_h[k] * tc[k];
            let d_c = d_h[k] * o * (1.0 - tc[k] * tc[k]);
            d_pre[k] = d_c * g * i * (1.0 - i); // d_i
            d_pre[h + k] = d_c * self.c_prev[k] * f * (1.0 - f); // d_f
            d_pre[2 * h + k] = d_c * i * (1.0 - g * g); // d_g
            d_pre[3 * h + k] = d_o * o * (1.0 - o); // d_o
        }
        let mut d_x = vec![0.0f32; e];
        let mut d_hprev = vec![0.0f32; h];
        for (r, &dp) in d_pre.iter().enumerate() {
            axpy(&mut self.gwx[r * e..r * e + e], x, dp);
            axpy(&mut d_x, &self.wx[r * e..r * e + e], dp);
            axpy(&mut self.gwh[r * h..r * h + h], &self.h_prev, dp);
            axpy(&mut d_hprev, &self.wh[r * h..r * h + h], dp);
            self.gb[r] += dp;
        }
        axpy(
            &mut self.emb[inb as usize * e..inb as usize * e + e],
            &d_x,
            1.0,
        );

        // advance recurrent state
        self.h_prev.copy_from_slice(&hcur);
        self.c_prev.copy_from_slice(&self.c_cur);
    }

    /// Apply the accumulated gradients (one SGD step per update window) and clear.
    fn update(&mut self, lr: f32) {
        for (w, g) in self.wx.iter_mut().zip(&self.gwx) {
            *w -= lr * g;
        }
        for (w, g) in self.wh.iter_mut().zip(&self.gwh) {
            *w -= lr * g;
        }
        for (w, g) in self.wo.iter_mut().zip(&self.gwo) {
            *w -= lr * g;
        }
        for (w, g) in self.b.iter_mut().zip(&self.gb) {
            *w -= lr * g;
        }
        for (w, g) in self.bo.iter_mut().zip(&self.gbo) {
            *w -= lr * g;
        }
        self.gwx.iter_mut().for_each(|g| *g = 0.0);
        self.gwh.iter_mut().for_each(|g| *g = 0.0);
        self.gwo.iter_mut().for_each(|g| *g = 0.0);
        self.gb.iter_mut().for_each(|g| *g = 0.0);
        self.gbo.iter_mut().for_each(|g| *g = 0.0);
    }

    /// Parameter count (the would-be `L(D)` if it were shipped — but online it is
    /// not, so this is informational: it sizes the per-byte FLOPs, not the blob).
    fn params(&self) -> usize {
        self.wx.len()
            + self.wh.len()
            + self.wo.len()
            + self.b.len()
            + self.bo.len()
            + self.emb.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// Measure forward+backward+update throughput across a few candidate cell
    /// sizes and project the single-core enwik9 ETA. Run:
    /// `cargo test --release lstm_throughput_spike -- --ignored --nocapture`
    /// Override the byte count per config with `LZR_N` (default 1_000_000).
    #[test]
    #[ignore = "compute spike: online-neural arm throughput / enwik9 ETA"]
    fn lstm_throughput_spike() {
        const WINDOW: usize = 16; // bytes per SGD update (frozen-within-window schedule)
        let n: usize = std::env::var("LZR_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_000_000);

        // Synthetic byte stream (compute is content-independent).
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let bytes: Vec<u8> = (0..=n)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                (s >> 56) as u8
            })
            .collect();

        // (embedding dim, hidden dim, layer multiplier note)
        let configs = [(64usize, 128usize), (64, 256), (96, 384)];
        println!(
            "\nonline-neural arm compute spike — {n} bytes/config, 1 SGD step / {WINDOW} bytes\n\
             {:>5} {:>5} {:>10} {:>10} {:>12} {:>14}",
            "e", "h", "params", "MB/s", "KB/s", "enwik9 ETA(h)"
        );
        for (e, h) in configs {
            let mut cell = Cell::new(e, h);
            let params = cell.params();
            let t0 = Instant::now();
            for i in 0..n {
                cell.step(bytes[i], bytes[i + 1]);
                if (i + 1) % WINDOW == 0 {
                    cell.update(0.01);
                }
            }
            let secs = t0.elapsed().as_secs_f64();
            let bps = n as f64 / secs;
            let eta_h = 1e9 / bps / 3600.0;
            println!(
                "{e:>5} {h:>5} {params:>10} {:>10.3} {:>10.1} {:>14.1}",
                bps / 1e6,
                bps / 1e3,
                eta_h
            );
        }
        println!(
            "\nNote: per-byte cost ~= forward + one backward step. A 2-layer cell\n\
             roughly doubles the LSTM matmul cost. Compare ETA against the\n\
             machine's `70000 / Geekbench5` hour limit (both directions)."
        );
    }
}
