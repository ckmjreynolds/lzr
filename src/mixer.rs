//! Logistic context mixing.
//!
//! [`stretch`] / [`squash`] move between the 12-bit probability domain and the
//! signed logit domain in which predictions are mixed. [`Mixer`] takes the
//! stretched outputs of every model and forms a single probability via a
//! weighted sum, then adapts its weights toward the observed bit.

use std::sync::OnceLock;

const PROB_BITS: i32 = 12;
const PROB_ONE: i32 = 1 << PROB_BITS; // 4096
const STRETCH_MAX: i32 = 2047;

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn squash_lut() -> &'static Vec<i32> {
    static LUT: OnceLock<Vec<i32>> = OnceLock::new();
    LUT.get_or_init(|| {
        // index = x + STRETCH_MAX, for x in [-STRETCH_MAX, STRETCH_MAX]
        (0..=2 * STRETCH_MAX)
            .map(|i| {
                let x = f64::from(i - STRETCH_MAX);
                let p = f64::from(PROB_ONE) / (1.0 + (-x / 256.0).exp());
                (p.round() as i32).clamp(1, PROB_ONE - 1)
            })
            .collect()
    })
}

/// Map a stretched logit `x` to a 12-bit probability in `1..=4095`.
#[allow(clippy::cast_sign_loss)]
pub(crate) fn squash(x: i32) -> i32 {
    let x = x.clamp(-STRETCH_MAX, STRETCH_MAX);
    squash_lut()[(x + STRETCH_MAX) as usize]
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
fn stretch_lut() -> &'static Vec<i32> {
    static LUT: OnceLock<Vec<i32>> = OnceLock::new();
    LUT.get_or_init(|| {
        let sq = squash_lut();
        let mut t = vec![0i32; PROB_ONE as usize];
        let mut x = -STRETCH_MAX;
        for (p, slot) in t.iter_mut().enumerate() {
            while x < STRETCH_MAX && sq[(x + STRETCH_MAX) as usize] < p as i32 {
                x += 1;
            }
            *slot = x;
        }
        t
    })
}

/// Inverse of [`squash`]: map a 12-bit probability to a logit in `[-2047, 2047]`.
#[allow(clippy::cast_sign_loss)]
pub(crate) fn stretch(p: i32) -> i32 {
    let p = p.clamp(0, PROB_ONE - 1);
    stretch_lut()[p as usize]
}

const BIT_POSITIONS: usize = 8;
const LR_SHIFT: i32 = 13; // re-tuned for the 5 averaged sub-mixers (was 12)

/// Shipped learning-rate shift, overridable by `LZR_LR` for offline sweeps only
/// (production never sets it, so the deterministic codec uses the const).
fn lr_shift() -> i32 {
    std::env::var("LZR_LR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(LR_SHIFT)
}

/// Adaptive logistic mixer: several sub-mixers, each with its own weight set
/// selected by a *different* local context (previous byte c1/c2/c3 and the
/// current word hash), all crossed with bit-position. Their logits are averaged
/// (then squashed once). Averaging decorrelated selectors — rather than learning
/// one global blend or a second-level meta-mixer over correlated selectors — is
/// the lever (v7): each selector specializes the blend for a different regime.
#[derive(Debug)]
pub(crate) struct Mixer {
    n: usize,
    cards: Vec<usize>, // contexts per sub-mixer (crossed with bit-position)
    w: Vec<Vec<i32>>,  // per sub-mixer weights, [cards[k] * 8 * n], 16.16 fixed
    inputs: Vec<i32>,  // stretched inputs from the last `mix`
    set: Vec<usize>,   // selected offset per sub-mixer from the last `mix`
    pr: i32,           // last squashed prediction (12-bit)
    lr: i32,           // learning-rate shift
}

impl Mixer {
    /// New mixer over `n` model inputs. Five sub-mixers, selected by c1, c2, c3,
    /// the word-hash byte, and the match-length bucket (all × bit-position).
    pub(crate) fn new(n: usize) -> Self {
        let cards = vec![256usize, 256, 256, 256, 64, 16, 16, 16];
        // mix() reduces a selector mod its card via `& (card - 1)`, which equals
        // `% card` exactly only for power-of-two cards — enforce that here.
        debug_assert!(cards.iter().all(|c| c.is_power_of_two()));
        let w = cards
            .iter()
            .map(|&c| vec![0i32; c * BIT_POSITIONS * n])
            .collect();
        let set = vec![0usize; cards.len()];
        Self {
            n,
            cards,
            w,
            inputs: vec![0; n],
            set,
            pr: PROB_ONE / 2,
            lr: lr_shift(),
        }
    }

    /// Mix the `stretched` model outputs into a 12-bit probability. `sel[k]` is
    /// the context selecting sub-mixer `k`'s weight set (crossed with `bpos`).
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::needless_range_loop
    )]
    pub(crate) fn mix(&mut self, stretched: &[i32], sel: &[usize], bpos: usize) -> i32 {
        self.inputs.copy_from_slice(stretched);
        let nsub = self.w.len();
        let n = self.n;
        let mut acc: i64 = 0;
        for k in 0..nsub {
            // `& (card - 1)` == `% card` for the power-of-two cards (asserted in
            // new()), turning 8 integer divisions/bit into masks.
            let off = ((sel[k] & (self.cards[k] - 1)) * BIT_POSITIONS + bpos) * n;
            self.set[k] = off;
            let row = &self.w[k][off..off + n];
            // Four independent i64 partials break the serial multiply-add
            // dependency chain (and let the backend issue lanes in parallel).
            // i64 addition is exact and associative, and the partials cannot
            // overflow (|wt*st| <= 2^31*2047 ~ 4.4e12, *22 terms ~ 9.7e13 <<
            // i64::MAX ~ 9.2e18), so any grouping yields the identical sum.
            let mut a = [0i64; 4];
            let mut wc = row.chunks_exact(4);
            let mut sc = stretched.chunks_exact(4);
            for (w4, s4) in wc.by_ref().zip(sc.by_ref()) {
                a[0] += i64::from(w4[0]) * i64::from(s4[0]);
                a[1] += i64::from(w4[1]) * i64::from(s4[1]);
                a[2] += i64::from(w4[2]) * i64::from(s4[2]);
                a[3] += i64::from(w4[3]) * i64::from(s4[3]);
            }
            let mut dot = a[0] + a[1] + a[2] + a[3];
            for (wt, st) in wc.remainder().iter().zip(sc.remainder()) {
                dot += i64::from(*wt) * i64::from(*st);
            }
            acc += dot >> 16;
        }
        self.pr = squash((acc / nsub as i64) as i32);
        self.pr
    }

    /// Adapt each sub-mixer's selected weight set toward the observed `bit`.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn update(&mut self, bit: u8) {
        let err = (i32::from(bit) << PROB_BITS) - self.pr;
        let lr = self.lr;
        let n = self.n;
        for k in 0..self.w.len() {
            let off = self.set[k];
            for (wt, st) in self.w[k][off..off + n].iter_mut().zip(&self.inputs) {
                *wt += (st * err) >> lr;
            }
        }
    }
}

const APM_KNOTS: usize = 33; // interpolation points across the stretch domain
const APM_RATE: i32 = 7; // adaptation shift

/// Secondary estimation (SSE): refines a probability through a per-context
/// adaptive curve over the stretch domain, correcting systematic miscalibration
/// of the mixer output. Knots start on the identity curve and adapt toward the
/// observed bits; `refine` interpolates between the two knots bracketing the
/// input.
#[derive(Debug)]
pub(crate) struct Apm {
    t: Vec<u16>, // [n * APM_KNOTS] knots, 16-bit probability
    idx: usize,  // lower knot index from the last `refine`
}

impl Apm {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap
    )]
    pub(crate) fn new(n: usize) -> Self {
        let mut t = vec![0u16; n * APM_KNOTS];
        for i in 0..n {
            for (j, slot) in t[i * APM_KNOTS..(i + 1) * APM_KNOTS].iter_mut().enumerate() {
                let s = (j as i32 - 16) * 128; // stretch value at this knot
                *slot = (squash(s) << 4) as u16; // 12-bit squash held in 16-bit
            }
        }
        Self { t, idx: 0 }
    }

    /// Map 12-bit `pr` to a refined 12-bit probability under context `cx`.
    #[allow(clippy::cast_sign_loss)]
    pub(crate) fn refine(&mut self, pr: i32, cx: usize) -> i32 {
        let s = (stretch(pr) + 2048).clamp(0, 4095);
        let w = s & 127;
        self.idx = cx * APM_KNOTS + (s >> 7) as usize;
        let lo = i32::from(self.t[self.idx]);
        let hi = i32::from(self.t[self.idx + 1]);
        (lo * (128 - w) + hi * w) >> 11
    }

    /// Adapt the two bracketing knots toward the observed `bit`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(crate) fn update(&mut self, bit: u8) {
        let g = i32::from(bit) * 65535;
        for k in [self.idx, self.idx + 1] {
            let cur = i32::from(self.t[k]);
            self.t[k] = (cur + ((g - cur) >> APM_RATE)) as u16;
        }
    }
}

/// Residual neural (MLP) mixing stage — a feasibility probe for the cmix-style
/// neural mixer. It sits between the linear [`Mixer`] and the [`Apm`]: a small
/// one-hidden-layer net reads the raw per-model stretched logits **plus the
/// linear mixer's own output logit as an anchor**, and emits an additive
/// *correction* to that anchor. The output layer initializes to zero, so at
/// step 0 the correction is exactly 0 and the stage is a pass-through of the
/// linear mixer — it can only improve as it learns, never structurally regress.
///
/// All-`f32`, deterministic (identical scalar ops on encode and decode, like the
/// LSTM arm), so it round-trips byte-exact. Trained online by SGD on the per-bit
/// logistic loss. This tests whether the ensemble's logits carry nonlinear
/// structure the (optimal-for-independent-predictors) linear log-odds mix misses.
#[derive(Debug)]
pub(crate) struct NeuralMixer {
    n: usize,     // number of model logits fed in
    nin: usize,   // n + 2 (the linear-mixer anchor logit copy, and a bias)
    hm: usize,    // hidden width
    nctx: usize,  // context-selected weight sets (1 = shared, the online-data optimum)
    w1: Vec<f32>, // [nctx][hm * nin]
    w2: Vec<f32>, // [nctx][hm]
    x: Vec<f32>,  // cached input from the last `refine`
    hh: Vec<f32>, // cached hidden activations from the last `refine`
    pf: f32,      // cached output probability in [0,1] from the last `refine`
    sel: usize,   // cached weight-set index from the last `refine`
    lr: f32,
}

// Scale raw logits (≈[-2047, 2047]) into [-1, 1] so the hidden tanh stays in its
// responsive range with small init weights.
const NMIX_XSCALE: f32 = 1.0 / 2047.0;

impl NeuralMixer {
    /// New residual mixer over `n` model logits, `hm` hidden units, `nctx`
    /// context-selected weight sets (e.g. one per bit-position), learning rate
    /// `lr`. Hidden weights get a small deterministic init; the output layer is
    /// zero (pass-through of the anchor at step 0).
    #[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
    pub(crate) fn new(n: usize, hm: usize, nctx: usize, lr: f32) -> Self {
        let nin = n + 2;
        let mut state = 0x853c_49e6_748f_ea9bu64;
        let mut rng = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            // ±0.1 uniform from the high bits (16-bit fraction is exact in f32).
            (((state >> 40) & 0xffff) as f32 / 65536.0 - 0.5) * 0.2
        };
        Self {
            n,
            nin,
            hm,
            nctx,
            w1: (0..nctx * hm * nin).map(|_| rng()).collect(),
            w2: vec![0.0; nctx * hm],
            x: vec![0.0; nin],
            hh: vec![0.0; hm],
            pf: 0.5,
            sel: 0,
            lr,
        }
    }

    /// Refine the linear mixer's 12-bit probability `pm` given the model logits,
    /// using weight set `sel`, returning a refined 12-bit probability. Caches the
    /// forward pass for [`update`](Self::update).
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub(crate) fn refine(&mut self, pm: i32, logits: &[i32], sel: usize) -> i32 {
        let sel = sel % self.nctx;
        self.sel = sel;
        let anchor = stretch(pm); // the linear mixer's logit, the residual baseline
        for (xi, &l) in self.x[..self.n].iter_mut().zip(logits) {
            *xi = l as f32 * NMIX_XSCALE;
        }
        self.x[self.n] = anchor as f32 * NMIX_XSCALE;
        self.x[self.nin - 1] = 1.0;
        let w1 = &self.w1[sel * self.hm * self.nin..(sel + 1) * self.hm * self.nin];
        let w2 = &self.w2[sel * self.hm..(sel + 1) * self.hm];
        let mut corr = 0.0f32;
        for j in 0..self.hm {
            let row = &w1[j * self.nin..(j + 1) * self.nin];
            let mut a = 0.0f32;
            for (w, xi) in row.iter().zip(&self.x) {
                a = w.mul_add(*xi, a);
            }
            let hj = a.tanh();
            self.hh[j] = hj;
            corr = w2[j].mul_add(hj, corr);
        }
        let out = (anchor as f32 + corr) as i32;
        let p = squash(out);
        self.pf = p as f32 / 4096.0;
        p
    }

    /// Adapt the last-used weight set toward the observed `bit` (SGD on the
    /// per-bit logistic loss). The `1/256` from the logit temperature is folded
    /// into `lr`.
    #[allow(clippy::suboptimal_flops)]
    pub(crate) fn update(&mut self, bit: u8) {
        let g = self.pf - f32::from(bit); // dL/d(out logit), up to the folded 1/256
        let glr = g * self.lr;
        let w1 = &mut self.w1[self.sel * self.hm * self.nin..(self.sel + 1) * self.hm * self.nin];
        let w2 = &mut self.w2[self.sel * self.hm..(self.sel + 1) * self.hm];
        for j in 0..self.hm {
            let w2j = w2[j];
            let hj = self.hh[j];
            w2[j] -= glr * hj;
            let da = glr * w2j * (1.0 - hj * hj);
            let row = &mut w1[j * self.nin..(j + 1) * self.nin];
            for (w, xi) in row.iter_mut().zip(&self.x) {
                *w -= da * *xi;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stretch_inverts_squash() {
        // `stretch` is defined as the inverse of `squash`, so re-squashing a
        // stretched probability recovers it to within `squash`'s largest step
        // (it saturates at the tails, so the other direction is not invertible).
        let mut max_err = 0;
        for p in 1..PROB_ONE {
            let err = (squash(stretch(p)) - p).abs();
            max_err = max_err.max(err);
        }
        assert!(max_err <= 5, "max squash(stretch(p)) error = {max_err}");
    }

    #[test]
    fn squash_is_monotone_and_bounded() {
        let mut prev = 0;
        for x in -STRETCH_MAX..=STRETCH_MAX {
            let p = squash(x);
            assert!((1..=PROB_ONE - 1).contains(&p));
            assert!(p >= prev);
            prev = p;
        }
    }
}
