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
        let cards = vec![256usize, 256, 256, 256, 64, 16];
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
        let mut acc: i64 = 0;
        for k in 0..nsub {
            let off = ((sel[k] % self.cards[k]) * BIT_POSITIONS + bpos) * self.n;
            self.set[k] = off;
            let mut dot: i64 = 0;
            for (wt, st) in self.w[k][off..off + self.n].iter().zip(stretched) {
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
        for k in 0..self.w.len() {
            let off = self.set[k];
            for (wt, st) in self.w[k][off..off + self.n].iter_mut().zip(&self.inputs) {
                *wt += (st * err) >> self.lr;
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
