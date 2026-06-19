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

const NUM_SETS: usize = 8; // one weight set per bit position
const LR_SHIFT: i32 = 10;

/// Adaptive logistic mixer with one weight set selected per call.
#[derive(Debug)]
pub(crate) struct Mixer {
    n: usize,
    weights: Vec<i32>, // [NUM_SETS * n], 16.16 fixed point
    inputs: Vec<i32>,  // stretched inputs from the last `mix`
    set: usize,        // offset into `weights` of the selected set
    pr: i32,           // last squashed prediction (12-bit)
}

impl Mixer {
    /// New mixer over `n` model inputs, weights zeroed.
    pub(crate) fn new(n: usize) -> Self {
        Self {
            n,
            weights: vec![0; NUM_SETS * n],
            inputs: vec![0; n],
            set: 0,
            pr: PROB_ONE / 2,
        }
    }

    /// Mix the `stretched` model outputs into a 12-bit probability. `select`
    /// (taken mod the set count) chooses which weight set to use and adapt.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn mix(&mut self, stretched: &[i32], select: usize) -> i32 {
        self.set = (select % NUM_SETS) * self.n;
        self.inputs.copy_from_slice(stretched);
        let mut dot: i64 = 0;
        for (w, x) in self.weights[self.set..self.set + self.n]
            .iter()
            .zip(stretched)
        {
            dot += i64::from(*w) * i64::from(*x);
        }
        self.pr = squash((dot >> 16) as i32);
        self.pr
    }

    /// Adapt the selected weight set toward the observed `bit`.
    pub(crate) fn update(&mut self, bit: u8) {
        let err = (i32::from(bit) << PROB_BITS) - self.pr;
        for (w, x) in self.weights[self.set..self.set + self.n]
            .iter_mut()
            .zip(&self.inputs)
        {
            *w += (x * err) >> LR_SHIFT;
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
