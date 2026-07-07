//! Adaptive state-to-probability map (the lpaq `StateMap`), shared by models.
//!
//! Maps a small integer state to a probability, calibrated online: the learning
//! rate decays with the per-state observation count (capped by [`LIMIT`], so it
//! stays responsive to nonstationarity). Probability is held in 16-bit precision
//! and returned stretched, ready for the mixer.

use std::sync::OnceLock;

use crate::mixer::stretch;

/// Observation-count cap in the adaptive rate. Lower = more adaptive (tracks
/// drift faster), which a single-pass online coder generally wants.
const LIMIT: usize = 11;

/// The count-decay rate table shared by every `StateMap`: `rate_table()[k] = (1 << 16) / (k + 2)`, the
/// learning rate after `k` observations. A pure function of [`LIMIT`], so it is built once rather than
/// recomputing 256 divisions (and allocating a `Vec`) per map.
fn rate_table() -> &'static [i32; LIMIT + 1] {
    static DT: OnceLock<[i32; LIMIT + 1]> = OnceLock::new();
    DT.get_or_init(|| {
        let mut dt = [0i32; LIMIT + 1];
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "`k` is at most LIMIT (255), so `k as i32` is exact."
        )]
        for (k, rate) in dt.iter_mut().enumerate() {
            *rate = (1i32 << 16) / (k as i32 + 2);
        }
        dt
    })
}

/// A probability estimate per integer state, adapted toward observed bits.
#[derive(Debug)]
pub(crate) struct StateMap {
    p: Vec<i32>, // 16-bit probability per state
    n: Vec<u16>, // observation count per state (capped at LIMIT)
}

impl StateMap {
    /// A map over `size` states, each initialized to P = 0.5.
    pub(crate) fn new(size: usize) -> Self {
        Self {
            p: vec![1 << 15; size],
            n: vec![0; size],
        }
    }

    /// The stretched (logit) prediction for state `s`.
    pub(crate) fn predict(&self, s: usize) -> i32 {
        stretch(self.p[s] >> 4)
    }

    /// Move state `s`'s probability toward the observed `bit` at its count-decayed
    /// rate, then bump the (capped) observation count.
    #[expect(clippy::cast_possible_truncation, reason = "the >>16 shifted product fits i32")]
    pub(crate) fn update(&mut self, s: usize, bit: u8) {
        let target = i32::from(bit) * 65535;
        // n[s] is only incremented while < LIMIT, so it indexes dt (length LIMIT+1)
        // in bounds without a redundant clamp.
        debug_assert!((self.n[s] as usize) <= LIMIT);
        let rate = rate_table()[self.n[s] as usize];
        self.p[s] += ((i64::from(target - self.p[s]) * i64::from(rate)) >> 16) as i32;
        if (self.n[s] as usize) < LIMIT {
            self.n[s] += 1;
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn converges_toward_repeated_bit() {
        let mut sm = StateMap::new(4);
        let before = sm.predict(1);
        // Feed state 1 a long run of 1s; its stretched prediction must rise.
        for _ in 0..500 {
            sm.update(1, 1);
        }
        let after = sm.predict(1);
        assert!(after > before, "before={before} after={after}");
        // An untouched state stays at the neutral midpoint (stretch(2048) ≈ 0).
        assert_eq!(sm.predict(2), stretch(1 << 11));
    }

    #[test]
    fn tracks_both_directions() {
        let mut sm = StateMap::new(2);
        for _ in 0..500 {
            sm.update(0, 0);
        }
        assert!(sm.predict(0) < 0, "a run of zeros should predict a negative logit");
    }
}
