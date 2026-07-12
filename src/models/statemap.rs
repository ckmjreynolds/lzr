//! Adaptive state-to-probability map (the lpaq `StateMap`), shared by models.
//!
//! Maps a small integer state to a probability, calibrated online: the learning
//! rate decays with the per-state observation count (capped by [`LIMIT`], so it
//! stays responsive to nonstationarity). Probability is held in 16-bit precision
//! and returned stretched, ready for the mixer.

use std::sync::OnceLock;

use crate::mixer::stretch;

/// Default observation-count cap in the adaptive rate. Lower = more adaptive (tracks drift faster),
/// which a single-pass online coder generally wants for a *per-context* map (few observations each).
const DEFAULT_LIMIT: u8 = 11;

/// Length of the shared rate table: one entry per observation count a `u8` limit can reach (`0..=255`).
/// A shared state→probability map (see [`crate::models::state`]) is visited millions of times and can
/// carry a high cap for a more precise, less jittery estimate, so the table covers the whole `u8` range.
const RATE_TABLE_LEN: usize = u8::MAX as usize + 1;

/// The count-decay rate table shared by every `StateMap`: `rate_table()[k] = (1 << 16) / (k + 2)`, the
/// learning rate after `k` observations. A pure function, so it is built once rather than recomputing
/// the divisions (and allocating a `Vec`) per map.
fn rate_table() -> &'static [i32; RATE_TABLE_LEN] {
    static DT: OnceLock<[i32; RATE_TABLE_LEN]> = OnceLock::new();
    DT.get_or_init(|| {
        let mut dt = [0i32; RATE_TABLE_LEN];
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "`k` is at most MAX_LIMIT (255), so `k as i32` is exact."
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
    p: Vec<u16>, // 16-bit probability per state (the full 0..=65535 range)
    n: Vec<u8>,  // observation count per state (capped at `limit`, which fits a byte)
    limit: u8,   // observation-count cap floor-ing the adaptive rate
}

impl StateMap {
    /// A per-context map over `size` states, each initialized to P = 0.5, with the default adaptive
    /// cap. Use [`StateMap::with_limit`] for a shared state map that wants a higher cap.
    pub(crate) fn new(size: usize) -> Self {
        Self::with_limit(size, DEFAULT_LIMIT)
    }

    /// A map over `size` states with an explicit observation-count `limit`. A higher limit yields a
    /// more precise but slower-adapting estimate — appropriate for the small, heavily-visited shared
    /// state→probability maps.
    pub(crate) fn with_limit(size: usize, limit: u8) -> Self {
        Self {
            p: vec![1 << 15; size],
            n: vec![0; size],
            limit,
        }
    }

    /// The stretched (logit) prediction for state `s`.
    pub(crate) fn predict(&self, s: usize) -> i32 {
        stretch(i32::from(self.p[s] >> 4))
    }

    /// Move state `s`'s probability toward the observed `bit` at its count-decayed
    /// rate, then bump the (capped) observation count.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the >>16 shifted product fits i32; `p` tracks toward target 0 or 65535 by a fraction \
                  (rate/65536 <= 1/2) and so never overshoots or goes negative, keeping the sum in \
                  0..=65535 for the u16 store"
    )]
    pub(crate) fn update(&mut self, s: usize, bit: u8) {
        let target = i32::from(bit) * 65535;
        // n[s] is only incremented while < limit (a u8), so it indexes the rate table (length 256,
        // covering the whole u8 range) in bounds without a redundant clamp.
        debug_assert!(self.n[s] <= self.limit);
        let rate = rate_table()[self.n[s] as usize];
        let p = i32::from(self.p[s]);
        self.p[s] = (p + ((i64::from(target - p) * i64::from(rate)) >> 16) as i32) as u16;
        if self.n[s] < self.limit {
            self.n[s] += 1;
        }
    }
}

/// A [`StateMap`] driven by an *abstaining* slot selector.
///
/// Several models key a `StateMap` on a context they sometimes have nothing to say about (`run`,
/// `xmltag`, `match`, `iddelta`). Each shares the identical wrapper: on `predict`, compute an
/// `Option<slot>`, predict from it (or abstain to a neutral stretched `0`), and remember it; on
/// `update`, adapt that same slot — or leave the map alone if the model abstained. `SlotMap` owns that
/// wrapper so a model supplies only its own `slot` logic instead of re-implementing the `idx` field
/// and the paired predict/update.
#[derive(Debug)]
pub(crate) struct SlotMap {
    sm: StateMap,
    /// Slot chosen by the last [`SlotMap::predict`], reused by the paired [`SlotMap::update`]; `None`
    /// when the model abstained, so `update` leaves the map alone.
    idx: Option<usize>,
}

impl SlotMap {
    /// A slot map over `size` states (see [`StateMap::new`]).
    pub(crate) fn new(size: usize) -> Self {
        Self {
            sm: StateMap::new(size),
            idx: None,
        }
    }

    /// Predict from `slot`, remembering it for the paired [`SlotMap::update`]. Abstains to a neutral
    /// stretched `0` when `slot` is `None`.
    pub(crate) fn predict(&mut self, slot: Option<usize>) -> i32 {
        self.idx = slot;
        self.idx.map_or(0, |s| self.sm.predict(s))
    }

    /// Adapt the slot chosen by the last [`SlotMap::predict`] toward the observed `bit`; a no-op when
    /// that prediction abstained.
    pub(crate) fn update(&mut self, bit: u8) {
        if let Some(s) = self.idx {
            self.sm.update(s, bit);
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
