//! Nonstationary bit-history state machine shared by the context models.
//!
//! A context model that keeps a full `(probability, count)` `StateMap` slot per context spends three
//! bytes on every context and, for the sparse high-order contexts seen only a handful of times, never
//! collects enough observations for that per-context probability to move far from ½. The lpaq-class
//! design instead stores a single **bit-history state** byte per context and shares one small
//! [`crate::models::statemap::StateMap`] that maps state → probability: every context reaching the same
//! state pools its statistics, so a context seen three times inherits a probability calibrated from the
//! millions of other contexts that passed through the same short history.
//!
//! The state is a capped, discounted `(n0, n1)` counter pair — how many 0 and 1 bits this context has
//! recently emitted — packed one nibble each into a byte, so there are exactly 256 states and the
//! transition table needs no enumeration. Incrementing the observed bit's count **discounts** the
//! opposite count (halving its excess above a small exact floor), so a fresh run of one bit forgets
//! stale evidence of the other — the nonstationarity a single-pass online coder wants. The whole thing
//! is integer and deterministic, so encode and decode transition in lock-step.

use std::sync::OnceLock;

use super::statemap::StateMap;

/// The starting state: no bits seen (`n0 == n1 == 0`).
pub(crate) const INIT_STATE: u8 = 0;

/// Observation-count cap for a shared state→probability map. Higher than the per-context default
/// because each of the 256 states is visited millions of times and wants a precise, stable estimate.
const STATE_MAP_LIMIT: u8 = 127;

/// Counts at or below this are kept exact; the excess above it is halved on each discount. Keeping the
/// short histories (the common case in sparse high-order contexts) exact is what lets the shared
/// `StateMap` distinguish "one 0 so far" from "three 0s so far".
const EXACT: u16 = 3;

/// Per-nibble count cap. Each of `n0`/`n1` is held in a nibble, so both saturate at 15.
const CAP: u16 = 15;

/// Discount a stale opposite-bit count when the other bit is observed: keep small counts intact, halve
/// the excess above [`EXACT`]. A run of one bit thus decays the opposite count 15→9→6→4→3, forgetting
/// old evidence gradually rather than all at once.
const fn discount(c: u16) -> u16 {
    if c > EXACT {
        EXACT + ((c - EXACT) >> 1)
    } else {
        c
    }
}

/// The `(n0, n1)` counts a state byte encodes (high nibble = `n0`, low nibble = `n1`).
const fn counts(state: u8) -> (u16, u16) {
    ((state >> 4) as u16, (state & 0x0F) as u16)
}

/// Pack `(n0, n1)` (each `0..=15`) back into a state byte.
#[expect(clippy::cast_possible_truncation, reason = "both counts are capped at CAP (15), so each fits a nibble")]
const fn state_of(n0: u16, n1: u16) -> u8 {
    ((n0 << 4) | n1) as u8
}

/// Precomputed transition table: `state_table()[state][bit]` is the next state after observing `bit` in
/// a context currently in `state`. Built once (a pure function of the constants above) and shared by
/// every model, so a change to the transition rule cannot silently desync encode from decode.
pub(crate) fn state_table() -> &'static [[u8; 2]; 256] {
    static T: OnceLock<[[u8; 2]; 256]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [[0u8; 2]; 256];
        for (s, row) in t.iter_mut().enumerate() {
            #[expect(clippy::cast_possible_truncation, reason = "loop index is 0..256, an exact u8")]
            let (n0, n1) = counts(s as u8);
            // Observe a 0: bump n0 (saturating), discount the stale n1.
            row[0] = state_of((n0 + 1).min(CAP), discount(n1));
            // Observe a 1: bump n1 (saturating), discount the stale n0.
            row[1] = state_of(discount(n0), (n1 + 1).min(CAP));
        }
        t
    })
}

/// A context model's bit-history predictor: a per-context [`INIT_STATE`]-initialized state table plus
/// one shared 256-entry [`StateMap`] mapping state → probability. A model picks a context slot each
/// bit; this reads that slot's state, predicts from the shared map, and on `update` both trains the map
/// and advances the slot's state along the observed bit. Shared by [`super::ordern::OrderN`] and
/// [`super::sparse::SparseModel`], whose only difference is how they compute the context slot.
#[derive(Debug)]
pub(crate) struct BitHistory {
    /// Bit-history state per context (`states.len()` = the table width the owning model chose).
    states: Vec<u8>,
    /// Shared state → probability map (256 states), the calibrated estimate pooled across contexts.
    sm: StateMap,
    /// Context slot from the last [`BitHistory::predict`], reused by `update` to advance its state.
    slot: usize,
    /// Bit-history state read at that slot, i.e. the shared-map index the paired `update` trains.
    state: usize,
}

impl BitHistory {
    /// A predictor over `table_size` contexts (a power of two the owning model sizes from capacity).
    pub(crate) fn new(table_size: usize) -> Self {
        Self {
            states: vec![INIT_STATE; table_size],
            sm: StateMap::with_limit(1 << u8::BITS, STATE_MAP_LIMIT),
            slot: 0,
            state: 0,
        }
    }

    /// The stretched prediction for the context at `slot` (caller guarantees `slot < table_size`):
    /// read its bit-history state and look that state up in the shared map.
    pub(crate) fn predict(&mut self, slot: usize) -> i32 {
        self.slot = slot;
        self.state = usize::from(self.states[slot]);
        self.sm.predict(self.state)
    }

    /// Train the shared map for the state just predicted and advance that context's state along `bit`.
    pub(crate) fn update(&mut self, bit: u8) {
        self.sm.update(self.state, bit);
        self.states[self.slot] = state_table()[self.state][usize::from(bit)];
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    /// Every transition lands on a valid state whose nibbles are in range, and the table is total
    /// (defined for all 256 states × both bits) — the property the lock-step decoder relies on.
    #[test]
    fn transitions_stay_in_range() {
        let t = state_table();
        for row in t {
            for &next in row {
                let (n0, n1) = counts(next);
                assert!(n0 <= CAP && n1 <= CAP);
            }
        }
    }

    /// A run of the same bit drives the count monotonically up and saturates, while the opposite count
    /// decays toward the exact floor — the nonstationary behaviour the state machine exists to provide.
    #[test]
    fn run_saturates_and_discounts_opposite() {
        let t = state_table();
        // Start from a state with plenty of the opposite bit, then feed zeros.
        let mut s = state_of(0, 15);
        let mut prev0 = 0;
        for _ in 0..64 {
            let (n0, n1) = counts(s);
            assert!(n0 >= prev0, "n0 must not decrease on a 0");
            prev0 = n0;
            let _ = n1;
            s = t[s as usize][0];
        }
        let (n0, n1) = counts(s);
        assert_eq!(n0, CAP, "a long run of zeros saturates n0");
        assert_eq!(n1, EXACT, "the opposite count decays to the exact floor");
    }

    /// The start state distinguishes the first bit's direction (recency), which is the whole point of a
    /// bit history over a symmetric count.
    #[test]
    fn first_bit_direction_differs() {
        let t = state_table();
        assert_ne!(t[INIT_STATE as usize][0], t[INIT_STATE as usize][1]);
    }

    /// A `BitHistory` fed a run of ones at one context raises that context's prediction, while a
    /// never-touched context stays neutral — the pooled state map calibrating from bit history.
    #[test]
    fn bithistory_learns_per_context() {
        let mut bh = BitHistory::new(16);
        let before = bh.predict(3);
        for _ in 0..64 {
            let _ = bh.predict(3);
            bh.update(1);
        }
        let after = bh.predict(3);
        assert!(after > before, "a run of ones should raise the prediction: before={before} after={after}");
        // A context never updated is still at the neutral start state.
        assert_eq!(bh.predict(9), bh.predict(10));
    }
}
