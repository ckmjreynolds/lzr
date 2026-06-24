//! Adaptive state-to-probability map (the lpaq `StateMap`), shared by models.
//!
//! Maps a small integer state to a probability, calibrated globally: the
//! learning rate decays with the per-state observation count (with a floor, so
//! it stays responsive to nonstationarity). Probability is held in 16-bit
//! precision and returned stretched, ready for the mixer.

use crate::mixer::stretch;

const LIMIT: usize = 255; // observation-count cap in the adaptive rate (was 1023:
// lower = more adaptive, which a single-pass online coder wants — full-enwik8
// sweep 1023/511/255 = 1.5000/1.4935/1.4915; enwik9 may want slightly higher).

/// Shipped count cap, overridable by `LZR_SMLIMIT` for offline sweeps only
/// (production uses the const). Lower = more adaptive (faster to track drift),
/// higher = more stable.
fn limit() -> usize {
    std::env::var("LZR_SMLIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(LIMIT)
}

#[derive(Debug)]
pub(crate) struct StateMap {
    p: Vec<i32>,  // 16-bit probability per state
    n: Vec<u16>,  // observation count per state (capped at LIMIT)
    dt: Vec<i32>, // dt[k] = (1<<16) / (k + 2): the rate after k observations
    limit: usize, // observation-count cap
}

impl StateMap {
    pub(crate) fn new(size: usize) -> Self {
        let limit = limit();
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let dt = (0..=limit).map(|k| (1i32 << 16) / (k as i32 + 2)).collect();
        Self {
            p: vec![1 << 15; size],
            n: vec![0; size],
            dt,
            limit,
        }
    }

    pub(crate) fn predict(&self, s: usize) -> i32 {
        stretch(self.p[s] >> 4)
    }

    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn update(&mut self, s: usize, bit: u8) {
        let target = i32::from(bit) * 65535;
        // n[s] is incremented only while < limit, so n[s] <= limit always and the
        // dt index (table length limit+1) is in bounds without a redundant clamp.
        debug_assert!(self.n[s] as usize <= self.limit);
        let rate = self.dt[self.n[s] as usize];
        self.p[s] += ((i64::from(target - self.p[s]) * i64::from(rate)) >> 16) as i32;
        if (self.n[s] as usize) < self.limit {
            self.n[s] += 1;
        }
    }
}
