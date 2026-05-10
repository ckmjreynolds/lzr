//! Deterministic byte-level prob sources for the codec.
//!
//! These provide a [`crate::codec::ProbSource`] backed not by a neural
//! model but by classical adaptive statistical models. They make the
//! codec usable end-to-end during phase-1 deterministic-stack
//! iteration, before any neural component is plugged back in. The
//! plan-of-record (see 2026-05-09 journal entry, when written) is to
//! grow this module from order-0 → order-N PPM → type-routed mixing
//! → LZ pre-pass, measuring each step against the fixed 5-offset
//! eval panel (see [`crate::bench`]).

use crate::ac::TOTAL;
use crate::arch::{CDF_LEN, VOCAB};
use crate::codec::ProbSource;
use crate::tokenizer::Token;

const _: () = assert!(VOCAB == 256, "predict module assumes byte-level VOCAB=256");

/// Three-way classification used by the type-routed codec
/// (`crate::routed`). Lowercase and uppercase letters share the
/// letter-stream predictor — uppercase positions get folded to
/// lowercase before encoding, so the case bit travels in the type map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub(crate) enum ByteClass {
    Lower = 0,
    Upper = 1,
    NonLetter = 2,
}

impl ByteClass {
    pub(crate) const fn classify(b: u8) -> Self {
        match b {
            b'a'..=b'z' => Self::Lower,
            b'A'..=b'Z' => Self::Upper,
            _ => Self::NonLetter,
        }
    }

    pub(crate) const fn as_token(self) -> Token {
        self as Token
    }

    pub(crate) const fn from_token(t: Token) -> Self {
        match t {
            0 => Self::Lower,
            1 => Self::Upper,
            _ => Self::NonLetter,
        }
    }
}

/// Order-0 adaptive byte model with Laplace +1 smoothing and online
/// halving when the running total approaches `TOTAL`.
///
/// Halving keeps the count totals bounded so the rescale to `TOTAL`
/// never collapses a slot to zero mass — the AC requires every
/// symbol to have ≥ 1 mass.
pub(crate) struct Order0Adaptive {
    counts: [u32; VOCAB],
    total: u64,
}

const RESCALE_THRESHOLD: u64 = (TOTAL / 2) as u64;

impl Order0Adaptive {
    pub(crate) const fn new() -> Self {
        Self {
            counts: [1; VOCAB],
            total: VOCAB as u64,
        }
    }

    // Counts are always ≤ TOTAL (=65 536) post-rescale, so `as u32` is
    // exact on every path. The CDF mass arithmetic is in `u64` to keep
    // the multiply from overflowing.
    #[allow(clippy::cast_possible_truncation)]
    fn cdf(&self) -> [u32; CDF_LEN] {
        let mut out = [0u32; CDF_LEN];
        let mut acc: u64 = 0;
        let total = self.total;
        for (i, &c) in self.counts.iter().enumerate() {
            out[i] = ((acc * u64::from(TOTAL)) / total) as u32;
            acc += u64::from(c);
        }
        out[VOCAB] = TOTAL;
        out
    }

    fn rescale_if_needed(&mut self) {
        if self.total <= RESCALE_THRESHOLD {
            return;
        }
        let mut new_total: u64 = 0;
        for c in &mut self.counts {
            *c = (*c >> 1).max(1);
            new_total += u64::from(*c);
        }
        self.total = new_total;
    }
}

impl Default for Order0Adaptive {
    fn default() -> Self {
        Self::new()
    }
}

impl ProbSource for Order0Adaptive {
    fn initial_cdf(&mut self) -> [u32; CDF_LEN] {
        self.cdf()
    }
    fn advance(&mut self, observed: Token) -> [u32; CDF_LEN] {
        self.counts[observed as usize] += 1;
        self.total += 1;
        self.rescale_if_needed();
        self.cdf()
    }
}

/// Order-1 adaptive byte model: per-previous-byte counts with Laplace
/// +1 smoothing and online per-row halving. Memory is `VOCAB × VOCAB`
/// `u32` counts plus `VOCAB` row totals — 256 KiB working state, well
/// under any realistic budget.
///
/// On the first byte (no history yet) we fall back to a uniform CDF;
/// the 1-byte mismeasurement is negligible on any window size we
/// actually use.
pub(crate) struct Order1Adaptive {
    counts: Vec<[u32; VOCAB]>,
    totals: Vec<u64>,
    last: Option<u8>,
}

impl Order1Adaptive {
    pub(crate) fn new() -> Self {
        Self {
            counts: vec![[1u32; VOCAB]; VOCAB],
            totals: vec![VOCAB as u64; VOCAB],
            last: None,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn cdf_for(&self, ctx: usize) -> [u32; CDF_LEN] {
        let counts = &self.counts[ctx];
        let total = self.totals[ctx];
        let mut out = [0u32; CDF_LEN];
        let mut acc: u64 = 0;
        for (i, &c) in counts.iter().enumerate() {
            out[i] = ((acc * u64::from(TOTAL)) / total) as u32;
            acc += u64::from(c);
        }
        out[VOCAB] = TOTAL;
        out
    }

    fn rescale_row(&mut self, ctx: usize) {
        if self.totals[ctx] <= RESCALE_THRESHOLD {
            return;
        }
        let mut new_total: u64 = 0;
        for c in &mut self.counts[ctx] {
            *c = (*c >> 1).max(1);
            new_total += u64::from(*c);
        }
        self.totals[ctx] = new_total;
    }
}

impl Default for Order1Adaptive {
    fn default() -> Self {
        Self::new()
    }
}

impl ProbSource for Order1Adaptive {
    fn initial_cdf(&mut self) -> [u32; CDF_LEN] {
        // After pre-warming the predictor through K bytes outside the
        // codec, `self.last` carries the conditional context for the
        // first measured byte. With no pre-warm we have no context and
        // fall back to uniform.
        self.last.map_or_else(crate::probs::uniform_cdf, |prev| {
            self.cdf_for(prev as usize)
        })
    }
    #[allow(clippy::cast_possible_truncation)]
    fn advance(&mut self, observed: Token) -> [u32; CDF_LEN] {
        let obs = observed as usize;
        if let Some(prev) = self.last {
            let p = prev as usize;
            self.counts[p][obs] += 1;
            self.totals[p] += 1;
            self.rescale_row(p);
        }
        self.last = Some(obs as u8);
        self.cdf_for(obs)
    }
}

/// Order-2 adaptive byte model: per-(prev2, prev1) byte-pair contexts.
/// Working state is `VOCAB² × VOCAB` u32 counts + `VOCAB²` u64 totals
/// — 64 MiB heap. Order-3 dense would need 16 GiB; that's where
/// PPM-style sparse contexts become mandatory.
///
/// Falls back to uniform when fewer than two bytes of history are
/// available; with the bench's 1 MiB pre-warm only the first byte of
/// each window pays this cost.
pub(crate) struct Order2Adaptive {
    counts: Vec<[u32; VOCAB]>,
    totals: Vec<u64>,
    last2: Option<u8>,
    last1: Option<u8>,
}

impl Order2Adaptive {
    pub(crate) fn new() -> Self {
        let n_ctx = VOCAB * VOCAB;
        Self {
            counts: vec![[1u32; VOCAB]; n_ctx],
            totals: vec![VOCAB as u64; n_ctx],
            last2: None,
            last1: None,
        }
    }

    const fn ctx_index(prev2: u8, prev1: u8) -> usize {
        (prev2 as usize) * VOCAB + (prev1 as usize)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn cdf_for(&self, ctx: usize) -> [u32; CDF_LEN] {
        let counts = &self.counts[ctx];
        let total = self.totals[ctx];
        let mut out = [0u32; CDF_LEN];
        let mut acc: u64 = 0;
        for (i, &c) in counts.iter().enumerate() {
            out[i] = ((acc * u64::from(TOTAL)) / total) as u32;
            acc += u64::from(c);
        }
        out[VOCAB] = TOTAL;
        out
    }

    fn rescale_row(&mut self, ctx: usize) {
        if self.totals[ctx] <= RESCALE_THRESHOLD {
            return;
        }
        let mut new_total: u64 = 0;
        for c in &mut self.counts[ctx] {
            *c = (*c >> 1).max(1);
            new_total += u64::from(*c);
        }
        self.totals[ctx] = new_total;
    }

    fn current_cdf(&self) -> [u32; CDF_LEN] {
        match (self.last2, self.last1) {
            (Some(p2), Some(p1)) => self.cdf_for(Self::ctx_index(p2, p1)),
            _ => crate::probs::uniform_cdf(),
        }
    }
}

impl Default for Order2Adaptive {
    fn default() -> Self {
        Self::new()
    }
}

impl ProbSource for Order2Adaptive {
    fn initial_cdf(&mut self) -> [u32; CDF_LEN] {
        self.current_cdf()
    }
    #[allow(clippy::cast_possible_truncation)]
    fn advance(&mut self, observed: Token) -> [u32; CDF_LEN] {
        let obs = observed as usize;
        if let (Some(p2), Some(p1)) = (self.last2, self.last1) {
            let ctx = Self::ctx_index(p2, p1);
            self.counts[ctx][obs] += 1;
            self.totals[ctx] += 1;
            self.rescale_row(ctx);
        }
        self.last2 = self.last1;
        self.last1 = Some(obs as u8);
        self.current_cdf()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdf_is_monotonic_and_total_matches() {
        let mut p = Order0Adaptive::new();
        let cdf = p.initial_cdf();
        assert_eq!(cdf[0], 0);
        assert_eq!(cdf[VOCAB], TOTAL);
        for i in 0..VOCAB {
            assert!(cdf[i + 1] > cdf[i], "sym {i} has zero mass at init");
        }
    }

    #[test]
    fn cdf_stays_valid_through_long_run() {
        let mut p = Order0Adaptive::new();
        let _ = p.initial_cdf();
        for step in 0u32..100_000 {
            let sym = (step % 256) as Token;
            let cdf = p.advance(sym);
            assert_eq!(cdf[VOCAB], TOTAL, "step {step}: total drift");
            for i in 0..VOCAB {
                assert!(cdf[i + 1] > cdf[i], "step {step}: sym {i} zero mass");
            }
        }
    }

    #[test]
    fn skewed_input_concentrates_mass() {
        let mut p = Order0Adaptive::new();
        let _ = p.initial_cdf();
        for _ in 0..10_000 {
            let _ = p.advance(Token::from(b'a'));
        }
        let cdf = p.advance(Token::from(b'a'));
        let a_mass = cdf[b'a' as usize + 1] - cdf[b'a' as usize];
        let other_mass = cdf[b'b' as usize + 1] - cdf[b'b' as usize];
        assert!(
            a_mass > 50 * other_mass,
            "expected 'a' to dominate; got a={a_mass} b={other_mass}"
        );
    }

    #[test]
    fn order1_cdf_stays_valid_through_long_run() {
        let mut p = Order1Adaptive::new();
        let _ = p.initial_cdf();
        for step in 0u32..50_000 {
            let sym = (step % 256) as Token;
            let cdf = p.advance(sym);
            assert_eq!(cdf[VOCAB], TOTAL, "step {step}: total drift");
            for i in 0..VOCAB {
                assert!(cdf[i + 1] > cdf[i], "step {step}: sym {i} zero mass");
            }
        }
    }

    #[test]
    fn order2_cdf_stays_valid_through_long_run() {
        let mut p = Order2Adaptive::new();
        let _ = p.initial_cdf();
        for step in 0u32..50_000 {
            let sym = (step % 256) as Token;
            let cdf = p.advance(sym);
            assert_eq!(cdf[VOCAB], TOTAL, "step {step}: total drift");
            for i in 0..VOCAB {
                assert!(cdf[i + 1] > cdf[i], "step {step}: sym {i} zero mass");
            }
        }
    }

    #[test]
    fn order2_learns_abc_cycle() {
        // Pattern: a→b→c→a→b→c→… After learning, CDF given (last2,last1)
        // == ('a','b') should put nearly all mass on 'c'.
        let mut p = Order2Adaptive::new();
        let _ = p.initial_cdf();
        let cycle = *b"abc";
        for i in 0..3_000 {
            let _ = p.advance(Token::from(cycle[i % 3]));
        }
        // After 3000 steps, last sym was cycle[2999 % 3] = 'c'. To probe
        // the (a,b)→? distribution we feed 'a' then 'b' to set up the
        // context; the assertion is on the CDF returned after the 'b'.
        let _ = p.advance(Token::from(b'a'));
        let cdf_ab = p.advance(Token::from(b'b'));
        let c_mass = cdf_ab[b'c' as usize + 1] - cdf_ab[b'c' as usize];
        let d_mass = cdf_ab[b'd' as usize + 1] - cdf_ab[b'd' as usize];
        assert!(
            c_mass > 100 * d_mass,
            "after a→b→c training, (a,b) should predict 'c' overwhelmingly; got c={c_mass} d={d_mass}"
        );
    }

    #[test]
    fn byte_class_covers_alphabet() {
        assert_eq!(ByteClass::classify(b'a'), ByteClass::Lower);
        assert_eq!(ByteClass::classify(b'z'), ByteClass::Lower);
        assert_eq!(ByteClass::classify(b'A'), ByteClass::Upper);
        assert_eq!(ByteClass::classify(b'Z'), ByteClass::Upper);
        assert_eq!(ByteClass::classify(b' '), ByteClass::NonLetter);
        assert_eq!(ByteClass::classify(b'9'), ByteClass::NonLetter);
        assert_eq!(ByteClass::classify(b'<'), ByteClass::NonLetter);
        assert_eq!(ByteClass::classify(0xFF), ByteClass::NonLetter);
    }

    #[test]
    fn byte_class_token_roundtrip() {
        for &c in &[ByteClass::Lower, ByteClass::Upper, ByteClass::NonLetter] {
            assert_eq!(ByteClass::from_token(c.as_token()), c);
        }
    }

    #[test]
    fn order1_learns_a_then_b_pattern() {
        // Pattern: a→b, b→a, repeated. After learning, CDF given last=='a'
        // should put nearly all mass on 'b'.
        let mut p = Order1Adaptive::new();
        let _ = p.initial_cdf();
        for _ in 0..1_000 {
            let _ = p.advance(Token::from(b'a'));
            let _ = p.advance(Token::from(b'b'));
        }
        let cdf_after_a = p.advance(Token::from(b'a'));
        let b_mass = cdf_after_a[b'b' as usize + 1] - cdf_after_a[b'b' as usize];
        let c_mass = cdf_after_a[b'c' as usize + 1] - cdf_after_a[b'c' as usize];
        assert!(
            b_mass > 100 * c_mass,
            "after a→b training, last='a' should predict 'b' overwhelmingly; got b={b_mass} c={c_mass}"
        );
    }
}
