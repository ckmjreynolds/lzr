//! Carryless binary range coder over 12-bit probabilities `p = P(bit == 1)`.
//!
//! The encoder and decoder track a 32-bit range `[x1, x2]` and narrow it by the
//! predicted probability of each bit. Output bytes are emitted only once the top
//! byte of the range is settled — the `(x1 ^ x2) & 0xff000000 == 0` test — so a
//! carry can never propagate into an already-emitted byte and no output buffering
//! is needed. Encoder and decoder are exact mirrors and round-trip with each
//! other; that self-consistency is the only guarantee the codec needs.

const PROB_BITS: u32 = 12;
const PROB_MAX: u32 = (1 << PROB_BITS) - 1;

/// Encodes a stream of bits, each with a 12-bit probability of being 1.
#[derive(Debug)]
pub(crate) struct Encoder {
    x1: u32,
    x2: u32,
    out: Vec<u8>,
}

impl Encoder {
    /// New encoder whose output buffer already holds `prefix` (e.g. a length
    /// header), reserving `extra_cap` further bytes for the coded stream so the
    /// renorm `push` never reallocates mid-stream. Seeding the buffer with a
    /// header lets a caller prepend one without a second buffer + copy: [`finish`]
    /// returns the prefix and the coded payload in one allocation. Pass an empty
    /// `prefix` for a plain capacity-reserved encoder.
    ///
    /// [`finish`]: Self::finish
    pub(crate) fn with_prefix(mut prefix: Vec<u8>, extra_cap: usize) -> Self {
        prefix.reserve(extra_cap);
        Self {
            x1: 0,
            x2: 0xffff_ffff,
            out: prefix,
        }
    }

    /// Encode `bit` given `p` = P(bit == 1), clamped to `1..=4095`.
    ///
    /// `xmid` stays strictly below `x2` because `p <= PROB_MAX < 2^PROB_BITS`, so
    /// the `x1 + …` sum cannot overflow `u32`.
    pub(crate) fn encode(&mut self, bit: u8, p: u32) {
        let p = p.clamp(1, PROB_MAX);
        let xmid = self.x1 + ((self.x2 - self.x1) >> PROB_BITS) * p;
        if bit == 1 {
            self.x2 = xmid;
        } else {
            self.x1 = xmid + 1;
        }
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.out.push((self.x2 >> 24) as u8);
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xff;
        }
    }

    /// Flush the remaining range state and return the coded bytes. Any value in
    /// `[x1, x2]` decodes correctly, so the four bytes of `x1` suffice.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        for _ in 0..4 {
            self.out.push((self.x1 >> 24) as u8);
            self.x1 <<= 8;
        }
        self.out
    }
}

/// Decodes a bit stream produced by [`Encoder`], using the same per-bit
/// probabilities supplied in the same order.
#[derive(Debug)]
pub(crate) struct Decoder<'a> {
    x1: u32,
    x2: u32,
    x: u32,
    input: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    /// New decoder over `input`, priming the 32-bit code window with the first
    /// four bytes (zero-padding past the end).
    pub(crate) fn new(input: &'a [u8]) -> Self {
        let mut d = Self {
            x1: 0,
            x2: 0xffff_ffff,
            x: 0,
            input,
            pos: 0,
        };
        for _ in 0..4 {
            d.x = (d.x << 8) | u32::from(d.next_byte());
        }
        d
    }

    /// Number of input bytes consumed so far, counting zero-padding read past the
    /// end. A valid stream consumes exactly its payload length, so a decoder that
    /// has read well past that is being driven by a corrupt token count.
    pub(crate) const fn consumed(&self) -> usize {
        self.pos
    }

    /// The next input byte, or `0` once the input is exhausted. Zero-padding past
    /// the end is safe (the encoder's flush bytes cover the final renorms) and,
    /// crucially, means a truncated or adversarial input never panics.
    fn next_byte(&mut self) -> u8 {
        let b = self.input.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }

    /// Decode one bit given `p` = P(bit == 1), clamped to `1..=4095`.
    pub(crate) fn decode(&mut self, p: u32) -> u8 {
        let p = p.clamp(1, PROB_MAX);
        let xmid = self.x1 + ((self.x2 - self.x1) >> PROB_BITS) * p;
        let bit = u8::from(self.x <= xmid);
        if bit == 1 {
            self.x2 = xmid;
        } else {
            self.x1 = xmid + 1;
        }
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xff;
            self.x = (self.x << 8) | u32::from(self.next_byte());
        }
        bit
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::*;

    /// Encode a long pseudo-random bit/probability stream, then decode it back
    /// with the same probabilities and assert every bit round-trips.
    #[test]
    fn roundtrip_random() {
        let mut state = 0x1234_5678_9abc_def0_u64;
        let mut bits = Vec::new();
        let mut probs = Vec::new();
        for _ in 0..50_000 {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            bits.push(((state >> 40) & 1) as u8);
            probs.push((((state >> 20) & 0xFFFF) as u32 % PROB_MAX).max(1));
        }

        let mut enc = Encoder::with_prefix(Vec::new(), 0);
        for (&b, &p) in bits.iter().zip(&probs) {
            enc.encode(b, p);
        }
        let coded = enc.finish();

        let mut dec = Decoder::new(&coded);
        for (&expected, &p) in bits.iter().zip(&probs) {
            assert_eq!(dec.decode(p), expected);
        }
    }

    proptest! {
        #[test]
        fn roundtrip_proptest(
            pairs in prop::collection::vec((any::<bool>(), 1u32..=PROB_MAX), 0..2000)
        ) {
            let mut enc = Encoder::with_prefix(Vec::new(), pairs.len());
            for &(b, p) in &pairs {
                enc.encode(u8::from(b), p);
            }
            let coded = enc.finish();

            let mut dec = Decoder::new(&coded);
            for &(b, p) in &pairs {
                prop_assert_eq!(dec.decode(p), u8::from(b));
            }
        }
    }
}
