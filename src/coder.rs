//! Binary arithmetic coder: an lpaq-style carryless range coder over 12-bit
//! probabilities `p = P(bit == 1)`.
//!
//! The encoder and decoder are exact mirrors and round-trip with each other —
//! that self-consistency is the only guarantee the codec needs.

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
    /// New encoder with the full `[0, 2^32)` range, pre-reserving `cap` output
    /// bytes so the renorm `push` never reallocates mid-stream (the coded output
    /// is well under one byte per input byte at the operating bpb). Capacity is
    /// invisible to the coded bytes.
    pub(crate) fn with_capacity(cap: usize) -> Self {
        Self {
            x1: 0,
            x2: 0xffff_ffff,
            out: Vec::with_capacity(cap),
        }
    }

    /// Encode `bit` given `p` = P(bit == 1), clamped to `1..=4095`.
    #[allow(clippy::cast_possible_truncation)]
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

    /// Coded bytes emitted so far (grows as the range renormalizes) — used for
    /// live progress on long encodes.
    pub(crate) fn output_len(&self) -> usize {
        self.out.len()
    }

    /// Flush remaining state and return the coded bytes.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn finish(mut self) -> Vec<u8> {
        // Emit the four bytes of `x1`; any value in `[x1, x2]` decodes correctly,
        // and these bytes feed the decoder's renormalization to the end.
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
    /// New decoder over `input`, priming the 32-bit code window.
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
mod tests {
    #![allow(clippy::cast_possible_truncation)]
    use super::*;

    #[test]
    fn roundtrip_random() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut bits = Vec::new();
        let mut probs = Vec::new();
        for _ in 0..50_000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            bits.push(((state >> 40) & 1) as u8);
            probs.push(((state >> 20) as u32 % PROB_MAX).max(1));
        }

        let mut enc = Encoder::with_capacity(0);
        for (&b, &p) in bits.iter().zip(&probs) {
            enc.encode(b, p);
        }
        let coded = enc.finish();

        let mut dec = Decoder::new(&coded);
        for (&expect, &p) in bits.iter().zip(&probs) {
            assert_eq!(dec.decode(p), expect);
        }
    }
}
