//! Arithmetic (range) coder.
//!
//! Byte-renormalizing 32-bit range coder with a `u64` `low` to absorb carry
//! propagation, matching the pattern used by the LZMA reference.
//!
//! Encoder emits one dummy leading byte (always `0x00`) plus four trailing
//! flush bytes. Caller provides a CDF slice with `cdf[0] = 0` and
//! `cdf[len-1] = TOTAL` (= `1 << 16`); the implied vocabulary size is
//! `cdf.len() - 1`. Each symbol's probability mass is `cdf[s+1] - cdf[s]`;
//! this must be > 0 for every symbol (no zero-probability symbols).
//!
//! Slice-based rather than `&[u32; CDF_LEN]` because PPM-style predictors
//! emit a different vocabulary size (escape + seen-not-excluded) at every
//! position. Fixed-size callers (the byte codec) pass `&full_cdf[..]`
//! and incur a single len-check.

#![allow(clippy::large_stack_arrays)]

use std::io::{self, Read, Write};

use crate::tokenizer::Token;

/// CDF total mass. All CDFs passed to [`Encoder::encode`] / [`Decoder::decode`]
/// must have `cdf[cdf.len() - 1] == TOTAL`.
pub(crate) const TOTAL: u32 = 1 << 16;

const TOP: u32 = 1 << 24;

/// Arithmetic encoder.
#[derive(Debug)]
pub(crate) struct Encoder<W: Write> {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    out: W,
}

impl<W: Write> Encoder<W> {
    /// Create a new encoder wrapping `out`.
    pub(crate) const fn new(out: W) -> Self {
        Self {
            low: 0,
            range: u32::MAX,
            cache: 0,
            cache_size: 1,
            out,
        }
    }

    /// Encode one symbol using the given CDF slice. `cdf.len() - 1` is the
    /// implied vocabulary size; `cdf[0] = 0`, `cdf[len-1] = TOTAL`.
    pub(crate) fn encode(&mut self, cdf: &[u32], sym: Token) -> io::Result<()> {
        debug_assert!(cdf.len() >= 2, "CDF must have at least one symbol");
        debug_assert_eq!(cdf[cdf.len() - 1], TOTAL);
        let s = sym as usize;
        debug_assert!(
            s + 1 < cdf.len(),
            "symbol {sym} out of range (vocab={})",
            cdf.len() - 1
        );
        let lo = cdf[s];
        let hi = cdf[s + 1];
        debug_assert!(hi > lo, "symbol {sym} has zero probability mass");
        let r = self.range >> 16; // divide by TOTAL = 1<<16
        self.low = self.low.wrapping_add(u64::from(r) * u64::from(lo));
        self.range = r * (hi - lo);
        while self.range < TOP {
            self.range <<= 8;
            self.shift_low()?;
        }
        Ok(())
    }

    /// Flush trailing bytes and return the wrapped writer.
    pub(crate) fn finish(mut self) -> io::Result<W> {
        for _ in 0..5 {
            self.shift_low()?;
        }
        Ok(self.out)
    }

    // The `u64 >> N as u8` casts drop bits we have just proven are the top
    // byte; the truncation is the whole point.
    #[allow(clippy::cast_possible_truncation)]
    fn shift_low(&mut self) -> io::Result<()> {
        // Top byte of `low` is stable if `low < 0xFF00_0000`, or a carry has
        // fired if `low > 0xFFFF_FFFF`. In either case we can flush cache +
        // pending 0xFF bytes with the now-known carry bit.
        if self.low < 0xFF00_0000_u64 || self.low > 0xFFFF_FFFF_u64 {
            let carry = ((self.low >> 32) & 1) as u8;
            let mut temp = self.cache;
            while self.cache_size > 0 {
                self.out.write_all(&[temp.wrapping_add(carry)])?;
                temp = 0xFF;
                self.cache_size -= 1;
            }
            self.cache = (self.low >> 24) as u8;
        }
        self.cache_size += 1;
        self.low = (self.low & 0x00FF_FFFF) << 8;
        Ok(())
    }
}

/// Arithmetic decoder.
#[derive(Debug)]
pub(crate) struct Decoder<R: Read> {
    range: u32,
    code: u32,
    inp: R,
}

impl<R: Read> Decoder<R> {
    /// Create a new decoder by priming the range and reading 5 bytes (first is
    /// the encoder's dummy leading byte, discarded).
    pub(crate) fn new(mut inp: R) -> io::Result<Self> {
        let mut buf = [0u8; 5];
        inp.read_exact(&mut buf)?;
        let code = u32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]);
        Ok(Self {
            range: u32::MAX,
            code,
            inp,
        })
    }

    /// Decode one symbol using the given CDF slice. `cdf.len() - 1` is the
    /// implied vocabulary size.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn decode(&mut self, cdf: &[u32]) -> io::Result<Token> {
        debug_assert!(cdf.len() >= 2, "CDF must have at least one symbol");
        debug_assert_eq!(cdf[cdf.len() - 1], TOTAL);
        let vocab = cdf.len() - 1;
        let r = self.range >> 16;
        let value = (self.code / r).min(TOTAL - 1);

        // Linear scan. PPM-style CDFs are short (a handful of seen
        // symbols + escape); the byte codec's CDF is 256 entries. Binary
        // search would only help materially if a large CDF showed up
        // with no skew — not the regime we're in.
        let mut sym: usize = 0;
        while sym < vocab && cdf[sym + 1] <= value {
            sym += 1;
        }
        debug_assert!(sym < vocab, "decoder value {value} exceeds CDF");

        let lo = cdf[sym];
        let hi = cdf[sym + 1];
        self.code = self.code.wrapping_sub(r * lo);
        self.range = r * (hi - lo);

        while self.range < TOP {
            self.range <<= 8;
            let mut b = [0u8; 1];
            match self.inp.read_exact(&mut b) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => b[0] = 0,
                Err(e) => return Err(e),
            }
            self.code = (self.code << 8) | u32::from(b[0]);
        }
        Ok(sym as Token)
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use crate::arch::{CDF_LEN, VOCAB};

    fn uniform_cdf() -> [u32; CDF_LEN] {
        let mut cdf = [0u32; CDF_LEN];
        let step_q = TOTAL / (VOCAB as u32);
        let step_r = TOTAL % (VOCAB as u32);
        let mut acc: u32 = 0;
        for (i, slot) in cdf.iter_mut().enumerate().take(VOCAB) {
            *slot = acc;
            acc += step_q + u32::from(u32::try_from(i).unwrap() < step_r);
        }
        cdf[VOCAB] = TOTAL;
        cdf
    }

    fn skewed_cdf(mode: Token) -> [u32; CDF_LEN] {
        let mut cdf = [0u32; CDF_LEN];
        let mut acc = 0u32;
        let v = u32::try_from(VOCAB).unwrap();
        for (i, slot) in cdf.iter_mut().enumerate().take(VOCAB) {
            let freq = if i == mode as usize {
                TOTAL - (v - 1)
            } else {
                1
            };
            *slot = acc;
            acc += freq;
        }
        cdf[VOCAB] = TOTAL;
        assert_eq!(acc, TOTAL);
        cdf
    }

    fn random_tokens(seed: u64, count: usize) -> Vec<Token> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..count)
            .map(|_| rng.random_range(0..VOCAB) as Token)
            .collect()
    }

    #[test]
    fn roundtrip_uniform_cdf() {
        let src = random_tokens(1, 4 * 1024);
        let cdf = uniform_cdf();

        let mut buf = Vec::new();
        let mut enc = Encoder::new(&mut buf);
        for &t in &src {
            enc.encode(&cdf, t).unwrap();
        }
        enc.finish().unwrap();

        let mut cur = &buf[..];
        let mut dec = Decoder::new(&mut cur).unwrap();
        let dec_out: Vec<Token> = (0..src.len()).map(|_| dec.decode(&cdf).unwrap()).collect();
        assert_eq!(dec_out, src);
    }

    #[test]
    fn roundtrip_skewed_cdf() {
        let mode: Token = 97;
        let cdf = skewed_cdf(mode);
        let mut src: Vec<Token> = vec![mode; 16 * 1024];
        // Sprinkle some rare symbols.
        for (i, s) in src.iter_mut().enumerate().step_by(257) {
            *s = (i % VOCAB) as Token;
        }

        let mut buf = Vec::new();
        let mut enc = Encoder::new(&mut buf);
        for &t in &src {
            enc.encode(&cdf, t).unwrap();
        }
        enc.finish().unwrap();

        let mut cur = &buf[..];
        let mut dec = Decoder::new(&mut cur).unwrap();
        let dec_out: Vec<Token> = (0..src.len()).map(|_| dec.decode(&cdf).unwrap()).collect();
        assert_eq!(dec_out, src);

        // Highly-compressible input should compress substantially.
        assert!(
            buf.len() < src.len(),
            "expected compression on skewed input, got {} / {}",
            buf.len(),
            src.len()
        );
    }

    #[test]
    fn roundtrip_varying_cdf_per_symbol() {
        let src = random_tokens(42, 2 * 1024);
        let cdfs: Vec<[u32; CDF_LEN]> = (0..src.len())
            .map(|i| skewed_cdf(((i * 7) % VOCAB) as Token))
            .collect();

        let mut buf = Vec::new();
        let mut enc = Encoder::new(&mut buf);
        for (cdf, &t) in cdfs.iter().zip(src.iter()) {
            enc.encode(cdf, t).unwrap();
        }
        enc.finish().unwrap();

        let mut cur = &buf[..];
        let mut dec = Decoder::new(&mut cur).unwrap();
        let dec_out: Vec<Token> = cdfs.iter().map(|cdf| dec.decode(cdf).unwrap()).collect();
        assert_eq!(dec_out, src);
    }

    #[test]
    fn single_token_roundtrip() {
        let cdf = uniform_cdf();
        let mut buf = Vec::new();
        let mut enc = Encoder::new(&mut buf);
        enc.encode(&cdf, 42).unwrap();
        enc.finish().unwrap();

        let mut cur = &buf[..];
        let mut dec = Decoder::new(&mut cur).unwrap();
        assert_eq!(dec.decode(&cdf).unwrap(), 42);
    }
}
