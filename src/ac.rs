//! Arithmetic (range) coder.
//!
//! Byte-renormalizing 32-bit range coder with a `u64` `low` to absorb carry
//! propagation, matching the pattern used by the LZMA reference.
//!
//! Encoder emits one dummy leading byte (always `0x00`) plus four trailing
//! flush bytes. Caller provides a 257-entry `cdf` with `cdf[0] = 0` and
//! `cdf[256] = TOTAL` (= `1 << 16`). Each symbol's probability mass is
//! `cdf[s+1] - cdf[s]`; this must be > 0 for every symbol (i.e. no
//! zero-probability symbols — [`crate::probs`] guarantees this).

use std::io::{self, Read, Write};

/// CDF total mass. All CDFs passed to [`Encoder::encode`] / [`Decoder::decode`]
/// must have `cdf[256] == TOTAL`.
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

    /// Encode one symbol using the given 257-entry CDF.
    pub(crate) fn encode(&mut self, cdf: &[u32; 257], sym: u8) -> io::Result<()> {
        debug_assert_eq!(cdf[256], TOTAL);
        let s = sym as usize;
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

    /// Decode one symbol using the given 257-entry CDF.
    ///
    /// The final `sym as u8` cast is safe: the loop bounds `sym` to `< 256`.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn decode(&mut self, cdf: &[u32; 257]) -> io::Result<u8> {
        debug_assert_eq!(cdf[256], TOTAL);
        let r = self.range >> 16;
        let value = (self.code / r).min(TOTAL - 1);

        // Linear scan is fastest for 256 symbols in skewed distributions —
        // the target value is usually near the peak of the CDF, and branch
        // prediction favors consecutive-increment comparisons.
        let mut sym: usize = 0;
        while sym < 256 && cdf[sym + 1] <= value {
            sym += 1;
        }
        debug_assert!(sym < 256, "decoder value {value} exceeds CDF");

        let lo = cdf[sym];
        let hi = cdf[sym + 1];
        self.code = self.code.wrapping_sub(r * lo);
        self.range = r * (hi - lo);

        while self.range < TOP {
            self.range <<= 8;
            let mut b = [0u8; 1];
            // Trailing bytes beyond the encoder's flush read as zero — the
            // range still narrows correctly.
            match self.inp.read_exact(&mut b) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => b[0] = 0,
                Err(e) => return Err(e),
            }
            self.code = (self.code << 8) | u32::from(b[0]);
        }
        Ok(sym as u8)
    }
}

#[cfg(test)]
// Test helpers build 257-entry CDFs by casting small indices to u32 / u8;
// `i` never exceeds 256 in either context.
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::*;

    fn uniform_cdf() -> [u32; 257] {
        // Equal 256-count buckets: TOTAL / 256 = 256.
        core::array::from_fn(|i| (i as u32) * 256)
    }

    fn skewed_cdf() -> [u32; 257] {
        // Symbol 'a' = 97 gets almost all the probability mass.
        let mut cdf = [0u32; 257];
        let mut acc = 0u32;
        for (i, slot) in cdf.iter_mut().enumerate().take(256) {
            let freq = if i == 97 { TOTAL - 255 } else { 1 };
            *slot = acc;
            acc += freq;
        }
        cdf[256] = TOTAL;
        assert_eq!(acc, TOTAL);
        cdf
    }

    #[test]
    fn roundtrip_uniform_cdf() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let src: Vec<u8> = (0..16 * 1024).map(|_| rng.random()).collect();
        let cdf = uniform_cdf();

        let mut buf = Vec::new();
        let mut enc = Encoder::new(&mut buf);
        for &b in &src {
            enc.encode(&cdf, b).unwrap();
        }
        enc.finish().unwrap();

        let mut cur = &buf[..];
        let mut dec = Decoder::new(&mut cur).unwrap();
        let dec_out: Vec<u8> = (0..src.len()).map(|_| dec.decode(&cdf).unwrap()).collect();
        assert_eq!(dec_out, src);
    }

    #[test]
    fn roundtrip_skewed_cdf() {
        let cdf = skewed_cdf();
        let mut src = vec![97u8; 16 * 1024];
        // Sprinkle some rare symbols.
        for (i, s) in src.iter_mut().enumerate().step_by(257) {
            *s = (i % 256) as u8;
        }

        let mut buf = Vec::new();
        let mut enc = Encoder::new(&mut buf);
        for &b in &src {
            enc.encode(&cdf, b).unwrap();
        }
        enc.finish().unwrap();

        let mut cur = &buf[..];
        let mut dec = Decoder::new(&mut cur).unwrap();
        let dec_out: Vec<u8> = (0..src.len()).map(|_| dec.decode(&cdf).unwrap()).collect();
        assert_eq!(dec_out, src);

        // Highly-compressible input should compress substantially.
        assert!(
            buf.len() < src.len() / 4,
            "expected strong compression on skewed input, got {} / {}",
            buf.len(),
            src.len()
        );
    }

    #[test]
    fn roundtrip_varying_cdf_per_symbol() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let src: Vec<u8> = (0..8 * 1024).map(|_| rng.random()).collect();

        // Build a sequence of CDFs: each step slightly shifts the mode.
        let cdfs: Vec<[u32; 257]> = (0..src.len())
            .map(|i| {
                let mode = u8::try_from(i % 256).unwrap();
                let mut cdf = [0u32; 257];
                let mut acc = 0u32;
                for (j, slot) in cdf.iter_mut().enumerate().take(256) {
                    let freq = if u8::try_from(j).unwrap() == mode {
                        TOTAL - 255
                    } else {
                        1
                    };
                    *slot = acc;
                    acc += freq;
                }
                cdf[256] = TOTAL;
                cdf
            })
            .collect();

        let mut buf = Vec::new();
        let mut enc = Encoder::new(&mut buf);
        for (cdf, &b) in cdfs.iter().zip(src.iter()) {
            enc.encode(cdf, b).unwrap();
        }
        enc.finish().unwrap();

        let mut cur = &buf[..];
        let mut dec = Decoder::new(&mut cur).unwrap();
        let dec_out: Vec<u8> = cdfs.iter().map(|cdf| dec.decode(cdf).unwrap()).collect();
        assert_eq!(dec_out, src);
    }

    #[test]
    fn single_byte_roundtrip() {
        // Smallest case: one symbol.
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
