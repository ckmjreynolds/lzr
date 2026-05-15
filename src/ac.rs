//! Bit-level arithmetic coder.
//!
//! Standard Howard-Vitter design with 32-bit precision and an
//! underflow handler. Both encoder and decoder operate on the
//! `BitWriter` / `BitReader` from [`crate::bits`], so AC bits pack
//! into the same archive byte stream as any other fractional-bit
//! emissions a codec might make.
//!
//! CDF contract: caller passes a `&[u32]` of length `alphabet + 1`
//! with `cdf[0] == 0`, `cdf[alphabet] == TOTAL`, and strictly
//! increasing entries (every symbol gets ≥ 1 unit of mass).
//! `TOTAL` is fixed at `1 << 16` to keep the multiply-divide steps
//! exact in u64 arithmetic.

use anyhow::{Result, bail};

use crate::bits::{BitReader, BitWriter};

/// Sum of probability mass across all symbols in any CDF the AC sees.
/// 16-bit precision keeps the `range * mass / TOTAL` multiplies inside
/// u64 with headroom.
pub(crate) const TOTAL: u32 = 1 << 16;

const PRECISION_BITS: u32 = 32;
const HALF: u32 = 1u32 << (PRECISION_BITS - 1);
const QUARTER: u32 = HALF >> 1;
const THREE_QUARTERS: u32 = HALF | QUARTER;

#[derive(Debug)]
pub(crate) struct AcEncoder<'a> {
    low: u32,
    high: u32,
    /// Count of underflow-deferred bits. Flushed by `emit_pending` on
    /// the next definitive bit, or by `finish` at stream end.
    pending: u64,
    out: &'a mut BitWriter,
}

impl<'a> AcEncoder<'a> {
    pub(crate) const fn new(out: &'a mut BitWriter) -> Self {
        Self {
            low: 0,
            high: u32::MAX,
            pending: 0,
            out,
        }
    }

    /// Encode `symbol` against `cdf`. Caller-supplied CDF must satisfy
    /// the contract documented at the module level.
    ///
    /// Arithmetic is done in u64 because at the initial state
    /// (`low=0, high=u32::MAX`), `range = 2^32` and `range * TOTAL`
    /// = `2^48`. The intermediate `(range * hi) / TOTAL` can reach
    /// `2^32`, which doesn't fit u32 — so the cast to u32 happens
    /// after subtracting 1, when the value is bounded by `u32::MAX`.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn encode(&mut self, cdf: &[u32], symbol: usize) {
        debug_assert!(symbol + 1 < cdf.len(), "symbol out of CDF range");
        debug_assert_eq!(cdf[cdf.len() - 1], TOTAL, "CDF must total to TOTAL");
        debug_assert!(
            cdf[symbol + 1] > cdf[symbol],
            "zero-mass symbol {symbol} (cdf[{symbol}]={}, cdf[{}]={}) — caller's model violated the AC's strictly-increasing-CDF contract",
            cdf[symbol],
            symbol + 1,
            cdf[symbol + 1],
        );
        let lo = u64::from(cdf[symbol]);
        let hi = u64::from(cdf[symbol + 1]);
        let low_u64 = u64::from(self.low);
        let range = u64::from(self.high - self.low) + 1;
        let scaled_lo = (range * lo) / u64::from(TOTAL);
        let scaled_hi = (range * hi) / u64::from(TOTAL);
        self.low = (low_u64 + scaled_lo) as u32;
        self.high = (low_u64 + scaled_hi - 1) as u32;
        self.renormalize();
    }

    fn renormalize(&mut self) {
        loop {
            if self.high < HALF {
                self.emit_bit(false);
                self.emit_pending(false);
                self.low <<= 1;
                self.high = (self.high << 1) | 1;
            } else if self.low >= HALF {
                self.emit_bit(true);
                self.emit_pending(true);
                self.low = (self.low - HALF) << 1;
                self.high = ((self.high - HALF) << 1) | 1;
            } else if self.low >= QUARTER && self.high < THREE_QUARTERS {
                self.pending += 1;
                self.low = (self.low - QUARTER) << 1;
                self.high = ((self.high - QUARTER) << 1) | 1;
            } else {
                break;
            }
        }
    }

    fn emit_bit(&mut self, b: bool) {
        self.out.write_bits(u64::from(b), 1);
    }

    fn emit_pending(&mut self, definitive: bool) {
        let inverse = u64::from(!definitive);
        for _ in 0..self.pending {
            self.out.write_bits(inverse, 1);
        }
        self.pending = 0;
    }

    /// Bits committed to the underlying writer so far. Excludes the
    /// AC's internal `low`/`high`/`pending` state that hasn't been
    /// emitted yet — useful for components that want to attribute the
    /// bit-delta of a single `encode` call but understand the tail
    /// flush happens lazily.
    pub(crate) fn bits_written(&self) -> u64 {
        self.out.bits_written()
    }

    /// Flush remaining state. After `finish` the encoder's bits are
    /// fully committed to the underlying writer.
    ///
    /// Per Howard-Vitter end-of-message: bump `pending` once, then
    /// emit either `0` followed by `pending` `1`s (if `low < QUARTER`)
    /// or `1` followed by `pending` `0`s otherwise. The
    /// `emit_pending(b)` helper writes `!b` so matching `b` between
    /// `emit_bit` and `emit_pending` produces the opposite-bit tail
    /// the algorithm needs.
    pub(crate) fn finish(mut self) {
        self.pending += 1;
        if self.low < QUARTER {
            self.emit_bit(false);
            self.emit_pending(false);
        } else {
            self.emit_bit(true);
            self.emit_pending(true);
        }
    }
}

#[derive(Debug)]
pub(crate) struct AcDecoder<'a, 'b> {
    low: u32,
    high: u32,
    code: u32,
    inp: &'b mut BitReader<'a>,
}

impl<'a, 'b> AcDecoder<'a, 'b> {
    pub(crate) fn new(inp: &'b mut BitReader<'a>) -> Self {
        let mut code = 0u32;
        for _ in 0..PRECISION_BITS {
            let bit = inp.read_bits(1).unwrap_or(0);
            code = (code << 1) | u32::try_from(bit).expect("1-bit read fits u32");
        }
        Self {
            low: 0,
            high: u32::MAX,
            code,
            inp,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn decode(&mut self, cdf: &[u32]) -> Result<usize> {
        debug_assert_eq!(cdf[cdf.len() - 1], TOTAL, "CDF must total to TOTAL");
        let range = u64::from(self.high - self.low) + 1;
        let scaled = ((u64::from(self.code - self.low) + 1) * u64::from(TOTAL) - 1) / range;
        let target = u32::try_from(scaled).expect("scaled value < TOTAL fits u32");
        // Binary search: rightmost `i` with `cdf[i] <= target`.
        let symbol = match cdf.binary_search(&target) {
            Ok(i) => {
                // Tie: skip duplicates to land on the rightmost match
                // (some CDFs can have consecutive equal entries, though
                // ours don't since gaps are ≥ 1).
                let mut i = i;
                while i + 1 < cdf.len() && cdf[i + 1] <= target {
                    i += 1;
                }
                i
            }
            Err(i) => i.saturating_sub(1),
        };
        if symbol + 1 >= cdf.len() {
            bail!("AC decode produced out-of-range symbol {symbol}");
        }
        let lo = u64::from(cdf[symbol]);
        let hi = u64::from(cdf[symbol + 1]);
        let low_u64 = u64::from(self.low);
        let scaled_lo = (range * lo) / u64::from(TOTAL);
        let scaled_hi = (range * hi) / u64::from(TOTAL);
        self.low = (low_u64 + scaled_lo) as u32;
        self.high = (low_u64 + scaled_hi - 1) as u32;
        self.renormalize();
        Ok(symbol)
    }

    fn renormalize(&mut self) {
        loop {
            if self.high < HALF {
                // No adjustment to code beyond shift.
            } else if self.low >= HALF {
                self.low -= HALF;
                self.high -= HALF;
                self.code -= HALF;
            } else if self.low >= QUARTER && self.high < THREE_QUARTERS {
                self.low -= QUARTER;
                self.high -= QUARTER;
                self.code -= QUARTER;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
            let bit = self.inp.read_bits(1).unwrap_or(0);
            self.code = (self.code << 1) | u32::try_from(bit).expect("1-bit read fits u32");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Uniform CDF over an alphabet of size `n`.
    #[allow(clippy::cast_possible_truncation)]
    fn uniform_cdf(n: usize) -> Vec<u32> {
        let mut cdf = vec![0u32; n + 1];
        let total = u64::from(TOTAL);
        let n_u64 = n as u64;
        for (i, slot) in cdf.iter_mut().enumerate().take(n) {
            *slot = ((i as u64 * total) / n_u64) as u32;
        }
        cdf[n] = TOTAL;
        cdf
    }

    #[test]
    fn ac_roundtrips_uniform_bytes() {
        let cdf = uniform_cdf(256);
        let input: Vec<u8> = (0u32..1000)
            .map(|i| u8::try_from((i * 17) % 256).unwrap())
            .collect();
        let mut writer = BitWriter::new();
        {
            let mut enc = AcEncoder::new(&mut writer);
            for &b in &input {
                enc.encode(&cdf, b as usize);
            }
            enc.finish();
        }
        let (buf, _pad) = writer.finish();
        let mut reader = BitReader::new(&buf);
        let mut dec = AcDecoder::new(&mut reader);
        let out: Vec<u8> = (0..input.len())
            .map(|_| {
                let s = dec.decode(&cdf).unwrap();
                u8::try_from(s).unwrap()
            })
            .collect();
        assert_eq!(out, input);
    }

    #[test]
    fn ac_roundtrips_skewed_distribution() {
        // Skewed CDF: byte 0 gets most of the mass.
        let mut cdf = vec![0u32; 257];
        cdf[1] = TOTAL - 255;
        for i in 1..256 {
            cdf[i + 1] = cdf[i] + 1;
        }
        let input: Vec<u8> = vec![0u8; 500]
            .into_iter()
            .chain((1..=20u8).flat_map(|b| std::iter::repeat_n(b, 5)))
            .collect();

        let mut writer = BitWriter::new();
        {
            let mut enc = AcEncoder::new(&mut writer);
            for &b in &input {
                enc.encode(&cdf, b as usize);
            }
            enc.finish();
        }
        let (buf, _pad) = writer.finish();

        let mut reader = BitReader::new(&buf);
        let mut dec = AcDecoder::new(&mut reader);
        let out: Vec<u8> = (0..input.len())
            .map(|_| u8::try_from(dec.decode(&cdf).unwrap()).unwrap())
            .collect();
        assert_eq!(out, input);

        // Skewed distribution should compress much better than 8 bpb.
        // 500 byte-0s carry near zero bits each; the 20 other bytes
        // carry ~8 bits each. Total ≲ 250 bytes, vs 600 raw.
        assert!(
            buf.len() < input.len() / 2,
            "skewed compression should beat 2x: {} vs {}",
            buf.len(),
            input.len(),
        );
    }

    #[test]
    fn ac_handles_two_symbol_alphabet() {
        // Mix of true/false events with adaptive-ish CDF.
        let cdf = [0u32, TOTAL / 4, TOTAL]; // 25% / 75%
        let pattern: Vec<usize> = (0..1000).map(|i| usize::from(i % 4 != 0)).collect();

        let mut writer = BitWriter::new();
        {
            let mut enc = AcEncoder::new(&mut writer);
            for &s in &pattern {
                enc.encode(&cdf, s);
            }
            enc.finish();
        }
        let (buf, _pad) = writer.finish();
        let mut reader = BitReader::new(&buf);
        let mut dec = AcDecoder::new(&mut reader);
        let out: Vec<usize> = (0..pattern.len())
            .map(|_| dec.decode(&cdf).unwrap())
            .collect();
        assert_eq!(out, pattern);
    }

    #[test]
    fn ac_stress_many_symbols_skewed_static() {
        // Long sequence through a static, very-skewed CDF — exercises
        // the renormalization paths that the short synthetic tests
        // don't reach (long runs of high-prob symbols + occasional
        // low-prob ones). Replaces the v3 Order-0 adaptive stress test
        // which depended on the now-deleted `models` module.
        let input: Vec<u8> = (0..50_000u32)
            .map(|i| {
                // 90% byte 0, 9% byte 1, 1% byte 2 — extreme skew.
                let r = (i.wrapping_mul(2_654_435_761)) % 100;
                if r < 90 {
                    0
                } else if r < 99 {
                    1
                } else {
                    2
                }
            })
            .collect();

        // CDF: byte 0 → 90%, byte 1 → 9%, byte 2 → 1%, all else → 0
        // (but AC requires strictly-increasing, so floor at 1 unit).
        let mut cdf = vec![0u32; 257];
        cdf[1] = (TOTAL * 90) / 100;
        cdf[2] = (TOTAL * 99) / 100;
        cdf[3] = TOTAL - 253; // floor remaining 253 entries at 1 unit each
        for i in 4..257 {
            cdf[i] = cdf[i - 1] + 1;
        }
        cdf[256] = TOTAL;

        let mut writer = BitWriter::new();
        {
            let mut enc = AcEncoder::new(&mut writer);
            for &b in &input {
                enc.encode(&cdf, b as usize);
            }
            enc.finish();
        }
        let (buf, _pad) = writer.finish();

        let mut reader = BitReader::new(&buf);
        let mut dec = AcDecoder::new(&mut reader);
        let out: Vec<u8> = (0..input.len())
            .map(|_| u8::try_from(dec.decode(&cdf).unwrap()).unwrap())
            .collect();
        assert_eq!(out, input);
    }

    #[test]
    fn ac_empty_stream_finishes_cleanly() {
        let mut writer = BitWriter::new();
        let enc = AcEncoder::new(&mut writer);
        enc.finish();
        let (buf, _pad) = writer.finish();
        // No content; finish should produce a tiny tail of state bits.
        let mut reader = BitReader::new(&buf);
        let _dec = AcDecoder::new(&mut reader);
    }
}
