//! Phase-11 codec: bit-level arithmetic coding with hash-keyed
//! context predictor.
//!
//! Decomposes each byte into 8 binary decisions and encodes each bit
//! through the existing arithmetic coder (with a 2-entry CDF derived
//! from the predicted `P(bit = 0)`). The context for each bit is a
//! 64-bit FNV-1a hash of:
//!
//! - The last 3 bytes (Order-3 byte history).
//! - The bit position within the byte (0-7).
//! - The bits of the current byte already emitted (the "partial byte").
//!
//! The partial-byte component is what makes bit-level encoding
//! genuinely different from a byte-level Order-N model: as each bit
//! is emitted within a byte, the context for the next bit narrows,
//! letting the predictor exploit intra-byte structure (e.g., "the
//! MSB was 0 so this is an ASCII byte; given the next bit is 1,
//! we're in the [64, 127] range, so high probability of letter
//! bits").
//!
//! This is the simplest possible PAQ-style architecture: one
//! predictor, no mixing, no secondary symbol estimation. The
//! infrastructure here is the substrate that mixed multi-predictor
//! variants will plug into.

use anyhow::{Context, Result, bail};

use crate::ac::{AcDecoder, AcEncoder, TOTAL};
use crate::bit_pred::{BitPredictor, FNV_OFFSET, fnv_mix};
use crate::bits::{BitReader, BitWriter};
use crate::codec::{Codec, Decomposition};

/// Bits of context-hash address space. `1 << K_BITS` predictor slots
/// — at `K_BITS = 20` the table is 4 MiB (2 × u16 per slot).
const K_BITS: u32 = 20;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PaqCodec;

/// Compute the context hash for the next bit. Inputs:
/// - `history`: last 3 bytes (oldest first); fewer if at start of stream.
/// - `bit_pos`: bit position within the byte, 7 (MSB) → 0 (LSB).
/// - `partial`: bits of the current byte already emitted, packed in
///   the low bits (`partial = byte >> (bit_pos + 1)`).
#[inline]
fn compute_context(history: &[u8], bit_pos: u8, partial: u8) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in history {
        h = fnv_mix(h, u64::from(b));
    }
    h = fnv_mix(h, u64::from(bit_pos));
    fnv_mix(h, u64::from(partial))
}

impl Codec for PaqCodec {
    fn name(&self) -> &'static str {
        "paq"
    }

    #[allow(clippy::cast_possible_truncation)]
    fn encode_window(&self, warm: &[u8], measure: &[u8]) -> Result<(Vec<u8>, Decomposition)> {
        let mut writer = BitWriter::new();
        let mut predictor = BitPredictor::new(K_BITS);

        // Sliding 3-byte history. Initialized from the tail of warm so
        // the encoder and decoder start at the same context.
        let mut hist = [0u8; 3];
        if warm.len() >= 3 {
            hist.copy_from_slice(&warm[warm.len() - 3..]);
        }

        // Prewarm: observe every bit of the warm prefix, updating
        // predictor state. Don't emit through the AC — this is
        // state-priming only.
        prewarm_predictor(&mut predictor, warm);

        let bits_before = writer.bits_written();
        {
            let mut enc = AcEncoder::new(&mut writer);
            for &byte in measure {
                let mut partial: u8 = 0;
                for bit_pos in (0..8u8).rev() {
                    let bit = u32::from((byte >> bit_pos) & 1);
                    let ctx = compute_context(&hist, bit_pos, partial);
                    let p_zero = predictor.predict_p_zero(ctx);
                    let cdf = [0u32, p_zero, TOTAL];
                    enc.encode(&cdf, bit as usize);
                    predictor.observe(ctx, bit);
                    partial = (partial << 1) | (bit as u8);
                }
                // Slide history.
                hist[0] = hist[1];
                hist[1] = hist[2];
                hist[2] = byte;
            }
            enc.finish();
        }
        let payload_bits = writer.bits_written() - bits_before;
        let (mut payload, pad_bits) = writer.finish();

        let measure_len =
            u32::try_from(measure.len()).context("measure window > 4 GiB — unsupported")?;
        let mut archive = Vec::with_capacity(4 + payload.len());
        archive.extend_from_slice(&measure_len.to_le_bytes());
        archive.append(&mut payload);

        let mut decomp = Decomposition::new();
        decomp.add("paq_bits", payload_bits);
        decomp.add("framing", 32);
        decomp.add("padding", u64::from(pad_bits));
        Ok((archive, decomp))
    }

    fn decode_window(&self, warm: &[u8], archive: &[u8]) -> Result<Vec<u8>> {
        if archive.len() < 4 {
            bail!(
                "archive too short for length prefix ({} bytes)",
                archive.len()
            );
        }
        let mut len_bytes = [0u8; 4];
        len_bytes.copy_from_slice(&archive[..4]);
        let measure_len = usize::try_from(u32::from_le_bytes(len_bytes))
            .expect("u32 measure_len fits usize on supported targets");

        let mut predictor = BitPredictor::new(K_BITS);
        let mut hist = [0u8; 3];
        if warm.len() >= 3 {
            hist.copy_from_slice(&warm[warm.len() - 3..]);
        }
        prewarm_predictor(&mut predictor, warm);

        let mut reader = BitReader::new(&archive[4..]);
        let mut dec = AcDecoder::new(&mut reader);
        let mut out = Vec::with_capacity(measure_len);
        for _ in 0..measure_len {
            let mut byte: u8 = 0;
            let mut partial: u8 = 0;
            for bit_pos in (0..8u8).rev() {
                let ctx = compute_context(&hist, bit_pos, partial);
                let p_zero = predictor.predict_p_zero(ctx);
                let cdf = [0u32, p_zero, TOTAL];
                let bit_sym = dec.decode(&cdf)?;
                let bit_u8 = u8::try_from(bit_sym).expect("bit-AC symbol ∈ {0,1}");
                predictor.observe(ctx, u32::from(bit_u8));
                byte = (byte << 1) | bit_u8;
                partial = (partial << 1) | bit_u8;
            }
            out.push(byte);
            hist[0] = hist[1];
            hist[1] = hist[2];
            hist[2] = byte;
        }
        Ok(out)
    }
}

/// Walk the warm prefix bit-by-bit, updating the predictor state but
/// not emitting through the AC. Both encoder and decoder run this
/// identically before the measured region begins.
#[allow(clippy::cast_possible_truncation)]
fn prewarm_predictor(predictor: &mut BitPredictor, warm: &[u8]) {
    let mut hist = [0u8; 3];
    for (i, &byte) in warm.iter().enumerate() {
        let mut partial: u8 = 0;
        for bit_pos in (0..8u8).rev() {
            let bit = u32::from((byte >> bit_pos) & 1);
            let ctx = compute_context(&hist, bit_pos, partial);
            predictor.observe(ctx, bit);
            partial = (partial << 1) | (bit as u8);
        }
        if i >= 2 {
            // Sliding update — once we've seen 3 bytes, push the
            // current one into the history window.
            hist[0] = hist[1];
            hist[1] = hist[2];
            hist[2] = byte;
        } else {
            // Fill from the start.
            hist[i] = byte;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(warm: &[u8], measure: &[u8]) {
        let codec = PaqCodec;
        let (archive, _decomp) = codec.encode_window(warm, measure).unwrap();
        let decoded = codec.decode_window(warm, &archive).unwrap();
        assert_eq!(decoded, measure);
    }

    #[test]
    fn paq_codec_roundtrips_short_text() {
        roundtrip(b"", b"The quick brown fox jumps over the lazy dog.");
    }

    #[test]
    fn paq_codec_roundtrips_with_warm() {
        let warm = b"<page><title>Foo</title></page>";
        let measure = b"some content following the warm prefix.";
        roundtrip(warm, measure);
    }

    #[test]
    fn paq_codec_roundtrips_long_repetition() {
        let mut s = Vec::new();
        for _ in 0..200 {
            s.extend_from_slice(b"abcdefghijklmnop");
        }
        roundtrip(b"", &s);
    }

    #[test]
    fn paq_codec_roundtrips_real_enwik8_window() {
        let Ok(bytes) = std::fs::read("assets/enwik8") else {
            return;
        };
        let warm_start = 4_194_304 - 4 * 1024 * 1024;
        let warm = &bytes[warm_start..4_194_304];
        let measure = &bytes[4_194_304..4_194_304 + 8192];
        roundtrip(warm, measure);
    }
}
