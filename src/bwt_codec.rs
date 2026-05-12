//! Phase-9 codec: BWT + MTF + adaptive Order-0 AC.
//!
//! Completely orthogonal architecture to the mode-routed codecs.
//! Block-based: the entire measure window is processed as one block
//! through BWT → MTF → arithmetic-coded byte stream. The `warm`
//! prefix is ignored — BWT carries no state across blocks, so
//! prewarm doesn't help.
//!
//! The point: BWT permutes the measure so that bytes following
//! similar contexts become adjacent. MTF then converts those clusters
//! into runs of small numbers (especially 0). An adaptive Order-0
//! byte model running on the MTF output captures the zero-heavy
//! distribution very efficiently — equivalent to running an
//! effectively-high-order model on the original input "for free."
//!
//! Archive layout:
//!   `[u32 measure_len LE] [u32 primary_index LE] [AC payload bytes]`
//! The two u32 prefixes are 64 bits of unavoidable framing per
//! window.

use anyhow::{Context, Result, bail};

use crate::ac::{AcDecoder, AcEncoder};
use crate::bits::{BitReader, BitWriter};
use crate::bwt;
use crate::codec::{Codec, Decomposition};
use crate::models::Order1Bytes;
use crate::mtf;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BwtCodec;

impl Codec for BwtCodec {
    fn name(&self) -> &'static str {
        "bwt"
    }

    #[allow(clippy::cast_possible_truncation)]
    fn encode_window(&self, _warm: &[u8], measure: &[u8]) -> Result<(Vec<u8>, Decomposition)> {
        let measure_len =
            u32::try_from(measure.len()).context("measure window > 4 GiB — unsupported")?;

        let (last_col, primary) = bwt::forward(measure);
        let mtf_out = mtf::forward(&last_col);

        let mut writer = BitWriter::new();
        let bits_before = writer.bits_written();
        // Order-1 byte model conditioned on the previous MTF symbol.
        // MTF output after BWT has strong "after 0, likely another 0"
        // structure that Order-0 can't capture — Order-1 directly
        // models the run-vs-not-run transition probability per
        // previous-symbol context.
        let mut model = Order1Bytes::new();
        let mut cdf = vec![0u32; 257];
        {
            let mut enc = AcEncoder::new(&mut writer);
            for &sym in &mtf_out {
                model.cdf_to(&mut cdf);
                enc.encode(&cdf, sym as usize);
                model.observe(sym);
            }
            enc.finish();
        }
        let payload_bits = writer.bits_written() - bits_before;
        let (mut payload, pad_bits) = writer.finish();

        let mut archive = Vec::with_capacity(8 + payload.len());
        archive.extend_from_slice(&measure_len.to_le_bytes());
        archive.extend_from_slice(&primary.to_le_bytes());
        archive.append(&mut payload);

        let mut decomp = Decomposition::new();
        decomp.add("bwt_mtf_ac", payload_bits);
        decomp.add("framing", 64);
        decomp.add("padding", u64::from(pad_bits));

        Ok((archive, decomp))
    }

    fn decode_window(&self, _warm: &[u8], archive: &[u8]) -> Result<Vec<u8>> {
        if archive.len() < 8 {
            bail!("archive too short for framing ({} bytes)", archive.len());
        }
        let measure_len = usize::try_from(u32::from_le_bytes(
            archive[0..4].try_into().expect("4 bytes"),
        ))
        .expect("u32 measure_len fits usize on supported targets");
        let primary = u32::from_le_bytes(archive[4..8].try_into().expect("4 bytes"));

        let mut reader = BitReader::new(&archive[8..]);
        let mut dec = AcDecoder::new(&mut reader);
        let mut model = Order1Bytes::new();
        let mut cdf = vec![0u32; 257];
        let mut mtf_buf = Vec::with_capacity(measure_len);
        for _ in 0..measure_len {
            model.cdf_to(&mut cdf);
            let sym = dec.decode(&cdf)?;
            let b = u8::try_from(sym).expect("byte symbol fits u8");
            mtf_buf.push(b);
            model.observe(b);
        }

        let last_col = mtf::inverse(&mtf_buf);
        let out = bwt::inverse(&last_col, primary);
        if out.len() != measure_len {
            bail!(
                "decode produced {} bytes; expected {}",
                out.len(),
                measure_len,
            );
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(measure: &[u8]) {
        let codec = BwtCodec;
        let (archive, _decomp) = codec.encode_window(b"", measure).unwrap();
        let decoded = codec.decode_window(b"", &archive).unwrap();
        assert_eq!(decoded, measure);
    }

    #[test]
    fn bwt_codec_roundtrips_simple_text() {
        roundtrip(b"The quick brown fox jumps over the lazy dog.");
    }

    #[test]
    fn bwt_codec_roundtrips_repetitive_text() {
        let mut s: Vec<u8> = Vec::new();
        for _ in 0..50 {
            s.extend_from_slice(b"the quick brown fox jumps.\n");
        }
        roundtrip(&s);
    }

    #[test]
    fn bwt_codec_roundtrips_real_enwik8_window() {
        let Ok(bytes) = std::fs::read("assets/enwik8") else {
            return;
        };
        let measure = &bytes[4_194_304..4_194_304 + 4096];
        roundtrip(measure);
    }
}
