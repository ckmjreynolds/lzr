//! Null codec — raw byte pass-through, no compression.
//!
//! Serves as the sanity floor for the eval infrastructure: any `Codec`
//! implementation must roundtrip cleanly, and `NullCodec` should
//! report exactly 8.000 bpb on every window. If the panel reports
//! anything else for `--codec null`, the infrastructure is wrong
//! before any real compressor has been touched.

use anyhow::Result;

use crate::codec::{Codec, Decomposition};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NullCodec;

impl Codec for NullCodec {
    fn name(&self) -> &'static str {
        "null"
    }

    fn encode_window(&self, _warm: &[u8], measure: &[u8]) -> Result<(Vec<u8>, Decomposition)> {
        let archive = measure.to_vec();
        let mut decomp = Decomposition::new();
        decomp.add("raw", 8 * archive.len() as u64);
        Ok((archive, decomp))
    }

    fn decode_window(&self, _warm: &[u8], archive: &[u8]) -> Result<Vec<u8>> {
        Ok(archive.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_codec_roundtrip_preserves_bytes() {
        let measure: &[u8] = b"hello world\x00\xFF\x01";
        let codec = NullCodec;
        let (archive, decomp) = codec.encode_window(b"", measure).unwrap();
        assert_eq!(archive, measure);
        assert_eq!(decomp.total(), 8 * measure.len() as u64);

        let decoded = codec.decode_window(b"", &archive).unwrap();
        assert_eq!(decoded, measure);
    }

    #[test]
    fn null_codec_decomposition_attributes_to_raw() {
        let measure = vec![0u8; 1024];
        let codec = NullCodec;
        let (_, decomp) = codec.encode_window(b"", &measure).unwrap();
        assert_eq!(decomp.by_component.get("raw"), Some(&8192));
        assert_eq!(decomp.by_component.len(), 1);
    }
}
