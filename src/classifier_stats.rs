//! Measurement codec that classifies every byte and attributes 8 bits
//! to the corresponding mode bucket. Doesn't compress — the archive is
//! the raw bytes, same as `NullCodec`. The point is that the panel's
//! `Decomposition` output now reports what fraction of each window
//! falls into each mode, which is the input for sizing the per-mode
//! bit budgets that downstream codecs are measured against.

use anyhow::Result;

use crate::classifier::Classifier;
use crate::codec::{Codec, Decomposition};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ClassifierStats;

impl Codec for ClassifierStats {
    fn name(&self) -> &'static str {
        "classifier-stats"
    }

    fn encode_window(&self, warm: &[u8], measure: &[u8]) -> Result<(Vec<u8>, Decomposition)> {
        let mut c = Classifier::new();
        // Prime classifier state through the warm prefix without
        // emitting bits — the encoder and decoder both do this so
        // the measure-region modes agree.
        for &b in warm {
            c.advance(b);
        }

        let mut decomp = Decomposition::new();
        for &b in measure {
            let mode = c.current_mode();
            decomp.add(mode.component_name(), 8);
            c.advance(b);
        }
        Ok((measure.to_vec(), decomp))
    }

    fn decode_window(&self, _warm: &[u8], archive: &[u8]) -> Result<Vec<u8>> {
        Ok(archive.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_attribute_simple_xml() {
        let measure = b"<a>hi</a>";
        let codec = ClassifierStats;
        let (archive, decomp) = codec.encode_window(b"", measure).unwrap();
        assert_eq!(archive, measure);
        // 1× '<' = content, 2× tag-name + '>' = 3 tag-structure,
        // 2× hi = content, 1× '<' = content, 3× '/a>' = tag-structure.
        // Totals: content = 4 bytes (32 bits), tag = 5 bytes (40 bits).
        assert_eq!(decomp.by_component.get("content"), Some(&32));
        assert_eq!(decomp.by_component.get("tag"), Some(&40));
        assert_eq!(decomp.total(), 8 * measure.len() as u64);
    }

    #[test]
    fn stats_attribute_with_attr_value() {
        let measure = b"<x a=\"bar\"/>";
        let codec = ClassifierStats;
        let (_archive, decomp) = codec.encode_window(b"", measure).unwrap();
        // '<' = content (1).
        // 'x', ' ', 'a', '=', '"' = tag (5).
        // 'b', 'a', 'r', '"' = attr (4).
        // '/', '>' = tag (2).
        assert_eq!(decomp.by_component.get("content"), Some(&8));
        assert_eq!(decomp.by_component.get("tag"), Some(&((5 + 2) * 8)));
        assert_eq!(decomp.by_component.get("attr"), Some(&(4 * 8)));
        assert_eq!(decomp.total(), 8 * measure.len() as u64);
    }
}
