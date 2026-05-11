//! XML-aware codec for the `TagStructure` and `AttrValue` modes.
//!
//! Walks the input, tracking mode via [`crate::classifier::Classifier`].
//! Each contiguous `TagStructure` byte run (terminated by `>` or `"`)
//! is matched against a small hardcoded dictionary of common enwik9
//! tag-structure runs. Matched runs collapse to a 1-bit marker plus a
//! 5-bit dictionary index (= 6 bits regardless of run length).
//! Unmatched runs emit a 0-bit marker followed by raw bytes — the
//! decoder advances the classifier on each byte and stops the
//! unmatched run when the classifier transitions out of `TagStructure`.
//!
//! `Content` and `AttrValue` bytes are passed through as raw 8 bpb in
//! this phase; the dedicated codecs for those modes arrive in Phase 3.
//!
//! Archive layout:
//!   `[u32 little-endian payload byte length] [bit-packed stream]`
//! The length prefix lets `decode_window` know when to stop without a
//! sentinel marker in the bit stream.

use anyhow::{Context, Result, bail};

use crate::bits::{BitReader, BitWriter};
use crate::classifier::{Classifier, Mode};
use crate::codec::{Codec, Decomposition};

/// Hardcoded enwik9 tag-structure dictionary. Indices 0..N-1 each
/// expand to the byte sequence on the right. Picked from a
/// frequency survey of `enwik8` (`<text xml:space="preserve">` etc.
/// are split at the `"` per the classifier's mode boundaries).
///
/// `N ≤ 32` so the dictionary index fits in 5 bits. Entries are
/// matched whole-tag-run (no partial-match fallback) so the dict
/// covers the high-frequency exact strings rather than fragments.
const DICTIONARY: &[&[u8]] = &[
    b"page>",
    b"/page>",
    b"title>",
    b"/title>",
    b"id>",
    b"/id>",
    b"revision>",
    b"/revision>",
    b"timestamp>",
    b"/timestamp>",
    b"contributor>",
    b"/contributor>",
    b"username>",
    b"/username>",
    b"ip>",
    b"/ip>",
    b"comment>",
    b"/comment>",
    b"minor />",
    b"text xml:space=\"",
    b">",
    b"/text>",
    b"restrictions>",
    b"/restrictions>",
    b"/namespace>",
    b"parentid>",
    b"/parentid>",
    b"sha1>",
    b"/sha1>",
    b"model>",
    b"/model>",
    b"format>",
];

const DICT_INDEX_BITS: u8 = 5;
const _: () = assert!(DICTIONARY.len() <= 1 << DICT_INDEX_BITS);

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct XmlCodec;

impl Codec for XmlCodec {
    fn name(&self) -> &'static str {
        "xml"
    }

    #[allow(clippy::cast_possible_truncation)]
    fn encode_window(&self, warm: &[u8], measure: &[u8]) -> Result<(Vec<u8>, Decomposition)> {
        let mut classifier = Classifier::new();
        for &b in warm {
            classifier.advance(b);
        }

        let mut writer = BitWriter::new();
        let mut decomp = Decomposition::new();

        // Walk `measure` looking for contiguous TagStructure runs and
        // emitting compactly; pass Content / AttrValue bytes through.
        let mut i = 0;
        while i < measure.len() {
            let mode = classifier.current_mode();
            match mode {
                Mode::Content => {
                    let bits_before = writer.bits_written();
                    writer.write_byte(measure[i]);
                    decomp.add("content_raw", writer.bits_written() - bits_before);
                    classifier.advance(measure[i]);
                    i += 1;
                }
                Mode::AttrValue => {
                    let bits_before = writer.bits_written();
                    writer.write_byte(measure[i]);
                    decomp.add("attr_raw", writer.bits_written() - bits_before);
                    classifier.advance(measure[i]);
                    i += 1;
                }
                Mode::TagStructure => {
                    // Identify the whole TagStructure run by peeking
                    // through `measure` with a probe classifier.
                    let run_end = find_tag_run_end(measure, i, classifier);
                    let run = &measure[i..run_end];
                    let bits_before = writer.bits_written();
                    if let Some(idx) = dict_lookup(run) {
                        // Known: 1 bit marker + 5 bit index.
                        writer.write_bits(1, 1);
                        writer.write_bits(
                            u64::try_from(idx).expect("dict index ≤ 32 fits u64"),
                            DICT_INDEX_BITS,
                        );
                        decomp.add("tag_dict", writer.bits_written() - bits_before);
                    } else {
                        // Unknown: 0 bit marker + raw bytes. The
                        // decoder's classifier transitions out of
                        // TagStructure exactly when ours does, so
                        // no length prefix is needed.
                        writer.write_bits(0, 1);
                        for &b in run {
                            writer.write_byte(b);
                        }
                        decomp.add("tag_raw", writer.bits_written() - bits_before);
                    }
                    for &b in run {
                        classifier.advance(b);
                    }
                    i = run_end;
                }
            }
        }

        // Prefix the archive with the decoded length so the decoder
        // knows when to stop. 4 bytes = 32 bits, accounted as framing.
        let measure_len =
            u32::try_from(measure.len()).context("measure window > 4 GiB — unsupported")?;
        let payload_byte_cap = usize::try_from(writer.bits_written().div_ceil(8))
            .expect("payload byte count fits usize");
        let mut archive = Vec::with_capacity(4 + payload_byte_cap);
        archive.extend_from_slice(&measure_len.to_le_bytes());
        let (payload, pad_bits) = writer.finish();
        archive.extend_from_slice(&payload);

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
        let mut reader = BitReader::new(&archive[4..]);

        let mut classifier = Classifier::new();
        for &b in warm {
            classifier.advance(b);
        }

        let mut out = Vec::with_capacity(measure_len);
        while out.len() < measure_len {
            let mode = classifier.current_mode();
            match mode {
                Mode::Content | Mode::AttrValue => {
                    let b = reader.read_byte()?;
                    out.push(b);
                    classifier.advance(b);
                }
                Mode::TagStructure => {
                    let marker = reader.read_bits(1)?;
                    if marker == 1 {
                        let idx = usize::try_from(reader.read_bits(DICT_INDEX_BITS)?)
                            .expect("DICT_INDEX_BITS ≤ 5 fits usize");
                        let entry = DICTIONARY
                            .get(idx)
                            .with_context(|| format!("dict index {idx} out of range"))?;
                        for &b in *entry {
                            out.push(b);
                            classifier.advance(b);
                        }
                    } else {
                        // Read raw bytes until classifier exits Tag mode.
                        loop {
                            let b = reader.read_byte()?;
                            out.push(b);
                            classifier.advance(b);
                            if classifier.current_mode() != Mode::TagStructure {
                                break;
                            }
                        }
                    }
                }
            }
        }

        if out.len() != measure_len {
            bail!(
                "decode produced {} bytes; expected {} from length prefix",
                out.len(),
                measure_len,
            );
        }
        Ok(out)
    }
}

/// Scan `measure` from `start` forward as long as the classifier
/// reports `TagStructure`. Return the index just past the last
/// `TagStructure` byte (so `&measure[start..end]` is the run).
fn find_tag_run_end(measure: &[u8], start: usize, mut probe: Classifier) -> usize {
    let mut i = start;
    while i < measure.len() && probe.current_mode() == Mode::TagStructure {
        probe.advance(measure[i]);
        i += 1;
    }
    i
}

/// Find the dictionary index for an exact byte-slice match.
fn dict_lookup(run: &[u8]) -> Option<usize> {
    DICTIONARY.iter().position(|entry| *entry == run)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(codec: XmlCodec, warm: &[u8], measure: &[u8]) {
        let (archive, _decomp) = codec.encode_window(warm, measure).unwrap();
        let decoded = codec.decode_window(warm, &archive).unwrap();
        assert_eq!(decoded, measure);
    }

    #[test]
    fn roundtrip_simple_dict_hit() {
        let codec = XmlCodec;
        roundtrip(codec, b"", b"<page>hello</page>");
    }

    #[test]
    fn roundtrip_with_attr_value() {
        let codec = XmlCodec;
        roundtrip(codec, b"", b"<text xml:space=\"preserve\">hi</text>");
    }

    #[test]
    fn roundtrip_unknown_tag_falls_back_to_raw() {
        let codec = XmlCodec;
        // <weirdtag> isn't in the dictionary.
        roundtrip(codec, b"", b"<weirdtag>body</weirdtag>");
    }

    #[test]
    fn roundtrip_with_warm_prefix() {
        let codec = XmlCodec;
        let warm = b"<mediawiki><page>x</page>";
        let measure = b"<page>second</page>";
        roundtrip(codec, warm, measure);
    }

    #[test]
    fn dict_hit_compresses_versus_raw() {
        let codec = XmlCodec;
        let (archive_known, _) = codec.encode_window(b"", b"<page>").unwrap();
        let (archive_unknown, _) = codec.encode_window(b"", b"<weird>").unwrap();
        // Dict hit: 4-byte framing + 8 bits content (`<`) + 6 bits
        // (1 marker + 5 index) = 4 bytes + 14 bits = 6 bytes (with pad).
        // Unknown: 4-byte framing + 8 bits content + 1 bit marker +
        // 6*8 bits raw = 4 + 57 = 12 bytes (with pad).
        assert!(
            archive_known.len() < archive_unknown.len(),
            "dict hit {} should beat raw {}",
            archive_known.len(),
            archive_unknown.len(),
        );
    }

    #[test]
    fn decomposition_sums_to_archive_bits() {
        let codec = XmlCodec;
        let measure = b"<page><title>x</title></page>";
        let (archive, decomp) = codec.encode_window(b"", measure).unwrap();
        let archive_bits = 8 * archive.len() as u64;
        assert_eq!(
            decomp.total(),
            archive_bits,
            "decomposition {} vs archive {}",
            decomp.total(),
            archive_bits,
        );
    }
}
