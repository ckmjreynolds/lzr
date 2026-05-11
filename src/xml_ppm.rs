//! Phase-3 codec: XML schema for tags + Order-1 AC for Content.
//!
//! All three modes share a single arithmetic-coded bit stream:
//!
//! - `TagStructure` runs are encoded as a single 33-symbol Order-0
//!   model decision (32 dictionary entries + 1 "unknown" escape). On
//!   the escape, individual bytes of the run are encoded via a
//!   separate 256-symbol Order-0 model until the classifier
//!   transitions out of Tag mode.
//! - `AttrValue` bytes go through a 256-symbol Order-0 model (their
//!   own context — values are mostly the literal `preserve` plus
//!   page-redirect titles).
//! - `Content` bytes go through a 256-symbol Order-1 model
//!   conditioned on the previous Content byte.
//!
//! All adaptive state is built fresh per `encode_window` call, primed
//! through the `warm` prefix so the encoder and decoder converge to
//! the same state before the measured region.

use anyhow::{Context, Result, bail};

use crate::ac::{AcDecoder, AcEncoder};
use crate::bits::{BitReader, BitWriter};
use crate::classifier::{Classifier, Mode};
use crate::codec::{Codec, Decomposition};
use crate::models::{Order0, Order1Bytes};

/// Hardcoded enwik9 tag-structure dictionary (Phase 2's table, kept
/// in sync). 32 entries → 5-bit raw index, but in this codec the
/// dictionary index is encoded through an adaptive 33-symbol model
/// so high-frequency entries pay sub-5-bit costs.
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

/// 33-symbol alphabet: 32 dict entries + 1 escape (index = 32).
const TAG_ESCAPE: usize = DICTIONARY.len();
const TAG_ALPHABET: usize = DICTIONARY.len() + 1;
const _: () = assert!(TAG_ALPHABET == 33);

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct XmlPpmCodec;

struct Models {
    tag_dict: Order0<TAG_ALPHABET>,
    tag_byte: Order0<256>,
    attr_byte: Order0<256>,
    content: Order1Bytes,
}

impl Models {
    fn new() -> Self {
        Self {
            tag_dict: Order0::new(),
            tag_byte: Order0::new(),
            attr_byte: Order0::new(),
            content: Order1Bytes::new(),
        }
    }
}

impl Codec for XmlPpmCodec {
    fn name(&self) -> &'static str {
        "xml-ppm"
    }

    #[allow(clippy::cast_possible_truncation)]
    fn encode_window(&self, warm: &[u8], measure: &[u8]) -> Result<(Vec<u8>, Decomposition)> {
        let mut classifier = Classifier::new();
        let mut models = Models::new();

        // Prime models through the warm prefix.
        prewarm(warm, &mut classifier, &mut models);

        let mut writer = BitWriter::new();
        let mut decomp = Decomposition::new();
        let mut tag_cdf = [0u32; TAG_ALPHABET + 1];
        let mut byte_cdf = [0u32; 257];

        let bits_before_payload = writer.bits_written();
        let mut comp_bits = ComponentBits::default();
        {
            let mut enc = AcEncoder::new(&mut writer);

            let mut i = 0;
            while i < measure.len() {
                let mode = classifier.current_mode();
                match mode {
                    Mode::Content => {
                        models.content.cdf_to(&mut byte_cdf);
                        let before = enc.bits_written();
                        enc.encode(&byte_cdf, measure[i] as usize);
                        comp_bits.content += enc.bits_written() - before;
                        models.content.observe(measure[i]);
                        classifier.advance(measure[i]);
                        i += 1;
                    }
                    Mode::AttrValue => {
                        models.attr_byte.cdf_to(&mut byte_cdf);
                        let before = enc.bits_written();
                        enc.encode(&byte_cdf, measure[i] as usize);
                        comp_bits.attr += enc.bits_written() - before;
                        models.attr_byte.observe(measure[i] as usize);
                        classifier.advance(measure[i]);
                        i += 1;
                    }
                    Mode::TagStructure => {
                        let run_end = find_tag_run_end(measure, i, classifier);
                        let run = &measure[i..run_end];
                        models.tag_dict.cdf_to(&mut tag_cdf);
                        let before = enc.bits_written();
                        if let Some(idx) = dict_lookup(run) {
                            enc.encode(&tag_cdf, idx);
                            models.tag_dict.observe(idx);
                        } else {
                            enc.encode(&tag_cdf, TAG_ESCAPE);
                            models.tag_dict.observe(TAG_ESCAPE);
                            // Encode each raw byte through tag_byte.
                            for &b in run {
                                models.tag_byte.cdf_to(&mut byte_cdf);
                                enc.encode(&byte_cdf, b as usize);
                                models.tag_byte.observe(b as usize);
                            }
                        }
                        comp_bits.tag += enc.bits_written() - before;
                        for &b in run {
                            classifier.advance(b);
                        }
                        i = run_end;
                    }
                }
            }

            enc.finish();
        }

        // Length prefix + final padding.
        let payload_bits = writer.bits_written() - bits_before_payload;
        let (mut payload, pad_bits) = writer.finish();

        let measure_len =
            u32::try_from(measure.len()).context("measure window > 4 GiB — unsupported")?;
        let mut archive = Vec::with_capacity(4 + payload.len());
        archive.extend_from_slice(&measure_len.to_le_bytes());
        archive.append(&mut payload);

        decomp.add("content_ac", comp_bits.content);
        decomp.add("tag_ac", comp_bits.tag);
        decomp.add("attr_ac", comp_bits.attr);
        // Bits emitted by the AC's `finish` flush are not attributable
        // to a specific symbol; bucket them as `ac_finish`. Difference
        // between payload_bits and the per-symbol totals isolates them.
        let attributed = comp_bits.content + comp_bits.tag + comp_bits.attr;
        decomp.add("ac_finish", payload_bits.saturating_sub(attributed));
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

        let mut classifier = Classifier::new();
        let mut models = Models::new();
        prewarm(warm, &mut classifier, &mut models);

        let mut reader = BitReader::new(&archive[4..]);
        let mut dec = AcDecoder::new(&mut reader);
        let mut tag_cdf = [0u32; TAG_ALPHABET + 1];
        let mut byte_cdf = [0u32; 257];

        let mut out = Vec::with_capacity(measure_len);
        while out.len() < measure_len {
            let mode = classifier.current_mode();
            match mode {
                Mode::Content => {
                    models.content.cdf_to(&mut byte_cdf);
                    let sym = dec.decode(&byte_cdf)?;
                    let b = u8::try_from(sym).expect("byte symbol fits u8");
                    out.push(b);
                    models.content.observe(b);
                    classifier.advance(b);
                }
                Mode::AttrValue => {
                    models.attr_byte.cdf_to(&mut byte_cdf);
                    let sym = dec.decode(&byte_cdf)?;
                    let b = u8::try_from(sym).expect("byte symbol fits u8");
                    out.push(b);
                    models.attr_byte.observe(sym);
                    classifier.advance(b);
                }
                Mode::TagStructure => {
                    models.tag_dict.cdf_to(&mut tag_cdf);
                    let idx = dec.decode(&tag_cdf)?;
                    models.tag_dict.observe(idx);
                    if idx == TAG_ESCAPE {
                        // Read raw bytes until classifier exits Tag mode.
                        loop {
                            models.tag_byte.cdf_to(&mut byte_cdf);
                            let sym = dec.decode(&byte_cdf)?;
                            let b = u8::try_from(sym).expect("byte symbol fits u8");
                            out.push(b);
                            models.tag_byte.observe(sym);
                            classifier.advance(b);
                            if classifier.current_mode() != Mode::TagStructure {
                                break;
                            }
                        }
                    } else {
                        let entry = DICTIONARY
                            .get(idx)
                            .with_context(|| format!("dict index {idx} out of range"))?;
                        for &b in *entry {
                            out.push(b);
                            classifier.advance(b);
                        }
                    }
                }
            }
        }

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

#[derive(Default)]
struct ComponentBits {
    content: u64,
    tag: u64,
    attr: u64,
}

/// Walk `warm` priming both the classifier and the per-mode models so
/// the measure region starts with converged state on both sides.
fn prewarm(warm: &[u8], classifier: &mut Classifier, models: &mut Models) {
    if warm.is_empty() {
        return;
    }
    let mut i = 0;
    while i < warm.len() {
        let mode = classifier.current_mode();
        match mode {
            Mode::Content => {
                let b = warm[i];
                models.content.observe(b);
                classifier.advance(b);
                i += 1;
            }
            Mode::AttrValue => {
                let b = warm[i];
                models.attr_byte.observe(b as usize);
                classifier.advance(b);
                i += 1;
            }
            Mode::TagStructure => {
                let run_end = find_tag_run_end(warm, i, *classifier);
                let run = &warm[i..run_end];
                if let Some(idx) = dict_lookup(run) {
                    models.tag_dict.observe(idx);
                } else {
                    models.tag_dict.observe(TAG_ESCAPE);
                    for &b in run {
                        models.tag_byte.observe(b as usize);
                    }
                }
                for &b in run {
                    classifier.advance(b);
                }
                i = run_end;
            }
        }
    }
}

fn find_tag_run_end(measure: &[u8], start: usize, mut probe: Classifier) -> usize {
    let mut i = start;
    while i < measure.len() && probe.current_mode() == Mode::TagStructure {
        probe.advance(measure[i]);
        i += 1;
    }
    i
}

fn dict_lookup(run: &[u8]) -> Option<usize> {
    DICTIONARY.iter().position(|entry| *entry == run)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(warm: &[u8], measure: &[u8]) {
        let codec = XmlPpmCodec;
        let (archive, _decomp) = codec.encode_window(warm, measure).unwrap();
        let decoded = codec.decode_window(warm, &archive).unwrap();
        assert_eq!(decoded, measure);
    }

    #[test]
    fn roundtrip_simple_xml() {
        roundtrip(b"", b"<page>hello world</page>");
    }

    #[test]
    fn roundtrip_with_attr_value() {
        roundtrip(b"", b"<text xml:space=\"preserve\">prose body</text>");
    }

    #[test]
    fn roundtrip_unknown_tag() {
        roundtrip(b"", b"<weird>body</weird>");
    }

    #[test]
    fn roundtrip_with_warm_prefix() {
        let warm = b"<mediawiki xmlns=\"http://example/\"><page>warm</page>";
        let measure = b"<page><title>X</title></page>";
        roundtrip(warm, measure);
    }

    #[test]
    fn roundtrip_long_repetitive_content() {
        // High-redundancy content; Order-1 model should compress
        // well, but the test just checks roundtrip correctness.
        let mut measure = Vec::new();
        for _ in 0..200 {
            measure.extend_from_slice(b"<page><title>foo</title><id>1</id></page>");
        }
        roundtrip(b"", &measure);
    }
}
