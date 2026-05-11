//! Phase-4 codec: XML schema for tags + LZ77 pre-pass + Order-1 AC
//! for Content literals.
//!
//! Builds on `xml-ppm`. At each Content-mode position the encoder
//! decides literal vs match using the LZ77 matcher; the choice is
//! emitted as a 1-bit flag through an adaptive 2-symbol model, then
//! either the byte (Order-1 AC) or the match record
//! (`bucketed-offset` + `length` Order-0 AC models). Tag and
//! `AttrValue` handling is unchanged from `xml-ppm`.
//!
//! Insertion timing: the matcher inserts position `p` once bytes
//! `p..=p+2` are available. The encoder uses the full input buffer
//! (warm + measure) for matching but commits to the same lazy-insert
//! cursor the decoder will use, so the two sides see identical
//! matcher state.

use anyhow::{Context, Result, bail};

use crate::ac::{AcDecoder, AcEncoder};
use crate::bits::{BitReader, BitWriter};
use crate::classifier::{Classifier, Mode};
use crate::codec::{Codec, Decomposition};
use crate::lz::{MIN_MATCH, Matcher};
use crate::models::{Order0, Order1Bytes};

/// Tag-structure dictionary, copied from Phase 3 to keep the codec
/// self-contained. 32 entries → adaptive 33-symbol model (entries +
/// 1 escape). High-frequency entries pay sub-5-bit costs.
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

const TAG_ESCAPE: usize = DICTIONARY.len();
const TAG_ALPHABET: usize = DICTIONARY.len() + 1;
const _: () = assert!(TAG_ALPHABET == 33);

/// Offset log-bucket alphabet. `offset` is in `[1, WINDOW_SIZE]` so
/// `bucket = log2_floor(offset)` is in `[0, 22]`. Allow 32 to
/// future-proof without changing the encoding.
const OFFSET_BUCKET_ALPHABET: usize = 32;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct XmlLzPpmCodec;

struct Models {
    tag_dict: Order0<TAG_ALPHABET>,
    tag_byte: Order0<256>,
    attr_byte: Order0<256>,
    content: Order1Bytes,
    /// Literal-vs-match flag at Content positions.
    lz_flag: Order0<2>,
    /// Log-bucket of the offset (= floor(log2(offset))).
    lz_offset_bucket: Order0<OFFSET_BUCKET_ALPHABET>,
    /// `length - MIN_MATCH` (always in `[0, 255]`).
    lz_length: Order0<256>,
}

impl Models {
    fn new() -> Self {
        Self {
            tag_dict: Order0::new(),
            tag_byte: Order0::new(),
            attr_byte: Order0::new(),
            content: Order1Bytes::new(),
            lz_flag: Order0::new(),
            lz_offset_bucket: Order0::new(),
            lz_length: Order0::new(),
        }
    }
}

impl Codec for XmlLzPpmCodec {
    fn name(&self) -> &'static str {
        "xml-lz-ppm"
    }

    #[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
    fn encode_window(&self, warm: &[u8], measure: &[u8]) -> Result<(Vec<u8>, Decomposition)> {
        // Build the full byte buffer the matcher sees: warm prefix
        // followed by the measure window. Matcher offsets reference
        // positions inside this combined view.
        let mut buf: Vec<u8> = Vec::with_capacity(warm.len() + measure.len());
        buf.extend_from_slice(warm);
        let warm_end = buf.len();
        buf.extend_from_slice(measure);

        let mut classifier = Classifier::new();
        let mut models = Models::new();
        let mut matcher = Matcher::new();

        // Prime the matcher + models + classifier through the warm
        // prefix. Order-1 model only observes Content-mode bytes; the
        // matcher inserts every position once we have three bytes of
        // context. `prewarm` returns the next-insert cursor so we
        // don't re-insert warm positions when extending past `warm`.
        let mut next_insert = prewarm(&buf[..warm_end], &mut classifier, &mut models, &mut matcher);
        next_insert = catch_up_inserts(&buf, warm_end, next_insert, &mut matcher);

        let mut writer = BitWriter::new();
        let mut comp_bits = ComponentBits::default();
        let bits_before_payload = writer.bits_written();
        let mut uniform_cdf_cache = UniformCdfCache::new();
        let mut tag_cdf = [0u32; TAG_ALPHABET + 1];
        let mut byte_cdf = [0u32; 257];
        let mut bucket_cdf = [0u32; OFFSET_BUCKET_ALPHABET + 1];
        let mut flag_cdf = [0u32; 3];
        let mut length_cdf = [0u32; 257];
        {
            let mut enc = AcEncoder::new(&mut writer);

            let mut i = warm_end;
            while i < buf.len() {
                next_insert = catch_up_inserts(&buf, i, next_insert, &mut matcher);
                let mode = classifier.current_mode();
                match mode {
                    Mode::Content => {
                        // Match decision with single-position lazy parse.
                        let m_here = matcher.find_match(&buf, i);
                        let take_match = match m_here {
                            Some((_, len_here)) if i + (len_here as usize) <= buf.len() => {
                                // Lazy parse: defer if a *strictly* longer match
                                // starts at the next position.
                                let m_next = if i + 1 + MIN_MATCH <= buf.len() {
                                    matcher.find_match(&buf, i + 1)
                                } else {
                                    None
                                };
                                !matches!(m_next, Some((_, len_next)) if len_next > len_here)
                            }
                            _ => false,
                        };

                        if take_match {
                            let (offset, length) = m_here.unwrap();
                            // Literal-vs-match flag.
                            models.lz_flag.cdf_to(&mut flag_cdf);
                            let before = enc.bits_written();
                            enc.encode(&flag_cdf, 1);
                            models.lz_flag.observe(1);

                            // Offset: bucket + bucket-relative bits.
                            let bucket = log2_floor(offset);
                            models.lz_offset_bucket.cdf_to(&mut bucket_cdf);
                            enc.encode(&bucket_cdf, bucket as usize);
                            models.lz_offset_bucket.observe(bucket as usize);
                            if bucket > 0 {
                                let rel = offset - (1u32 << bucket);
                                encode_uniform_bits(&mut enc, &mut uniform_cdf_cache, rel, bucket);
                            }

                            // Length - MIN_MATCH.
                            let len_token = (length as usize) - MIN_MATCH;
                            models.lz_length.cdf_to(&mut length_cdf);
                            enc.encode(&length_cdf, len_token);
                            models.lz_length.observe(len_token);

                            comp_bits.lz_match += enc.bits_written() - before;

                            // Advance: classifier through all matched bytes,
                            // Order-1 model is NOT updated on match bytes (they
                            // were encoded by the match record, not the model).
                            let len_us = length as usize;
                            for &b in &buf[i..i + len_us] {
                                classifier.advance(b);
                            }
                            i += len_us;
                        } else {
                            models.lz_flag.cdf_to(&mut flag_cdf);
                            let before = enc.bits_written();
                            enc.encode(&flag_cdf, 0);
                            models.lz_flag.observe(0);
                            models.content.cdf_to(&mut byte_cdf);
                            enc.encode(&byte_cdf, buf[i] as usize);
                            comp_bits.content += enc.bits_written() - before;
                            models.content.observe(buf[i]);
                            classifier.advance(buf[i]);
                            i += 1;
                        }
                    }
                    Mode::AttrValue => {
                        models.attr_byte.cdf_to(&mut byte_cdf);
                        let before = enc.bits_written();
                        enc.encode(&byte_cdf, buf[i] as usize);
                        comp_bits.attr += enc.bits_written() - before;
                        models.attr_byte.observe(buf[i] as usize);
                        classifier.advance(buf[i]);
                        i += 1;
                    }
                    Mode::TagStructure => {
                        let run_end = find_tag_run_end(&buf, i, classifier);
                        let run = &buf[i..run_end];
                        models.tag_dict.cdf_to(&mut tag_cdf);
                        let before = enc.bits_written();
                        if let Some(idx) = dict_lookup(run) {
                            enc.encode(&tag_cdf, idx);
                            models.tag_dict.observe(idx);
                        } else {
                            enc.encode(&tag_cdf, TAG_ESCAPE);
                            models.tag_dict.observe(TAG_ESCAPE);
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

        let payload_bits = writer.bits_written() - bits_before_payload;
        let (mut payload, pad_bits) = writer.finish();

        let measure_len =
            u32::try_from(measure.len()).context("measure window > 4 GiB — unsupported")?;
        let mut archive = Vec::with_capacity(4 + payload.len());
        archive.extend_from_slice(&measure_len.to_le_bytes());
        archive.append(&mut payload);

        let mut decomp = Decomposition::new();
        decomp.add("content_ac", comp_bits.content);
        decomp.add("tag_ac", comp_bits.tag);
        decomp.add("attr_ac", comp_bits.attr);
        decomp.add("lz_match", comp_bits.lz_match);
        let attributed = comp_bits.content + comp_bits.tag + comp_bits.attr + comp_bits.lz_match;
        decomp.add("ac_finish", payload_bits.saturating_sub(attributed));
        decomp.add("framing", 32);
        decomp.add("padding", u64::from(pad_bits));

        Ok((archive, decomp))
    }

    #[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
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

        // The decoder builds the same combined buffer the encoder did,
        // growing the measure portion byte by byte as it decodes.
        let mut buf: Vec<u8> = Vec::with_capacity(warm.len() + measure_len);
        buf.extend_from_slice(warm);
        let warm_end = buf.len();

        let mut classifier = Classifier::new();
        let mut models = Models::new();
        let mut matcher = Matcher::new();
        let mut next_insert = prewarm(&buf, &mut classifier, &mut models, &mut matcher);
        next_insert = catch_up_inserts(&buf, warm_end, next_insert, &mut matcher);

        let mut reader = BitReader::new(&archive[4..]);
        let mut dec = AcDecoder::new(&mut reader);
        let mut uniform_cdf_cache = UniformCdfCache::new();
        let mut tag_cdf = [0u32; TAG_ALPHABET + 1];
        let mut byte_cdf = [0u32; 257];
        let mut bucket_cdf = [0u32; OFFSET_BUCKET_ALPHABET + 1];
        let mut flag_cdf = [0u32; 3];
        let mut length_cdf = [0u32; 257];

        while buf.len() - warm_end < measure_len {
            next_insert = catch_up_inserts(&buf, buf.len(), next_insert, &mut matcher);
            let mode = classifier.current_mode();
            match mode {
                Mode::Content => {
                    models.lz_flag.cdf_to(&mut flag_cdf);
                    let flag = dec.decode(&flag_cdf)?;
                    models.lz_flag.observe(flag);
                    if flag == 1 {
                        models.lz_offset_bucket.cdf_to(&mut bucket_cdf);
                        let bucket = dec.decode(&bucket_cdf)?;
                        models.lz_offset_bucket.observe(bucket);
                        let bucket_u32 = u32::try_from(bucket).expect("bucket fits u32");
                        let offset = if bucket_u32 == 0 {
                            1
                        } else {
                            let rel = decode_uniform_bits(
                                &mut dec,
                                &mut uniform_cdf_cache,
                                bucket_u32,
                            )?;
                            (1u32 << bucket_u32) + rel
                        };

                        models.lz_length.cdf_to(&mut length_cdf);
                        let len_token = dec.decode(&length_cdf)?;
                        models.lz_length.observe(len_token);
                        let length = MIN_MATCH + len_token;

                        let pos = buf.len();
                        let src_start = pos - offset as usize;
                        for k in 0..length {
                            let b = buf[src_start + k];
                            buf.push(b);
                            classifier.advance(b);
                        }
                    } else {
                        models.content.cdf_to(&mut byte_cdf);
                        let sym = dec.decode(&byte_cdf)?;
                        let b = u8::try_from(sym).expect("byte symbol fits u8");
                        buf.push(b);
                        models.content.observe(b);
                        classifier.advance(b);
                    }
                }
                Mode::AttrValue => {
                    models.attr_byte.cdf_to(&mut byte_cdf);
                    let sym = dec.decode(&byte_cdf)?;
                    let b = u8::try_from(sym).expect("byte symbol fits u8");
                    buf.push(b);
                    models.attr_byte.observe(sym);
                    classifier.advance(b);
                }
                Mode::TagStructure => {
                    models.tag_dict.cdf_to(&mut tag_cdf);
                    let idx = dec.decode(&tag_cdf)?;
                    models.tag_dict.observe(idx);
                    if idx == TAG_ESCAPE {
                        loop {
                            models.tag_byte.cdf_to(&mut byte_cdf);
                            let sym = dec.decode(&byte_cdf)?;
                            let b = u8::try_from(sym).expect("byte symbol fits u8");
                            buf.push(b);
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
                            buf.push(b);
                            classifier.advance(b);
                        }
                    }
                }
            }
        }

        let out = buf.split_off(warm_end);
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
    lz_match: u64,
}

/// Cache of uniform CDFs for power-of-2 alphabets, keyed by bit count
/// `k` (alphabet size `2^k`). Only `k ∈ [1, 8]` is supported — for
/// larger bit counts, callers must split the value into ≤8-bit chunks
/// and call the AC once per chunk. (At `TOTAL = 1 << 16`, alphabets
/// larger than 256 force per-symbol mass below `TOTAL/alphabet`,
/// which approaches the integer-zero floor and breaks the AC's
/// strictly-increasing-CDF contract.)
struct UniformCdfCache {
    cdfs: [Option<Vec<u32>>; 9],
}

impl UniformCdfCache {
    const fn new() -> Self {
        Self {
            cdfs: [const { None }; 9],
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn get(&mut self, bits: u32) -> &[u32] {
        let idx = bits as usize;
        assert!(
            bits >= 1 && bits <= 8,
            "uniform CDF requested for {bits}-bit alphabet; use chunked encoding for >8 bits",
        );
        if self.cdfs[idx].is_none() {
            let n: usize = 1 << bits;
            let mut cdf = vec![0u32; n + 1];
            let n_u64 = n as u64;
            for (i, slot) in cdf.iter_mut().enumerate().take(n) {
                *slot = ((i as u64 * u64::from(crate::ac::TOTAL)) / n_u64) as u32;
            }
            cdf[n] = crate::ac::TOTAL;
            self.cdfs[idx] = Some(cdf);
        }
        self.cdfs[idx].as_ref().expect("just populated").as_slice()
    }
}

/// Encode `value` as `n_bits` raw bits through the AC by chunking
/// into ≤8-bit symbols. Each chunk costs exactly `chunk_bits` bits,
/// so the total cost is `n_bits` — equivalent to a raw bit emission,
/// but routed through the same AC stream so framing is consistent.
fn encode_uniform_bits(
    enc: &mut AcEncoder,
    cache: &mut UniformCdfCache,
    value: u32,
    n_bits: u32,
) {
    let mut remaining = n_bits;
    let mut v = value;
    while remaining > 0 {
        let chunk = remaining.min(8);
        let mask = (1u32 << chunk) - 1;
        let cdf = cache.get(chunk);
        let symbol = (v & mask) as usize;
        enc.encode(cdf, symbol);
        v >>= chunk;
        remaining -= chunk;
    }
}

fn decode_uniform_bits(
    dec: &mut AcDecoder,
    cache: &mut UniformCdfCache,
    n_bits: u32,
) -> Result<u32> {
    let mut value: u32 = 0;
    let mut shift: u32 = 0;
    let mut remaining = n_bits;
    while remaining > 0 {
        let chunk = remaining.min(8);
        let cdf = cache.get(chunk);
        let sym = dec.decode(cdf)?;
        value |= u32::try_from(sym).expect("chunk ≤ 8 bits fits u32") << shift;
        shift += chunk;
        remaining -= chunk;
    }
    Ok(value)
}

const fn log2_floor(x: u32) -> u32 {
    x.ilog2()
}

/// Run the classifier, the per-mode models, and the matcher's hash
/// chain through `warm` so the encoder and decoder converge to the
/// same state before the measured region. Returns the
/// `next_insert` cursor the caller should resume from when bytes
/// extend past `warm`.
fn prewarm(
    warm: &[u8],
    classifier: &mut Classifier,
    models: &mut Models,
    matcher: &mut Matcher,
) -> usize {
    if warm.is_empty() {
        return 0;
    }
    let mut i = 0;
    while i < warm.len() {
        let mode = classifier.current_mode();
        match mode {
            Mode::Content => {
                models.content.observe(warm[i]);
                classifier.advance(warm[i]);
                i += 1;
            }
            Mode::AttrValue => {
                models.attr_byte.observe(warm[i] as usize);
                classifier.advance(warm[i]);
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
    catch_up_inserts(warm, warm.len(), 0, matcher)
}

/// Insert all positions `q` with `q + 3 <= up_to` and `q >= next`.
/// Returns the new `next_insert` cursor.
fn catch_up_inserts(buf: &[u8], up_to: usize, mut next: usize, matcher: &mut Matcher) -> usize {
    let bound = up_to.min(buf.len());
    while next + 3 <= bound {
        matcher.insert(buf, next);
        next += 1;
    }
    next
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
        let codec = XmlLzPpmCodec;
        let (archive, _decomp) = codec.encode_window(warm, measure).unwrap();
        let decoded = codec.decode_window(warm, &archive).unwrap();
        assert_eq!(decoded, measure);
    }

    #[test]
    fn roundtrip_short_no_matches() {
        roundtrip(b"", b"<page>quick</page>");
    }

    #[test]
    fn roundtrip_long_repetition() {
        let head = b"This prefix repeats and repeats and repeats again.\n";
        let mut measure: Vec<u8> = Vec::new();
        measure.extend_from_slice(b"<page><text xml:space=\"preserve\">");
        for _ in 0..4 {
            measure.extend_from_slice(head);
        }
        measure.extend_from_slice(b"</text></page>");
        roundtrip(b"", &measure);
    }

    #[test]
    fn roundtrip_with_warm_prefix_match() {
        let warm = b"<page><text xml:space=\"preserve\">Lorem ipsum dolor sit amet.</text></page>";
        let measure =
            b"<page><text xml:space=\"preserve\">Lorem ipsum dolor sit amet.</text></page>";
        roundtrip(warm, measure);
    }

    #[test]
    fn roundtrip_unknown_tag() {
        roundtrip(b"", b"<weird>body content</weird>");
    }

    #[test]
    fn roundtrip_real_enwik8_window() {
        // 4 MiB warm + 4 KiB measure — matches the bench's pre-warm
        // size, isolates the codec from bench plumbing.
        let bytes = match std::fs::read("assets/enwik8") {
            Ok(b) => b,
            Err(_) => return,
        };
        let warm_start = 4_194_304;
        let warm = &bytes[warm_start - 4 * 1024 * 1024..warm_start];
        let measure = &bytes[warm_start..warm_start + 4096];
        roundtrip(warm, measure);
    }

    #[test]
    fn roundtrip_attr_value_with_match_in_content() {
        let warm = b"<page><text xml:space=\"preserve\">abcdefghijklmnop</text></page>";
        let measure = b"<page><text xml:space=\"preserve\">abcdefghijklmnop</text></page>";
        roundtrip(warm, measure);
    }
}
