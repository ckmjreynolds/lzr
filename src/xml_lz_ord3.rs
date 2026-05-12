//! Phase-6 codec: XML schema + LZ77 + Order-3 letter / Order-2
//! non-letter for Content literals.
//!
//! Same structure as `xml-lz-word`, but the letter and non-letter
//! sub-streams now condition on more bytes of history:
//!
//! - `letter`: 52-alphabet, conditioned on the last **3 letters**
//!   (skipping non-letter Content bytes in context). `52^3 = 140 608`
//!   contexts in a dense table (~29 MiB).
//! - `nonletter`: 256-alphabet, conditioned on the last **2 Content
//!   bytes** (regardless of letter class). `256^2 = 65 536` contexts
//!   (~67 MiB).
//!
//! Both with Laplace `+1` smoothing — no escape mechanism. Unseen
//! contexts predict near-uniformly via the `+1` initialization, which
//! the AC absorbs as 8 bpb cost on the rare misses. The dense table
//! approach gives v2 a chunk of v1's PPM-D gain without the
//! escape-bookkeeping complexity.
//!
//! Memory: peak `~96 MiB` per codec instance (encode and decode are
//! sequential per panel offset, so the bench's RSS stays bounded).

use anyhow::{Context, Result, bail};

use crate::ac::{AcDecoder, AcEncoder};
use crate::bits::{BitReader, BitWriter};
use crate::classifier::{Classifier, Mode};
use crate::codec::{Codec, Decomposition};
use crate::lz::{MIN_MATCH, Matcher};
use crate::models::{Order0, Order1Ctx};

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

const OFFSET_BUCKET_ALPHABET: usize = 32;

const N_LETTERS: usize = 52;
const LETTER_NCTX: usize = N_LETTERS * N_LETTERS * N_LETTERS;
const NONLETTER_NCTX: usize = 256 * 256;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct XmlLzOrd3Codec;

struct Models {
    tag_dict: Order0<TAG_ALPHABET>,
    tag_byte: Order0<256>,
    attr_byte: Order0<256>,
    lz_flag: Order0<2>,
    lz_offset_bucket: Order0<OFFSET_BUCKET_ALPHABET>,
    lz_length: Order0<256>,
    is_letter: Order1Ctx<256, 2>,
    letter: Order1Ctx<LETTER_NCTX, N_LETTERS>,
    nonletter: Order1Ctx<NONLETTER_NCTX, 256>,
}

impl Models {
    fn new() -> Self {
        Self {
            tag_dict: Order0::new(),
            tag_byte: Order0::new(),
            attr_byte: Order0::new(),
            lz_flag: Order0::new(),
            lz_offset_bucket: Order0::new(),
            lz_length: Order0::new(),
            is_letter: Order1Ctx::new(),
            letter: Order1Ctx::new(),
            nonletter: Order1Ctx::new(),
        }
    }
}

/// Sliding history for context computation. `last_letters` is a
/// 3-deep ring (oldest first); `last_bytes_2` is a 2-deep ring of
/// all Content bytes regardless of class.
#[allow(clippy::struct_field_names)]
#[derive(Clone, Copy, Debug, Default)]
struct ContentState {
    last_byte: u8,
    last_letters: [Option<u8>; 3],
    last_bytes_2: [u8; 2],
}

impl ContentState {
    const fn advance(&mut self, byte: u8) {
        self.last_byte = byte;
        self.last_bytes_2[0] = self.last_bytes_2[1];
        self.last_bytes_2[1] = byte;
        if let Some(idx) = letter_to_idx(byte) {
            self.last_letters[0] = self.last_letters[1];
            self.last_letters[1] = self.last_letters[2];
            self.last_letters[2] = Some(idx);
        }
    }

    fn letter_ctx(&self) -> usize {
        let a = self.last_letters[0].map_or(0, |x| x as usize);
        let b = self.last_letters[1].map_or(0, |x| x as usize);
        let c = self.last_letters[2].map_or(0, |x| x as usize);
        (a * N_LETTERS + b) * N_LETTERS + c
    }

    const fn nonletter_ctx(&self) -> usize {
        (self.last_bytes_2[0] as usize) * 256 + (self.last_bytes_2[1] as usize)
    }
}

const fn letter_to_idx(b: u8) -> Option<u8> {
    match b {
        b'a'..=b'z' => Some(b - b'a'),
        b'A'..=b'Z' => Some(26 + (b - b'A')),
        _ => None,
    }
}

#[allow(clippy::cast_possible_truncation)]
const fn idx_to_letter(idx: usize) -> u8 {
    if idx < 26 {
        b'a' + idx as u8
    } else {
        b'A' + (idx - 26) as u8
    }
}

impl Codec for XmlLzOrd3Codec {
    fn name(&self) -> &'static str {
        "xml-lz-ord3"
    }

    #[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
    fn encode_window(&self, warm: &[u8], measure: &[u8]) -> Result<(Vec<u8>, Decomposition)> {
        let mut buf: Vec<u8> = Vec::with_capacity(warm.len() + measure.len());
        buf.extend_from_slice(warm);
        let warm_end = buf.len();
        buf.extend_from_slice(measure);

        let mut classifier = Classifier::new();
        let mut models = Models::new();
        let mut matcher = Matcher::new();
        let mut content_state = ContentState::default();

        let mut next_insert = prewarm(
            &buf[..warm_end],
            &mut classifier,
            &mut models,
            &mut matcher,
            &mut content_state,
        );
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
        let mut is_letter_cdf = [0u32; 3];
        let mut letter_cdf = [0u32; N_LETTERS + 1];
        {
            let mut enc = AcEncoder::new(&mut writer);

            let mut i = warm_end;
            while i < buf.len() {
                next_insert = catch_up_inserts(&buf, i, next_insert, &mut matcher);
                let mode = classifier.current_mode();
                match mode {
                    Mode::Content => {
                        let m_here = matcher.find_match(&buf, i);
                        let take_match = match m_here {
                            Some((_, len_here)) if i + (len_here as usize) <= buf.len() => {
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
                            models.lz_flag.cdf_to(&mut flag_cdf);
                            let before = enc.bits_written();
                            enc.encode(&flag_cdf, 1);
                            models.lz_flag.observe(1);

                            let bucket = log2_floor(offset);
                            models.lz_offset_bucket.cdf_to(&mut bucket_cdf);
                            enc.encode(&bucket_cdf, bucket as usize);
                            models.lz_offset_bucket.observe(bucket as usize);
                            if bucket > 0 {
                                let rel = offset - (1u32 << bucket);
                                encode_uniform_bits(&mut enc, &mut uniform_cdf_cache, rel, bucket);
                            }
                            let len_token = (length as usize) - MIN_MATCH;
                            models.lz_length.cdf_to(&mut length_cdf);
                            enc.encode(&length_cdf, len_token);
                            models.lz_length.observe(len_token);

                            comp_bits.lz_match += enc.bits_written() - before;

                            let len_us = length as usize;
                            for &b in &buf[i..i + len_us] {
                                classifier.advance(b);
                                content_state.advance(b);
                            }
                            i += len_us;
                        } else {
                            models.lz_flag.cdf_to(&mut flag_cdf);
                            let before = enc.bits_written();
                            enc.encode(&flag_cdf, 0);
                            models.lz_flag.observe(0);

                            let byte = buf[i];
                            let ctx_byte = content_state.last_byte as usize;
                            let is_letter_val = usize::from(letter_to_idx(byte).is_some());
                            models.is_letter.cdf_to(ctx_byte, &mut is_letter_cdf);
                            enc.encode(&is_letter_cdf, is_letter_val);
                            models.is_letter.observe(ctx_byte, is_letter_val);

                            if is_letter_val == 1 {
                                let letter_idx = letter_to_idx(byte)
                                    .expect("just classified as letter")
                                    as usize;
                                let ctx_letter = content_state.letter_ctx();
                                models.letter.cdf_to(ctx_letter, &mut letter_cdf);
                                enc.encode(&letter_cdf, letter_idx);
                                models.letter.observe(ctx_letter, letter_idx);
                            } else {
                                let ctx_nonletter = content_state.nonletter_ctx();
                                models.nonletter.cdf_to(ctx_nonletter, &mut byte_cdf);
                                enc.encode(&byte_cdf, byte as usize);
                                models.nonletter.observe(ctx_nonletter, byte as usize);
                            }

                            comp_bits.content += enc.bits_written() - before;
                            classifier.advance(byte);
                            content_state.advance(byte);
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

        let mut buf: Vec<u8> = Vec::with_capacity(warm.len() + measure_len);
        buf.extend_from_slice(warm);
        let warm_end = buf.len();

        let mut classifier = Classifier::new();
        let mut models = Models::new();
        let mut matcher = Matcher::new();
        let mut content_state = ContentState::default();
        let mut next_insert = prewarm(
            &buf,
            &mut classifier,
            &mut models,
            &mut matcher,
            &mut content_state,
        );
        next_insert = catch_up_inserts(&buf, warm_end, next_insert, &mut matcher);

        let mut reader = BitReader::new(&archive[4..]);
        let mut dec = AcDecoder::new(&mut reader);
        let mut uniform_cdf_cache = UniformCdfCache::new();
        let mut tag_cdf = [0u32; TAG_ALPHABET + 1];
        let mut byte_cdf = [0u32; 257];
        let mut bucket_cdf = [0u32; OFFSET_BUCKET_ALPHABET + 1];
        let mut flag_cdf = [0u32; 3];
        let mut length_cdf = [0u32; 257];
        let mut is_letter_cdf = [0u32; 3];
        let mut letter_cdf = [0u32; N_LETTERS + 1];

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
                            let rel =
                                decode_uniform_bits(&mut dec, &mut uniform_cdf_cache, bucket_u32)?;
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
                            content_state.advance(b);
                        }
                    } else {
                        let ctx_byte = content_state.last_byte as usize;
                        models.is_letter.cdf_to(ctx_byte, &mut is_letter_cdf);
                        let is_letter_val = dec.decode(&is_letter_cdf)?;
                        models.is_letter.observe(ctx_byte, is_letter_val);

                        let b = if is_letter_val == 1 {
                            let ctx_letter = content_state.letter_ctx();
                            models.letter.cdf_to(ctx_letter, &mut letter_cdf);
                            let letter_idx = dec.decode(&letter_cdf)?;
                            models.letter.observe(ctx_letter, letter_idx);
                            idx_to_letter(letter_idx)
                        } else {
                            let ctx_nonletter = content_state.nonletter_ctx();
                            models.nonletter.cdf_to(ctx_nonletter, &mut byte_cdf);
                            let sym = dec.decode(&byte_cdf)?;
                            models.nonletter.observe(ctx_nonletter, sym);
                            u8::try_from(sym).expect("byte symbol fits u8")
                        };

                        buf.push(b);
                        classifier.advance(b);
                        content_state.advance(b);
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
            (1..=8).contains(&bits),
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

fn encode_uniform_bits(
    enc: &mut AcEncoder<'_>,
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
    dec: &mut AcDecoder<'_, '_>,
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

fn prewarm(
    warm: &[u8],
    classifier: &mut Classifier,
    models: &mut Models,
    matcher: &mut Matcher,
    content_state: &mut ContentState,
) -> usize {
    if warm.is_empty() {
        return 0;
    }
    let mut i = 0;
    while i < warm.len() {
        let mode = classifier.current_mode();
        match mode {
            Mode::Content => {
                let byte = warm[i];
                let ctx_byte = content_state.last_byte as usize;
                let is_letter_val = usize::from(letter_to_idx(byte).is_some());
                models.is_letter.observe(ctx_byte, is_letter_val);
                if is_letter_val == 1 {
                    let letter_idx =
                        letter_to_idx(byte).expect("just classified as letter") as usize;
                    let ctx_letter = content_state.letter_ctx();
                    models.letter.observe(ctx_letter, letter_idx);
                } else {
                    let ctx_nonletter = content_state.nonletter_ctx();
                    models.nonletter.observe(ctx_nonletter, byte as usize);
                }
                classifier.advance(byte);
                content_state.advance(byte);
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
        let codec = XmlLzOrd3Codec;
        let (archive, _decomp) = codec.encode_window(warm, measure).unwrap();
        let decoded = codec.decode_window(warm, &archive).unwrap();
        assert_eq!(decoded, measure);
    }

    #[test]
    fn roundtrip_short_mixed_letters() {
        roundtrip(b"", b"<page>The quick brown fox jumps.</page>");
    }

    #[test]
    fn roundtrip_with_attr_value() {
        roundtrip(
            b"",
            b"<text xml:space=\"preserve\">Hello, World! 123 ABC.</text>",
        );
    }

    #[test]
    fn roundtrip_real_enwik8_window() {
        let Ok(bytes) = std::fs::read("assets/enwik8") else {
            return;
        };
        let warm_start = 4_194_304;
        let warm = &bytes[warm_start - 4 * 1024 * 1024..warm_start];
        let measure = &bytes[warm_start..warm_start + 8192];
        roundtrip(warm, measure);
    }
}
