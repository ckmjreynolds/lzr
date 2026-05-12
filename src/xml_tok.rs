//! Phase-13 codec: word/separator tokenization with online dictionary.
//!
//! Departs from the byte-level LZ77 → adaptive-Order-N stack that
//! shapes Phases 4–12. Content is tokenized into alternating word
//! (`[a-zA-Z]+`) and separator (`[^a-zA-Z]+`) runs; each token is
//! either a hit against a u16-indexed dictionary built online via
//! first-occurrence emission, or an OOV that emits length + bytes
//! and is then inserted into the dictionary. Encoder and decoder
//! grow the dictionary in lockstep, so no dictionary is shipped.
//!
//! Word tokens are case-folded: the dictionary stores only the
//! lowercase byte sequence, and an out-of-band case pattern code
//! tells the decoder how to reconstruct the original case. Per
//! CDR's insight, without folding the dictionary fragments on
//! "The"/"the"/"THE"-style variants and burns ~40-50% of its slots.
//!
//! Per-token case stats live on the dictionary entry from day one
//! ("United" is title-case ~98% of the time it appears; "the" is
//! title-case ~10% of the time). The case-pattern probability comes
//! from those per-token counts with Laplace `+1` smoothing.
//!
//! Tag and attr modes are unchanged from Phase 7 (xml-lz-cp).
//!
//! ## Pre-measurement prediction
//!
//! Conditional entropy on a u16 word alphabet with Zipfian
//! distribution: top-256 words get ~3-5 bits each; tail (256-65535)
//! pays ~8-16 bits depending on frequency; OOV pays per-byte cost
//! at ~5 bits/letter. Target: **2.1-2.3 bpb** on the quick panel
//! (vs xml-lz-cp at 2.506), an 8-15% relative reduction.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};

use crate::ac::{AcDecoder, AcEncoder, TOTAL};
use crate::bit_pred::{BitPredictor, FNV_OFFSET, fnv_mix};
use crate::bits::{BitReader, BitWriter};
use crate::classifier::{Classifier, Mode};
use crate::codec::{Codec, Decomposition};
use crate::models::{Order0, Order1Ctx};
use crate::tok_lz::{self, TokenMatcher};
use crate::tokenizer::{
    CasePattern, TokenClass, apply_case, classify_case, lowercase, mixed_mask, run_end,
};

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

/// Maximum dictionary entries. v3 Phase 1: widened from 1 << 16
/// (the u16 packaging cap that the survey showed costs ~0.527 bpb
/// of pure-entropy headroom at enwik8 scale due to 7% OOV) to
/// 1 << 18 = 262,144 — eliminates OOV on enwik8 measure windows
/// and leaves headroom for enwik9's ~3× larger unique-type count.
/// IDs are emitted as 18 bits per dictionary entry; the bit-level
/// predictor sees the prev-id context at 18-bit resolution.
const DICT_CAPACITY: usize = 1 << 18;
/// Number of bits used to emit each dictionary id.
const ID_BITS: u32 = 18;

/// Class-bit context: 0 = no prior token (Content-run start), 1 =
/// last was Word, 2 = last was Separator. Forced alternation
/// collapses the within-run cost to near-zero after warm-up.
const CLASS_CTX_NONE: usize = 0;
const CLASS_CTX_WORD: usize = 1;
const CLASS_CTX_SEP: usize = 2;
const CLASS_NCTX: usize = 3;

/// Letter context for OOV-word byte emission: index 26 = "start of
/// token"; 0-25 = previous lowercase letter.
const LETTER_START: usize = 26;
const LETTER_NCTX: usize = 27;

/// Byte context for OOV-separator emission: index 256 = "start of
/// token"; 0-255 = previous byte.
const BYTE_START: usize = 256;
const BYTE_NCTX: usize = 257;

/// Hash-table size (log₂) for the bit-level dict-id predictor.
/// Phase 15 folds `prev_id` into the context to make this an
/// Order-1 word model; the `(prev_id, bit_pos, prefix)` space is far
/// larger than `(bit_pos, prefix)` alone. v3 Phase 1 widens IDs to
/// 18 bits and bumps the table to 2^24 = 16 M slots (64 MiB) to
/// keep collision rate near zero at the new context-space size.
const ID_BIT_K: u32 = 24;
/// Hash-table size for the bit-level length predictor. Length is
/// independent of `prev_id`, so the context space stays as in
/// Phase 14: 2^18 slots = 1 MiB.
const LEN_BIT_K: u32 = 18;

/// LZ-on-tokens offset bucketing: bucket = `log₂(offset)`. Offsets
/// up to `WINDOW_SIZE` (1 M tokens) need 20 buckets. Use 21 to give
/// the model room when the offset hits the upper end.
const OFFSET_BUCKET_ALPHABET: usize = 21;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct XmlTokCodec;

#[derive(Debug)]
struct DictEntry {
    lower: Vec<u8>,
    case_counts: [u32; 4],
}

impl DictEntry {
    const fn new(lower: Vec<u8>) -> Self {
        Self {
            lower,
            case_counts: [0; 4],
        }
    }
}

#[derive(Debug, Default)]
struct Dict {
    by_bytes: HashMap<Vec<u8>, u32>,
    entries: Vec<DictEntry>,
}

impl Dict {
    fn get(&self, key: &[u8]) -> Option<u32> {
        self.by_bytes.get(key).copied()
    }

    /// Insert a new entry if there's room. Returns the new id, or
    /// `None` if the dictionary is full.
    fn insert(&mut self, key: Vec<u8>) -> Option<u32> {
        if self.entries.len() >= DICT_CAPACITY {
            return None;
        }
        let id = u32::try_from(self.entries.len()).expect("entries.len() < DICT_CAPACITY");
        self.by_bytes.insert(key.clone(), id);
        self.entries.push(DictEntry::new(key));
        Some(id)
    }

    fn entry_mut(&mut self, id: u32) -> &mut DictEntry {
        &mut self.entries[id as usize]
    }

    fn entry(&self, id: u32) -> &DictEntry {
        &self.entries[id as usize]
    }
}

struct Models {
    token_class: Order1Ctx<CLASS_NCTX, 2>,
    token_oov: Order0<2>,

    /// Bit-level adaptive predictor for the 16-bit dictionary id.
    /// Context = (bit position, prefix bits emitted so far). The MSB
    /// is emitted first; once a hot token's high bits collapse to a
    /// predictable 0-prefix the bit cost converges to Shannon. Replaces
    /// the Phase-13 2-byte chunked encoding that starved tail rows.
    token_id_bit: BitPredictor,
    /// Same construction for OOV token lengths.
    token_length_bit: BitPredictor,

    /// Phase 16: LZ-on-tokens match-vs-literal flag (1 bit emitted
    /// per Content-token boundary).
    lz_flag: Order0<2>,
    /// `log₂(offset)` bucket. Bucket-relative low bits emitted via
    /// the uniform CDF cache.
    lz_offset_bucket: Order0<OFFSET_BUCKET_ALPHABET>,
    /// `length - MIN_MATCH` token, Order-0 over 0..=255.
    lz_length: Order0<256>,

    oov_word_letter: Order1Ctx<LETTER_NCTX, 26>,
    oov_sep_byte: Order1Ctx<BYTE_NCTX, 256>,

    case_pattern_global: Order0<4>,
    case_mixed_bit: Order0<2>,

    tag_dict: Order0<TAG_ALPHABET>,
    tag_byte: Order0<256>,
    attr_byte: Order0<256>,
}

impl Models {
    fn new() -> Self {
        Self {
            token_class: Order1Ctx::new(),
            token_oov: Order0::new(),
            token_id_bit: BitPredictor::new(ID_BIT_K),
            token_length_bit: BitPredictor::new(LEN_BIT_K),
            lz_flag: Order0::new(),
            lz_offset_bucket: Order0::new(),
            lz_length: Order0::new(),
            oov_word_letter: Order1Ctx::new(),
            oov_sep_byte: Order1Ctx::new(),
            case_pattern_global: Order0::new(),
            case_mixed_bit: Order0::new(),
            tag_dict: Order0::new(),
            tag_byte: Order0::new(),
            attr_byte: Order0::new(),
        }
    }
}

#[derive(Default)]
struct ComponentBits {
    token_hit: u64,
    token_oov: u64,
    token_class: u64,
    case: u64,
    lz_match: u64,
    tag: u64,
    attr: u64,
}

impl Codec for XmlTokCodec {
    fn name(&self) -> &'static str {
        "xml-tok"
    }

    #[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
    fn encode_window(&self, warm: &[u8], measure: &[u8]) -> Result<(Vec<u8>, Decomposition)> {
        let mut buf: Vec<u8> = Vec::with_capacity(warm.len() + measure.len());
        buf.extend_from_slice(warm);
        let warm_end = buf.len();
        buf.extend_from_slice(measure);

        let mut classifier = Classifier::new();
        let mut models = Models::new();
        let mut dict = Dict::default();
        let mut matcher = TokenMatcher::new();

        prewarm(
            &buf[..warm_end],
            &mut classifier,
            &mut models,
            &mut dict,
            &mut matcher,
        );

        let mut writer = BitWriter::new();
        let mut comp = ComponentBits::default();
        let bits_before_payload = writer.bits_written();
        {
            let mut enc = AcEncoder::new(&mut writer);
            let mut last_class_ctx = CLASS_CTX_NONE;
            let mut prev_id: Option<u32> = None;
            let mut uniform_cdf_cache = UniformCdfCache::new();
            let mut i = warm_end;
            while i < buf.len() {
                let mode = classifier.current_mode();
                match mode {
                    Mode::Content => {
                        // Try LZ match on the upcoming Content tokens.
                        let lookahead = lookahead_tokens(&buf, i, &dict);
                        let lz_match = if lookahead.len() >= tok_lz::MIN_MATCH {
                            let ids: Vec<u32> = lookahead.iter().map(|t| t.id).collect();
                            matcher.find_match(&ids).and_then(|(off, len)| {
                                let len_us = len as usize;
                                if len_us <= lookahead.len() && len_us >= tok_lz::MIN_MATCH {
                                    Some((off, len))
                                } else {
                                    None
                                }
                            })
                        } else {
                            None
                        };

                        let mut flag_cdf = [0u32; 3];
                        models.lz_flag.cdf_to(&mut flag_cdf);
                        let flag_sym = usize::from(lz_match.is_some());
                        let flag_before = enc.bits_written();
                        enc.encode(&flag_cdf, flag_sym);
                        models.lz_flag.observe(flag_sym);
                        comp.lz_match += enc.bits_written() - flag_before;

                        if let Some((offset, length)) = lz_match {
                            let length_us = length as usize;
                            let lz_before = enc.bits_written();

                            let bucket = log2_floor(offset);
                            let mut bucket_cdf = [0u32; OFFSET_BUCKET_ALPHABET + 1];
                            models.lz_offset_bucket.cdf_to(&mut bucket_cdf);
                            enc.encode(&bucket_cdf, bucket as usize);
                            models.lz_offset_bucket.observe(bucket as usize);
                            if bucket > 0 {
                                let rel = offset - (1u32 << bucket);
                                encode_uniform_bits(&mut enc, &mut uniform_cdf_cache, rel, bucket);
                            }

                            let length_token = length_us - tok_lz::MIN_MATCH;
                            let mut length_cdf = [0u32; 257];
                            models.lz_length.cdf_to(&mut length_cdf);
                            enc.encode(&length_cdf, length_token);
                            models.lz_length.observe(length_token);

                            comp.lz_match += enc.bits_written() - lz_before;

                            // Per-token case for each matched word token.
                            for k in 0..length_us {
                                let lhk = lookahead[k];
                                if lhk.class == TokenClass::Word {
                                    let token_start = if k == 0 { i } else { lookahead[k - 1].end };
                                    let token_bytes = &buf[token_start..lhk.end];
                                    let case_pat = classify_case(token_bytes);
                                    let cb = enc.bits_written();
                                    encode_case_with_token_prior(
                                        &mut enc,
                                        &mut models,
                                        &mut dict,
                                        lhk.id,
                                        case_pat,
                                        token_bytes,
                                    );
                                    comp.case += enc.bits_written() - cb;
                                }
                                matcher.push(lhk.id);
                            }

                            let new_i = lookahead[length_us - 1].end;
                            for &b in &buf[i..new_i] {
                                classifier.advance(b);
                            }
                            i = new_i;
                            let last = lookahead[length_us - 1];
                            last_class_ctx = match last.class {
                                TokenClass::Word => CLASS_CTX_WORD,
                                TokenClass::Separator => CLASS_CTX_SEP,
                            };
                            prev_id = Some(last.id);
                            continue;
                        }

                        // No LZ match — emit a regular token.
                        let class = TokenClass::from_byte(buf[i]);
                        let class_sym = class_to_sym(class);
                        let mut class_cdf = [0u32; 3];
                        models.token_class.cdf_to(last_class_ctx, &mut class_cdf);
                        let before = enc.bits_written();
                        enc.encode(&class_cdf, class_sym);
                        models.token_class.observe(last_class_ctx, class_sym);
                        comp.token_class += enc.bits_written() - before;

                        let end = run_end(&buf, i);
                        let token = &buf[i..end];
                        let new_id = encode_token(
                            &mut enc,
                            &mut models,
                            &mut dict,
                            &mut comp,
                            class,
                            token,
                            prev_id,
                        );
                        if let Some(id) = new_id {
                            matcher.push(id);
                        }
                        prev_id = new_id;

                        for &b in token {
                            classifier.advance(b);
                        }
                        i = end;
                        last_class_ctx = match class {
                            TokenClass::Word => CLASS_CTX_WORD,
                            TokenClass::Separator => CLASS_CTX_SEP,
                        };
                    }
                    Mode::AttrValue => {
                        let mut byte_cdf = [0u32; 257];
                        models.attr_byte.cdf_to(&mut byte_cdf);
                        let before = enc.bits_written();
                        enc.encode(&byte_cdf, buf[i] as usize);
                        comp.attr += enc.bits_written() - before;
                        models.attr_byte.observe(buf[i] as usize);
                        classifier.advance(buf[i]);
                        i += 1;
                        last_class_ctx = CLASS_CTX_NONE;
                        prev_id = None;
                    }
                    Mode::TagStructure => {
                        let tag_end = find_tag_run_end(&buf, i, classifier);
                        let run = &buf[i..tag_end];
                        let mut tag_cdf = [0u32; TAG_ALPHABET + 1];
                        models.tag_dict.cdf_to(&mut tag_cdf);
                        let before = enc.bits_written();
                        if let Some(idx) = dict_lookup(run) {
                            enc.encode(&tag_cdf, idx);
                            models.tag_dict.observe(idx);
                        } else {
                            enc.encode(&tag_cdf, TAG_ESCAPE);
                            models.tag_dict.observe(TAG_ESCAPE);
                            let mut byte_cdf = [0u32; 257];
                            for &b in run {
                                models.tag_byte.cdf_to(&mut byte_cdf);
                                enc.encode(&byte_cdf, b as usize);
                                models.tag_byte.observe(b as usize);
                            }
                        }
                        comp.tag += enc.bits_written() - before;
                        for &b in run {
                            classifier.advance(b);
                        }
                        i = tag_end;
                        last_class_ctx = CLASS_CTX_NONE;
                        prev_id = None;
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
        decomp.add("token_hit_ac", comp.token_hit);
        decomp.add("token_oov_ac", comp.token_oov);
        decomp.add("token_class_ac", comp.token_class);
        decomp.add("case_ac", comp.case);
        decomp.add("lz_match_ac", comp.lz_match);
        decomp.add("tag_ac", comp.tag);
        decomp.add("attr_ac", comp.attr);
        let attributed = comp.token_hit
            + comp.token_oov
            + comp.token_class
            + comp.case
            + comp.lz_match
            + comp.tag
            + comp.attr;
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
        let mut dict = Dict::default();
        let mut matcher = TokenMatcher::new();
        prewarm(&buf, &mut classifier, &mut models, &mut dict, &mut matcher);

        let mut reader = BitReader::new(&archive[4..]);
        let mut dec = AcDecoder::new(&mut reader);
        let mut last_class_ctx = CLASS_CTX_NONE;
        let mut prev_id: Option<u32> = None;
        let mut uniform_cdf_cache = UniformCdfCache::new();
        while buf.len() - warm_end < measure_len {
            let mode = classifier.current_mode();
            match mode {
                Mode::Content => {
                    let mut flag_cdf = [0u32; 3];
                    models.lz_flag.cdf_to(&mut flag_cdf);
                    let flag = dec.decode(&flag_cdf)?;
                    models.lz_flag.observe(flag);

                    if flag == 1 {
                        // LZ match record.
                        let mut bucket_cdf = [0u32; OFFSET_BUCKET_ALPHABET + 1];
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

                        let mut length_cdf = [0u32; 257];
                        models.lz_length.cdf_to(&mut length_cdf);
                        let length_token = dec.decode(&length_cdf)?;
                        models.lz_length.observe(length_token);
                        let length = tok_lz::MIN_MATCH + length_token;

                        let stream_len = matcher.stream_len();
                        let src_start =
                            stream_len.checked_sub(offset as usize).with_context(|| {
                                format!("match offset {offset} exceeds stream length {stream_len}")
                            })?;

                        // Collect matched ids, then emit them. Read ids
                        // first so the matcher's stream isn't mutated
                        // mid-loop (would invalidate the source range).
                        let copied: Vec<u32> =
                            (0..length).map(|k| matcher.at(src_start + k)).collect();

                        for id in copied {
                            let lower = dict.entry(id).lower.clone();
                            let class = TokenClass::from_byte(lower[0]);
                            let bytes = if class == TokenClass::Word {
                                let pat = decode_case_with_token_prior(
                                    &mut dec,
                                    &mut models,
                                    &dict,
                                    id,
                                    lower.len(),
                                )?;
                                case_counts_inc(&mut dict.entry_mut(id).case_counts, pat.idx());
                                apply_case_decoded(&lower, pat, &mut dec, &mut models)?
                            } else {
                                lower
                            };
                            for &b in &bytes {
                                buf.push(b);
                                classifier.advance(b);
                            }
                            matcher.push(id);
                            last_class_ctx = match class {
                                TokenClass::Word => CLASS_CTX_WORD,
                                TokenClass::Separator => CLASS_CTX_SEP,
                            };
                            prev_id = Some(id);
                        }
                        continue;
                    }

                    let mut class_cdf = [0u32; 3];
                    models.token_class.cdf_to(last_class_ctx, &mut class_cdf);
                    let class_sym = dec.decode(&class_cdf)?;
                    models.token_class.observe(last_class_ctx, class_sym);
                    let class = sym_to_class(class_sym)
                        .with_context(|| format!("invalid class symbol {class_sym}"))?;

                    let (token_bytes, new_prev) =
                        decode_token(&mut dec, &mut models, &mut dict, class, prev_id)?;
                    for &b in &token_bytes {
                        buf.push(b);
                        classifier.advance(b);
                    }
                    if let Some(id) = new_prev {
                        matcher.push(id);
                    }
                    last_class_ctx = match class {
                        TokenClass::Word => CLASS_CTX_WORD,
                        TokenClass::Separator => CLASS_CTX_SEP,
                    };
                    prev_id = new_prev;
                }
                Mode::AttrValue => {
                    let mut byte_cdf = [0u32; 257];
                    models.attr_byte.cdf_to(&mut byte_cdf);
                    let sym = dec.decode(&byte_cdf)?;
                    let b = u8::try_from(sym).expect("byte symbol fits u8");
                    buf.push(b);
                    models.attr_byte.observe(sym);
                    classifier.advance(b);
                    last_class_ctx = CLASS_CTX_NONE;
                    prev_id = None;
                }
                Mode::TagStructure => {
                    let mut tag_cdf = [0u32; TAG_ALPHABET + 1];
                    models.tag_dict.cdf_to(&mut tag_cdf);
                    let idx = dec.decode(&tag_cdf)?;
                    models.tag_dict.observe(idx);
                    if idx == TAG_ESCAPE {
                        let mut byte_cdf = [0u32; 257];
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
                    last_class_ctx = CLASS_CTX_NONE;
                    prev_id = None;
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

const fn class_to_sym(c: TokenClass) -> usize {
    match c {
        TokenClass::Word => 0,
        TokenClass::Separator => 1,
    }
}

const fn sym_to_class(s: usize) -> Option<TokenClass> {
    match s {
        0 => Some(TokenClass::Word),
        1 => Some(TokenClass::Separator),
        _ => None,
    }
}

/// Compose `(case_counts + 1)` into a 5-entry CDF for the 4-symbol
/// case-pattern alphabet, scaled to `TOTAL`.
#[allow(clippy::cast_possible_truncation)]
fn case_cdf_from_counts(counts: &[u32; 4], out: &mut [u32; 5]) {
    let total: u64 = u64::from(counts[0])
        + u64::from(counts[1])
        + u64::from(counts[2])
        + u64::from(counts[3])
        + 4;
    let mut acc: u64 = 0;
    for (slot, &c) in out.iter_mut().zip(counts.iter()).take(4) {
        *slot = ((acc * u64::from(TOTAL)) / total) as u32;
        acc += u64::from(c) + 1;
    }
    out[4] = TOTAL;
}

/// Rescale threshold for per-dict-entry case counts. Above this the
/// counts get halved (with `.max(1)`) so the smoothed total stays
/// below `TOTAL`, which is what `case_cdf_from_counts` needs to keep
/// every symbol at ≥ 1 unit of mass.
///
/// Without this, hot entries (e.g. "the" with ~1 M `AllLower`
/// observations and a handful of `TitleCase` ones) produce a CDF
/// where the rare patterns collapse to zero mass. Encoding that
/// rare pattern then yields `new_high < new_low` and the AC
/// permanently desyncs from the decoder (manifests at ~75 MB on
/// enwik8).
const CASE_COUNT_RESCALE: u32 = 1 << 15;

/// Increment one case-pattern bucket and rescale if the bucket totals
/// would push the smoothed sum past `TOTAL/2`. Encoder, decoder and
/// prewarm all funnel through this so the case CDF stays in lock-step
/// across all three.
fn case_counts_inc(counts: &mut [u32; 4], idx: usize) {
    counts[idx] = counts[idx].saturating_add(1);
    let total = counts[0]
        .saturating_add(counts[1])
        .saturating_add(counts[2])
        .saturating_add(counts[3]);
    if total > CASE_COUNT_RESCALE {
        for c in counts.iter_mut() {
            *c = (*c >> 1).max(1);
        }
    }
}

/// Encode one Content token and return the dictionary id assigned
/// to it (for use as the next token's `prev_id`). Returns `None`
/// when the token was OOV and the dictionary was already full —
/// in that case the next token has no usable conditional context.
fn encode_token(
    enc: &mut AcEncoder<'_>,
    models: &mut Models,
    dict: &mut Dict,
    comp: &mut ComponentBits,
    class: TokenClass,
    token: &[u8],
    prev_id: Option<u32>,
) -> Option<u32> {
    // Word tokens carry case info; the dict key is the lowercase form.
    // Separator tokens use their raw bytes as the dict key.
    let (case_pat, key): (Option<CasePattern>, Vec<u8>) = match class {
        TokenClass::Word => {
            let pat = classify_case(token);
            (Some(pat), lowercase(token))
        }
        TokenClass::Separator => (None, token.to_vec()),
    };

    let hit_id = dict.get(&key);
    let mut oov_cdf = [0u32; 3];
    models.token_oov.cdf_to(&mut oov_cdf);
    let oov_sym = usize::from(hit_id.is_none());
    let before = enc.bits_written();
    enc.encode(&oov_cdf, oov_sym);
    models.token_oov.observe(oov_sym);
    comp.token_hit += enc.bits_written() - before;

    if let Some(id) = hit_id {
        let before = enc.bits_written();
        encode_dict_id(enc, models, prev_id, id);
        comp.token_hit += enc.bits_written() - before;

        if let (TokenClass::Word, Some(pat)) = (class, case_pat) {
            let cb = enc.bits_written();
            encode_case_with_token_prior(enc, models, dict, id, pat, token);
            comp.case += enc.bits_written() - cb;
        }
        Some(id)
    } else {
        // OOV path: emit length + bytes, then insert into dict.
        let oov_before = enc.bits_written();
        encode_length(enc, models, key.len());
        match class {
            TokenClass::Word => encode_oov_word_bytes(enc, models, &key),
            TokenClass::Separator => encode_oov_sep_bytes(enc, models, &key),
        }
        comp.token_oov += enc.bits_written() - oov_before;

        if let (TokenClass::Word, Some(pat)) = (class, case_pat) {
            let cb = enc.bits_written();
            encode_case_global(enc, models, pat, token);
            comp.case += enc.bits_written() - cb;
        }

        let inserted = dict.insert(key);
        if let (Some(new_id), Some(pat)) = (inserted, case_pat) {
            // Bootstrap the new entry's case stats with this occurrence.
            case_counts_inc(&mut dict.entry_mut(new_id).case_counts, pat.idx());
        }
        inserted
    }
}

/// Decode one Content token. Returns the reconstructed bytes and
/// the dictionary id assigned (for the next token's `prev_id`).
#[allow(clippy::too_many_lines)]
fn decode_token(
    dec: &mut AcDecoder<'_, '_>,
    models: &mut Models,
    dict: &mut Dict,
    class: TokenClass,
    prev_id: Option<u32>,
) -> Result<(Vec<u8>, Option<u32>)> {
    let mut oov_cdf = [0u32; 3];
    models.token_oov.cdf_to(&mut oov_cdf);
    let oov_sym = dec.decode(&oov_cdf)?;
    models.token_oov.observe(oov_sym);

    if oov_sym == 0 {
        let id = decode_dict_id(dec, models, prev_id)?;
        let lower = dict.entry(id).lower.clone();
        let bytes = if class == TokenClass::Word {
            let pat = decode_case_with_token_prior(dec, models, dict, id, lower.len())?;
            case_counts_inc(&mut dict.entry_mut(id).case_counts, pat.idx());
            apply_case_decoded(&lower, pat, dec, models)?
        } else {
            lower
        };
        Ok((bytes, Some(id)))
    } else {
        let length = decode_length(dec, models)?;
        let bytes = match class {
            TokenClass::Word => decode_oov_word_bytes(dec, models, length)?,
            TokenClass::Separator => decode_oov_sep_bytes(dec, models, length)?,
        };
        if class == TokenClass::Word {
            let pat = decode_case_global(dec, models, length)?;
            let mask = if pat == CasePattern::Mixed {
                Some(decode_mixed_mask(dec, models, length)?)
            } else {
                None
            };
            let word = apply_case(&bytes, pat, mask.as_deref());
            let inserted = dict.insert(bytes);
            if let Some(new_id) = inserted {
                case_counts_inc(&mut dict.entry_mut(new_id).case_counts, pat.idx());
            }
            Ok((word, inserted))
        } else {
            let inserted = dict.insert(bytes.clone());
            Ok((bytes, inserted))
        }
    }
}

fn apply_case_decoded(
    lower: &[u8],
    pat: CasePattern,
    dec: &mut AcDecoder<'_, '_>,
    models: &mut Models,
) -> Result<Vec<u8>> {
    let mask = if pat == CasePattern::Mixed {
        Some(decode_mixed_mask(dec, models, lower.len())?)
    } else {
        None
    };
    Ok(apply_case(lower, pat, mask.as_deref()))
}

/// Hash `(prev_id, prefix, bit_pos)` into the bit predictor's context
/// space. `prev_id` is the dictionary id of the previous token in
/// the Content stream (or `None` at the start of a Content run / when
/// the previous token was an OOV that couldn't be inserted). Phase 15
/// folds this into the FNV mix to make the bit predictor an Order-1
/// word model. For contexts independent of `prev_id` (e.g. OOV
/// length emission) the caller passes `None` to use a single
/// "no-prev" sub-table.
///
/// `u64::MAX` is the sentinel for `None` — real prev ids live in
/// `0..=65535`, so the high bits of `u64::MAX` make the sentinel
/// unambiguous.
fn id_bit_ctx(prev_id: Option<u32>, prefix: u32, bit_pos: u32) -> u64 {
    let prev_field = prev_id.map_or(u64::MAX, u64::from);
    let h = fnv_mix(FNV_OFFSET, prev_field);
    let h = fnv_mix(h, u64::from(prefix));
    fnv_mix(h, u64::from(bit_pos))
}

/// Encode `id` MSB-first as `ID_BITS` bit-level AC emissions, each
/// conditioned on `(prev_id, bit_pos, prefix_so_far)`. v3 emits 18
/// bits per id (up from 16 in v2) to cover the widened 262 K
/// dictionary.
fn encode_dict_id(enc: &mut AcEncoder<'_>, models: &mut Models, prev_id: Option<u32>, id: u32) {
    let mut prefix: u32 = 0;
    for bit_pos in (0..ID_BITS).rev() {
        let bit = (id >> bit_pos) & 1;
        let ctx = id_bit_ctx(prev_id, prefix, bit_pos);
        let p_zero = models.token_id_bit.predict_p_zero(ctx);
        let cdf = [0u32, p_zero, TOTAL];
        enc.encode(&cdf, bit as usize);
        models.token_id_bit.observe(ctx, bit);
        prefix = (prefix << 1) | bit;
    }
}

fn decode_dict_id(
    dec: &mut AcDecoder<'_, '_>,
    models: &mut Models,
    prev_id: Option<u32>,
) -> Result<u32> {
    let mut prefix: u32 = 0;
    for bit_pos in (0..ID_BITS).rev() {
        let ctx = id_bit_ctx(prev_id, prefix, bit_pos);
        let p_zero = models.token_id_bit.predict_p_zero(ctx);
        let cdf = [0u32, p_zero, TOTAL];
        let bit = u32::try_from(dec.decode(&cdf)?).expect("bit symbol fits u32");
        models.token_id_bit.observe(ctx, bit);
        prefix = (prefix << 1) | bit;
    }
    Ok(prefix)
}

fn encode_length(enc: &mut AcEncoder<'_>, models: &mut Models, length: usize) {
    assert!(length <= 0xFFFF, "token length {length} exceeds u16 cap");
    let mut prefix: u32 = 0;
    for bit_pos in (0..16).rev() {
        let bit = u32::try_from((length >> bit_pos) & 1).expect("bit fits u32");
        let ctx = id_bit_ctx(None, prefix, bit_pos);
        let p_zero = models.token_length_bit.predict_p_zero(ctx);
        let cdf = [0u32, p_zero, TOTAL];
        enc.encode(&cdf, bit as usize);
        models.token_length_bit.observe(ctx, bit);
        prefix = (prefix << 1) | bit;
    }
}

fn decode_length(dec: &mut AcDecoder<'_, '_>, models: &mut Models) -> Result<usize> {
    let mut prefix: u32 = 0;
    for bit_pos in (0..16).rev() {
        let ctx = id_bit_ctx(None, prefix, bit_pos);
        let p_zero = models.token_length_bit.predict_p_zero(ctx);
        let cdf = [0u32, p_zero, TOTAL];
        let bit = u32::try_from(dec.decode(&cdf)?).expect("bit symbol fits u32");
        models.token_length_bit.observe(ctx, bit);
        prefix = (prefix << 1) | bit;
    }
    Ok(usize::try_from(prefix).expect("u32 prefix fits usize"))
}

fn encode_oov_word_bytes(enc: &mut AcEncoder<'_>, models: &mut Models, lower: &[u8]) {
    let mut prev_ctx = LETTER_START;
    let mut cdf = [0u32; 27];
    for &b in lower {
        let idx = (b - b'a') as usize;
        models.oov_word_letter.cdf_to(prev_ctx, &mut cdf);
        enc.encode(&cdf, idx);
        models.oov_word_letter.observe(prev_ctx, idx);
        prev_ctx = idx;
    }
}

fn decode_oov_word_bytes(
    dec: &mut AcDecoder<'_, '_>,
    models: &mut Models,
    length: usize,
) -> Result<Vec<u8>> {
    let mut prev_ctx = LETTER_START;
    let mut cdf = [0u32; 27];
    let mut out = Vec::with_capacity(length);
    for _ in 0..length {
        models.oov_word_letter.cdf_to(prev_ctx, &mut cdf);
        let idx = dec.decode(&cdf)?;
        models.oov_word_letter.observe(prev_ctx, idx);
        let b = b'a' + u8::try_from(idx).expect("letter idx fits u8");
        out.push(b);
        prev_ctx = idx;
    }
    Ok(out)
}

fn encode_oov_sep_bytes(enc: &mut AcEncoder<'_>, models: &mut Models, bytes: &[u8]) {
    let mut prev_ctx = BYTE_START;
    let mut cdf = [0u32; 257];
    for &b in bytes {
        let idx = b as usize;
        models.oov_sep_byte.cdf_to(prev_ctx, &mut cdf);
        enc.encode(&cdf, idx);
        models.oov_sep_byte.observe(prev_ctx, idx);
        prev_ctx = idx;
    }
}

fn decode_oov_sep_bytes(
    dec: &mut AcDecoder<'_, '_>,
    models: &mut Models,
    length: usize,
) -> Result<Vec<u8>> {
    let mut prev_ctx = BYTE_START;
    let mut cdf = [0u32; 257];
    let mut out = Vec::with_capacity(length);
    for _ in 0..length {
        models.oov_sep_byte.cdf_to(prev_ctx, &mut cdf);
        let idx = dec.decode(&cdf)?;
        models.oov_sep_byte.observe(prev_ctx, idx);
        let b = u8::try_from(idx).expect("byte symbol fits u8");
        out.push(b);
        prev_ctx = idx;
    }
    Ok(out)
}

fn encode_case_with_token_prior(
    enc: &mut AcEncoder<'_>,
    models: &mut Models,
    dict: &mut Dict,
    id: u32,
    pat: CasePattern,
    original: &[u8],
) {
    let mut cdf = [0u32; 5];
    case_cdf_from_counts(&dict.entry(id).case_counts, &mut cdf);
    enc.encode(&cdf, pat.idx());
    case_counts_inc(&mut dict.entry_mut(id).case_counts, pat.idx());
    if pat == CasePattern::Mixed {
        encode_mixed_mask(enc, models, original);
    }
}

fn decode_case_with_token_prior(
    dec: &mut AcDecoder<'_, '_>,
    _models: &mut Models,
    dict: &Dict,
    id: u32,
    _length: usize,
) -> Result<CasePattern> {
    let mut cdf = [0u32; 5];
    case_cdf_from_counts(&dict.entry(id).case_counts, &mut cdf);
    let idx = dec.decode(&cdf)?;
    CasePattern::from_idx(idx).with_context(|| format!("invalid case index {idx}"))
}

fn encode_case_global(
    enc: &mut AcEncoder<'_>,
    models: &mut Models,
    pat: CasePattern,
    original: &[u8],
) {
    let mut cdf = [0u32; 5];
    models.case_pattern_global.cdf_to(&mut cdf);
    enc.encode(&cdf, pat.idx());
    models.case_pattern_global.observe(pat.idx());
    if pat == CasePattern::Mixed {
        encode_mixed_mask(enc, models, original);
    }
}

fn decode_case_global(
    dec: &mut AcDecoder<'_, '_>,
    models: &mut Models,
    _length: usize,
) -> Result<CasePattern> {
    let mut cdf = [0u32; 5];
    models.case_pattern_global.cdf_to(&mut cdf);
    let idx = dec.decode(&cdf)?;
    models.case_pattern_global.observe(idx);
    CasePattern::from_idx(idx).with_context(|| format!("invalid case index {idx}"))
}

fn encode_mixed_mask(enc: &mut AcEncoder<'_>, models: &mut Models, original: &[u8]) {
    let mask = mixed_mask(original);
    let mut cdf = [0u32; 3];
    for upper in mask {
        models.case_mixed_bit.cdf_to(&mut cdf);
        let sym = usize::from(upper);
        enc.encode(&cdf, sym);
        models.case_mixed_bit.observe(sym);
    }
}

fn decode_mixed_mask(
    dec: &mut AcDecoder<'_, '_>,
    models: &mut Models,
    length: usize,
) -> Result<Vec<bool>> {
    let mut cdf = [0u32; 3];
    let mut out = Vec::with_capacity(length);
    for _ in 0..length {
        models.case_mixed_bit.cdf_to(&mut cdf);
        let sym = dec.decode(&cdf)?;
        models.case_mixed_bit.observe(sym);
        out.push(sym == 1);
    }
    Ok(out)
}

/// One lookahead token used by the LZ matcher's encode-time search.
/// `end` is the byte position one past the token; `id` is the
/// already-known dictionary id; `class` lets the encoder pick the
/// case-emission path per token without re-classifying.
#[derive(Clone, Copy, Debug)]
struct TokenLookahead {
    end: usize,
    id: u32,
    class: TokenClass,
}

/// Walk forward from `start`, tokenizing Content bytes into a
/// sequence of `(end_pos, dict_id, class)` triples for the LZ
/// matcher. Stops at:
///   - `tok_lz::MAX_MATCH` collected tokens (the matcher won't
///     extend past this anyway).
///   - An OOV token (no dict id) — a match record can't bridge
///     a slot that hasn't been assigned an id yet.
///   - A separator that contains `<` — the next byte transitions
///     into `TagStructure` mode and the Content run ends.
fn lookahead_tokens(buf: &[u8], start: usize, dict: &Dict) -> Vec<TokenLookahead> {
    let mut out = Vec::new();
    let mut p = start;
    while p < buf.len() && out.len() < tok_lz::MAX_MATCH {
        let end = run_end(buf, p);
        let token_bytes = &buf[p..end];
        let class = TokenClass::from_byte(buf[p]);
        let key = match class {
            TokenClass::Word => lowercase(token_bytes),
            TokenClass::Separator => token_bytes.to_vec(),
        };
        match dict.get(&key) {
            Some(id) => out.push(TokenLookahead { end, id, class }),
            None => break,
        }
        if class == TokenClass::Separator && token_bytes.contains(&b'<') {
            // Include this token but stop further lookahead — the
            // next byte is in TagStructure mode.
            break;
        }
        p = end;
    }
    out
}

/// Cache of uniform CDFs for chunked raw-bit emission via the AC.
/// Mirrors the helper in `xml_lz_cp` so LZ offset bucket-relative
/// bits can be coded against the shared AC stream without
/// materializing a 2^k CDF every emission.
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
                *slot = ((i as u64 * u64::from(TOTAL)) / n_u64) as u32;
            }
            cdf[n] = TOTAL;
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

fn prewarm(
    warm: &[u8],
    classifier: &mut Classifier,
    models: &mut Models,
    dict: &mut Dict,
    matcher: &mut TokenMatcher,
) {
    let mut last_class_ctx = CLASS_CTX_NONE;
    let mut prev_id: Option<u32> = None;
    let mut i = 0;
    while i < warm.len() {
        let mode = classifier.current_mode();
        match mode {
            Mode::Content => {
                let class = TokenClass::from_byte(warm[i]);
                let class_sym = class_to_sym(class);
                models.token_class.observe(last_class_ctx, class_sym);

                let end = run_end(warm, i);
                let token = &warm[i..end];
                let new_id = observe_token(models, dict, class, token, prev_id);
                if let Some(id) = new_id {
                    matcher.push(id);
                }
                prev_id = new_id;
                // Also observe the lz_flag=0 (no-match) for prewarm
                // so the model arrives at measure with a calibrated
                // P(no-match) prior.
                models.lz_flag.observe(0);

                for &b in token {
                    classifier.advance(b);
                }
                i = end;
                last_class_ctx = match class {
                    TokenClass::Word => CLASS_CTX_WORD,
                    TokenClass::Separator => CLASS_CTX_SEP,
                };
            }
            Mode::AttrValue => {
                models.attr_byte.observe(warm[i] as usize);
                classifier.advance(warm[i]);
                i += 1;
                last_class_ctx = CLASS_CTX_NONE;
                prev_id = None;
            }
            Mode::TagStructure => {
                let run_end_ = find_tag_run_end(warm, i, *classifier);
                let run = &warm[i..run_end_];
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
                i = run_end_;
                last_class_ctx = CLASS_CTX_NONE;
                prev_id = None;
            }
        }
    }
}

/// Prewarm observe path — must mirror the encode/decode bit-level
/// operations exactly (including conditioning on `prev_id`) so the
/// model state at the start of measure is identical on both sides.
/// Returns the assigned id for the next token's `prev_id`, matching
/// the encode-time return convention.
fn observe_token(
    models: &mut Models,
    dict: &mut Dict,
    class: TokenClass,
    token: &[u8],
    prev_id: Option<u32>,
) -> Option<u32> {
    let (pat, key) = match class {
        TokenClass::Word => (Some(classify_case(token)), lowercase(token)),
        TokenClass::Separator => (None, token.to_vec()),
    };
    let hit_id = dict.get(&key);
    let oov_sym = usize::from(hit_id.is_none());
    models.token_oov.observe(oov_sym);

    if let Some(id) = hit_id {
        let mut prefix: u32 = 0;
        for bit_pos in (0..ID_BITS).rev() {
            let bit = (id >> bit_pos) & 1;
            let ctx = id_bit_ctx(prev_id, prefix, bit_pos);
            models.token_id_bit.observe(ctx, bit);
            prefix = (prefix << 1) | bit;
        }

        if let Some(p) = pat {
            case_counts_inc(&mut dict.entry_mut(id).case_counts, p.idx());
            if p == CasePattern::Mixed {
                for upper in mixed_mask(token) {
                    models.case_mixed_bit.observe(usize::from(upper));
                }
            }
        }
        Some(id)
    } else {
        let length = key.len();
        assert!(length <= 0xFFFF);
        let mut prefix: u32 = 0;
        for bit_pos in (0..16).rev() {
            let bit = u32::try_from((length >> bit_pos) & 1).expect("bit fits u32");
            let ctx = id_bit_ctx(None, prefix, bit_pos);
            models.token_length_bit.observe(ctx, bit);
            prefix = (prefix << 1) | bit;
        }

        match class {
            TokenClass::Word => {
                let mut prev = LETTER_START;
                for &b in &key {
                    let idx = (b - b'a') as usize;
                    models.oov_word_letter.observe(prev, idx);
                    prev = idx;
                }
            }
            TokenClass::Separator => {
                let mut prev = BYTE_START;
                for &b in &key {
                    let idx = b as usize;
                    models.oov_sep_byte.observe(prev, idx);
                    prev = idx;
                }
            }
        }

        if let Some(p) = pat {
            models.case_pattern_global.observe(p.idx());
            if p == CasePattern::Mixed {
                for upper in mixed_mask(token) {
                    models.case_mixed_bit.observe(usize::from(upper));
                }
            }
        }

        let inserted = dict.insert(key);
        if let (Some(new_id), Some(p)) = (inserted, pat) {
            case_counts_inc(&mut dict.entry_mut(new_id).case_counts, p.idx());
        }
        inserted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(warm: &[u8], measure: &[u8]) {
        let codec = XmlTokCodec;
        let (archive, _decomp) = codec.encode_window(warm, measure).unwrap();
        let decoded = codec.decode_window(warm, &archive).unwrap();
        assert_eq!(decoded, measure);
    }

    #[test]
    fn roundtrip_short_mixed_letters() {
        roundtrip(b"", b"<page>The quick brown fox jumps.</page>");
    }

    #[test]
    fn roundtrip_repeated_words() {
        roundtrip(
            b"",
            b"<page>the the the cat sat on the mat the cat sat.</page>",
        );
    }

    #[test]
    fn roundtrip_with_case_variants() {
        roundtrip(
            b"",
            b"<page>The cat is THE animal. the cat sat. The Cat sat.</page>",
        );
    }

    #[test]
    fn roundtrip_with_mixed_case_token() {
        roundtrip(b"", b"<page>I use an iPhone and a MacBook.</page>");
    }

    #[test]
    fn roundtrip_with_attr_value() {
        roundtrip(
            b"",
            b"<text xml:space=\"preserve\">Hello, World! 123 ABC.</text>",
        );
    }

    #[test]
    fn roundtrip_with_long_separator() {
        roundtrip(b"", b"<page>word\n\n\n\n  ,,..!!  next.</page>");
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

    /// Regression test for the case-CDF zero-mass AC desync. At
    /// ~75 MB into enwik8, hot dict entries (e.g. "the") accumulate
    /// O(10^6) `AllLower` observations and a handful of `TitleCase`
    /// ones; without rescaling, the smoothed case CDF gives the rare
    /// patterns zero mass, breaking the AC's `new_high < new_low`
    /// invariant. Encodes an 80 MB chunk in one shot and verifies
    /// the roundtrip.
    #[test]
    #[ignore = "slow; encodes 80 MB end-to-end to exercise the case-CDF rescale path"]
    fn roundtrip_enwik8_large_chunk() {
        let Ok(bytes) = std::fs::read("assets/enwik8") else {
            return;
        };
        let chunk = &bytes[..80 * 1024 * 1024];
        roundtrip(b"", chunk);
    }
}
