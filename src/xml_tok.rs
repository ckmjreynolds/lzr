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

/// Maximum dictionary entries — fits a u16 index.
const DICT_CAPACITY: usize = 1 << 16;

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

/// Hash-table size (log₂) for the bit-level dict-id predictor. 2^20
/// slots = 4 MiB of u16 count pairs. K=22 gave byte-identical
/// output in the 5-window panel, so collisions aren't blurring the
/// Zipfian peaks — the Order-0 model is already at Shannon and the
/// table size doesn't move the floor.
const ID_BIT_K: u32 = 20;
/// Hash-table size for the bit-level length predictor. Length
/// distribution is narrower (most tokens < 32 bytes), so 2^18 slots
/// = 1 MiB are plenty.
const LEN_BIT_K: u32 = 18;

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
    by_bytes: HashMap<Vec<u8>, u16>,
    entries: Vec<DictEntry>,
}

impl Dict {
    fn get(&self, key: &[u8]) -> Option<u16> {
        self.by_bytes.get(key).copied()
    }

    /// Insert a new entry if there's room. Returns the new id, or
    /// `None` if the dictionary is full.
    fn insert(&mut self, key: Vec<u8>) -> Option<u16> {
        if self.entries.len() >= DICT_CAPACITY {
            return None;
        }
        let id = u16::try_from(self.entries.len()).expect("entries.len() < DICT_CAPACITY");
        self.by_bytes.insert(key.clone(), id);
        self.entries.push(DictEntry::new(key));
        Some(id)
    }

    fn entry_mut(&mut self, id: u16) -> &mut DictEntry {
        &mut self.entries[id as usize]
    }

    fn entry(&self, id: u16) -> &DictEntry {
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

        prewarm(&buf[..warm_end], &mut classifier, &mut models, &mut dict);

        let mut writer = BitWriter::new();
        let mut comp = ComponentBits::default();
        let bits_before_payload = writer.bits_written();
        {
            let mut enc = AcEncoder::new(&mut writer);
            let mut last_class_ctx = CLASS_CTX_NONE;
            let mut i = warm_end;
            while i < buf.len() {
                let mode = classifier.current_mode();
                match mode {
                    Mode::Content => {
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
                        encode_token(&mut enc, &mut models, &mut dict, &mut comp, class, token);

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
        decomp.add("tag_ac", comp.tag);
        decomp.add("attr_ac", comp.attr);
        let attributed =
            comp.token_hit + comp.token_oov + comp.token_class + comp.case + comp.tag + comp.attr;
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
        prewarm(&buf, &mut classifier, &mut models, &mut dict);

        let mut reader = BitReader::new(&archive[4..]);
        let mut dec = AcDecoder::new(&mut reader);
        let mut last_class_ctx = CLASS_CTX_NONE;

        while buf.len() - warm_end < measure_len {
            let mode = classifier.current_mode();
            match mode {
                Mode::Content => {
                    let mut class_cdf = [0u32; 3];
                    models.token_class.cdf_to(last_class_ctx, &mut class_cdf);
                    let class_sym = dec.decode(&class_cdf)?;
                    models.token_class.observe(last_class_ctx, class_sym);
                    let class = sym_to_class(class_sym)
                        .with_context(|| format!("invalid class symbol {class_sym}"))?;

                    let token_bytes = decode_token(&mut dec, &mut models, &mut dict, class)?;
                    for &b in &token_bytes {
                        buf.push(b);
                        classifier.advance(b);
                    }
                    last_class_ctx = match class {
                        TokenClass::Word => CLASS_CTX_WORD,
                        TokenClass::Separator => CLASS_CTX_SEP,
                    };
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

fn encode_token(
    enc: &mut AcEncoder<'_>,
    models: &mut Models,
    dict: &mut Dict,
    comp: &mut ComponentBits,
    class: TokenClass,
    token: &[u8],
) {
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
        encode_dict_id(enc, models, id);
        comp.token_hit += enc.bits_written() - before;

        if let (TokenClass::Word, Some(pat)) = (class, case_pat) {
            let cb = enc.bits_written();
            encode_case_with_token_prior(enc, models, dict, id, pat, token);
            comp.case += enc.bits_written() - cb;
        }
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

        if let Some(new_id) = dict.insert(key)
            && let Some(pat) = case_pat
        {
            // Bootstrap the new entry's case stats with this occurrence.
            dict.entry_mut(new_id).case_counts[pat.idx()] += 1;
        }
    }
}

#[allow(clippy::too_many_lines)]
fn decode_token(
    dec: &mut AcDecoder<'_, '_>,
    models: &mut Models,
    dict: &mut Dict,
    class: TokenClass,
) -> Result<Vec<u8>> {
    let mut oov_cdf = [0u32; 3];
    models.token_oov.cdf_to(&mut oov_cdf);
    let oov_sym = dec.decode(&oov_cdf)?;
    models.token_oov.observe(oov_sym);

    if oov_sym == 0 {
        let id = decode_dict_id(dec, models)?;
        let lower = dict.entry(id).lower.clone();
        if class == TokenClass::Word {
            let pat = decode_case_with_token_prior(dec, models, dict, id, lower.len())?;
            dict.entry_mut(id).case_counts[pat.idx()] += 1;
            apply_case_decoded(&lower, pat, dec, models)
        } else {
            Ok(lower)
        }
    } else {
        let length = decode_length(dec, models)?;
        let bytes = match class {
            TokenClass::Word => decode_oov_word_bytes(dec, models, length)?,
            TokenClass::Separator => decode_oov_sep_bytes(dec, models, length)?,
        };
        let reconstructed = if class == TokenClass::Word {
            let pat = decode_case_global(dec, models, length)?;
            let mask = if pat == CasePattern::Mixed {
                Some(decode_mixed_mask(dec, models, length)?)
            } else {
                None
            };
            let word = apply_case(&bytes, pat, mask.as_deref());
            if let Some(new_id) = dict.insert(bytes) {
                dict.entry_mut(new_id).case_counts[pat.idx()] += 1;
            }
            word
        } else {
            let _ = dict.insert(bytes.clone());
            bytes
        };
        Ok(reconstructed)
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

/// Hash `(prefix, bit_pos)` into the bit predictor's context space.
/// Both arguments are u16 so the FNV mix stays cheap; the high bits
/// of `prefix` collapse to 0 quickly during MSB-first emission, and
/// `bit_pos` disambiguates contexts at different stages of the same
/// id.
fn id_bit_ctx(prefix: u16, bit_pos: u32) -> u64 {
    let h = fnv_mix(FNV_OFFSET, u64::from(prefix));
    fnv_mix(h, u64::from(bit_pos))
}

/// Encode `id` MSB-first as 16 bit-level AC emissions, each
/// conditioned on `(bit_pos, prefix_so_far)` via `BitPredictor`.
fn encode_dict_id(enc: &mut AcEncoder<'_>, models: &mut Models, id: u16) {
    let mut prefix: u32 = 0;
    for bit_pos in (0..16).rev() {
        let bit = u32::from((id >> bit_pos) & 1);
        let ctx = id_bit_ctx(
            u16::try_from(prefix).expect("prefix < 2^16 within first 16 iterations"),
            bit_pos,
        );
        let p_zero = models.token_id_bit.predict_p_zero(ctx);
        let cdf = [0u32, p_zero, TOTAL];
        enc.encode(&cdf, bit as usize);
        models.token_id_bit.observe(ctx, bit);
        prefix = (prefix << 1) | bit;
    }
}

fn decode_dict_id(dec: &mut AcDecoder<'_, '_>, models: &mut Models) -> Result<u16> {
    let mut prefix: u32 = 0;
    for bit_pos in (0..16).rev() {
        let ctx = id_bit_ctx(
            u16::try_from(prefix).expect("prefix < 2^16 within first 16 iterations"),
            bit_pos,
        );
        let p_zero = models.token_id_bit.predict_p_zero(ctx);
        let cdf = [0u32, p_zero, TOTAL];
        let bit = u32::try_from(dec.decode(&cdf)?).expect("bit symbol fits u32");
        models.token_id_bit.observe(ctx, bit);
        prefix = (prefix << 1) | bit;
    }
    Ok(u16::try_from(prefix).expect("16-bit id fits u16"))
}

fn encode_length(enc: &mut AcEncoder<'_>, models: &mut Models, length: usize) {
    assert!(length <= 0xFFFF, "token length {length} exceeds u16 cap");
    let mut prefix: u32 = 0;
    for bit_pos in (0..16).rev() {
        let bit = u32::from(u16::try_from((length >> bit_pos) & 1).expect("bit fits u16"));
        let ctx = id_bit_ctx(
            u16::try_from(prefix).expect("prefix < 2^16 within first 16 iterations"),
            bit_pos,
        );
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
        let ctx = id_bit_ctx(
            u16::try_from(prefix).expect("prefix < 2^16 within first 16 iterations"),
            bit_pos,
        );
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
    id: u16,
    pat: CasePattern,
    original: &[u8],
) {
    let mut cdf = [0u32; 5];
    case_cdf_from_counts(&dict.entry(id).case_counts, &mut cdf);
    enc.encode(&cdf, pat.idx());
    dict.entry_mut(id).case_counts[pat.idx()] += 1;
    if pat == CasePattern::Mixed {
        encode_mixed_mask(enc, models, original);
    }
}

fn decode_case_with_token_prior(
    dec: &mut AcDecoder<'_, '_>,
    _models: &mut Models,
    dict: &Dict,
    id: u16,
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

fn prewarm(warm: &[u8], classifier: &mut Classifier, models: &mut Models, dict: &mut Dict) {
    let mut last_class_ctx = CLASS_CTX_NONE;
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
                observe_token(models, dict, class, token);

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
            }
        }
    }
}

fn observe_token(models: &mut Models, dict: &mut Dict, class: TokenClass, token: &[u8]) {
    let (pat, key) = match class {
        TokenClass::Word => (Some(classify_case(token)), lowercase(token)),
        TokenClass::Separator => (None, token.to_vec()),
    };
    let hit_id = dict.get(&key);
    let oov_sym = usize::from(hit_id.is_none());
    models.token_oov.observe(oov_sym);

    if let Some(id) = hit_id {
        let mut prefix: u32 = 0;
        for bit_pos in (0..16).rev() {
            let bit = u32::from((id >> bit_pos) & 1);
            let ctx = id_bit_ctx(
                u16::try_from(prefix).expect("prefix < 2^16 within first 16 iterations"),
                bit_pos,
            );
            models.token_id_bit.observe(ctx, bit);
            prefix = (prefix << 1) | bit;
        }

        if let Some(p) = pat {
            dict.entry_mut(id).case_counts[p.idx()] += 1;
            if p == CasePattern::Mixed {
                for upper in mixed_mask(token) {
                    models.case_mixed_bit.observe(usize::from(upper));
                }
            }
        }
    } else {
        let length = key.len();
        assert!(length <= 0xFFFF);
        let mut prefix: u32 = 0;
        for bit_pos in (0..16).rev() {
            let bit = u32::from(u16::try_from((length >> bit_pos) & 1).expect("bit fits u16"));
            let ctx = id_bit_ctx(
                u16::try_from(prefix).expect("prefix < 2^16 within first 16 iterations"),
                bit_pos,
            );
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

        if let Some(new_id) = dict.insert(key) {
            if let Some(p) = pat {
                dict.entry_mut(new_id).case_counts[p.idx()] += 1;
            }
        }
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
}
