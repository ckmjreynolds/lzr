//! Phase-18 codec: cost-aware predictor switching layered on `xml-tok`.
//!
//! Maintains N independent bit-level dictionary-id predictors (Order-1
//! and Order-2 word context to start; the design extends to more
//! without code-shape changes). For each Content hit-token the
//! encoder:
//!
//! 1. Computes the 18-bit emission cost under each predictor — a
//!    `predict_p_zero`-only pass, no AC state mutated.
//! 2. Adds a small **stickiness bias** to non-current predictors to
//!    prevent flip-flop on near-tied costs (the router stream
//!    compresses well only if switches are locally rare).
//! 3. Picks the cheapest, emits a `log2(N)`-bit **router** decision
//!    via an adaptive `BitPredictor` keyed on `(prev_id, current)`.
//!    The predictor learns `P(stay)`, which approaches 1 in stable
//!    regions and collapses the router cost to ~0.
//! 4. Emits the 18-bit `dict_id` using the chosen predictor.
//! 5. Updates **all** predictors with the observed bits — they stay
//!    in lock-step with the decoder, which maintains the same set.
//!
//! The key bet vs PPM-D-style escape (a special case of switching
//! where the router decision is deterministic from context-count):
//! the encoder's per-token hindsight on actual emission cost lets
//! it pick the right predictor even when both have warm context.
//! Phase-3's Order-2 word loss (-0.079 bpb at panel) was driven by
//! cold bigrams; Phase 18 routes those tokens to Order-1 instead.
//!
//! Calibration discipline (Phase 17 lesson): validate end-to-end on
//! enwik8 before claiming, and on enwik9 before committing the
//! design. Panel results for changes that compete with predictor
//! saturation can be misleading in either direction.
//!
//! Below this block the codec is `xml-tok` verbatim plus the route
//! layer in `encode_token`/`decode_token`. Original xml-tok docs:
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
use crate::mixer::LogitMixer;
use crate::models::{Order0, Order1Ctx};
use crate::neural::{OnlineMLP, SparseLR};
use crate::tok_lz::{self, TokenMatcher};
use crate::tokenizer::{
    CasePattern, TokenClass, apply_case, classify_case, lowercase, mixed_mask, run_end,
};
use crate::wiki_classifier::WikiFineClassifier;

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
/// Phase-20: joint context for the Order-4 OOV word letter model.
/// `ctx = p4 * 27^3 + p3 * 27^2 + p2 * 27 + p1`. 531441 rows × 26
/// syms × 4 bytes = ~55 MiB. The Order-2 → Order-3 → Order-4 sweep
/// gained -0.0207, -0.0063, -0.0013 bpb on enwik8 e2e respectively;
/// Order-5 (14M rows, ~1.5 GiB) regressed +0.003 because cells went
/// too cold for the OOV byte volume. Phase-22 tested per-wiki-sub-
/// mode splitting (5 × 531 K cells, ~276 MiB): regressed +0.005 bpb
/// on enwik8 — 1.5 M OOV letters / 2.66 M cells = 0.56 obs/cell,
/// too sparse no matter how the wiki mode partitions.
const LETTER_NCTX_O4: usize = LETTER_NCTX * LETTER_NCTX * LETTER_NCTX * LETTER_NCTX;

/// Byte context for OOV-separator emission: index 256 = "start of
/// token"; 0-255 = previous byte.
const BYTE_START: usize = 256;
const BYTE_NCTX: usize = 257;
/// Phase-20g: joint context for the Order-2 OOV separator byte
/// model. `ctx = prev_prev * 257 + prev` over `(BYTE_NCTX, BYTE_NCTX)`.
/// 66049 rows × 256 syms × 4 bytes = ~64 MiB. Phase-20o tested
/// Order-3 hashed (64 K rows × 256 = 16 M cells, hashed via FNV)
/// and it regressed +0.018 bpb on enwik8 — the OOV sep byte volume
/// (~1 M bytes on enwik8) over 16 M cells is ~0.06 obs/cell, too
/// sparse to converge regardless of how the trigram is mapped.
const BYTE_NCTX_O2: usize = BYTE_NCTX * BYTE_NCTX;

/// Hash-table size (log₂) for the bit-level dict-id predictor.
/// Phase 15 folds `prev_id` into the context to make this an
/// Order-1 word model; the `(prev_id, bit_pos, prefix)` space is far
/// larger than `(bit_pos, prefix)` alone. v3 Phase 1 widens IDs to
/// 18 bits and bumps the table to 2^24 = 16 M slots (64 MiB) to
/// keep collision rate near zero at the new context-space size.
/// Phase-19k: bumped from 2 ^ 24 to 2 ^ 25 (128 MiB → 256 MiB) so
/// the long tail of Order-1 contexts has less collision pressure.
const ID_BIT_K: u32 = 27;
/// Hash-table size for the bit-level length predictor. Length is
/// independent of `prev_id`, so the context space stays as in
/// Phase 14: 2^18 slots = 1 MiB.
const LEN_BIT_K: u32 = 18;

/// LZ-on-tokens offset bucketing: bucket = `log₂(offset)`. Offsets
/// up to `WINDOW_SIZE` (1 M tokens) need 20 buckets. Use 21 to give
/// the model room when the offset hits the upper end.
const OFFSET_BUCKET_ALPHABET: usize = 21;

/// Hash-table size for the Order-2 `dict_id` bit predictor. The joint
/// `(prev_prev_id, prev_id, prefix, bit_pos)` context space is much
/// larger than Order-1's. Phase-19i bumped from 2 ^ 26 to 2 ^ 27 to
/// reduce collisions on the long tail of warm bigrams; the table is
/// 1 GiB now (within the 10 GiB judging-machine budget). Phase-20e
/// bumped to K=29 (2 GiB).
const ID_BIT_O2_K: u32 = 29;

/// Phase-22: hash-table size for the wiki-mode-conditioned `dict_id`
/// bit predictor. Context = `(wiki_sub_mode, prev_id, prefix,
/// bit_pos)`. The 5-mode `WikiFineClassifier` (`Plain` /
/// `LinkTarget` / `LinkDisplay` / `TemplateName` / `TemplateArg`)
/// multiplies the Order-1 context space by 5, so K=26 (64 MiB)
/// keeps roughly Order-1's per-slot density per wiki mode.
const ID_BIT_WIKI_K: u32 = 26;

/// Phase-21: logit-space mixer of the Order-1 and Order-2 `dict_id`
/// bit predictors. Replaces the Phase-18 router. The mixer is
/// keyed on `(bit_pos, prefix)` (16 × 65 K ≤ 2^21 distinct
/// contexts; K=18 covers the populated subset with collision
/// pressure on the cold tail). Each context holds two `f32`
/// weights and learns via SGD on log-loss — no signaling bits.
const MIXER_K: u32 = 18;
/// SGD step size for the mixer. PAQ-like mixers typically use
/// 0.005 – 0.05; we start at the upper end to converge fast inside
/// the panel windows and tighten down on enwik9 if the e2e regresses.
const MIXER_LR: f32 = 0.02;

/// Phase-19f: hash-table size for the Order-1 `lz_flag` predictor.
/// Context = `(prev_id)`. Brought back in Phase-21 so the mixer
/// can blend it with the Order-2 `lz_flag` predictor (warm bigrams
/// favor Order-2; cold ones — fresh `prev_id`s — favor Order-1).
/// 2^20 = 1 MiB.
const LZ_FLAG_BIT_O1_K: u32 = 20;
/// Phase-20l: hash-table size for the Order-2 `lz_flag` predictor.
/// Context = `(prev_prev_id, prev_id)`. 2^23 = 8 MiB.
const LZ_FLAG_BIT_O2_K: u32 = 23;
/// Phase-21: mixer table size for the `lz_flag` stream. Context is
/// `(prev_id)`, hashed to K=10 = 1024 slots. The mixer learns per-
/// prev-id whether Order-1 or Order-2 wins.
const LZ_FLAG_MIXER_K: u32 = 10;

/// Order-1 `token_oov` predictor. Context = `(prev_id)`.
/// 2^20 = 1 MiB. Phase-21 keeps this as one arm of a mixer
/// blending with an Order-2 variant.
const TOKEN_OOV_BIT_K: u32 = 20;
/// Phase-21: Order-2 `token_oov` predictor — `(prev_prev_id,
/// prev_id)`. Phase-20m's unconditional swap to Order-2 regressed
/// +0.003 bpb on enwik8 because cold bigrams starved cells while
/// the global Order-1 prior already had `P(OOV | prev_id)`
/// calibrated. Under the mixer, the Order-2 arm contributes only
/// where the bigram is warm; weights collapse to Order-1 on cold
/// `prev_id`s. 2^23 = 8 MiB.
const TOKEN_OOV_BIT_O2_K: u32 = 23;
/// Mixer table size for the `token_oov` bit. Keyed on `(prev_id)`,
/// 2^10 = 1024 slots. Same shape as `lz_flag_mixer`.
const TOKEN_OOV_MIXER_K: u32 = 10;

/// Context-row count for the Order-1 LZ match predictors
/// (`lz_offset_bucket` and `lz_length`). `prev_id` is hashed into
/// `LZ_MATCH_NCTX` rows so the `Order1Ctx` tables stay small enough
/// to fit (86 KiB for the 21-symbol bucket, 1 MiB for the 256-
/// symbol length) while still giving the model meaningful prev-id
/// stratification — after-`{` tokens have different match offset
/// and length distributions than after-prose-words. Phase-20n
/// confirmed 1024 is the saturation point on enwik8 e2e too — 4096
/// regressed +0.016 bpb because the length CDF cells went cold.
const LZ_MATCH_NCTX: usize = 1024;

/// Phase 24a: bit-level Order-1 LZ-length predictor. Context =
/// `(prev_id, prefix, bit_pos)` for 8-bit MSB-first length emission.
/// Length emission is once-per-match with the match count ~30 % of
/// Content tokens, so the per-slot observation rate is ~3× lower
/// than `token_id_bit` (16-bit × every token). K=24 → 16 Mi slots
/// × 4 bytes = 64 MiB. Mirrors the dict-id Order-1 size budget,
/// 3 bits less than dict-id because the observation rate is ~8× lower.
const LZ_LENGTH_BIT_K: u32 = 24;
/// Phase 24a: bit-level Order-2 LZ-length predictor. Context =
/// `(p2, prev_id, prefix, bit_pos)`. K=27 (512 MiB), 3 bits more
/// than Order-1 — matches the dict-id Order-1→Order-2 ratio. Phase
/// 24e tested K=28 and confirmed observation-limit saturation: -536
/// bytes / ~0 bpb at +512 MiB memory, so K=27 is retained.
const LZ_LENGTH_BIT_O2_K: u32 = 27;
/// Phase 24a: hash-table size for the `lz_length` bit-level mixer.
/// Keyed on `(bit_pos, prefix)` — 8 × 128 = 1 K cells. K=10 covers
/// it without collisions.
const LZ_LENGTH_MIXER_K: u32 = 10;
/// Phase 24a: bit count for LZ-length MSB-first emission.
/// `length_token = length - MIN_MATCH` ranges 0..=255 → 8 bits.
const LZ_LENGTH_BITS: u32 = 8;
/// Phase 24a → 24c: arm count of the `lz_length` mixer.
/// `[Order-1, Order-2, SparseLR, MLP]`.
const N_MIX_LZ_LENGTH: usize = 4;

/// Phase 24c: weight-table size per feature for the `lz_length` /
/// `lz_offset_bucket` sparse-LR and MLP. Length emits 8 bits per
/// match × ~30 % match rate; per-bit-pos observation rate is
/// ~3.75 % per token. At K=18, with the 3-feature `id_lr_features`
/// shape and `(prefix, bit_pos)` in the hash, slots see ~1-2 obs
/// over the 1 GiB warm.
const LZ_LENGTH_LR_K: u32 = 18;
/// Phase 24c: SGD step size for `lz_length` / `lz_offset_bucket`
/// LR/MLP — matches the rest of the codec's SGD predictors.
const LZ_LENGTH_LR_LR: f32 = 0.02;
/// Phase 24c: hidden dim for the MLP arms. Same as dict-id.
const LZ_LENGTH_MLP_H: usize = 8;

/// Phase 24b: bit-level Order-1 `lz_offset_bucket` predictor. The
/// offset bucket is 21-way (`OFFSET_BUCKET_ALPHABET`); 5 MSB-first
/// bits cover it (`2^5 = 32`), with buckets 21-31 unreachable. The
/// predictor learns `P(bit=1)=0` for impossible branches during
/// warm-up; decoder produces identical bits, so roundtrip is safe.
/// K=22 (4 Mi slots, 16 MiB) — smaller than `lz_length`'s K=24
/// because the alphabet is 21 vs 256 and the observation rate is
/// identical (one per match), so the per-slot density is similar
/// at this lower K.
const LZ_OFFSET_BUCKET_BIT_K: u32 = 22;
/// Phase 24b: Order-2 `lz_offset_bucket` predictor. K=25 (32 Mi
/// slots, 128 MiB), 3 bits more than Order-1 — same ratio as
/// `lz_length`.
const LZ_OFFSET_BUCKET_BIT_O2_K: u32 = 25;
/// Phase 24b: hash-table size for the `lz_offset_bucket` bit-level
/// mixer. Keyed on `(bit_pos, prefix)`. 5 × 32 = 160 cells; K=8
/// covers it without collisions.
const LZ_OFFSET_BUCKET_MIXER_K: u32 = 8;
/// Phase 24b: bit count for `lz_offset_bucket` MSB-first emission.
/// `2^5 = 32` covers the 21 legal buckets.
const LZ_OFFSET_BUCKET_BITS: u32 = 5;
/// Phase 24b → 24c: arm count of the `lz_offset_bucket` mixer.
/// `[Order-1, Order-2, SparseLR, MLP]`.
const N_MIX_LZ_OFFSET_BUCKET: usize = 4;

/// Phase-22 / Phase-23A / Phase-23D: number of predictors in the
/// `dict_id` mixer. `[Order-1, Order-2, Wiki, SparseLR, MLP]`. The
/// 5th arm is a small online MLP ([`crate::neural::OnlineMLP`])
/// that adds a `tanh` nonlinearity and a learned hidden dimension
/// over the same hashed feature set the sparse-LR uses.
const N_MIX_ID: usize = 5;

/// Phase-23A: weight-table size per feature in the `dict_id`
/// sparse-LR. K=24 → 16 Mi slots × 4 bytes × 3 features = 192 MiB.
/// Phase-23A used K=22 (48 MiB); Phase 23C bumped K=24 to test
/// whether the Order-3 hashed feature was still
/// saturation-limited. Memory is fine — the headroom against the
/// 10 GiB Hutter cap is huge.
const ID_SPARSE_LR_K: u32 = 24;
/// Phase-23A: SGD learning rate for the `dict_id` sparse-LR. Same
/// rate the `LogitMixer` uses; tuned more carefully if the predictor
/// pulls weight from the existing arms.
const ID_SPARSE_LR_LR: f32 = 0.02;
/// Phase-23A: number of features the `dict_id` sparse-LR carries.
/// `[Order-3-hashed, Wiki × Order-2, (bit_pos, prefix) bias]`.
const N_LR_ID_FEATS: usize = 3;

/// Phase-23D: weight-table size per feature in the `dict_id` MLP.
/// K=20 (1 Mi slots) × `H=8` × `f32` × 3 features = 96 MiB. Smaller
/// than the LR's K=24 table because each MLP slot stores an
/// `H`-dim embedding row rather than a single scalar.
const ID_MLP_K: u32 = 20;
/// Phase-23D: SGD learning rate for the `dict_id` MLP.
const ID_MLP_LR: f32 = 0.02;
/// Phase-23D: hidden dimension of the `dict_id` MLP. Phase 23I
/// tested `H=16` to diagnose feature- vs capacity-limitation; it
/// landed +0.0002 bpb on enwik8 at +13 % encode and +192 MiB,
/// confirming the diminishing-returns curve is feature-limited.
/// Reverted to `H=8`.
const ID_MLP_H: usize = 8;
/// Phase-23D → Phase-23Q: feature count of the `dict_id` MLP.
/// Layout: `[Order-3 (LR-shared), Wiki × Order-2 (LR-shared),
/// bias (LR-shared), p1_len_bucket (23F), p2_len_bucket (23G),
/// (p1_len, p2_len) joint (23H), tokens_since_match_bucket (23J),
/// p1_class (23K), p2_class (23L), p1_first_byte_class (23M),
/// lz_match_length_last_bucket (23N), page_offset_bucket (23O),
/// wiki_depth_bucket (23Q)]`. Phase 23E (Order-4) and 23I (H=16)
/// both regressed; the "genuinely-new information" path is the
/// productive lever — see each journal entry.
const N_MLP_ID_FEATS: usize = 13;
/// Phase-21 / Phase-23B / Phase-23P: number of predictors in the
/// `lz_flag` mixer. `[Order-1, Order-2, SparseLR, MLP]`. The 4th arm
/// (Phase-23P) mirrors the dict-id MLP pattern: same feature shape as
/// the LR, but with a learned `H`-dim hidden layer and a `tanh`
/// nonlinearity for cheap nonlinear feature interactions.
const N_MIX_LZ_FLAG: usize = 4;
/// Phase-21 / Phase-23B / Phase-23P: number of predictors in the
/// `token_oov_bit` mixer. `[Order-1, Order-2, SparseLR, MLP]`. Mirrors
/// the `lz_flag` mix — same context, different target bit.
const N_MIX_TOKEN_OOV: usize = 4;

/// Phase-23B: weight-table size per feature in the binary-decision
/// sparse-LRs (`lz_flag`, `token_oov_bit`). K=20 → 1 Mi slots ×
/// 4 bytes × 3 features × 2 streams = 24 MiB total. Smaller than
/// the `dict_id` LR because each stream emits one bit per token, so
/// the per-feature observation rate is ~16× lower than the `dict_id`
/// stream; K=20 keeps the average obs/slot near 1.
const FLAG_SPARSE_LR_K: u32 = 20;
/// Phase-23B: SGD learning rate for the binary-decision sparse-LRs.
const FLAG_SPARSE_LR_LR: f32 = 0.02;
/// Phase-23B: number of features each binary-decision sparse-LR
/// carries. `[Order-3-hashed, Wiki × Order-2, Wiki × Order-1]`.
/// `lz_flag` and `token_oov_bit` share the feature shape; the
/// per-stream weight tables differ because the targets differ. The
/// Phase-23P MLP arms reuse the same feature shape.
const N_LR_FLAG_FEATS: usize = 3;

/// Phase-23P: weight-table size per feature for the flag-stream
/// MLPs. K=18 → 256 Ki slots × 8 H × 4 bytes × 3 features × 2
/// streams = 48 MiB total. Smaller than the dict-id MLP's K=20
/// because the flag streams emit one bit per token (~16× lower
/// observation density); K=18 matches the LR's parameter count at
/// `H=8` so per-slot saturation stays near the LR's regime.
const FLAG_MLP_K: u32 = 18;
/// Phase-23P: SGD learning rate for the flag-stream MLPs. Same as
/// every other SGD-trained predictor in this codec.
const FLAG_MLP_LR: f32 = 0.02;
/// Phase-23P: hidden dimension of the flag-stream MLPs. Matches the
/// dict-id MLP's `H=8`. The dict-id 23I probe at `H=16` regressed
/// (capacity wasn't the limit), so we start at `H=8` here too.
const FLAG_MLP_H: usize = 8;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct XmlTokRouteCodec;

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
    /// Kept as a placeholder; Phase-19g replaces with `token_oov_bit`.
    #[allow(dead_code)]
    token_oov: Order0<2>,

    /// Bit-level adaptive predictor for the 16-bit dictionary id.
    /// Context = (bit position, prefix bits emitted so far). The MSB
    /// is emitted first; once a hot token's high bits collapse to a
    /// predictable 0-prefix the bit cost converges to Shannon. Replaces
    /// the Phase-13 2-byte chunked encoding that starved tail rows.
    token_id_bit: BitPredictor,
    /// Same construction for OOV token lengths.
    token_length_bit: BitPredictor,

    /// Phase 16 (Order-0, kept for backward compat with the
    /// non-Order-1 baseline of xml-tok-route). Phase-19f migrates
    /// the `lz_flag` emission to `lz_flag_bit` (`BitPredictor`).
    #[allow(dead_code)]
    lz_flag: Order0<2>,
    /// Phase-19h → Phase-24b: legacy Order-1 `lz_offset_bucket`
    /// 21-way count CDF. Retained for future LR/MLP-arm extensions;
    /// the encode/decode path uses [`Self::lz_offset_bucket_bit_o1`]
    /// / `_o2` bit-level predictors as of Phase 24b.
    #[allow(dead_code)]
    lz_offset_bucket_o1: Order1Ctx<LZ_MATCH_NCTX, OFFSET_BUCKET_ALPHABET>,
    /// Phase-19h → Phase-24a: legacy Order-1 `lz_length` 256-way
    /// count CDF. Retained only to support the panel-historical
    /// baseline regression; the actual encode/decode path uses
    /// [`Self::lz_length_bit_o1`] / `_o2` bit-level predictors as of
    /// Phase 24a. Reserved for the deferred LR/MLP-arm extension
    /// in 24c.
    #[allow(dead_code)]
    lz_length_o1: Order1Ctx<LZ_MATCH_NCTX, 256>,
    /// Phase-24a: bit-level Order-1 `lz_length` predictor — context
    /// = `(prev_id, prefix, bit_pos)`. Replaces the 256-way count
    /// CDF with an MSB-first 8-bit chain, allowing per-bit-position
    /// conditional distributions and Order-2 mixing.
    lz_length_bit_o1: BitPredictor,
    /// Phase-24a: Order-2 `lz_length` predictor — `(p2, prev_id,
    /// prefix, bit_pos)`. Wins on warm bigrams where the
    /// `prev_prev_id` adds genuine information beyond `prev_id`.
    lz_length_bit_o2: BitPredictor,
    /// Phase-24a: mixer for `lz_length` bit-level predictors.
    /// Blends Order-1 / Order-2 per `(bit_pos, prefix)`.
    lz_length_mixer: LogitMixer<N_MIX_LZ_LENGTH>,
    /// Phase-24b: bit-level Order-1 `lz_offset_bucket` predictor —
    /// context = `(prev_id, prefix, bit_pos)`. 5-bit MSB-first chain
    /// over the 21-bucket alphabet.
    lz_offset_bucket_bit_o1: BitPredictor,
    /// Phase-24b: Order-2 `lz_offset_bucket` predictor.
    lz_offset_bucket_bit_o2: BitPredictor,
    /// Phase-24b: mixer for `lz_offset_bucket` bit-level predictors.
    lz_offset_bucket_mixer: LogitMixer<N_MIX_LZ_OFFSET_BUCKET>,
    /// Phase-24c: sparse-LR arm for `lz_length` bit emission.
    /// Same 3-feature `id_lr_features` shape as the dict-id LR;
    /// independent weight table because the target stream differs.
    lz_length_sparse_lr: SparseLR<N_LR_ID_FEATS>,
    /// Phase-24c: MLP arm for `lz_length` bit emission. Adds a
    /// learned `H`-dim hidden layer and `tanh` over the LR's
    /// hashed feature shape.
    lz_length_mlp: OnlineMLP<N_LR_ID_FEATS, LZ_LENGTH_MLP_H>,
    /// Phase-24c: sparse-LR arm for `lz_offset_bucket` bit emission.
    lz_offset_bucket_sparse_lr: SparseLR<N_LR_ID_FEATS>,
    /// Phase-24c: MLP arm for `lz_offset_bucket` bit emission.
    lz_offset_bucket_mlp: OnlineMLP<N_LR_ID_FEATS, LZ_LENGTH_MLP_H>,

    /// Phase-20j: Order-4 OOV word letter model. Context =
    /// `p4 * 27^3 + p3 * 27^2 + p2 * 27 + p1`. 531441 rows × 26
    /// syms × 4 bytes = ~55 MiB.
    oov_word_letter: Order1Ctx<LETTER_NCTX_O4, 26>,
    /// Phase-20g: Order-2 OOV separator byte model. Context =
    /// `prev_prev * 257 + prev`. 66049 rows × 256 syms × 4 bytes =
    /// ~64 MiB.
    oov_sep_byte: Order1Ctx<BYTE_NCTX_O2, 256>,

    case_pattern_global: Order0<4>,
    case_mixed_bit: Order0<2>,

    tag_dict: Order0<TAG_ALPHABET>,
    tag_byte: Order0<256>,
    attr_byte: Order0<256>,

    /// Phase-18 / Phase-21: Order-2 `dict_id` bit predictor —
    /// `(prev_prev_id, prev_id, prefix, bit_pos)` context.
    token_id_bit_o2: BitPredictor,
    /// Phase-19 / Phase-22: wiki-mode-conditioned `dict_id`
    /// predictor — `(wiki_sub_mode, prev_id, prefix, bit_pos)`.
    /// Mixed with Order-1 + Order-2 by `token_id_mixer`.
    token_id_bit_wiki: BitPredictor,
    /// Phase-19f / Phase-21: Order-1 `lz_flag` predictor — context
    /// = `(prev_id)`. Mixed with the Order-2 variant by
    /// `lz_flag_mixer`.
    lz_flag_bit_o1: BitPredictor,
    /// Phase-20l: Order-2 `lz_flag` predictor — context =
    /// `(prev_prev_id, prev_id)`.
    lz_flag_bit: BitPredictor,
    /// Phase-21: `lz_flag` mixer. Blends Order-1 + Order-2 per
    /// `prev_id` (warm bigrams favor Order-2; cold `prev_id`s
    /// favor Order-1).
    lz_flag_mixer: LogitMixer<N_MIX_LZ_FLAG>,
    /// Phase-19g: Order-1 `token_oov` predictor — context =
    /// `(prev_id)`.
    token_oov_bit: BitPredictor,
    /// Phase-21: Order-2 `token_oov` predictor — context =
    /// `(prev_prev_id, prev_id)`.
    token_oov_bit_o2: BitPredictor,
    /// Phase-21: mixer blending Order-1 + Order-2 OOV bit per
    /// `prev_id`.
    token_oov_mixer: LogitMixer<N_MIX_TOKEN_OOV>,
    /// Phase-21: logit-space mixer for the `dict_id` bit stream.
    /// Takes the four base predictors' `P(bit = 0)` and produces a
    /// per-context blend learned by SGD on log-loss. Replaces the
    /// Phase-18 router (which paid 1 router bit/token plus its own
    /// `BitPredictor` table).
    token_id_mixer: LogitMixer<N_MIX_ID>,
    /// Phase-23A: sparse logistic regression that joins the
    /// `dict_id` mixer as the 4th arm. Features are deliberately
    /// chosen to be signals the count-table base predictors miss:
    /// hashed Order-3, wiki × Order-2, and a position-prefix bias.
    token_id_sparse_lr: SparseLR<N_LR_ID_FEATS>,
    /// Phase-23B: sparse-LR for the `lz_flag` binary decision —
    /// 3rd arm of `lz_flag_mixer`. Features: hashed Order-3, wiki ×
    /// Order-2, wiki × Order-1. Same shape used by
    /// `token_oov_sparse_lr`.
    lz_flag_sparse_lr: SparseLR<N_LR_FLAG_FEATS>,
    /// Phase-23B: sparse-LR for the `token_oov_bit` binary decision —
    /// 3rd arm of `token_oov_mixer`. Same feature shape as the
    /// `lz_flag` LR with independent weights (the target bit differs).
    token_oov_sparse_lr: SparseLR<N_LR_FLAG_FEATS>,
    /// Phase-23D: small online MLP that joins the `dict_id` mixer as
    /// the 5th arm. Shares the Phase-23A LR's 3-feature shape but
    /// adds a learned `H`-dim hidden layer and a `tanh`
    /// nonlinearity, letting it capture interactions a single
    /// linear layer can't.
    token_id_mlp: OnlineMLP<N_MLP_ID_FEATS, ID_MLP_H>,
    /// Phase-23P: small online MLP joining the `lz_flag` mixer as the
    /// 4th arm. Same 3-feature shape as `lz_flag_sparse_lr`, but with
    /// a learned `H`-dim hidden layer and `tanh` nonlinearity.
    lz_flag_mlp: OnlineMLP<N_LR_FLAG_FEATS, FLAG_MLP_H>,
    /// Phase-23P: small online MLP joining the `token_oov_bit` mixer
    /// as the 4th arm. Mirrors `lz_flag_mlp` with independent weights
    /// (different target bit).
    token_oov_mlp: OnlineMLP<N_LR_FLAG_FEATS, FLAG_MLP_H>,
}

impl Models {
    fn new() -> Self {
        Self {
            token_class: Order1Ctx::new(),
            token_oov: Order0::new(),
            token_id_bit: BitPredictor::new(ID_BIT_K),
            token_length_bit: BitPredictor::new(LEN_BIT_K),
            lz_flag: Order0::new(),
            lz_offset_bucket_o1: Order1Ctx::new(),
            lz_length_o1: Order1Ctx::new(),
            lz_length_bit_o1: BitPredictor::new(LZ_LENGTH_BIT_K),
            lz_length_bit_o2: BitPredictor::new(LZ_LENGTH_BIT_O2_K),
            lz_length_mixer: LogitMixer::new(LZ_LENGTH_MIXER_K, MIXER_LR),
            lz_offset_bucket_bit_o1: BitPredictor::new(LZ_OFFSET_BUCKET_BIT_K),
            lz_offset_bucket_bit_o2: BitPredictor::new(LZ_OFFSET_BUCKET_BIT_O2_K),
            lz_offset_bucket_mixer: LogitMixer::new(LZ_OFFSET_BUCKET_MIXER_K, MIXER_LR),
            lz_length_sparse_lr: SparseLR::new(LZ_LENGTH_LR_K, LZ_LENGTH_LR_LR),
            lz_length_mlp: OnlineMLP::new(LZ_LENGTH_LR_K, LZ_LENGTH_LR_LR),
            lz_offset_bucket_sparse_lr: SparseLR::new(LZ_LENGTH_LR_K, LZ_LENGTH_LR_LR),
            lz_offset_bucket_mlp: OnlineMLP::new(LZ_LENGTH_LR_K, LZ_LENGTH_LR_LR),
            oov_word_letter: Order1Ctx::new(),
            oov_sep_byte: Order1Ctx::new(),
            case_pattern_global: Order0::new(),
            case_mixed_bit: Order0::new(),
            tag_dict: Order0::new(),
            tag_byte: Order0::new(),
            attr_byte: Order0::new(),
            token_id_bit_o2: BitPredictor::new(ID_BIT_O2_K),
            token_id_bit_wiki: BitPredictor::new(ID_BIT_WIKI_K),
            lz_flag_bit_o1: BitPredictor::new(LZ_FLAG_BIT_O1_K),
            lz_flag_bit: BitPredictor::new(LZ_FLAG_BIT_O2_K),
            token_oov_bit: BitPredictor::new(TOKEN_OOV_BIT_K),
            token_oov_bit_o2: BitPredictor::new(TOKEN_OOV_BIT_O2_K),
            token_id_mixer: LogitMixer::new(MIXER_K, MIXER_LR),
            lz_flag_mixer: LogitMixer::new(LZ_FLAG_MIXER_K, MIXER_LR),
            token_oov_mixer: LogitMixer::new(TOKEN_OOV_MIXER_K, MIXER_LR),
            token_id_sparse_lr: SparseLR::new(ID_SPARSE_LR_K, ID_SPARSE_LR_LR),
            lz_flag_sparse_lr: SparseLR::new(FLAG_SPARSE_LR_K, FLAG_SPARSE_LR_LR),
            token_oov_sparse_lr: SparseLR::new(FLAG_SPARSE_LR_K, FLAG_SPARSE_LR_LR),
            token_id_mlp: OnlineMLP::new(ID_MLP_K, ID_MLP_LR),
            lz_flag_mlp: OnlineMLP::new(FLAG_MLP_K, FLAG_MLP_LR),
            token_oov_mlp: OnlineMLP::new(FLAG_MLP_K, FLAG_MLP_LR),
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

impl Codec for XmlTokRouteCodec {
    fn name(&self) -> &'static str {
        "xml-tok-route"
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
        let mut wiki_class = WikiFineClassifier::new();
        prewarm(
            &buf[..warm_end],
            &mut classifier,
            &mut models,
            &mut dict,
            &mut matcher,
            &mut wiki_class,
        );

        let mut writer = BitWriter::new();
        let mut comp = ComponentBits::default();
        let bits_before_payload = writer.bits_written();
        {
            let mut enc = AcEncoder::new(&mut writer);
            let mut last_class_ctx = CLASS_CTX_NONE;
            let mut ctx = PredCtx::NONE;
            let mut uniform_cdf_cache = UniformCdfCache::new();
            // Phase 23O: token counter since the most recent `<page>`
            // boundary; reset to 0 when `tag_dict` emits idx 0
            // (`page>`). Bucketed into `ctx.page_offset_bucket` at
            // every Content-mode PredCtx update.
            let mut page_token_offset: u32 = 0;
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

                        let (flag_p_zeros, flag_p_mixed) = lz_flag_p_mixed(&models, ctx);
                        let flag_cdf = [0u32, flag_p_mixed, TOTAL];
                        let flag_sym = u32::from(lz_match.is_some());
                        let flag_before = enc.bits_written();
                        enc.encode(&flag_cdf, flag_sym as usize);
                        observe_lz_flag_bit(&mut models, ctx, flag_p_zeros, flag_sym);
                        comp.lz_match += enc.bits_written() - flag_before;

                        if let Some((offset, length)) = lz_match {
                            let length_us = length as usize;
                            let lz_before = enc.bits_written();

                            let bucket = log2_floor(offset);
                            encode_lz_offset_bucket(&mut enc, &mut models, ctx, bucket);
                            if bucket > 0 {
                                let rel = offset - (1u32 << bucket);
                                encode_uniform_bits(&mut enc, &mut uniform_cdf_cache, rel, bucket);
                            }

                            let length_token = length_us - tok_lz::MIN_MATCH;
                            encode_lz_length(
                                &mut enc,
                                &mut models,
                                ctx,
                                u32::try_from(length_token).expect("length_token fits u32"),
                            );

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
                                wiki_class.advance(b);
                            }
                            i = new_i;
                            let last = lookahead[length_us - 1];
                            last_class_ctx = match last.class {
                                TokenClass::Word => CLASS_CTX_WORD,
                                TokenClass::Separator => CLASS_CTX_SEP,
                            };
                            let last_id = lookahead[length_us - 1].id;
                            let p2_new = if length_us >= 2 {
                                Some(lookahead[length_us - 2].id)
                            } else {
                                ctx.p1
                            };
                            let p3_new = if length_us >= 3 {
                                Some(lookahead[length_us - 3].id)
                            } else if length_us >= 2 {
                                ctx.p1
                            } else {
                                ctx.p2
                            };
                            let p1_len_new =
                                u8::try_from(dict.entry(last_id).lower.len()).unwrap_or(u8::MAX);
                            let p2_len_new = if length_us >= 2 {
                                u8::try_from(dict.entry(lookahead[length_us - 2].id).lower.len())
                                    .unwrap_or(u8::MAX)
                            } else {
                                ctx.p1_len
                            };
                            let p1_class_new = class_to_p1_class(TokenClass::from_byte(
                                dict.entry(last_id).lower[0],
                            ));
                            let p2_class_new = if length_us >= 2 {
                                class_to_p1_class(TokenClass::from_byte(
                                    dict.entry(lookahead[length_us - 2].id).lower[0],
                                ))
                            } else {
                                ctx.p1_class
                            };
                            let p1_first_byte_class_new =
                                first_byte_subclass(dict.entry(last_id).lower[0]);
                            let lz_match_length_last_new =
                                u8::try_from(length_us).unwrap_or(u8::MAX);
                            page_token_offset = page_token_offset
                                .saturating_add(u32::try_from(length_us).unwrap_or(u32::MAX));
                            ctx = PredCtx {
                                p3: p3_new,
                                p2: p2_new,
                                p1: Some(last_id),
                                p1_len: p1_len_new,
                                p2_len: p2_len_new,
                                tokens_since_match: 0,
                                lz_match_length_last: lz_match_length_last_new,
                                p1_class: p1_class_new,
                                p2_class: p2_class_new,
                                p1_first_byte_class: p1_first_byte_class_new,
                                page_offset_bucket: log2_bucket(page_token_offset),
                                wiki: ctx.wiki,
                                wiki_depth_bucket: ctx.wiki_depth_bucket,
                            };
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
                        // Snapshot wiki sub-mode before consuming the
                        // token's bytes — that's the sub-mode that
                        // applies to the token being emitted.
                        let token_ctx = PredCtx {
                            wiki: u8::try_from(wiki_class.current_sub().idx())
                                .expect("WikiSub::idx() < 256"),
                            wiki_depth_bucket: wiki_depth_bucket_u8(wiki_class.depth_total()),
                            ..ctx
                        };
                        let new_id = encode_token(
                            &mut enc,
                            &mut models,
                            &mut dict,
                            &mut comp,
                            class,
                            token,
                            token_ctx,
                        );
                        if let Some(id) = new_id {
                            matcher.push(id);
                        }
                        let p1_len_new =
                            new_id.map_or(0, |_| u8::try_from(token.len()).unwrap_or(u8::MAX));
                        let p1_class_new = new_id.map_or(0, |_| class_to_p1_class(class));
                        let p1_first_byte_class_new =
                            new_id.map_or(0, |_| first_byte_subclass(token[0]));
                        page_token_offset = page_token_offset.saturating_add(1);
                        ctx = PredCtx {
                            p3: ctx.p2,
                            p2: ctx.p1,
                            p1: new_id,
                            p1_len: p1_len_new,
                            p2_len: ctx.p1_len,
                            tokens_since_match: ctx.tokens_since_match.saturating_add(1),
                            lz_match_length_last: ctx.lz_match_length_last,
                            p1_class: p1_class_new,
                            p2_class: ctx.p1_class,
                            p1_first_byte_class: p1_first_byte_class_new,
                            page_offset_bucket: log2_bucket(page_token_offset),
                            wiki: token_ctx.wiki,
                            wiki_depth_bucket: token_ctx.wiki_depth_bucket,
                        };

                        for &b in token {
                            classifier.advance(b);
                            wiki_class.advance(b);
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
                        wiki_class.advance(buf[i]);
                        i += 1;
                        last_class_ctx = CLASS_CTX_NONE;
                        ctx = PredCtx::NONE;
                    }
                    Mode::TagStructure => {
                        let tag_end = find_tag_run_end(&buf, i, classifier);
                        let run = &buf[i..tag_end];
                        let mut tag_cdf = [0u32; TAG_ALPHABET + 1];
                        models.tag_dict.cdf_to(&mut tag_cdf);
                        let before = enc.bits_written();
                        let dict_idx = dict_lookup(run);
                        if let Some(idx) = dict_idx {
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
                            wiki_class.advance(b);
                        }
                        i = tag_end;
                        last_class_ctx = CLASS_CTX_NONE;
                        ctx = PredCtx::NONE;
                        // Phase 23O: `page>` is dictionary index 0 — reset
                        // the page-offset counter on every page start.
                        if dict_idx == Some(0) {
                            page_token_offset = 0;
                        }
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
        let mut wiki_class = WikiFineClassifier::new();
        prewarm(
            &buf,
            &mut classifier,
            &mut models,
            &mut dict,
            &mut matcher,
            &mut wiki_class,
        );

        let mut reader = BitReader::new(&archive[4..]);
        let mut dec = AcDecoder::new(&mut reader);
        let mut last_class_ctx = CLASS_CTX_NONE;
        let mut ctx = PredCtx::NONE;
        let mut uniform_cdf_cache = UniformCdfCache::new();
        // Phase 23O: token counter since the most recent `<page>`
        // boundary; mirrors the encode side.
        let mut page_token_offset: u32 = 0;
        while buf.len() - warm_end < measure_len {
            let mode = classifier.current_mode();
            match mode {
                Mode::Content => {
                    let (flag_p_zeros, flag_p_mixed) = lz_flag_p_mixed(&models, ctx);
                    let flag_cdf = [0u32, flag_p_mixed, TOTAL];
                    let flag = dec.decode(&flag_cdf)?;
                    let flag_u32 = u32::try_from(flag).expect("bit fits u32");
                    observe_lz_flag_bit(&mut models, ctx, flag_p_zeros, flag_u32);

                    if flag == 1 {
                        // LZ match record.
                        let bucket_u32 = decode_lz_offset_bucket(&mut dec, &mut models, ctx)?;
                        if bucket_u32 >= OFFSET_BUCKET_ALPHABET as u32 {
                            bail!(
                                "decoded lz_offset_bucket {bucket_u32} >= alphabet \
                                 {OFFSET_BUCKET_ALPHABET}"
                            );
                        }
                        let offset = if bucket_u32 == 0 {
                            1
                        } else {
                            let rel =
                                decode_uniform_bits(&mut dec, &mut uniform_cdf_cache, bucket_u32)?;
                            (1u32 << bucket_u32) + rel
                        };

                        let length_token = decode_lz_length(&mut dec, &mut models, ctx)?;
                        let length =
                            tok_lz::MIN_MATCH + usize::try_from(length_token).expect("length fits");

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

                        // Snapshot the last three ids before pushing,
                        // for the post-match context.
                        let len_match = copied.len();
                        let last_id = copied[len_match - 1];
                        let p2_new = if len_match >= 2 {
                            Some(copied[len_match - 2])
                        } else {
                            ctx.p1
                        };
                        let p3_new = if len_match >= 3 {
                            Some(copied[len_match - 3])
                        } else if len_match >= 2 {
                            ctx.p1
                        } else {
                            ctx.p2
                        };
                        // Snapshot p2_len before the loop consumes `copied`.
                        // p1_len is read from `last_id` after the loop, which
                        // is fine because `last_id` is a `u32` (Copy).
                        let p2_id_opt = if len_match >= 2 {
                            Some(copied[len_match - 2])
                        } else {
                            None
                        };
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
                                wiki_class.advance(b);
                            }
                            matcher.push(id);
                            last_class_ctx = match class {
                                TokenClass::Word => CLASS_CTX_WORD,
                                TokenClass::Separator => CLASS_CTX_SEP,
                            };
                        }
                        let p1_len_new =
                            u8::try_from(dict.entry(last_id).lower.len()).unwrap_or(u8::MAX);
                        let p2_len_new = p2_id_opt.map_or(ctx.p1_len, |id| {
                            u8::try_from(dict.entry(id).lower.len()).unwrap_or(u8::MAX)
                        });
                        let p1_class_new =
                            class_to_p1_class(TokenClass::from_byte(dict.entry(last_id).lower[0]));
                        let p2_class_new = p2_id_opt.map_or(ctx.p1_class, |id| {
                            class_to_p1_class(TokenClass::from_byte(dict.entry(id).lower[0]))
                        });
                        let p1_first_byte_class_new =
                            first_byte_subclass(dict.entry(last_id).lower[0]);
                        let lz_match_length_last_new = u8::try_from(len_match).unwrap_or(u8::MAX);
                        page_token_offset = page_token_offset
                            .saturating_add(u32::try_from(len_match).unwrap_or(u32::MAX));
                        ctx = PredCtx {
                            p3: p3_new,
                            p2: p2_new,
                            p1: Some(last_id),
                            p1_len: p1_len_new,
                            p2_len: p2_len_new,
                            tokens_since_match: 0,
                            lz_match_length_last: lz_match_length_last_new,
                            p1_class: p1_class_new,
                            p2_class: p2_class_new,
                            p1_first_byte_class: p1_first_byte_class_new,
                            page_offset_bucket: log2_bucket(page_token_offset),
                            wiki: ctx.wiki,
                            wiki_depth_bucket: ctx.wiki_depth_bucket,
                        };
                        continue;
                    }

                    let mut class_cdf = [0u32; 3];
                    models.token_class.cdf_to(last_class_ctx, &mut class_cdf);
                    let class_sym = dec.decode(&class_cdf)?;
                    models.token_class.observe(last_class_ctx, class_sym);
                    let class = sym_to_class(class_sym)
                        .with_context(|| format!("invalid class symbol {class_sym}"))?;

                    let token_ctx = PredCtx {
                        wiki: u8::try_from(wiki_class.current_sub().idx())
                            .expect("WikiSub::idx() < 256"),
                        wiki_depth_bucket: wiki_depth_bucket_u8(wiki_class.depth_total()),
                        ..ctx
                    };
                    let (token_bytes, new_prev) =
                        decode_token(&mut dec, &mut models, &mut dict, class, token_ctx)?;
                    for &b in &token_bytes {
                        buf.push(b);
                        classifier.advance(b);
                        wiki_class.advance(b);
                    }
                    if let Some(id) = new_prev {
                        matcher.push(id);
                    }
                    last_class_ctx = match class {
                        TokenClass::Word => CLASS_CTX_WORD,
                        TokenClass::Separator => CLASS_CTX_SEP,
                    };
                    let p1_len_new =
                        new_prev.map_or(0, |_| u8::try_from(token_bytes.len()).unwrap_or(u8::MAX));
                    let p1_class_new = new_prev.map_or(0, |_| class_to_p1_class(class));
                    let p1_first_byte_class_new =
                        new_prev.map_or(0, |_| first_byte_subclass(token_bytes[0]));
                    page_token_offset = page_token_offset.saturating_add(1);
                    ctx = PredCtx {
                        p3: ctx.p2,
                        p2: ctx.p1,
                        p1: new_prev,
                        p1_len: p1_len_new,
                        p2_len: ctx.p1_len,
                        tokens_since_match: ctx.tokens_since_match.saturating_add(1),
                        lz_match_length_last: ctx.lz_match_length_last,
                        p1_class: p1_class_new,
                        p2_class: ctx.p1_class,
                        p1_first_byte_class: p1_first_byte_class_new,
                        page_offset_bucket: log2_bucket(page_token_offset),
                        wiki: token_ctx.wiki,
                        wiki_depth_bucket: token_ctx.wiki_depth_bucket,
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
                    wiki_class.advance(b);
                    last_class_ctx = CLASS_CTX_NONE;
                    ctx = PredCtx::NONE;
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
                            wiki_class.advance(b);
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
                            wiki_class.advance(b);
                        }
                    }
                    last_class_ctx = CLASS_CTX_NONE;
                    ctx = PredCtx::NONE;
                    // Phase 23O: `page>` is dictionary index 0 — reset
                    // the page-offset counter on every page start.
                    if idx == 0 {
                        page_token_offset = 0;
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

const fn class_to_sym(c: TokenClass) -> usize {
    match c {
        TokenClass::Word => 0,
        TokenClass::Separator => 1,
    }
}

/// Phase 23K: `TokenClass` → `PredCtx.p1_class` encoding. 0 reserved
/// for "no token" (i.e. `p1 == None`), so Word/Separator map to 1/2.
const fn class_to_p1_class(c: TokenClass) -> u8 {
    match c {
        TokenClass::Word => 1,
        TokenClass::Separator => 2,
    }
}

/// Phase 23O: 16-bucket log₂ index of a non-negative count. Used
/// for `page_offset_bucket` so the natural unbounded counter folds
/// into a small categorical feature. Bucket 0 = count zero;
/// bucket k = count in `[2^(k-1), 2^k)`; bucket 15 is the saturating
/// top (count ≥ 16384). Compact enough to keep the hashed feature
/// dense; coarse enough to absorb the long-tailed page-size
/// distribution.
#[allow(clippy::cast_possible_truncation)]
const fn log2_bucket(x: u32) -> u8 {
    let raw = 32u32 - x.leading_zeros();
    if raw >= 15 { 15 } else { raw as u8 }
}

/// Phase 23Q: bucket the [`WikiFineClassifier`]'s combined nesting
/// depth (link + template) into a 4-bin categorical: 0 (Plain), 1
/// (single link or template), 2 (one nested level), 3+ (deeper).
/// Coarse on purpose — deep nesting is rare; the feature's value is
/// mostly in distinguishing Plain / depth-1 / depth-2-or-more rather
/// than resolving the long tail.
#[allow(clippy::cast_possible_truncation)]
const fn wiki_depth_bucket_u8(depth: u16) -> u8 {
    if depth >= 3 { 3 } else { depth as u8 }
}

/// Phase 23M: subdivide the first byte of `p1`'s token into a small
/// category. Subclass 0 is the cold "no `p1`" case; Word tokens fall
/// into the alpha bucket (1); the Separator buckets (2..=6) split
/// by character family. The split is chosen for structurally-
/// meaningful boundaries in Wikipedia / XML: whitespace separates
/// prose runs; digits introduce numerical content; ASCII
/// punctuation marks clause/sentence boundaries; brackets/symbols
/// mark tag and template structure.
const fn first_byte_subclass(b: u8) -> u8 {
    match b {
        b'a'..=b'z' | b'A'..=b'Z' => 1,
        b' ' | b'\t' | b'\n' | b'\r' => 2,
        b'0'..=b'9' => 3,
        b'.' | b',' | b'!' | b'?' | b';' | b':' | b'\'' | b'"' | b'-' => 4,
        b'<' | b'>' | b'{' | b'}' | b'[' | b']' | b'(' | b')' | b'=' | b'&' | b'|' | b'/'
        | b'\\' => 5,
        _ => 6,
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

/// Encode one Content token and return `(new_id, new_current_predictor)`.
/// `current_predictor` is the predictor index that was active at the
/// **start** of this token; after encoding it may have switched.
fn encode_token(
    enc: &mut AcEncoder<'_>,
    models: &mut Models,
    dict: &mut Dict,
    comp: &mut ComponentBits,
    class: TokenClass,
    token: &[u8],
    ctx: PredCtx,
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
    let (oov_p_zeros, oov_p_mixed) = token_oov_p_mixed(models, ctx);
    let oov_cdf = [0u32, oov_p_mixed, TOTAL];
    let oov_sym = u32::from(hit_id.is_none());
    let before = enc.bits_written();
    enc.encode(&oov_cdf, oov_sym as usize);
    observe_token_oov_bit(models, ctx, oov_p_zeros, oov_sym);
    comp.token_hit += enc.bits_written() - before;

    if let Some(id) = hit_id {
        // Phase-21: mixer-blended emission of `id` — no router bit.
        let before = enc.bits_written();
        encode_dict_id(enc, models, ctx, id);
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
fn decode_token(
    dec: &mut AcDecoder<'_, '_>,
    models: &mut Models,
    dict: &mut Dict,
    class: TokenClass,
    ctx: PredCtx,
) -> Result<(Vec<u8>, Option<u32>)> {
    let (oov_p_zeros, oov_p_mixed) = token_oov_p_mixed(models, ctx);
    let oov_cdf = [0u32, oov_p_mixed, TOTAL];
    let oov_sym = dec.decode(&oov_cdf)?;
    let oov_sym_u32 = u32::try_from(oov_sym).expect("bit fits u32");
    observe_token_oov_bit(models, ctx, oov_p_zeros, oov_sym_u32);

    if oov_sym == 0 {
        let id = decode_dict_id(dec, models, ctx)?;
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

/// Hash `(prev_id, prefix, bit_pos)` into the Order-1 bit predictor's
/// context space. `prev_id` is the dictionary id of the previous
/// token in the Content stream (or `None` at the start of a Content
/// run / when the previous token was an OOV that couldn't be
/// inserted). Phase 15 folds this into the FNV mix to make the bit
/// predictor an Order-1 word model. For contexts independent of
/// `prev_id` (e.g. OOV length emission) the caller passes `None` to
/// use a single "no-prev" sub-table.
///
/// `u64::MAX` is the sentinel for `None` — real prev ids live in
/// `0..=262143`, so the high bits of `u64::MAX` make the sentinel
/// unambiguous.
fn id_bit_ctx(prev_id: Option<u32>, prefix: u32, bit_pos: u32) -> u64 {
    let prev_field = prev_id.map_or(u64::MAX, u64::from);
    let h = fnv_mix(FNV_OFFSET, prev_field);
    let h = fnv_mix(h, u64::from(prefix));
    fnv_mix(h, u64::from(bit_pos))
}

/// Hash `(prev_prev_id, prev_id, prefix, bit_pos)` into the Order-2
/// `dict_id` predictor's context space. Sentinels for missing context
/// match those of `id_bit_ctx`.
fn id_bit_ctx_o2(
    prev_prev_id: Option<u32>,
    prev_id: Option<u32>,
    prefix: u32,
    bit_pos: u32,
) -> u64 {
    let pp = prev_prev_id.map_or(u64::MAX, u64::from);
    let p = prev_id.map_or(u64::MAX, u64::from);
    let h = fnv_mix(FNV_OFFSET, pp);
    let h = fnv_mix(h, p);
    let h = fnv_mix(h, u64::from(prefix));
    fnv_mix(h, u64::from(bit_pos))
}

/// Hash `(prev_id)` into the Order-1 `lz_flag` and `token_oov`
/// predictor's context. (The `token_oov_bit` shares this hash —
/// it is binary and saturates fast on prev-id alone.)
fn lz_flag_ctx(prev_id: Option<u32>) -> u64 {
    let p = prev_id.map_or(u64::MAX, u64::from);
    fnv_mix(FNV_OFFSET, p)
}

/// Phase-20l: Order-2 `lz_flag` context — `(prev_prev_id, prev_id)`.
fn lz_flag_ctx_o2(prev_prev_id: Option<u32>, prev_id: Option<u32>) -> u64 {
    let pp = prev_prev_id.map_or(u64::MAX, u64::from);
    let p = prev_id.map_or(u64::MAX, u64::from);
    let h = fnv_mix(FNV_OFFSET, pp);
    fnv_mix(h, p)
}

/// Hash `(prev_id)` into a small context-row index for the legacy
/// Order-1 `lz_offset_bucket` and `lz_length` count CDFs. Retained
/// alongside the legacy [`Order1Ctx`] fields for the deferred
/// LR/MLP-arm extension where the count CDF can land as a 3rd
/// mixer arm. Phase 24a/24b's bit-level path uses
/// `id_bit_ctx`/`id_bit_ctx_o2` directly.
#[allow(clippy::cast_possible_truncation, dead_code)]
fn lz_match_row(prev_id: Option<u32>) -> usize {
    let p = prev_id.map_or(u64::MAX, u64::from);
    let h = fnv_mix(FNV_OFFSET, p);
    (h as usize) & (LZ_MATCH_NCTX - 1)
}

/// Bundle of context features available to the predictor stack.
/// `p1`/`p2`/`p3` are the trailing dict-id history; `wiki` is the
/// `wiki_classifier` sub-mode (0=Plain, 1=Link, 2=Template);
/// `p1_len`/`p2_len` are the byte lengths of `p1`/`p2`'s tokens
/// (capped at 255; 0 when the corresponding id is `None`);
/// `tokens_since_match` is the count of tokens emitted since the
/// most recent LZ-match (Phase 23J); `p1_class`/`p2_class` are
/// the token classes of `p1`/`p2` (0=None, 1=Word, 2=Separator,
/// Phases 23K/23L) — implicit in the dict ids but not explicit
/// anywhere the MLP can see. Phase 23F/G/H added the length
/// fields; 23J/K/L continued the "genuinely-new information"
/// pattern that 23I's diagnostic motivated.
#[derive(Clone, Copy, Debug)]
struct PredCtx {
    p3: Option<u32>,
    p2: Option<u32>,
    p1: Option<u32>,
    p1_len: u8,
    p2_len: u8,
    tokens_since_match: u8,
    /// Phase 23N: token length of the most-recent LZ-match (1..=255).
    /// Persists across non-match tokens; updates only on a match.
    /// 0 means "no LZ-match has happened yet on this page".
    lz_match_length_last: u8,
    p1_class: u8,
    p2_class: u8,
    p1_first_byte_class: u8,
    /// Phase 23O: log₂-bucketed token offset within the current
    /// `<page>` block (0 right after page start, growing through 15
    /// for "deep into a long page"). Tracked outside `PredCtx` via
    /// the `page_token_offset` loop variable and snapshotted into
    /// this field at every Content-mode `PredCtx` update.
    page_offset_bucket: u8,
    wiki: u8,
    /// Phase 23Q: total wiki nesting depth (`link_depth` +
    /// `template_depth`), saturating to 3. `wiki` only reveals the
    /// innermost mode; this field distinguishes shallow (depth 1)
    /// from nested (depth 2 / 3+) syntactic contexts where dict-id
    /// distributions differ (e.g. depth-2 templates are mostly
    /// citation-template arguments).
    wiki_depth_bucket: u8,
}

impl PredCtx {
    const NONE: Self = Self {
        p3: None,
        p2: None,
        p1: None,
        p1_len: 0,
        p2_len: 0,
        tokens_since_match: 0,
        lz_match_length_last: 0,
        p1_class: 0,
        p2_class: 0,
        p1_first_byte_class: 0,
        page_offset_bucket: 0,
        wiki: 0,
        wiki_depth_bucket: 0,
    };
}

/// Phase-21: mixer-side context for the `dict_id` bit stream —
/// `(bit_pos, prefix)`. Tested `(bit_pos, prefix, prev_id)` at
/// K=18 and it regressed +0.007 bpb on enwik8: the bigram-
/// conditioned context space (16 × 65 K × `prev_id`) hashed to
/// 250 K slots starves weight learning. The base predictors
/// already carry `prev_id` / `prev_prev_id`; the mixer adds value
/// by learning which base wins per *bit position*, not per bigram.
const fn mixer_id_ctx(bit_pos: u32, prefix: u32) -> u64 {
    let h = fnv_mix(FNV_OFFSET, bit_pos as u64);
    fnv_mix(h, prefix as u64)
}

/// Phase-23A: feature hashes for the `dict_id` sparse-LR. Each
/// feature lives in its own table; together they cover signals the
/// count-based base predictors can't afford to materialize
/// directly — hashed Order-3, wiki × Order-2, and a position-prefix
/// bias.
fn id_lr_features(ctx: PredCtx, prefix: u32, bit_pos: u32) -> [u64; N_LR_ID_FEATS] {
    let p1 = ctx.p1.map_or(u64::MAX, u64::from);
    let p2 = ctx.p2.map_or(u64::MAX, u64::from);
    let p3 = ctx.p3.map_or(u64::MAX, u64::from);
    let prefix64 = u64::from(prefix);
    let bit_pos64 = u64::from(bit_pos);

    // Order-3 hashed: (p3, p2, p1, prefix, bit_pos). Natural cell
    // count is ~2^57, far past any direct table; SGD with hash
    // collisions averages noisy updates the same way it averages
    // within each surviving cell.
    let f_o3 = {
        let h = fnv_mix(FNV_OFFSET, p3);
        let h = fnv_mix(h, p2);
        let h = fnv_mix(h, p1);
        let h = fnv_mix(h, prefix64);
        fnv_mix(h, bit_pos64)
    };

    // Wiki × Order-2: lets the wiki sub-mode bend the bigram
    // distribution. The Phase-22 wiki predictor stopped at Order-1
    // because its count table couldn't afford the cross.
    let f_wiki_o2 = {
        let h = fnv_mix(FNV_OFFSET, u64::from(ctx.wiki));
        let h = fnv_mix(h, p2);
        let h = fnv_mix(h, p1);
        let h = fnv_mix(h, prefix64);
        fnv_mix(h, bit_pos64)
    };

    // (bit_pos, prefix) bias — a position-level marginal that
    // applies uniformly across contexts. Lets the LR carry an
    // honest intercept even when both higher-order features are in
    // cold slots.
    let f_bias = {
        let h = fnv_mix(FNV_OFFSET, prefix64);
        fnv_mix(h, bit_pos64)
    };

    [f_o3, f_wiki_o2, f_bias]
}

/// Phase-23F → Phase-23J: feature hashes for the `dict_id` MLP.
/// Superset of [`id_lr_features`] plus four coarse warm features:
/// `p1_len_bucket` (23F), `p2_len_bucket` (23G), `(p1_len, p2_len)`
/// joint (23H), and `tokens_since_match_bucket` (23J). The last is
/// the first feature carrying information **not derivable from the
/// dict-id history** — it tells the MLP how recently the codec
/// emitted an LZ-match, which correlates strongly with whether
/// we're in a heavily-templated region or unique prose.
fn id_mlp_features(ctx: PredCtx, prefix: u32, bit_pos: u32) -> [u64; N_MLP_ID_FEATS] {
    let [f_o3, f_wiki_o2, f_bias] = id_lr_features(ctx, prefix, bit_pos);
    let prefix64 = u64::from(prefix);
    let bit_pos64 = u64::from(bit_pos);

    let p1_len_bucket = u64::from(ctx.p1_len.min(15));
    let p2_len_bucket = u64::from(ctx.p2_len.min(15));
    let recency_bucket = u64::from(ctx.tokens_since_match.min(15));
    let f_p1_len = {
        let h = fnv_mix(FNV_OFFSET, p1_len_bucket);
        let h = fnv_mix(h, prefix64);
        fnv_mix(h, bit_pos64)
    };
    let f_p2_len = {
        let h = fnv_mix(FNV_OFFSET, p2_len_bucket);
        let h = fnv_mix(h, prefix64);
        fnv_mix(h, bit_pos64)
    };
    let f_len_pair = {
        let h = fnv_mix(FNV_OFFSET, p1_len_bucket);
        let h = fnv_mix(h, p2_len_bucket);
        let h = fnv_mix(h, prefix64);
        fnv_mix(h, bit_pos64)
    };
    let f_recency = {
        let h = fnv_mix(FNV_OFFSET, recency_bucket);
        let h = fnv_mix(h, prefix64);
        fnv_mix(h, bit_pos64)
    };
    // Phase 23K: `p1_class` is 3-valued — densest "new info" feature
    // so far. Cell count 3 × 65 K × 16 ≈ 2^21, ~2000 contexts/slot
    // at K=20 — SGD trains weights almost immediately.
    let f_p1_class = {
        let h = fnv_mix(FNV_OFFSET, u64::from(ctx.p1_class));
        let h = fnv_mix(h, prefix64);
        fnv_mix(h, bit_pos64)
    };
    // Phase 23L: sibling of `p1_class` one step back. Same shape;
    // the marginal information value is smaller because of
    // Word/Separator alternation, but boundary cases at structural
    // breaks still carry independent signal.
    let f_p2_class = {
        let h = fnv_mix(FNV_OFFSET, u64::from(ctx.p2_class));
        let h = fnv_mix(h, prefix64);
        fnv_mix(h, bit_pos64)
    };
    // Phase 23M: subdivide the first byte of `p1`'s token into
    // Alpha / Whitespace / Digit / Punctuation / Symbol / Other.
    // Refines `p1_class` — same machinery, finer resolution at
    // separator boundaries (which is where Word/Sep alternation
    // leaves the most predictive uncertainty).
    let f_p1_first_byte = {
        let h = fnv_mix(FNV_OFFSET, u64::from(ctx.p1_first_byte_class));
        let h = fnv_mix(h, prefix64);
        fnv_mix(h, bit_pos64)
    };
    // Phase 23N: length of the most-recent LZ-match. Pairs with
    // `tokens_since_match` to encode "how big was the last match,
    // and how long ago" — distinguishes a templated region (long
    // matches near zero recency) from sporadic phrase repetition
    // (short matches, larger recency).
    let lz_match_length_bucket = u64::from(ctx.lz_match_length_last.min(15));
    let f_lz_match_length = {
        let h = fnv_mix(FNV_OFFSET, lz_match_length_bucket);
        let h = fnv_mix(h, prefix64);
        fnv_mix(h, bit_pos64)
    };
    // Phase 23O: log₂-bucketed token offset within the current
    // `<page>`. Genuinely orthogonal to every other PredCtx field —
    // captures structural position (early infobox / mid prose /
    // late references) the dict-id history can't reconstruct.
    let f_page_offset = {
        let h = fnv_mix(FNV_OFFSET, u64::from(ctx.page_offset_bucket));
        let h = fnv_mix(h, prefix64);
        fnv_mix(h, bit_pos64)
    };
    // Phase 23Q: 4-bin wiki nesting depth (link_depth +
    // template_depth, saturating to 3). `ctx.wiki` only carries the
    // innermost sub-mode; this distinguishes "depth-1 inside Plain"
    // from "depth-2 template inside a link", which have measurably
    // different dict-id distributions (deep templates are mostly
    // citation arguments — high digit / punct content).
    let f_wiki_depth = {
        let h = fnv_mix(FNV_OFFSET, u64::from(ctx.wiki_depth_bucket));
        let h = fnv_mix(h, prefix64);
        fnv_mix(h, bit_pos64)
    };

    [
        f_o3,
        f_wiki_o2,
        f_bias,
        f_p1_len,
        f_p2_len,
        f_len_pair,
        f_recency,
        f_p1_class,
        f_p2_class,
        f_p1_first_byte,
        f_lz_match_length,
        f_page_offset,
        f_wiki_depth,
    ]
}

/// Read all base predictors' `P(bit = 0)` without touching any
/// state. Used on the encode path before we know the bit value.
fn id_bit_p_zeros(models: &Models, ctx: PredCtx, prefix: u32, bit_pos: u32) -> [u32; N_MIX_ID] {
    let lr_feats = id_lr_features(ctx, prefix, bit_pos);
    let mlp_feats = id_mlp_features(ctx, prefix, bit_pos);
    [
        models
            .token_id_bit
            .predict_p_zero(id_bit_ctx(ctx.p1, prefix, bit_pos)),
        models
            .token_id_bit_o2
            .predict_p_zero(id_bit_ctx_o2(ctx.p2, ctx.p1, prefix, bit_pos)),
        models
            .token_id_bit_wiki
            .predict_p_zero(id_bit_ctx_wiki(ctx.wiki, ctx.p1, prefix, bit_pos)),
        models.token_id_sparse_lr.predict(&lr_feats),
        models.token_id_mlp.predict(&mlp_feats),
    ]
}

/// Observe one `id` bit in every base predictor *and* the mixer,
/// in the order encode/decode see them — MSB-first, with the
/// running `prefix` carried through.
fn observe_id_bit(
    models: &mut Models,
    ctx: PredCtx,
    prefix: u32,
    bit_pos: u32,
    bit: u32,
    p_zeros: [u32; N_MIX_ID],
) {
    models
        .token_id_bit
        .observe(id_bit_ctx(ctx.p1, prefix, bit_pos), bit);
    models
        .token_id_bit_o2
        .observe(id_bit_ctx_o2(ctx.p2, ctx.p1, prefix, bit_pos), bit);
    models
        .token_id_bit_wiki
        .observe(id_bit_ctx_wiki(ctx.wiki, ctx.p1, prefix, bit_pos), bit);
    let lr_feats = id_lr_features(ctx, prefix, bit_pos);
    let mlp_feats = id_mlp_features(ctx, prefix, bit_pos);
    models.token_id_sparse_lr.observe(&lr_feats, bit);
    models.token_id_mlp.observe(&mlp_feats, bit);
    models
        .token_id_mixer
        .observe(mixer_id_ctx(bit_pos, prefix), &p_zeros, bit);
}

/// Hash `(wiki_sub_mode, prev_id, prefix, bit_pos)` into the
/// wiki-mode-conditioned `dict_id` predictor's context space.
fn id_bit_ctx_wiki(wiki: u8, prev_id: Option<u32>, prefix: u32, bit_pos: u32) -> u64 {
    let p = prev_id.map_or(u64::MAX, u64::from);
    let h = fnv_mix(FNV_OFFSET, u64::from(wiki));
    let h = fnv_mix(h, p);
    let h = fnv_mix(h, u64::from(prefix));
    fnv_mix(h, u64::from(bit_pos))
}

/// Observe `id` in the entire `dict_id` predictor stack (used
/// during prewarm where we don't emit AC bits). Order matches
/// encode/decode.
fn observe_id_all(models: &mut Models, ctx: PredCtx, id: u32) {
    let mut prefix: u32 = 0;
    for bit_pos in (0..ID_BITS).rev() {
        let bit = (id >> bit_pos) & 1;
        let p_zeros = id_bit_p_zeros(models, ctx, prefix, bit_pos);
        observe_id_bit(models, ctx, prefix, bit_pos, bit, p_zeros);
        prefix = (prefix << 1) | bit;
    }
}

/// Encode `id` MSB-first using the mixer-blended `P(bit = 0)` from
/// the Order-1 and Order-2 base predictors. Updates all three models
/// (the two base predictors + the mixer weights) per bit.
fn encode_dict_id(enc: &mut AcEncoder<'_>, models: &mut Models, ctx: PredCtx, id: u32) {
    let mut prefix: u32 = 0;
    for bit_pos in (0..ID_BITS).rev() {
        let bit = (id >> bit_pos) & 1;
        let p_zeros = id_bit_p_zeros(models, ctx, prefix, bit_pos);
        let p_mixed = models
            .token_id_mixer
            .predict(mixer_id_ctx(bit_pos, prefix), &p_zeros);
        let cdf = [0u32, p_mixed, TOTAL];
        enc.encode(&cdf, bit as usize);
        observe_id_bit(models, ctx, prefix, bit_pos, bit, p_zeros);
        prefix = (prefix << 1) | bit;
    }
}

/// Phase-24a / 24c: read all four `lz_length` bit predictors'
/// `P(bit = 0)` (Order-1, Order-2, LR, MLP) and the mixer-blended
/// value. The LR/MLP share the dict-id `id_lr_features` feature
/// shape; each predictor has its own backing storage so the hash
/// reuse doesn't cause weight collisions.
fn length_bit_p_zeros(
    models: &Models,
    ctx: PredCtx,
    prefix: u32,
    bit_pos: u32,
) -> [u32; N_MIX_LZ_LENGTH] {
    let ctx_o1 = id_bit_ctx(ctx.p1, prefix, bit_pos);
    let ctx_o2 = id_bit_ctx_o2(ctx.p2, ctx.p1, prefix, bit_pos);
    let feats = id_lr_features(ctx, prefix, bit_pos);
    [
        models.lz_length_bit_o1.predict_p_zero(ctx_o1),
        models.lz_length_bit_o2.predict_p_zero(ctx_o2),
        models.lz_length_sparse_lr.predict(&feats),
        models.lz_length_mlp.predict(&feats),
    ]
}

/// Phase-24a / 24c: observe one `lz_length` bit in all base
/// predictors and the mixer.
fn observe_length_bit(
    models: &mut Models,
    ctx: PredCtx,
    prefix: u32,
    bit_pos: u32,
    bit: u32,
    p_zeros: [u32; N_MIX_LZ_LENGTH],
) {
    let ctx_o1 = id_bit_ctx(ctx.p1, prefix, bit_pos);
    let ctx_o2 = id_bit_ctx_o2(ctx.p2, ctx.p1, prefix, bit_pos);
    let feats = id_lr_features(ctx, prefix, bit_pos);
    let mixer_ctx = mixer_id_ctx(bit_pos, prefix);
    models.lz_length_bit_o1.observe(ctx_o1, bit);
    models.lz_length_bit_o2.observe(ctx_o2, bit);
    models.lz_length_sparse_lr.observe(&feats, bit);
    models.lz_length_mlp.observe(&feats, bit);
    models.lz_length_mixer.observe(mixer_ctx, &p_zeros, bit);
}

/// Phase-24a: encode `length_token` (0..=255) MSB-first using the
/// mixer-blended `P(bit=0)`. Replaces the Phase-19h
/// `lz_length_o1.cdf_to` count CDF, capturing per-bit-position
/// conditional structure that an Order-1 count CDF can't.
fn encode_lz_length(enc: &mut AcEncoder<'_>, models: &mut Models, ctx: PredCtx, length_token: u32) {
    let mut prefix: u32 = 0;
    for bit_pos in (0..LZ_LENGTH_BITS).rev() {
        let bit = (length_token >> bit_pos) & 1;
        let p_zeros = length_bit_p_zeros(models, ctx, prefix, bit_pos);
        let p_mixed = models
            .lz_length_mixer
            .predict(mixer_id_ctx(bit_pos, prefix), &p_zeros);
        let cdf = [0u32, p_mixed, TOTAL];
        enc.encode(&cdf, bit as usize);
        observe_length_bit(models, ctx, prefix, bit_pos, bit, p_zeros);
        prefix = (prefix << 1) | bit;
    }
}

/// Phase-24a: decode an 8-bit `length_token` MSB-first using the
/// mixer-blended `P(bit=0)`. Mirror of `encode_lz_length`.
fn decode_lz_length(dec: &mut AcDecoder<'_, '_>, models: &mut Models, ctx: PredCtx) -> Result<u32> {
    let mut prefix: u32 = 0;
    for bit_pos in (0..LZ_LENGTH_BITS).rev() {
        let p_zeros = length_bit_p_zeros(models, ctx, prefix, bit_pos);
        let p_mixed = models
            .lz_length_mixer
            .predict(mixer_id_ctx(bit_pos, prefix), &p_zeros);
        let cdf = [0u32, p_mixed, TOTAL];
        let bit = u32::try_from(dec.decode(&cdf)?).expect("bit symbol fits u32");
        observe_length_bit(models, ctx, prefix, bit_pos, bit, p_zeros);
        prefix = (prefix << 1) | bit;
    }
    Ok(prefix)
}

/// Phase-24b / 24c: read all four `lz_offset_bucket` predictors'
/// `P(bit=0)` and the mixer-blended value.
fn offset_bucket_bit_p_zeros(
    models: &Models,
    ctx: PredCtx,
    prefix: u32,
    bit_pos: u32,
) -> [u32; N_MIX_LZ_OFFSET_BUCKET] {
    let ctx_o1 = id_bit_ctx(ctx.p1, prefix, bit_pos);
    let ctx_o2 = id_bit_ctx_o2(ctx.p2, ctx.p1, prefix, bit_pos);
    let feats = id_lr_features(ctx, prefix, bit_pos);
    [
        models.lz_offset_bucket_bit_o1.predict_p_zero(ctx_o1),
        models.lz_offset_bucket_bit_o2.predict_p_zero(ctx_o2),
        models.lz_offset_bucket_sparse_lr.predict(&feats),
        models.lz_offset_bucket_mlp.predict(&feats),
    ]
}

/// Phase-24b / 24c: observe one `lz_offset_bucket` bit in all base
/// predictors and the mixer.
fn observe_offset_bucket_bit(
    models: &mut Models,
    ctx: PredCtx,
    prefix: u32,
    bit_pos: u32,
    bit: u32,
    p_zeros: [u32; N_MIX_LZ_OFFSET_BUCKET],
) {
    let ctx_o1 = id_bit_ctx(ctx.p1, prefix, bit_pos);
    let ctx_o2 = id_bit_ctx_o2(ctx.p2, ctx.p1, prefix, bit_pos);
    let feats = id_lr_features(ctx, prefix, bit_pos);
    let mixer_ctx = mixer_id_ctx(bit_pos, prefix);
    models.lz_offset_bucket_bit_o1.observe(ctx_o1, bit);
    models.lz_offset_bucket_bit_o2.observe(ctx_o2, bit);
    models.lz_offset_bucket_sparse_lr.observe(&feats, bit);
    models.lz_offset_bucket_mlp.observe(&feats, bit);
    models
        .lz_offset_bucket_mixer
        .observe(mixer_ctx, &p_zeros, bit);
}

/// Phase-24b: encode `bucket` (0..=20) MSB-first as 5 bits with
/// mixer-blended `P(bit=0)`. Replaces the Phase-19h
/// `lz_offset_bucket_o1.cdf_to` count CDF. The 21-vs-32 alphabet
/// gap is handled by letting the predictor learn `P(bit=1)=0` on
/// impossible branches during warm-up.
fn encode_lz_offset_bucket(
    enc: &mut AcEncoder<'_>,
    models: &mut Models,
    ctx: PredCtx,
    bucket: u32,
) {
    let mut prefix: u32 = 0;
    for bit_pos in (0..LZ_OFFSET_BUCKET_BITS).rev() {
        let bit = (bucket >> bit_pos) & 1;
        let p_zeros = offset_bucket_bit_p_zeros(models, ctx, prefix, bit_pos);
        let p_mixed = models
            .lz_offset_bucket_mixer
            .predict(mixer_id_ctx(bit_pos, prefix), &p_zeros);
        let cdf = [0u32, p_mixed, TOTAL];
        enc.encode(&cdf, bit as usize);
        observe_offset_bucket_bit(models, ctx, prefix, bit_pos, bit, p_zeros);
        prefix = (prefix << 1) | bit;
    }
}

/// Phase-24b: decode a 5-bit `bucket` MSB-first. Mirror of
/// `encode_lz_offset_bucket`.
fn decode_lz_offset_bucket(
    dec: &mut AcDecoder<'_, '_>,
    models: &mut Models,
    ctx: PredCtx,
) -> Result<u32> {
    let mut prefix: u32 = 0;
    for bit_pos in (0..LZ_OFFSET_BUCKET_BITS).rev() {
        let p_zeros = offset_bucket_bit_p_zeros(models, ctx, prefix, bit_pos);
        let p_mixed = models
            .lz_offset_bucket_mixer
            .predict(mixer_id_ctx(bit_pos, prefix), &p_zeros);
        let cdf = [0u32, p_mixed, TOTAL];
        let bit = u32::try_from(dec.decode(&cdf)?).expect("bit symbol fits u32");
        observe_offset_bucket_bit(models, ctx, prefix, bit_pos, bit, p_zeros);
        prefix = (prefix << 1) | bit;
    }
    Ok(prefix)
}

/// Phase-23B: feature hashes shared by the `lz_flag` and
/// `token_oov_bit` sparse-LRs. Both are binary decisions made once
/// per token, with identical available context (`p3..p1`, `wiki`),
/// so they share the feature shape; the per-stream weight tables
/// differ because the targets differ.
fn flag_lr_features(ctx: PredCtx) -> [u64; N_LR_FLAG_FEATS] {
    let p1 = ctx.p1.map_or(u64::MAX, u64::from);
    let p2 = ctx.p2.map_or(u64::MAX, u64::from);
    let p3 = ctx.p3.map_or(u64::MAX, u64::from);
    let wiki = u64::from(ctx.wiki);

    // Order-3 hashed: (p3, p2, p1). The natural ~2^48 logical-cell
    // count is unreachable for a count table; SGD averages noisy
    // updates across hash collisions instead.
    let f_o3 = {
        let h = fnv_mix(FNV_OFFSET, p3);
        let h = fnv_mix(h, p2);
        fnv_mix(h, p1)
    };

    // Wiki × Order-2: lets the wiki sub-mode bend the bigram
    // match/OOV rates without paying for a dedicated count table.
    let f_wiki_o2 = {
        let h = fnv_mix(FNV_OFFSET, wiki);
        let h = fnv_mix(h, p2);
        fnv_mix(h, p1)
    };

    // Wiki × Order-1: coarser, denser cross — captures the wiki
    // sub-mode unigram interaction even when p2 lands in a cold
    // slot of the higher-order feature.
    let f_wiki_o1 = {
        let h = fnv_mix(FNV_OFFSET, wiki);
        fnv_mix(h, p1)
    };

    [f_o3, f_wiki_o2, f_wiki_o1]
}

/// Phase-21 / Phase-23B: read all `lz_flag` base predictors'
/// `P(no-match)` and the mixer-blended value. State is not
/// mutated; call `observe_lz_flag_bit` after the bit is known.
fn lz_flag_p_mixed(models: &Models, ctx: PredCtx) -> ([u32; N_MIX_LZ_FLAG], u32) {
    let ctx_o1 = lz_flag_ctx(ctx.p1);
    let ctx_o2 = lz_flag_ctx_o2(ctx.p2, ctx.p1);
    let feats = flag_lr_features(ctx);
    let p_zeros = [
        models.lz_flag_bit_o1.predict_p_zero(ctx_o1),
        models.lz_flag_bit.predict_p_zero(ctx_o2),
        models.lz_flag_sparse_lr.predict(&feats),
        models.lz_flag_mlp.predict(&feats),
    ];
    let p_mixed = models.lz_flag_mixer.predict(ctx_o1, &p_zeros);
    (p_zeros, p_mixed)
}

/// Phase-21 / Phase-23B / Phase-23P: observe one `lz_flag` bit in
/// every base predictor and the mixer. `p_zeros` should be the same
/// array returned by `lz_flag_p_mixed` for the same context.
fn observe_lz_flag_bit(models: &mut Models, ctx: PredCtx, p_zeros: [u32; N_MIX_LZ_FLAG], bit: u32) {
    let ctx_o1 = lz_flag_ctx(ctx.p1);
    let ctx_o2 = lz_flag_ctx_o2(ctx.p2, ctx.p1);
    let feats = flag_lr_features(ctx);
    models.lz_flag_bit_o1.observe(ctx_o1, bit);
    models.lz_flag_bit.observe(ctx_o2, bit);
    models.lz_flag_sparse_lr.observe(&feats, bit);
    models.lz_flag_mlp.observe(&feats, bit);
    models.lz_flag_mixer.observe(ctx_o1, &p_zeros, bit);
}

/// Phase-21 / Phase-23B / Phase-23P: read all `token_oov` base
/// predictors' `P(hit)` and the mixer-blended value. Same shape as
/// `lz_flag_p_mixed`.
fn token_oov_p_mixed(models: &Models, ctx: PredCtx) -> ([u32; N_MIX_TOKEN_OOV], u32) {
    let ctx_o1 = lz_flag_ctx(ctx.p1);
    let ctx_o2 = lz_flag_ctx_o2(ctx.p2, ctx.p1);
    let feats = flag_lr_features(ctx);
    let p_zeros = [
        models.token_oov_bit.predict_p_zero(ctx_o1),
        models.token_oov_bit_o2.predict_p_zero(ctx_o2),
        models.token_oov_sparse_lr.predict(&feats),
        models.token_oov_mlp.predict(&feats),
    ];
    let p_mixed = models.token_oov_mixer.predict(ctx_o1, &p_zeros);
    (p_zeros, p_mixed)
}

fn observe_token_oov_bit(
    models: &mut Models,
    ctx: PredCtx,
    p_zeros: [u32; N_MIX_TOKEN_OOV],
    bit: u32,
) {
    let ctx_o1 = lz_flag_ctx(ctx.p1);
    let ctx_o2 = lz_flag_ctx_o2(ctx.p2, ctx.p1);
    let feats = flag_lr_features(ctx);
    models.token_oov_bit.observe(ctx_o1, bit);
    models.token_oov_bit_o2.observe(ctx_o2, bit);
    models.token_oov_sparse_lr.observe(&feats, bit);
    models.token_oov_mlp.observe(&feats, bit);
    models.token_oov_mixer.observe(ctx_o1, &p_zeros, bit);
}

fn decode_dict_id(dec: &mut AcDecoder<'_, '_>, models: &mut Models, ctx: PredCtx) -> Result<u32> {
    let mut prefix: u32 = 0;
    for bit_pos in (0..ID_BITS).rev() {
        let p_zeros = id_bit_p_zeros(models, ctx, prefix, bit_pos);
        let p_mixed = models
            .token_id_mixer
            .predict(mixer_id_ctx(bit_pos, prefix), &p_zeros);
        let cdf = [0u32, p_mixed, TOTAL];
        let bit = u32::try_from(dec.decode(&cdf)?).expect("bit symbol fits u32");
        observe_id_bit(models, ctx, prefix, bit_pos, bit, p_zeros);
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
    let mut p4 = LETTER_START;
    let mut p3 = LETTER_START;
    let mut p2 = LETTER_START;
    let mut p1 = LETTER_START;
    let mut cdf = [0u32; 27];
    for &b in lower {
        let idx = (b - b'a') as usize;
        let ctx = ((p4 * LETTER_NCTX + p3) * LETTER_NCTX + p2) * LETTER_NCTX + p1;
        models.oov_word_letter.cdf_to(ctx, &mut cdf);
        enc.encode(&cdf, idx);
        models.oov_word_letter.observe(ctx, idx);
        p4 = p3;
        p3 = p2;
        p2 = p1;
        p1 = idx;
    }
}

fn decode_oov_word_bytes(
    dec: &mut AcDecoder<'_, '_>,
    models: &mut Models,
    length: usize,
) -> Result<Vec<u8>> {
    let mut p4 = LETTER_START;
    let mut p3 = LETTER_START;
    let mut p2 = LETTER_START;
    let mut p1 = LETTER_START;
    let mut cdf = [0u32; 27];
    let mut out = Vec::with_capacity(length);
    for _ in 0..length {
        let ctx = ((p4 * LETTER_NCTX + p3) * LETTER_NCTX + p2) * LETTER_NCTX + p1;
        models.oov_word_letter.cdf_to(ctx, &mut cdf);
        let idx = dec.decode(&cdf)?;
        models.oov_word_letter.observe(ctx, idx);
        let b = b'a' + u8::try_from(idx).expect("letter idx fits u8");
        out.push(b);
        p4 = p3;
        p3 = p2;
        p2 = p1;
        p1 = idx;
    }
    Ok(out)
}

fn encode_oov_sep_bytes(enc: &mut AcEncoder<'_>, models: &mut Models, bytes: &[u8]) {
    let mut prev_prev = BYTE_START;
    let mut prev = BYTE_START;
    let mut cdf = [0u32; 257];
    for &b in bytes {
        let idx = b as usize;
        let ctx = prev_prev * BYTE_NCTX + prev;
        models.oov_sep_byte.cdf_to(ctx, &mut cdf);
        enc.encode(&cdf, idx);
        models.oov_sep_byte.observe(ctx, idx);
        prev_prev = prev;
        prev = idx;
    }
}

fn decode_oov_sep_bytes(
    dec: &mut AcDecoder<'_, '_>,
    models: &mut Models,
    length: usize,
) -> Result<Vec<u8>> {
    let mut prev_prev = BYTE_START;
    let mut prev = BYTE_START;
    let mut cdf = [0u32; 257];
    let mut out = Vec::with_capacity(length);
    for _ in 0..length {
        let ctx = prev_prev * BYTE_NCTX + prev;
        models.oov_sep_byte.cdf_to(ctx, &mut cdf);
        let idx = dec.decode(&cdf)?;
        models.oov_sep_byte.observe(ctx, idx);
        let b = u8::try_from(idx).expect("byte symbol fits u8");
        out.push(b);
        prev_prev = prev;
        prev = idx;
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
    wiki_class: &mut WikiFineClassifier,
) {
    let mut last_class_ctx = CLASS_CTX_NONE;
    let mut ctx = PredCtx::NONE;
    // Phase 23O: token counter since the most recent `<page>`
    // boundary; same semantics as in encode/decode.
    let mut page_token_offset: u32 = 0;
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
                let token_ctx = PredCtx {
                    wiki: u8::try_from(wiki_class.current_sub().idx())
                        .expect("WikiSub::idx() < 256"),
                    wiki_depth_bucket: wiki_depth_bucket_u8(wiki_class.depth_total()),
                    ..ctx
                };
                let new_id = observe_token(models, dict, class, token, token_ctx);
                if let Some(id) = new_id {
                    matcher.push(id);
                }
                let p1_len_new = new_id.map_or(0, |_| u8::try_from(token.len()).unwrap_or(u8::MAX));
                let p1_class_new = new_id.map_or(0, |_| class_to_p1_class(class));
                let p1_first_byte_class_new = new_id.map_or(0, |_| first_byte_subclass(token[0]));
                page_token_offset = page_token_offset.saturating_add(1);
                ctx = PredCtx {
                    p3: ctx.p2,
                    p2: ctx.p1,
                    p1: new_id,
                    p1_len: p1_len_new,
                    p2_len: ctx.p1_len,
                    tokens_since_match: ctx.tokens_since_match.saturating_add(1),
                    lz_match_length_last: ctx.lz_match_length_last,
                    p1_class: p1_class_new,
                    p2_class: ctx.p1_class,
                    p1_first_byte_class: p1_first_byte_class_new,
                    page_offset_bucket: log2_bucket(page_token_offset),
                    wiki: token_ctx.wiki,
                    wiki_depth_bucket: token_ctx.wiki_depth_bucket,
                };
                // Also observe the lz_flag=0 (no-match) for prewarm
                // so the model arrives at measure with a calibrated
                // P(no-match) prior. Both base predictors *and* the
                // mixer must observe so encode/decode see identical
                // state when measure begins.
                let (flag_p_zeros, _) = lz_flag_p_mixed(models, ctx);
                observe_lz_flag_bit(models, ctx, flag_p_zeros, 0);

                for &b in token {
                    classifier.advance(b);
                    wiki_class.advance(b);
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
                wiki_class.advance(warm[i]);
                i += 1;
                last_class_ctx = CLASS_CTX_NONE;
                ctx = PredCtx::NONE;
            }
            Mode::TagStructure => {
                let run_end_ = find_tag_run_end(warm, i, *classifier);
                let run = &warm[i..run_end_];
                let dict_idx = dict_lookup(run);
                if let Some(idx) = dict_idx {
                    models.tag_dict.observe(idx);
                } else {
                    models.tag_dict.observe(TAG_ESCAPE);
                    for &b in run {
                        models.tag_byte.observe(b as usize);
                    }
                }
                for &b in run {
                    classifier.advance(b);
                    wiki_class.advance(b);
                }
                i = run_end_;
                last_class_ctx = CLASS_CTX_NONE;
                ctx = PredCtx::NONE;
                // Phase 23O: mirror encode/decode page-offset reset.
                if dict_idx == Some(0) {
                    page_token_offset = 0;
                }
            }
        }
    }
}

/// Prewarm observe path — must mirror encode/decode's bit-level
/// operations exactly so the model state at the start of measure
/// is identical on both sides. Returns the assigned id for the
/// next token's `prev_id`, matching the encode-time convention.
fn observe_token(
    models: &mut Models,
    dict: &mut Dict,
    class: TokenClass,
    token: &[u8],
    ctx: PredCtx,
) -> Option<u32> {
    let (pat, key) = match class {
        TokenClass::Word => (Some(classify_case(token)), lowercase(token)),
        TokenClass::Separator => (None, token.to_vec()),
    };
    let hit_id = dict.get(&key);
    let oov_sym = u32::from(hit_id.is_none());
    let (oov_p_zeros, _) = token_oov_p_mixed(models, ctx);
    observe_token_oov_bit(models, ctx, oov_p_zeros, oov_sym);

    if let Some(id) = hit_id {
        // Mirror encode_token: observe each id bit through the base
        // predictors *and* the mixer so its weights land at the
        // same state on encode and decode.
        observe_id_all(models, ctx, id);

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
                let mut p4 = LETTER_START;
                let mut p3 = LETTER_START;
                let mut p2 = LETTER_START;
                let mut p1 = LETTER_START;
                for &b in &key {
                    let idx = (b - b'a') as usize;
                    let ctx = ((p4 * LETTER_NCTX + p3) * LETTER_NCTX + p2) * LETTER_NCTX + p1;
                    models.oov_word_letter.observe(ctx, idx);
                    p4 = p3;
                    p3 = p2;
                    p2 = p1;
                    p1 = idx;
                }
            }
            TokenClass::Separator => {
                let mut prev_prev = BYTE_START;
                let mut prev = BYTE_START;
                for &b in &key {
                    let idx = b as usize;
                    let ctx = prev_prev * BYTE_NCTX + prev;
                    models.oov_sep_byte.observe(ctx, idx);
                    prev_prev = prev;
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
        let codec = XmlTokRouteCodec;
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
