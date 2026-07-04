//! The encode/decode driver: the framing-agnostic codec core.
//!
//! [`Compressor`] wires the pluggable stages — byte preprocessors, the tokenizer,
//! token preprocessors — around the entropy stage. [`encode_tokens`] /
//! [`decode_tokens`] are the entropy stage itself: one per-bit loop over the
//! models, the [`Mixer`], and the carryless range [`Encoder`]/[`Decoder`]. Encode
//! and decode build identical fresh predictor state (the models are deterministic
//! and the hashed-table sizes derive from the framed token count), so their
//! predictions match bit-for-bit and the stream round-trips.
//!
//! This layer carries no container header — just a ULEB128 token-count frame so
//! the decoder knows when to stop and how to size its tables. The self-describing
//! container (magic, version, checksum) lives in [`crate::container`].

use anyhow::{Context as _, Result, anyhow, bail, ensure};
use arbitrary_int::u15;
use static_assertions::const_assert;

use crate::coder::{Decoder, Encoder};
use crate::mixer::Mixer;
use crate::models::null::NullModel;
use crate::models::order0::Order0;
use crate::models::{Context, SYMBOL_BITS, TokenModel};
use crate::preprocessors::{CaseFolding, EntityFolding, NullBytes, NullTokens, Pipeline, Transform};
use crate::tokenizers::{NullTokenizer, RepairTokenizer, Tokenize};
use crate::uleb128;

/// Upper bound on decoded tokens per payload byte. The range coder spends a
/// strictly positive number of bits per symbol, so a genuine stream stays far
/// under this; the cap only bounds the work a corrupt or malicious count field
/// can demand (a decompression-ratio ceiling). A production build would expose a
/// configurable absolute output limit instead.
const MAX_TOKENS_PER_PAYLOAD_BYTE: usize = 4096;

/// The online predictor: every model's stretched prediction for the next bit,
/// mixed into one probability, plus the shared [`Context`] the models read.
struct Predictor {
    models: Vec<Box<dyn TokenModel>>,
    stretched: Vec<i32>,
    mixer: Mixer,
    ctx: Context,
}

impl Predictor {
    /// The model set for `profile`: the always-on order-0 model plus every enabled optional model
    /// (hashed models are sized from `capacity` = the token count, so encode and decode agree).
    fn new(capacity: usize, profile: Profile) -> Self {
        let models = profile.models(capacity);
        let n = models.len();
        Self {
            models,
            stretched: vec![0; n],
            mixer: Mixer::new(n, SYMBOL_BITS as usize),
            ctx: Context::new(),
        }
    }

    /// The mixed 12-bit probability that the next bit is 1.
    #[expect(clippy::cast_sign_loss, reason = "`squash` returns a positive 12-bit probability")]
    fn predict(&mut self) -> u32 {
        for (m, s) in self.models.iter_mut().zip(&mut self.stretched) {
            *s = m.predict(&self.ctx);
        }
        self.mixer.mix(&self.stretched, usize::from(self.ctx.bpos)) as u32
    }

    /// Learn from the actual `bit` (mixer first, then models, all on the pre-bit
    /// context), then advance the context by one bit.
    fn commit(&mut self, bit: u8) {
        self.mixer.update(bit);
        for m in &mut self.models {
            m.update(&self.ctx, bit);
        }
        self.ctx.push_bit(bit);
    }

    /// Close the current symbol once all its bits are coded.
    const fn end_symbol(&mut self) {
        self.ctx.push_symbol();
    }
}

/// Entropy-encode `tokens` into the codec core: a ULEB128 token-count frame
/// followed by the range-coded 15-bit symbols, using the models `profile` selects.
fn encode_tokens(tokens: &[u15], profile: Profile) -> Vec<u8> {
    let mut out = Vec::new();
    uleb128::encode_u64(tokens.len() as u64, &mut out);

    let mut pred = Predictor::new(tokens.len(), profile);
    let mut enc = Encoder::with_capacity(tokens.len() * 2);
    for &token in tokens {
        let value = token.value();
        for k in (0..SYMBOL_BITS).rev() {
            let bit = ((value >> k) & 1) as u8;
            let p = pred.predict();
            enc.encode(bit, p);
            pred.commit(bit);
        }
        pred.end_symbol();
    }
    out.extend_from_slice(&enc.finish());
    out
}

/// Entropy-decode a codec core produced by [`encode_tokens`] back into tokens.
///
/// # Errors
///
/// Returns an error if the token-count frame is malformed, the count exceeds
/// this platform's `usize`, or the count implies more tokens than the payload
/// could plausibly encode (a decompression-bomb guard).
fn decode_tokens(input: &[u8], profile: Profile) -> Result<Vec<u15>> {
    let mut pos = 0;
    let count = uleb128::decode_u64(input, &mut pos)?;
    let count = usize::try_from(count).context("token count exceeds this platform's usize")?;

    let payload = &input[pos..];
    let limit = payload.len().saturating_mul(MAX_TOKENS_PER_PAYLOAD_BYTE).saturating_add(1024);
    if count > limit {
        bail!("token count {count} exceeds the {limit} a {}-byte payload can encode", payload.len());
    }

    let mut pred = Predictor::new(count, profile);
    let mut dec = Decoder::new(payload);
    // Cap the up-front reservation so a large (but in-limit) count cannot demand a
    // huge allocation before a single symbol is decoded; the Vec grows as needed.
    let mut out = Vec::with_capacity(count.min(1 << 16));
    for i in 0..count {
        // A valid stream's decoder never reads past its payload (bar a small flush
        // window). Once it has, the remaining claimed tokens are only zero-padding
        // artifacts of a corrupt count — stop rather than churn through them. This
        // bounds decode work to O(payload) for incompressible/adversarial input.
        if dec.consumed() > payload.len() + 8 {
            bail!("token count {count} is inconsistent with the payload (exhausted at token {i})");
        }
        let mut value = 0u16;
        for _ in 0..SYMBOL_BITS {
            let p = pred.predict();
            let bit = dec.decode(p);
            pred.commit(bit);
            value = (value << 1) | u16::from(bit);
        }
        pred.end_symbol();
        out.push(u15::new(value));
    }
    Ok(out)
}

/// The pluggable compression pipeline: byte preprocessors → tokenizer → token
/// preprocessors → entropy coding. Every stage is chosen by a [`Profile`]; the
/// order-0 entropy model is always present, everything else is a toggleable feature.
pub(crate) struct Compressor {
    byte_pre: Pipeline<u8>,
    tokenizer: Box<dyn Tokenize>,
    token_pre: Pipeline<u15>,
    profile: Profile,
}

impl Compressor {
    /// Builds the pipeline `profile` selects.
    pub(crate) fn from_profile(profile: Profile) -> Self {
        let token_stages: Vec<Box<dyn Transform<u15>>> = vec![Box::new(NullTokens)];
        Self {
            byte_pre: Pipeline::new(profile.byte_preprocessors()),
            tokenizer: profile.tokenizer(),
            token_pre: Pipeline::new(token_stages),
            profile,
        }
    }

    /// Encode `input` into the framing-agnostic codec core.
    pub(crate) fn encode(&self, input: &[u8]) -> Vec<u8> {
        let bytes = self.byte_pre.forward(input);
        let tokens = self.tokenizer.forward(&bytes);
        let tokens = self.token_pre.forward(&tokens);
        encode_tokens(&tokens, self.profile)
    }

    /// Decode a codec core produced by [`Compressor::encode`] back to bytes.
    ///
    /// # Errors
    ///
    /// Propagates entropy-decode, token-preprocessor, and tokenizer failures on
    /// corrupt input.
    pub(crate) fn decode(&self, core: &[u8]) -> Result<Vec<u8>> {
        let tokens = decode_tokens(core, self.profile)?;
        let tokens = self.token_pre.inverse(&tokens)?;
        let bytes = self.tokenizer.inverse(&tokens)?;
        self.byte_pre.inverse(&bytes)
    }
}

/// Name of the tokenizer feature: enabled selects Re-Pair, disabled the identity tokenizer.
const FEATURE_REPAIR: &str = "repair";

/// Constructs the [`NullModel`] for the model registry.
fn build_null_model(_capacity: usize) -> Box<dyn TokenModel> {
    Box::new(NullModel::new())
}

/// Constructs the [`CaseFolding`] byte preprocessor for the feature registry.
fn build_casefold() -> Box<dyn Transform<u8>> {
    Box::new(CaseFolding)
}

/// Constructs the [`EntityFolding`] byte preprocessor for the feature registry.
fn build_entities() -> Box<dyn Transform<u8>> {
    Box::new(EntityFolding)
}

/// What enabling a pipeline feature does.
#[derive(Clone, Copy)]
enum Kind {
    /// Select the Re-Pair tokenizer (disabled → the identity NULL tokenizer).
    Tokenizer,
    /// Add an optional entropy model, mixed alongside the always-on order-0 model.
    Model(fn(usize) -> Box<dyn TokenModel>),
    /// Add a byte preprocessor, applied (in registry order) before tokenization.
    BytePre(fn() -> Box<dyn Transform<u8>>),
}

/// One toggleable pipeline feature. The order-0 entropy model is always present and is *not* a
/// feature. A feature's bit position is its index in [`FEATURES`], so this table is append-only —
/// never reorder or remove an entry, or existing containers become unreadable.
struct FeatureSpec {
    /// CLI/serialization name.
    name: &'static str,
    /// Whether the default profile enables it.
    default_on: bool,
    /// What it does when enabled.
    kind: Kind,
}

/// The pipeline feature registry. Adding a stage is one line here plus its implementation.
const FEATURES: &[FeatureSpec] = &[
    FeatureSpec {
        name: FEATURE_REPAIR,
        default_on: true,
        kind: Kind::Tokenizer,
    },
    FeatureSpec {
        name: "null",
        default_on: false,
        kind: Kind::Model(build_null_model),
    },
    FeatureSpec {
        name: "casefold",
        default_on: true,
        kind: Kind::BytePre(build_casefold),
    },
    // Appended after `casefold` so it applies after it: casefold claims its spare
    // control bytes first, then entity folding picks its five from what remains.
    FeatureSpec {
        name: "entities",
        default_on: true,
        kind: Kind::BytePre(build_entities),
    },
];

const_assert!(FEATURES.len() <= 64);

/// The set of enabled pipeline features.
///
/// Serialized as the container's ULEB128 profile field (a feature bitmask over the pipeline's
/// feature registry). The CLI toggles features by name; each feature carries its own default.
#[derive(Debug, Clone, Copy)]
pub struct Profile {
    /// Enabled-feature bitmask; bit `i` corresponds to `FEATURES[i]`.
    bits: u64,
}

impl Profile {
    /// The index (bit position) of the feature named `name`, or an error listing the known names.
    fn index(name: &str) -> Result<usize> {
        FEATURES
            .iter()
            .position(|feature| feature.name == name)
            .ok_or_else(|| anyhow!("unknown feature {name:?}; known features: {}", feature_names()))
    }

    /// Whether the feature at bit `index` is enabled.
    const fn is_set(self, index: usize) -> bool {
        self.bits & (1 << index) != 0
    }

    /// Whether the feature named `name` is enabled (unknown names read as disabled).
    fn enabled(self, name: &str) -> bool {
        Self::index(name).is_ok_and(|i| self.is_set(i))
    }

    /// Enables `feature`.
    ///
    /// # Errors
    ///
    /// Returns an error if `feature` is not a known feature name.
    pub fn enable(&mut self, feature: &str) -> Result<()> {
        self.bits |= 1 << Self::index(feature)?;
        Ok(())
    }

    /// Disables `feature`.
    ///
    /// # Errors
    ///
    /// Returns an error if `feature` is not a known feature name.
    pub fn disable(&mut self, feature: &str) -> Result<()> {
        self.bits &= !(1 << Self::index(feature)?);
        Ok(())
    }

    /// The profile bitmask, for serialization.
    pub(crate) const fn to_bits(self) -> u64 {
        self.bits
    }

    /// Parses a profile bitmask, rejecting any bit that is not a known feature.
    pub(crate) fn from_bits(bits: u64) -> Result<Self> {
        let known = if FEATURES.len() >= 64 {
            u64::MAX
        } else {
            (1 << FEATURES.len()) - 1
        };
        ensure!(bits & !known == 0, "profile enables unknown feature bits ({bits:#x})");
        Ok(Self {
            bits,
        })
    }

    /// Builds the compressor pipeline this profile selects.
    pub(crate) fn compressor(self) -> Compressor {
        Compressor::from_profile(self)
    }

    /// The byte preprocessors this profile selects: the always-present identity
    /// [`NullBytes`] stage plus every enabled [`Kind::BytePre`] feature, applied in
    /// registry order before tokenization.
    fn byte_preprocessors(self) -> Vec<Box<dyn Transform<u8>>> {
        let mut stages: Vec<Box<dyn Transform<u8>>> = vec![Box::new(NullBytes)];
        for (i, feature) in FEATURES.iter().enumerate() {
            if let Kind::BytePre(build) = feature.kind
                && self.is_set(i)
            {
                stages.push(build());
            }
        }
        stages
    }

    /// The tokenizer this profile selects.
    fn tokenizer(self) -> Box<dyn Tokenize> {
        if self.enabled(FEATURE_REPAIR) {
            Box::new(RepairTokenizer::default())
        } else {
            Box::new(NullTokenizer)
        }
    }

    /// The entropy model set: the always-on order-0 model plus every enabled [`Kind::Model`]
    /// feature (hashed models sized from `capacity`).
    fn models(self, capacity: usize) -> Vec<Box<dyn TokenModel>> {
        let mut models: Vec<Box<dyn TokenModel>> = vec![Box::new(Order0::new())];
        for (i, feature) in FEATURES.iter().enumerate() {
            if let Kind::Model(build) = feature.kind
                && self.is_set(i)
            {
                models.push(build(capacity));
            }
        }
        models
    }
}

impl Default for Profile {
    fn default() -> Self {
        let mut bits = 0;
        for (i, feature) in FEATURES.iter().enumerate() {
            if feature.default_on {
                bits |= 1 << i;
            }
        }
        Self {
            bits,
        }
    }
}

/// The comma-separated list of known feature names, for error messages.
fn feature_names() -> String {
    FEATURES.iter().map(|feature| feature.name).collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    use super::*;

    /// An identity-tokenizer, order-0-only pipeline for exercising the entropy layer directly.
    fn null_pipeline() -> Compressor {
        Compressor::from_profile(Profile::from_bits(0).unwrap())
    }

    proptest! {
        /// The entropy layer must round-trip arbitrary 15-bit tokens — including
        /// values ≥ 256, which exercise the full symbol width and would break
        /// under a wrong 8-bit sentinel mask in `Context::push_symbol`.
        #[test]
        fn tokens_roundtrip_full_range(raw in prop::collection::vec(0u16..=0x7FFF, 0..500)) {
            let tokens: Vec<u15> = raw.iter().copied().map(u15::new).collect();
            let profile = Profile::default();
            let core = encode_tokens(&tokens, profile);
            prop_assert_eq!(decode_tokens(&core, profile).unwrap(), tokens);
        }

        /// The full pluggable pipeline must round-trip arbitrary byte input.
        #[test]
        fn compressor_roundtrip(data in prop::collection::vec(any::<u8>(), 0..2048)) {
            let c = null_pipeline();
            prop_assert_eq!(c.decode(&c.encode(&data)).unwrap(), data);
        }

        /// Decoding arbitrary bytes must never panic — only return Ok or Err.
        #[test]
        fn decode_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..2048)) {
            let c = null_pipeline();
            drop(c.decode(&bytes));
        }
    }

    #[test]
    fn roundtrips_empty_and_single() {
        let c = null_pipeline();
        assert_eq!(c.decode(&c.encode(b"")).unwrap(), b"");
        assert_eq!(c.decode(&c.encode(b"A")).unwrap(), b"A");
    }

    #[test]
    fn compresses_repetitive_input() {
        let data = vec![b'a'; 10_000];
        let c = null_pipeline();
        let core = c.encode(&data);
        assert!(core.len() < data.len() / 10, "weak compression: {} bytes", core.len());
        assert_eq!(c.decode(&core).unwrap(), data);
    }

    #[test]
    fn enabling_the_null_model_still_round_trips() {
        // Toggling an optional entropy model on must keep the pipeline reversible.
        let mut profile = Profile::from_bits(0).unwrap();
        profile.enable("null").unwrap();
        let c = Compressor::from_profile(profile);
        let data = b"the quick brown fox";
        assert_eq!(c.decode(&c.encode(data)).unwrap(), data.to_vec());
    }

    #[test]
    fn rejects_absurd_token_count() {
        // A tiny payload claiming a huge token count must be rejected, not looped.
        let mut core = Vec::new();
        uleb128::encode_u64(u64::MAX, &mut core);
        core.extend_from_slice(&[0u8; 8]);
        assert!(decode_tokens(&core, Profile::default()).is_err());
    }

    #[test]
    fn profile_default_has_repair_on_null_off_casefold_on() {
        let profile = Profile::default();
        assert!(profile.enabled("repair"));
        assert!(!profile.enabled("null"));
        assert!(profile.enabled("casefold"));
        assert!(profile.enabled("entities"));
        // bit 0 (repair) + bit 2 (casefold) + bit 3 (entities), bit 1 (null) off.
        assert_eq!(profile.to_bits(), 0b1101);
    }

    #[test]
    fn profile_toggles_by_name() {
        let mut profile = Profile::default(); // 0b1101: repair + casefold + entities
        profile.disable("casefold").unwrap();
        profile.disable("repair").unwrap();
        profile.disable("entities").unwrap();
        assert_eq!(profile.to_bits(), 0b0000);
        profile.enable("null").unwrap();
        assert_eq!(profile.to_bits(), 0b0010);
        profile.enable("repair").unwrap();
        assert_eq!(profile.to_bits(), 0b0011);
        profile.enable("casefold").unwrap();
        assert_eq!(profile.to_bits(), 0b0111);
        profile.enable("entities").unwrap();
        assert_eq!(profile.to_bits(), 0b1111);
    }

    #[test]
    fn profile_rejects_unknown_feature() {
        let mut profile = Profile::default();
        assert!(profile.enable("nope").is_err());
        assert!(profile.disable("nope").is_err());
    }

    #[test]
    fn profile_bits_round_trip_and_reject_unknown() {
        // bits 0..3 (repair, null, casefold, entities) are all known features.
        for bits in 0..=0b1111 {
            assert_eq!(Profile::from_bits(bits).unwrap().to_bits(), bits);
        }
        assert!(Profile::from_bits(0b10000).is_err()); // bit 4 is not a known feature
        assert!(Profile::from_bits(u64::MAX).is_err());
    }
}
