//! The encode/decode driver: the framing-agnostic codec core.
//!
//! [`Compressor`] chains the pluggable stages — the byte preprocessors, the Re-Pair tokenizer, and
//! the entropy coder — into one [`Pipeline`] of byte→byte [`Transform`]s. Every stage is optional
//! and chosen by a [`Profile`] (a feature bitmask). The pipeline is assembled in a fixed *canonical*
//! order so the entropy coder is always last and the tokenizer always precedes it, regardless of a
//! feature's bit position. Turning tokenization off still leaves a working (entropy-only) coder;
//! turning entropy off still compresses (the tokenizer's grammar shrinks the stream on its own).
//!
//! This layer carries no container header — that lives in [`crate::container`].

use anyhow::{Result, anyhow, ensure};
use static_assertions::const_assert;

use crate::entropy::{EntropyCoder, ModelBuilder};
use crate::models::TokenModel;
use crate::models::null::NullModel;
use crate::models::order0::Order0;
use crate::preprocessors::{CaseFolding, EntityFolding};
use crate::tokenizers::{DEFAULT_NUM_TOKENS, MIN_NUM_TOKENS, RepairTokenizer};
use crate::transform::{Pipeline, Transform};

/// Encode-side options that are *not* serialized into the container — the decoder recovers everything
/// it needs from the stream itself (the Re-Pair vocabulary is self-describing in the token stream).
#[derive(Debug, Clone, Copy)]
pub struct EncodeOptions {
    /// The Re-Pair vocabulary cap for this run (see [`RepairTokenizer::num_tokens`]).
    num_tokens: u32,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            num_tokens: DEFAULT_NUM_TOKENS,
        }
    }
}

impl EncodeOptions {
    /// Options with an explicit Re-Pair vocabulary cap.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_tokens` is outside `256..=0x3F_FFFE` — the cap must admit up to 256
    /// byte terminals and stay one below the reserved NIL sentinel (`u22::MAX`).
    pub fn new(num_tokens: u32) -> Result<Self> {
        ensure!(
            (MIN_NUM_TOKENS..=DEFAULT_NUM_TOKENS).contains(&num_tokens),
            "num-tokens must be in {MIN_NUM_TOKENS}..={DEFAULT_NUM_TOKENS}, got {num_tokens}"
        );
        Ok(Self {
            num_tokens,
        })
    }
}

/// Constructs the always-on order-0 entropy model.
fn build_order0(_capacity: usize) -> Box<dyn TokenModel> {
    Box::new(Order0::new())
}

/// Constructs the optional [`NullModel`] for the model registry.
fn build_null_model(_capacity: usize) -> Box<dyn TokenModel> {
    Box::new(NullModel::new())
}

/// Builds a byte→byte pipeline stage from the encode profile and options. Non-capturing so it
/// coerces to a function pointer stored in the registry.
type StageBuilder = fn(&Profile, &EncodeOptions) -> Box<dyn Transform>;

/// What enabling a pipeline feature does.
#[derive(Clone, Copy)]
enum Kind {
    /// A byte→byte pipeline stage. The registry order of the [`Kind::Stage`] entries *is* the
    /// pipeline order, so the entropy coder's entry must come last.
    Stage(StageBuilder),
    /// An optional entropy model, mixed alongside the always-on order-0 model.
    Model(ModelBuilder),
}

/// One toggleable pipeline feature. A feature's bit position is its index in [`FEATURES`]. This is a
/// pre-1.0 format, so the table may be reassigned freely — the container `version` is bumped whenever
/// it is, so an old stream is rejected rather than misread under new bit meanings.
struct FeatureSpec {
    /// CLI/serialization name.
    name: &'static str,
    /// Whether the default profile enables it.
    default_on: bool,
    /// What it does when enabled.
    kind: Kind,
}

/// The pipeline feature registry. Bit position = index here, and for [`Kind::Stage`] entries this
/// order is also the pipeline order (entropy coder last). Adding or reordering a stage is a
/// single edit here — nothing re-keys stages by name.
const FEATURES: &[FeatureSpec] = &[
    FeatureSpec {
        name: "casefold",
        default_on: true,
        kind: Kind::Stage(|_, _| Box::new(CaseFolding)),
    },
    FeatureSpec {
        name: "entities",
        default_on: true,
        kind: Kind::Stage(|_, _| Box::new(EntityFolding)),
    },
    FeatureSpec {
        name: "repair",
        default_on: true,
        kind: Kind::Stage(|_, options| {
            Box::new(RepairTokenizer {
                num_tokens: options.num_tokens,
            })
        }),
    },
    FeatureSpec {
        name: "entropy",
        default_on: true,
        kind: Kind::Stage(|profile, _| Box::new(EntropyCoder::new(profile.model_builders()))),
    },
    FeatureSpec {
        name: "null",
        default_on: false,
        kind: Kind::Model(build_null_model),
    },
];

const_assert!(FEATURES.len() <= 64);

/// The pluggable compression pipeline: byte preprocessors → tokenizer → entropy coder, each a
/// byte→byte [`Transform`] chosen by a [`Profile`].
pub(crate) struct Compressor {
    pipeline: Pipeline,
}

impl Compressor {
    /// Builds the decode-side pipeline `profile` selects. The tokenizer's `num_tokens` is irrelevant
    /// to `inverse` (the vocabulary is read from the stream), so the default is used.
    pub(crate) fn from_profile(profile: Profile) -> Self {
        Self {
            pipeline: profile.pipeline_with(EncodeOptions::default()),
        }
    }

    /// Builds the encode-side pipeline `profile` selects with the given `options`.
    pub(crate) fn with_options(profile: Profile, options: EncodeOptions) -> Self {
        Self {
            pipeline: profile.pipeline_with(options),
        }
    }

    /// Encode `input` into the framing-agnostic codec core. Takes ownership so each stage can free
    /// its buffer as it consumes it — at gigabyte scale the tokenizer frees ~1 GB before its build.
    pub(crate) fn encode(&self, input: Vec<u8>) -> Vec<u8> {
        self.pipeline.forward(input)
    }

    /// Decode a codec core produced by [`Compressor::encode`] back to bytes.
    ///
    /// # Errors
    ///
    /// Propagates entropy-decode, tokenizer, and preprocessor failures on corrupt input.
    pub(crate) fn decode(&self, core: &[u8]) -> Result<Vec<u8>> {
        self.pipeline.inverse(core.to_vec())
    }
}

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

    /// Whether the feature named `name` is enabled (unknown names read as disabled). Test-only:
    /// the pipeline builds stages straight from the registry rather than looking them up by name.
    #[cfg(test)]
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
    ///
    /// # Errors
    ///
    /// Returns an error if `bits` enables a bit with no assigned feature.
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

    /// Builds the decode-side compressor pipeline this profile selects.
    pub(crate) fn compressor(self) -> Compressor {
        Compressor::from_profile(self)
    }

    /// Builds the encode-side compressor with the given `options`.
    pub(crate) fn compressor_with(self, options: EncodeOptions) -> Compressor {
        Compressor::with_options(self, options)
    }

    /// Assembles the pipeline in CANONICAL order — imposed here, not by feature bit position:
    /// casefold → entities → repair → \[future byte LZ77\] → entropy (always strictly last). This
    /// single explicit assembly is the only place stage order is decided, so it cannot be broken by
    /// editing the registry.
    fn pipeline_with(self, options: EncodeOptions) -> Pipeline {
        let mut stages: Vec<Box<dyn Transform>> = Vec::new();
        for (i, feature) in FEATURES.iter().enumerate() {
            if let Kind::Stage(build) = feature.kind
                && self.is_set(i)
            {
                stages.push(build(&self, &options));
            }
        }
        Pipeline::new(stages)
    }

    /// The entropy model builders: the always-on order-0 model first, then every enabled
    /// [`Kind::Model`] feature (each sized per-input inside the [`EntropyCoder`]).
    fn model_builders(self) -> Vec<ModelBuilder> {
        let mut builders: Vec<ModelBuilder> = vec![build_order0];
        for (i, feature) in FEATURES.iter().enumerate() {
            if let Kind::Model(build) = feature.kind
                && self.is_set(i)
            {
                builders.push(build);
            }
        }
        builders
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

    /// An entropy-only pipeline (no preprocessors, no tokenizer) for exercising the whole
    /// `Compressor` around the entropy stage directly.
    fn entropy_only() -> Compressor {
        let mut profile = Profile::from_bits(0).unwrap();
        profile.enable("entropy").unwrap();
        Compressor::from_profile(profile)
    }

    proptest! {
        /// The full pluggable pipeline must round-trip arbitrary byte input.
        #[test]
        fn compressor_roundtrip(data in prop::collection::vec(any::<u8>(), 0..2048)) {
            let c = entropy_only();
            prop_assert_eq!(c.decode(&c.encode(data.clone())).unwrap(), data);
        }

        /// Decoding arbitrary bytes must never panic — only return Ok or Err.
        #[test]
        fn decode_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..2048)) {
            let c = entropy_only();
            drop(c.decode(&bytes));
        }

        /// Every stage-toggle combination must round-trip a sample that exercises all stages
        /// (uppercase for casefold, entities for entity-folding, repetition for Re-Pair).
        #[test]
        fn each_stage_toggle_round_trips(bits in 0u64..=0b1_1111) {
            let sample =
                b"The QUICK brown fox &lt;tag&gt; JUMPS. The QUICK brown fox &lt;tag&gt; JUMPS.".to_vec();
            let c = Compressor::from_profile(Profile::from_bits(bits).unwrap());
            prop_assert_eq!(c.decode(&c.encode(sample.clone())).unwrap(), sample);
        }
    }

    #[test]
    fn roundtrips_empty_and_single() {
        let c = entropy_only();
        assert_eq!(c.decode(&c.encode(b"".to_vec())).unwrap(), b"");
        assert_eq!(c.decode(&c.encode(b"A".to_vec())).unwrap(), b"A");
    }

    #[test]
    fn entropy_compresses_repetitive_input() {
        let data = vec![b'a'; 10_000];
        let c = entropy_only();
        let core = c.encode(data.clone());
        assert!(core.len() < data.len() / 10, "weak compression: {} bytes", core.len());
        assert_eq!(c.decode(&core).unwrap(), data);
    }

    #[test]
    fn tokenize_only_still_compresses_and_round_trips() {
        // Entropy off, tokenizer on: the Re-Pair grammar alone must shrink the stream and round-trip.
        let mut profile = Profile::from_bits(0).unwrap();
        profile.enable("repair").unwrap();
        assert!(!profile.enabled("entropy"));
        let c = Compressor::from_profile(profile);
        let data = vec![b'a'; 10_000];
        let core = c.encode(data.clone());
        assert!(core.len() < data.len(), "tokenization alone should shrink: {} bytes", core.len());
        assert_eq!(c.decode(&core).unwrap(), data);
    }

    #[test]
    fn enabling_the_null_model_still_round_trips() {
        // Toggling an optional entropy model on must keep the pipeline reversible.
        let mut profile = Profile::from_bits(0).unwrap();
        profile.enable("entropy").unwrap();
        profile.enable("null").unwrap();
        let c = Compressor::from_profile(profile);
        let data = b"the quick brown fox";
        assert_eq!(c.decode(&c.encode(data.to_vec())).unwrap(), data.to_vec());
    }

    #[test]
    fn profile_default_enables_the_full_stack() {
        let profile = Profile::default();
        assert!(profile.enabled("casefold"));
        assert!(profile.enabled("entities"));
        assert!(profile.enabled("repair"));
        assert!(profile.enabled("entropy"));
        assert!(!profile.enabled("null"));
        // bits 0..=3 (casefold, entities, repair, entropy) on, bit 4 (null) off.
        assert_eq!(profile.to_bits(), 0b0_1111);
    }

    #[test]
    fn profile_toggles_by_name() {
        let mut profile = Profile::default(); // 0b0_1111: casefold + entities + repair + entropy
        profile.disable("casefold").unwrap();
        profile.disable("entities").unwrap();
        profile.disable("repair").unwrap();
        profile.disable("entropy").unwrap();
        assert_eq!(profile.to_bits(), 0b0_0000);
        profile.enable("casefold").unwrap();
        assert_eq!(profile.to_bits(), 0b0_0001);
        profile.enable("entities").unwrap();
        assert_eq!(profile.to_bits(), 0b0_0011);
        profile.enable("repair").unwrap();
        assert_eq!(profile.to_bits(), 0b0_0111);
        profile.enable("entropy").unwrap();
        assert_eq!(profile.to_bits(), 0b0_1111);
        profile.enable("null").unwrap();
        assert_eq!(profile.to_bits(), 0b1_1111);
    }

    #[test]
    fn profile_rejects_unknown_feature() {
        let mut profile = Profile::default();
        assert!(profile.enable("nope").is_err());
        assert!(profile.disable("nope").is_err());
        assert!(profile.enable("lz77").is_err()); // removed feature
    }

    #[test]
    fn profile_bits_round_trip_and_reject_unknown() {
        // bits 0..=4 (casefold, entities, repair, entropy, null) are all known features.
        for bits in 0..=0b1_1111 {
            assert_eq!(Profile::from_bits(bits).unwrap().to_bits(), bits);
        }
        assert!(Profile::from_bits(0b10_0000).is_err()); // bit 5 is not a known feature
        assert!(Profile::from_bits(u64::MAX).is_err());
    }

    #[test]
    fn encode_options_validate_num_tokens() {
        assert!(EncodeOptions::new(256).is_ok());
        assert!(EncodeOptions::new(0x3F_FFFE).is_ok());
        assert!(EncodeOptions::new(255).is_err()); // below the terminal floor
        assert!(EncodeOptions::new(0x3F_FFFF).is_err()); // the reserved NIL value
        assert_eq!(EncodeOptions::default().num_tokens, DEFAULT_NUM_TOKENS);
    }
}
