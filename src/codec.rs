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
use crate::models::match_model::MatchModel;
use crate::models::null::NullModel;
use crate::models::ordern::OrderN;
use crate::models::sparse::SparseModel;
use crate::models::varint::VarintModel;
use crate::preprocessors::{
    CaseFolding, DEFAULT_NUM_TOKENS, EntityFolding, Lz77, MAX_MATCH_LEN, MIN_MATCH_LEN, MIN_NUM_TOKENS, RepairTokenizer,
};
use crate::transform::{ModelTrace, Pipeline, StageTrace, Transform};

/// Encode-side options that are *not* serialized into the container — the decoder recovers everything
/// it needs from the stream itself (the Re-Pair vocabulary is self-describing in the token stream).
#[derive(Debug, Clone, Copy)]
pub struct EncodeOptions {
    /// The Re-Pair vocabulary cap for this run (see [`RepairTokenizer::num_tokens`]).
    num_tokens: u32,
    /// Whether Re-Pair prunes its built grammar by the serialized-byte cost model (the default) rather
    /// than to the fixed `num_tokens` count. A caller-pinned `num_tokens` turns it off.
    cost_stop: bool,
    /// The LZ77 minimum match length, in tokens: `None` uses the dynamic "only emit a winning match"
    /// rule, `Some(n)` forces every match of `>= n` tokens (which may expand the stream).
    min_match: Option<u32>,
    /// Whether the Re-Pair stage re-parses its top-level sequence into a minimum-cost cover (MGP). An
    /// encode-only, opt-in choice — the grammar rules are unchanged, so the stream stays
    /// self-describing and decodes without any format or decoder change.
    mgp: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            num_tokens: DEFAULT_NUM_TOKENS,
            cost_stop: true,
            min_match: None,
            mgp: false,
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
        // An explicit vocabulary cap asks for exactly that many tokens, so grammar pruning switches
        // from the cost model to a hard post-build count cap at `num_tokens`.
        Ok(Self {
            num_tokens,
            cost_stop: false,
            ..Self::default()
        })
    }

    /// Sets a fixed LZ77 minimum match length, overriding the default dynamic rule.
    ///
    /// # Errors
    ///
    /// Returns an error if `min_match` is outside `2..=0x3F_FFFF` — a match must cover at least two
    /// tokens (length 0 is the escape, one token never wins) and fit a u22 varint.
    pub fn with_min_match(mut self, min_match: u32) -> Result<Self> {
        ensure!(
            (MIN_MATCH_LEN..=MAX_MATCH_LEN).contains(&min_match),
            "min-match must be in {MIN_MATCH_LEN}..={MAX_MATCH_LEN}, got {min_match}"
        );
        self.min_match = Some(min_match);
        Ok(self)
    }

    /// Enables the Re-Pair minimal-grammar-parse (MGP) sequence re-parse. Encode-only and always
    /// reversible, so it needs no validation.
    #[must_use]
    pub const fn with_mgp(mut self) -> Self {
        self.mgp = true;
        self
    }
}

/// The fallback [`NullModel`] builder — the internal no-op predictor [`Profile::models`] injects when
/// the entropy stage is on but the profile selects no model. Not a registry feature.
const NULL_MODEL: ModelBuilder = |_| Box::new(NullModel::new());

/// Builds a byte→byte pipeline stage from the encode profile and options. Non-capturing so it
/// coerces to a function pointer stored in the registry.
type StageBuilder = fn(&Profile, &EncodeOptions) -> Box<dyn Transform>;

/// What enabling a pipeline feature does.
#[derive(Clone, Copy)]
enum Kind {
    /// A byte→byte pipeline stage. The registry order of the [`Kind::Stage`] entries *is* the
    /// pipeline order, so the entropy coder's entry must come last.
    Stage(StageBuilder),
    /// An entropy model, mixed into the entropy stage. Each is independently toggleable; at least
    /// one must be enabled when the entropy stage is (enforced by [`Profile::validate`]).
    Model(ModelBuilder),
    /// A boolean configuration flag with no builder, read via [`Profile::enabled`] by the stage it
    /// configures (e.g. `sse` enables the entropy coder's APM). Ignored by stage and model assembly.
    Flag,
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
                cost_stop: options.cost_stop,
                mgp: options.mgp,
            })
        }),
    },
    FeatureSpec {
        // Off by default: the token-aware LZ77 stage is net-negative at the small, dense Re-Pair
        // vocabularies that compress best under the context-mixing coder — the entropy stage's own
        // match model already captures the token repeats, and LZ77's match records inject
        // less-predictable bytes that hurt the higher-order models. It still helps at large
        // vocabularies, so it stays available via `--enable lz77` (and the `--auto` search toggles it).
        name: "lz77",
        default_on: false,
        kind: Kind::Stage(|_, options| {
            Box::new(Lz77 {
                min_match: options.min_match,
            })
        }),
    },
    FeatureSpec {
        name: "entropy",
        default_on: true,
        kind: Kind::Stage(|profile, _| {
            Box::new(EntropyCoder::new(profile.model_builders(), profile.model_names(), profile.enabled("sse")))
        }),
    },
    // Entropy models follow the stages. Registry order here is the model order in the mix (immaterial
    // to correctness — the mixer weights per input). The sweep winners are default-on — `order0`,
    // `order1`, `order2`, `sparse2`, `sparse24`, `varint`, and (registered after `sse`) the `match`
    // model. `order2` helps once the Re-Pair vocabulary is capped small enough for a dense token
    // stream (the tuned operating point); `order3`..`order7` stay default-off (they only help on an
    // even smaller/denser alphabet and otherwise hurt). If *every* model is disabled
    // while the entropy stage is on, [`Profile::model_builders`] silently supplies the internal null
    // model (a reversible no-op), so "entropy on, no models" still codes rather than erroring. The null
    // model is deliberately *not* a registry feature — it is an implementation fallback.
    FeatureSpec {
        name: "order0",
        default_on: true,
        kind: Kind::Model(|c| Box::new(OrderN::new(0, c))),
    },
    FeatureSpec {
        name: "order1",
        default_on: true,
        kind: Kind::Model(|c| Box::new(OrderN::new(1, c))),
    },
    FeatureSpec {
        name: "order2",
        default_on: true,
        kind: Kind::Model(|c| Box::new(OrderN::new(2, c))),
    },
    FeatureSpec {
        name: "order3",
        default_on: false,
        kind: Kind::Model(|c| Box::new(OrderN::new(3, c))),
    },
    FeatureSpec {
        name: "order4",
        default_on: false,
        kind: Kind::Model(|c| Box::new(OrderN::new(4, c))),
    },
    FeatureSpec {
        name: "order5",
        default_on: false,
        kind: Kind::Model(|c| Box::new(OrderN::new(5, c))),
    },
    FeatureSpec {
        name: "order6",
        default_on: false,
        kind: Kind::Model(|c| Box::new(OrderN::new(6, c))),
    },
    FeatureSpec {
        name: "order7",
        default_on: false,
        kind: Kind::Model(|c| Box::new(OrderN::new(7, c))),
    },
    FeatureSpec {
        name: "sparse2",
        default_on: true,
        kind: Kind::Model(|c| Box::new(SparseModel::new(&[2], c))),
    },
    FeatureSpec {
        name: "sparse24",
        default_on: true,
        kind: Kind::Model(|c| Box::new(SparseModel::new(&[2, 4], c))),
    },
    FeatureSpec {
        name: "varint",
        default_on: true,
        kind: Kind::Model(|_| Box::new(VarintModel::new())),
    },
    // `sse` is a config flag (not a model): it enables the entropy coder's APM refinement stage.
    // Default-on — it lowers bpb on enwik8/enwik9.
    FeatureSpec {
        name: "sse",
        default_on: true,
        kind: Kind::Flag,
    },
    // LZ77-aware match model: reconstructs the pre-LZ77 stream and predicts recurring content through
    // the LZ77 encoding (repeats that LZ77 folded, which the token models miss because the record's
    // distance differs between occurrences). A small net win on enwik9, so default-on. Appended after
    // `sse` so its bit does not shift the existing feature bits.
    FeatureSpec {
        name: "match",
        default_on: true,
        kind: Kind::Model(|c| Box::new(MatchModel::new(c))),
    },
];

const_assert!(FEATURES.len() <= 64);

/// What a listed [`FeatureInfo`] does, for the CLI's feature listing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureKind {
    /// A byte→byte pipeline stage (casefold, entities, repair, lz77, entropy).
    Stage,
    /// An entropy model mixed into the entropy stage.
    Model,
    /// A boolean configuration flag (e.g. `sse`).
    Flag,
}

/// A public description of one toggleable pipeline feature, for `--list-features`.
#[derive(Debug, Clone, Copy)]
pub struct FeatureInfo {
    /// The `--enable` / `--disable` name.
    pub name: &'static str,
    /// Whether the default profile enables it.
    pub default_on: bool,
    /// Whether it is fundamental and cannot be disabled (only `repair`).
    pub mandatory: bool,
    /// Whether it is a pipeline stage, an entropy model, or a config flag.
    pub kind: FeatureKind,
}

/// The full list of toggleable pipeline features, in registry order (which, for stages, is also the
/// pipeline order). For the CLI's `--list-features` output.
#[must_use]
pub fn features() -> Vec<FeatureInfo> {
    FEATURES
        .iter()
        .map(|feature| FeatureInfo {
            name: feature.name,
            default_on: feature.default_on,
            mandatory: MANDATORY.contains(&feature.name),
            kind: match feature.kind {
                Kind::Stage(_) => FeatureKind::Stage,
                Kind::Model(_) => FeatureKind::Model,
                Kind::Flag => FeatureKind::Flag,
            },
        })
        .collect()
}

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

    /// Like [`Compressor::encode`] but also returns each pipeline stage's `(name, input length, output
    /// length, optional detail)` and the entropy stage's per-model scorecard, for the CLI's per-stage
    /// and per-model reports.
    pub(crate) fn encode_traced(&self, input: Vec<u8>) -> (Vec<u8>, Vec<StageTrace>, Vec<ModelTrace>) {
        let (core, stages) = self.pipeline.forward_traced(input);
        let scores = self.pipeline.model_scores();
        (core, stages, scores)
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

/// Fundamental features that cannot be disabled. Re-Pair is the compressor's tokenization core — the
/// rest of the pipeline (LZ77, the entropy models) is built to consume its token stream — so it is
/// always on; `--disable repair` is rejected rather than silently emitting a byte-level stream. This
/// is enforced encode-side (via [`Profile::disable`]/[`Profile::validate`]/[`Profile::default`]); the
/// low-level [`Profile::from_bits`] stays permissive so internal callers and decode can still name any
/// bit combination (a hand-crafted repair-less container simply fails the container's checksum check).
const MANDATORY: &[&str] = &["repair"];

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
    /// Returns an error if `feature` is not a known feature name, or if it is a fundamental feature
    /// that cannot be disabled (see [`MANDATORY`]).
    pub fn disable(&mut self, feature: &str) -> Result<()> {
        let index = Self::index(feature)?;
        ensure!(
            !MANDATORY.contains(&FEATURES[index].name),
            "feature {feature:?} is fundamental and cannot be disabled"
        );
        self.bits &= !(1 << index);
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
    /// casefold → entities → repair → lz77 → entropy (always strictly last). This single explicit
    /// assembly is the only place stage order is decided, so it cannot be broken by editing the
    /// registry.
    fn pipeline_with(self, options: EncodeOptions) -> Pipeline {
        let mut stages: Vec<Box<dyn Transform>> = Vec::new();
        let mut names: Vec<&'static str> = Vec::new();
        for (i, feature) in FEATURES.iter().enumerate() {
            if let Kind::Stage(build) = feature.kind
                && self.is_set(i)
            {
                stages.push(build(&self, &options));
                names.push(feature.name);
            }
        }
        Pipeline::new(stages, names)
    }

    /// The enabled entropy models as aligned `(name, builder)` pairs, in mix (registry) order — the
    /// single scan that [`Profile::model_builders`] and [`Profile::model_names`] both project from, so
    /// their two lists cannot drift out of alignment. When the profile selects no model, falls back to
    /// the internal null model so the entropy stage always has at least one predictor; the fallback is
    /// a pure function of the profile, so encode and decode inject it identically and stay in step.
    fn models(self) -> Vec<(&'static str, ModelBuilder)> {
        let mut models: Vec<(&'static str, ModelBuilder)> = FEATURES
            .iter()
            .enumerate()
            .filter_map(|(i, feature)| match feature.kind {
                Kind::Model(build) if self.is_set(i) => Some((feature.name, build)),
                _ => None,
            })
            .collect();
        if models.is_empty() {
            models.push(("null", NULL_MODEL));
        }
        models
    }

    /// The entropy model builders (each sized per-input inside the [`EntropyCoder`]), aligned with
    /// [`Profile::model_names`]. See [`Profile::models`].
    fn model_builders(self) -> Vec<ModelBuilder> {
        self.models().into_iter().map(|(_, build)| build).collect()
    }

    /// The names of the models the entropy stage will mix, aligned with [`Profile::model_builders`].
    /// Assumes the entropy stage is on (it is wherever this is used). See [`Profile::models`].
    fn model_names(self) -> Vec<&'static str> {
        self.models().into_iter().map(|(name, _)| name).collect()
    }

    /// The names of the entropy models active for this profile, in mix (registry) order — for a caller
    /// (e.g. the CLI's run report) to name what runs: the enabled models, or `["null"]` (the injected
    /// fallback) when the entropy stage is on but no model is selected. Empty when the entropy stage is
    /// off, since no models run then.
    #[must_use]
    pub fn active_models(self) -> Vec<&'static str> {
        if self.enabled("entropy") {
            self.model_names()
        } else {
            Vec::new()
        }
    }

    /// The names of the config flags that actually take effect for this profile (e.g. `sse`), in
    /// registry order — for the CLI's run report. Flags are pipeline configuration, not predictors.
    /// `sse` (the only flag) configures the entropy stage, so it is inert — and not reported — when the
    /// entropy stage is off. Revisit if a flag configuring a different stage is ever added.
    #[must_use]
    pub fn active_flags(self) -> Vec<&'static str> {
        if !self.enabled("entropy") {
            return Vec::new();
        }
        FEATURES
            .iter()
            .enumerate()
            .filter(|(i, feature)| matches!(feature.kind, Kind::Flag) && self.is_set(*i))
            .map(|(_, feature)| feature.name)
            .collect()
    }

    /// Validates that the profile describes a usable pipeline.
    ///
    /// The one constraint not expressible as a single bit: every fundamental feature (see [`MANDATORY`])
    /// must be enabled. The entropy stage needs no explicit model — when the profile selects none,
    /// [`Profile::model_builders`] supplies the internal null model — so entropy-with-no-models is valid.
    ///
    /// # Errors
    ///
    /// Returns an error if a fundamental feature is disabled.
    pub fn validate(self) -> Result<()> {
        for name in MANDATORY {
            ensure!(self.enabled(name), "the {name} stage is fundamental and must stay enabled");
        }
        Ok(())
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

    /// An entropy-only pipeline (no preprocessors, no tokenizer) with the order-0 model, for
    /// exercising the whole `Compressor` around the entropy stage directly.
    fn entropy_only() -> Compressor {
        let mut profile = Profile::from_bits(0).unwrap();
        profile.enable("entropy").unwrap();
        profile.enable("order0").unwrap();
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

        /// Every stage- and model-toggle combination must round-trip a sample that exercises all
        /// stages (uppercase for casefold, entities for entity-folding, repetition for Re-Pair and
        /// LZ77). The 17-bit range spans the four preprocessors, the entropy stage, all eleven
        /// entropy models (order0..7, sparse2, sparse24, varint), and the `sse` flag — including
        /// entropy-with-no-model, which codes reversibly via the injected null fallback.
        #[test]
        fn each_stage_toggle_round_trips(bits in 0u64..=0x1_FFFF) {
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
    fn entropy_with_no_model_round_trips_via_null_fallback() {
        // Entropy on with no model selected must validate and still round-trip: `model_builders`
        // silently injects the internal null model (a reversible no-op predictor).
        let mut profile = Profile::from_bits(0).unwrap();
        profile.enable("repair").unwrap();
        profile.enable("entropy").unwrap();
        assert!(profile.validate().is_ok(), "entropy needs no explicit model");
        let c = Compressor::from_profile(profile);
        let data = b"the quick brown fox";
        assert_eq!(c.decode(&c.encode(data.to_vec())).unwrap(), data.to_vec());
    }

    #[test]
    fn enabling_order1_still_round_trips() {
        // The order-1 model (default on) must keep the pipeline reversible.
        let mut profile = Profile::from_bits(0).unwrap();
        profile.enable("entropy").unwrap();
        profile.enable("order0").unwrap();
        profile.enable("order1").unwrap();
        let c = Compressor::from_profile(profile);
        let data = b"the quick brown fox the quick brown fox";
        assert_eq!(c.decode(&c.encode(data.to_vec())).unwrap(), data.to_vec());
    }

    #[test]
    fn profile_default_enables_stages_and_winning_models() {
        let profile = Profile::default();
        assert!(profile.enabled("casefold"));
        assert!(profile.enabled("entities"));
        assert!(profile.enabled("repair"));
        // LZ77 (bit 3) is default-off: it is net-negative at the small dense vocabularies that compress
        // best; it stays available via `--enable lz77` and the `--auto` search.
        assert!(!profile.enabled("lz77"));
        assert!(profile.enabled("entropy"));
        // The sweep winners are default-on; order3..order7 are default-off.
        assert!(profile.enabled("order0"));
        assert!(profile.enabled("order1"));
        assert!(profile.enabled("order2"));
        assert!(profile.enabled("sparse2"));
        assert!(profile.enabled("sparse24"));
        assert!(profile.enabled("varint"));
        assert!(profile.enabled("sse"));
        assert!(profile.enabled("match"));
        assert!(!profile.enabled("order3"));
        // bits 0,1,2 (casefold/entities/repair) and 4 (entropy) — lz77 (bit 3) is default-off — plus
        // order0..2 (5,6,7) and sparse2(13), sparse24(14), varint(15), sse(16), match(17).
        assert_eq!(profile.to_bits(), 0xF7 | (1 << 13) | (1 << 14) | (1 << 15) | (1 << 16) | (1 << 17));
    }

    #[test]
    fn profile_toggles_by_name() {
        let mut profile = Profile::default(); // stages + order0/order1/order2/sparse2/sparse24/varint
        // `repair` is fundamental (bit 2) and cannot be disabled, so it stays set throughout. Disabling
        // every other default-on feature leaves only the repair bit.
        for feature in [
            "casefold", "entities", "lz77", "entropy", "order0", "order1", "order2", "sparse2", "sparse24", "varint",
            "sse", "match",
        ] {
            profile.disable(feature).unwrap();
        }
        assert_eq!(profile.to_bits(), 0b0000_0100); // only the mandatory repair bit remains
        assert!(profile.disable("repair").is_err());
        profile.enable("casefold").unwrap();
        assert_eq!(profile.to_bits(), 0b0000_0101);
        profile.enable("entities").unwrap();
        assert_eq!(profile.to_bits(), 0b0000_0111);
        profile.enable("lz77").unwrap();
        assert_eq!(profile.to_bits(), 0b0000_1111);
        profile.enable("entropy").unwrap();
        assert_eq!(profile.to_bits(), 0b0001_1111);
        profile.enable("order0").unwrap();
        assert_eq!(profile.to_bits(), 0b0011_1111);
        profile.enable("order1").unwrap();
        assert_eq!(profile.to_bits(), 0b0111_1111);
        profile.enable("order2").unwrap();
        assert_eq!(profile.to_bits(), 0b1111_1111);
        // `varint` is bit 15; the seven models between order2 and it (order3..7, sparse2, sparse24)
        // stay clear here, so enabling it sets only bit 15.
        profile.enable("varint").unwrap();
        assert_eq!(profile.to_bits(), 0xFF | (1 << 15));
    }

    #[test]
    fn repair_is_fundamental_and_cannot_be_disabled() {
        let mut profile = Profile::default();
        assert!(profile.disable("repair").is_err(), "repair must not be disableable");
        assert!(profile.enabled("repair"));
        // A profile with repair cleared (built via the low-level, permissive from_bits) fails validate.
        let without_repair = Profile::from_bits(Profile::default().to_bits() & !(1 << 2)).unwrap();
        assert!(without_repair.validate().is_err());
    }

    #[test]
    fn entropy_needs_no_explicit_model() {
        // Entropy on with no model selected is valid — `model_builders` injects the null model — so the
        // only validate constraint left is the mandatory `repair` stage.
        let mut profile = Profile::from_bits(0).unwrap();
        profile.enable("repair").unwrap();
        profile.enable("entropy").unwrap();
        assert!(profile.validate().is_ok());
        // A model may still be enabled explicitly.
        profile.enable("order1").unwrap();
        assert!(profile.validate().is_ok());
        // The default profile validates too (its only hard requirement is the repair stage).
        assert!(Profile::default().validate().is_ok());
        // The one remaining constraint: `repair` must be enabled.
        let no_repair = Profile::from_bits(0).unwrap();
        assert!(no_repair.validate().is_err());
    }

    #[test]
    fn profile_rejects_unknown_feature() {
        let mut profile = Profile::default();
        assert!(profile.enable("nope").is_err());
        assert!(profile.disable("nope").is_err());
    }

    #[test]
    fn active_models_reports_enabled_models_and_null_fallback() {
        // Default profile: the winning model set, listed in registry (mix) order (`match` last).
        assert_eq!(
            Profile::default().active_models(),
            vec!["order0", "order1", "order2", "sparse2", "sparse24", "varint", "match"]
        );
        // Enabling a further model keeps registry order regardless of enable order.
        let mut p = Profile::default();
        p.enable("order3").unwrap();
        assert_eq!(
            p.active_models(),
            vec!["order0", "order1", "order2", "order3", "sparse2", "sparse24", "varint", "match"]
        );
        // Entropy on with every model disabled -> the injected null fallback is reported.
        let mut none = Profile::default();
        for m in ["order0", "order1", "order2", "sparse2", "sparse24", "varint", "match"] {
            none.disable(m).unwrap();
        }
        assert_eq!(none.active_models(), vec!["null"]);
        // Entropy disabled -> no models run, so the list is empty (nothing to report).
        let mut off = Profile::default();
        off.disable("entropy").unwrap();
        assert!(off.active_models().is_empty());
    }

    #[test]
    fn active_flags_lists_enabled_flags() {
        // `sse` is the only flag and is default-on, so the default profile reports it.
        assert_eq!(Profile::default().active_flags(), vec!["sse"]);
        let mut p = Profile::default();
        p.disable("sse").unwrap();
        assert!(p.active_flags().is_empty());
        // `sse` configures the entropy stage, so it is inert (and unreported) when entropy is off.
        let mut off = Profile::default();
        off.disable("entropy").unwrap();
        assert!(off.active_flags().is_empty());
    }

    #[test]
    fn profile_bits_round_trip_and_reject_unknown() {
        // bits 0..=17 (preprocessors, entropy, order0..7, sparse2, sparse24, varint, sse, match) are known.
        for bits in 0..=0x3_FFFF {
            assert_eq!(Profile::from_bits(bits).unwrap().to_bits(), bits);
        }
        assert!(Profile::from_bits(1 << 18).is_err()); // bit 18 is not a known feature
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

    #[test]
    fn encode_options_validate_min_match() {
        assert_eq!(EncodeOptions::default().min_match, None);
        assert_eq!(EncodeOptions::default().with_min_match(2).unwrap().min_match, Some(2));
        assert_eq!(EncodeOptions::default().with_min_match(0x3F_FFFF).unwrap().min_match, Some(0x3F_FFFF));
        assert!(EncodeOptions::default().with_min_match(1).is_err()); // below the two-byte floor
        assert!(EncodeOptions::default().with_min_match(0x40_0000).is_err()); // beyond u22
    }

    #[test]
    fn encode_options_toggle_mgp() {
        assert!(!EncodeOptions::default().mgp);
        assert!(EncodeOptions::default().with_mgp().mgp);
    }
}
