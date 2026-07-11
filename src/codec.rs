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
use crate::models::iddelta::IdDeltaModel;
use crate::models::match_model::MatchModel;
use crate::models::nnlm::NnlmModel;
use crate::models::null::NullModel;
use crate::models::ordern::OrderN;
use crate::models::run::RunModel;
use crate::models::sparse::SparseModel;
use crate::models::xmltag::XmlTagModel;
use crate::preprocessors::{CaseFolding, EntityFolding, RepairTokenizer};
use crate::transform::{ModelTrace, Pipeline, StageTrace, Transform};

/// The fallback [`NullModel`] builder — the internal no-op predictor [`Profile::models`] injects when
/// the entropy stage is on but the profile selects no model. Not a registry feature.
const NULL_MODEL: ModelBuilder = |_| Box::new(NullModel::new());

/// Builds a byte→byte pipeline stage from the encode profile. Non-capturing so it coerces to a
/// function pointer stored in the registry.
type StageBuilder = fn(&Profile) -> Box<dyn Transform>;

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
/// pre-1.0 format, so the table may be reassigned freely — an old stream is simply rejected rather
/// than misread under new bit meanings.
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
        kind: Kind::Stage(|_| Box::new(CaseFolding)),
    },
    FeatureSpec {
        name: "entities",
        default_on: true,
        kind: Kind::Stage(|_| Box::new(EntityFolding)),
    },
    // Byte-BPE Re-Pair tokenizer. **Default-off**: on enwik8 it now *regresses* whole-stream bpb by
    // ~0.022 (1.6339 -> 1.6561) — the strengthened CM (order0..11 + match + per-context mixer + SSE)
    // subsumes its ~71 rules, and the injected token boundaries disrupt the models more than the
    // grammar shrinks the stream. Kept as an opt-in (`--enable repair`) for corpora where it still pays.
    FeatureSpec {
        name: "repair",
        default_on: false,
        kind: Kind::Stage(|_| Box::new(RepairTokenizer)),
    },
    FeatureSpec {
        name: "entropy",
        default_on: true,
        kind: Kind::Stage(|profile| {
            Box::new(EntropyCoder::new(
                profile.model_builders(),
                profile.model_names(),
                profile.enabled("sse"),
                profile.enabled("sse2"),
                profile.enabled("mix2"),
            ))
        }),
    },
    // Entropy models follow the stages. Registry order here is the model order in the mix (immaterial
    // to correctness — the mixer weights per input). The default set was chosen by a cumulative sweep
    // on enwik8 (add a model, keep it only if it lowered whole-stream bpb): the order-0..11 chain,
    // `sparse2`, the byte-level `match` model, the `word` model, and the `xmltag` model each earn their
    // place and are **default-on** (together with the `sse`/`sse2` APM chain: 1.6339 -> 1.6092 bpb). A
    // wider `sparse24` (positions 2 and 4) was also tried but regressed the mix — the mixer drove its
    // weight negative — so it was dropped; re-confirmed after the order-0..10 / `u8`-`StateMap` rework
    // (still `w<0`, whole-stream bpb up), so it stays out. If
    // *every* model is disabled while the entropy stage is on, [`Profile::model_builders`] silently
    // supplies the internal null model (a reversible no-op), so "entropy on, no models" still codes
    // rather than erroring. The null model is deliberately *not* a registry feature — it is an
    // implementation fallback.
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
        default_on: true,
        kind: Kind::Model(|c| Box::new(OrderN::new(3, c))),
    },
    FeatureSpec {
        name: "order4",
        default_on: true,
        kind: Kind::Model(|c| Box::new(OrderN::new(4, c))),
    },
    FeatureSpec {
        name: "order5",
        default_on: true,
        kind: Kind::Model(|c| Box::new(OrderN::new(5, c))),
    },
    FeatureSpec {
        name: "order6",
        default_on: true,
        kind: Kind::Model(|c| Box::new(OrderN::new(6, c))),
    },
    FeatureSpec {
        name: "order7",
        default_on: true,
        kind: Kind::Model(|c| Box::new(OrderN::new(7, c))),
    },
    FeatureSpec {
        name: "order8",
        default_on: true,
        kind: Kind::Model(|c| Box::new(OrderN::new(8, c))),
    },
    FeatureSpec {
        name: "order9",
        default_on: true,
        kind: Kind::Model(|c| Box::new(OrderN::new(9, c))),
    },
    FeatureSpec {
        name: "order10",
        default_on: true,
        kind: Kind::Model(|c| Box::new(OrderN::new(10, c))),
    },
    FeatureSpec {
        name: "order11",
        default_on: true,
        kind: Kind::Model(|c| Box::new(OrderN::new(11, c))),
    },
    FeatureSpec {
        name: "sparse2",
        default_on: true,
        kind: Kind::Model(|c| Box::new(SparseModel::new(&[2], c))),
    },
    // `sse` is a config flag (not a model): it enables the entropy coder's APM refinement stage.
    // Default-on — it is coder configuration, not a predictor.
    FeatureSpec {
        name: "sse",
        default_on: true,
        kind: Kind::Flag,
    },
    // Byte-level match model: predicts long recurrences the order-N context models miss. Default-on —
    // it earned the largest single-model win in the sweep after the order-N chain.
    FeatureSpec {
        name: "match",
        default_on: true,
        kind: Kind::Model(|c| Box::new(MatchModel::new(c))),
    },
    // Word model: keys on the current partial word (trailing ASCII letters), via the generic
    // char-class `RunModel`. Appended after `match` to keep the existing features' bit positions
    // stable. **Default-on**: the largest single win of this sweep on enwik8 (1.6339 -> 1.6186 bpb,
    // -0.0153) — its letter-only context generalises a word prefix across the punctuation it follows
    // and reaches past the order-11 chain on long words. (A digit-class `RunModel` was also swept but
    // measured neutral — the structure of ids/numbers is cross-token, not in the trailing digits — so
    // it was dropped.)
    FeatureSpec {
        name: "word",
        default_on: true,
        kind: Kind::Model(|c| Box::new(RunModel::new(u8::is_ascii_alphabetic, c))),
    },
    // XML-tag model: keys on the nearest enclosing `MediaWiki` element name. **Default-on**: -0.0010 bpb
    // on enwik8 atop word+sse2 — it gives `<id>`/`<timestamp>`/`<title>` bodies a per-field context the
    // order-N models lose once the content runs past the tag.
    FeatureSpec {
        name: "xmltag",
        default_on: true,
        kind: Kind::Model(|c| Box::new(XmlTagModel::new(c))),
    },
    // Second SSE stage: chains a second APM (keyed on previous-byte × bit-tree node, an order-1
    // context) after the `sse` APM. Only meaningful with `sse` on. **Default-on**: -0.0084 bpb on
    // enwik8 atop word — it corrects order-1 miscalibration the single order-0 `sse` APM leaves.
    FeatureSpec {
        name: "sse2",
        default_on: true,
        kind: Kind::Flag,
    },
    // ID-delta model: predicts the current number's digits from the previous completed number.
    // **Default-on**: -0.0006 bpb on enwik8 atop the rest — small but ~6x the neutral trailing-digit
    // model, confirming the predictable structure of ids/timestamps is *cross-token* (the shared
    // high-order prefix of near-monotonic sequences), not in the digits of a single number.
    FeatureSpec {
        name: "iddelta",
        default_on: true,
        kind: Kind::Model(|_| Box::new(IdDeltaModel::new())),
    },
    // Two-layer mixing network: replaces the single-layer logistic mixer with a small MLP trained
    // online by backprop — four sub-mixers over an order-1..4 context progression, blended by a learned
    // second layer (see `TwoLayerMixer`). **Default-on**: the largest single win of the recent sweep —
    // enwik8 1.6086 -> 1.5586 (−0.0500) and enwik9 1.3252 -> 1.2772 (−0.0480), both byte-exact
    // roundtrips, for +7% time and pure-integer determinism (no float). Disable with `--disable mix2`.
    FeatureSpec {
        name: "mix2",
        default_on: true,
        kind: Kind::Flag,
    },
    // Online neural language model: a feed-forward net over learned byte embeddings, trained online by
    // back-propagation. Generalises across contexts the exact order-N models cannot. **Default-off**
    // while under evaluation (`f32`, so it also relaxes cross-toolchain reproducibility); `--enable nnlm`.
    FeatureSpec {
        name: "nnlm",
        default_on: false,
        kind: Kind::Model(|_| Box::new(NnlmModel::new(false))),
    },
    // Recurrent variant of the neural model (Elman RNN): the hidden state carries memory beyond the
    // K-byte window — the long-range signal the fixed-order context models cannot hold. **Default-off**
    // while under evaluation; `--enable nnrnn`.
    FeatureSpec {
        name: "nnrnn",
        default_on: false,
        kind: Kind::Model(|_| Box::new(NnlmModel::new(true))),
    },
];

const_assert!(FEATURES.len() <= 64);

/// What a listed [`FeatureInfo`] does, for the CLI's feature listing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureKind {
    /// A byte→byte pipeline stage (casefold, entities, repair, entropy).
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
    /// Whether it is fundamental and cannot be disabled (currently none — every stage is optional).
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
    /// Builds the pipeline `profile` selects. The same assembly serves encode and decode — the
    /// tokenizer is parameterless and every stage recovers what it needs from the stream.
    pub(crate) fn from_profile(profile: Profile) -> Self {
        Self {
            pipeline: profile.pipeline(),
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

/// Fundamental features that cannot be disabled. In the byte-centric reset every stage is optional —
/// Re-Pair included, since the entropy models are byte-level and code the raw (or preprocessed) byte
/// stream directly rather than a token stream — so this is currently empty. The mechanism is kept:
/// listing a feature here makes [`Profile::disable`] reject it and [`Profile::validate`] require it,
/// should a future stage become structurally required. [`Profile::from_bits`] stays permissive
/// regardless, so decode can name any bit combination.
const MANDATORY: &[&str] = &[];

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

    /// Builds the compressor pipeline this profile selects (used for both encode and decode).
    pub(crate) fn compressor(self) -> Compressor {
        Compressor::from_profile(self)
    }

    /// Assembles the pipeline in registry order, which *is* the canonical stage order:
    /// casefold → entities → repair → entropy (always strictly last). Reordering stages is a
    /// matter of reordering their [`Kind::Stage`] entries in [`FEATURES`].
    fn pipeline(self) -> Pipeline {
        let mut stages: Vec<Box<dyn Transform>> = Vec::new();
        let mut names: Vec<&'static str> = Vec::new();
        for (i, feature) in FEATURES.iter().enumerate() {
            if let Kind::Stage(build) = feature.kind
                && self.is_set(i)
            {
                stages.push(build(&self));
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
    /// Every fundamental feature (see [`MANDATORY`]) must be enabled. That list is currently empty —
    /// every stage is optional — so this only fails if a future feature is marked mandatory and left
    /// off. The entropy stage needs no explicit model: when the profile selects none,
    /// [`Profile::model_builders`] supplies the internal null model, so entropy-with-no-models is valid.
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
        /// stages (uppercase for casefold, entities for entity-folding, repetition for Re-Pair). The
        /// 24-bit range spans the three text/tokenizer stages (casefold, entities, repair), the entropy
        /// stage, all sixteen entropy models (order0..11, sparse2, match, word, xmltag, iddelta), and
        /// the `sse`/`sse2`/`mix2` flags — including entropy-with-no-model, which codes reversibly via
        /// the injected null fallback.
        #[test]
        fn each_stage_toggle_round_trips(bits in 0u64..=0xFF_FFFF) {
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
    fn mix2_round_trips() {
        // The two-layer mixing network must keep the pipeline reversible with the full default model set.
        let mut profile = Profile::default();
        profile.enable("mix2").unwrap();
        let c = Compressor::from_profile(profile);
        let data = b"The QUICK brown fox &lt;tag&gt; JUMPS over the lazy dog 12345. ".repeat(40);
        assert_eq!(c.decode(&c.encode(data.clone())).unwrap(), data);
    }

    #[test]
    fn neural_models_round_trip() {
        // The f32 neural models must keep the pipeline reversible on a given build (same-binary
        // encode/decode determinism), both feed-forward and recurrent, alone and with mix2.
        let data = b"The QUICK brown fox &lt;tag&gt; JUMPS over the lazy dog 12345. ".repeat(40);
        for feats in [&["nnlm"][..], &["nnrnn"][..], &["nnrnn", "mix2"][..]] {
            let mut profile = Profile::default();
            for f in feats {
                profile.enable(f).unwrap();
            }
            let c = Compressor::from_profile(profile);
            assert_eq!(c.decode(&c.encode(data.clone())).unwrap(), data, "features {feats:?} did not round-trip");
        }
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
    fn profile_default_bits() {
        // Default-on: casefold(0)/entities(1)/entropy(3), the order-0..11 chain (bits 4..=15),
        // sparse2(16), sse(17), match(18), word(19), xmltag(20), sse2(21), iddelta(22), mix2(23).
        // Default-off: repair(2) (regresses the strengthened CM) and the experimental neural models
        // nnlm(24)/nnrnn(25). So the default is bits 0..=23 except bit 2 = 0x00FF_FFFF & !(1 << 2) =
        // 0x00FF_FFFB.
        assert_eq!(Profile::default().to_bits(), 0x00FF_FFFB);
    }

    #[test]
    fn profile_toggles_by_name() {
        // Build from an empty profile: enable the four stages, then two models, checking bits.
        let mut profile = Profile::from_bits(0).unwrap();
        profile.enable("casefold").unwrap();
        assert_eq!(profile.to_bits(), 1 << 0);
        profile.enable("entities").unwrap();
        assert_eq!(profile.to_bits(), 0b0000_0011);
        profile.enable("repair").unwrap();
        assert_eq!(profile.to_bits(), 0b0000_0111);
        // Every stage is optional — repair can be toggled off and back on like any other.
        profile.disable("repair").unwrap();
        assert_eq!(profile.to_bits(), 0b0000_0011);
        profile.enable("repair").unwrap();
        assert_eq!(profile.to_bits(), 0b0000_0111);
        profile.enable("entropy").unwrap();
        assert_eq!(profile.to_bits(), 0b0000_1111);
        profile.enable("order0").unwrap();
        assert_eq!(profile.to_bits(), 0b0001_1111);
        // `match` is bit 18 (the last feature); enabling it sets only that bit.
        profile.enable("match").unwrap();
        assert_eq!(profile.to_bits(), 0b0001_1111 | (1 << 18));
    }

    #[test]
    fn repair_is_optional_and_default_off() {
        // Repair is now default-off (it regresses the strengthened CM). The default profile omits it,
        // enabling it is valid, and disabling it again is valid — no feature is mandatory.
        let mut profile = Profile::default();
        assert!(!profile.enabled("repair"), "repair is default-off");
        profile.enable("repair").unwrap();
        assert!(profile.enabled("repair"));
        assert!(profile.validate().is_ok(), "a repair-enabled profile is valid");
        profile.disable("repair").unwrap();
        assert!(!profile.enabled("repair"));
        assert!(profile.validate().is_ok(), "a repair-less profile is valid");
    }

    #[test]
    fn entropy_needs_no_explicit_model() {
        // Entropy on with no model selected is valid — `model_builders` injects the null model.
        let mut profile = Profile::from_bits(0).unwrap();
        profile.enable("entropy").unwrap();
        assert!(profile.validate().is_ok());
        // A model may still be enabled explicitly.
        profile.enable("order1").unwrap();
        assert!(profile.validate().is_ok());
        // The default profile validates too.
        assert!(Profile::default().validate().is_ok());
        // With no feature mandatory, even the empty profile validates.
        assert!(Profile::from_bits(0).unwrap().validate().is_ok());
    }

    #[test]
    fn profile_rejects_unknown_feature() {
        let mut profile = Profile::default();
        assert!(profile.enable("nope").is_err());
        assert!(profile.disable("nope").is_err());
    }

    #[test]
    fn active_models_reports_enabled_models_and_null_fallback() {
        // Default profile: the swept-in set (order-0..11, sparse2, match, word, xmltag, iddelta), in
        // registry order.
        assert_eq!(
            Profile::default().active_models(),
            vec![
                "order0", "order1", "order2", "order3", "order4", "order5", "order6", "order7", "order8", "order9",
                "order10", "order11", "sparse2", "match", "word", "xmltag", "iddelta"
            ]
        );
        // With no model selected, entropy codes via the injected null fallback.
        let mut none = Profile::from_bits(0).unwrap();
        none.enable("entropy").unwrap();
        assert_eq!(none.active_models(), vec!["null"]);
        // Enabling models lists them in registry (mix) order regardless of enable order.
        let mut p = Profile::from_bits(0).unwrap();
        p.enable("entropy").unwrap();
        p.enable("order2").unwrap();
        p.enable("order0").unwrap();
        p.enable("match").unwrap();
        assert_eq!(p.active_models(), vec!["order0", "order2", "match"]);
        // Entropy disabled -> no models run, so the list is empty (nothing to report).
        let mut off = Profile::default();
        off.disable("entropy").unwrap();
        assert!(off.active_models().is_empty());
    }

    #[test]
    fn active_flags_lists_enabled_flags() {
        // `sse`, `sse2`, and `mix2` are the flags, all default-on, reported in registry order.
        assert_eq!(Profile::default().active_flags(), vec!["sse", "sse2", "mix2"]);
        let mut p = Profile::default();
        p.disable("sse").unwrap();
        p.disable("sse2").unwrap();
        p.disable("mix2").unwrap();
        assert!(p.active_flags().is_empty());
        // the flags configure the entropy stage, so they are inert (and unreported) when entropy is off.
        let mut off = Profile::default();
        off.disable("entropy").unwrap();
        assert!(off.active_flags().is_empty());
    }

    #[test]
    fn profile_bits_round_trip_and_reject_unknown() {
        // bits 0..=25 known (casefold/entities/repair/entropy, order0..11, sparse2, sse, match, word,
        // xmltag, sse2, iddelta, mix2, nnlm, nnrnn). Sample rather than enumerate all 2^26 combos.
        for bits in (0..=0x3FF_FFFF).step_by(7) {
            assert_eq!(Profile::from_bits(bits).unwrap().to_bits(), bits);
        }
        assert!(Profile::from_bits(1 << 26).is_err()); // bit 26 is not a known feature
        assert!(Profile::from_bits(u64::MAX).is_err());
    }

    #[test]
    fn default_profile_enables_stages_and_swept_models() {
        // The default ships with casefold/entities/entropy plus the full swept-in model set.
        let profile = Profile::default();
        for stage in ["casefold", "entities", "entropy"] {
            assert!(profile.enabled(stage), "{stage} should be default-on");
        }
        assert!(!profile.enabled("repair"), "repair is default-off (regresses the strengthened CM)");
        for model in [
            "order0", "order1", "order2", "order3", "order4", "order5", "order6", "order7", "order8", "order9",
            "order10", "order11", "sparse2", "match", "word", "xmltag", "iddelta",
        ] {
            assert!(profile.enabled(model), "{model} should be default-on");
        }
        assert!(profile.enabled("sse"), "sse flag should be default-on");
        assert!(profile.enabled("sse2"), "sse2 flag should be default-on");
        assert!(profile.enabled("mix2"), "mix2 flag should be default-on");
        // The experimental neural models stay default-off.
        assert!(!profile.enabled("nnlm"), "nnlm is default-off (experimental)");
        assert!(!profile.enabled("nnrnn"), "nnrnn is default-off (experimental)");
    }
}
