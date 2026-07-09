//! LZR Compression library.

#![allow(unused_features, reason = "Required for coverage_attribute feature.")]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

// The `bench-internals` feature widens these modules (and, via `visibility::make`
// on their items, the types/fns within) to `pub` so `benches/` can reach them.
// `visibility::make` can't be applied to a file-module declaration, so the module
// visibility is switched with `cfg` instead.
#[cfg(feature = "bench-internals")]
pub mod adler32;

#[cfg(not(feature = "bench-internals"))]
#[cfg_attr(not(test), expect(dead_code, reason = "Development."))]
pub(crate) mod adler32;

#[cfg(feature = "bench-internals")]
pub mod uleb128;

#[cfg(not(feature = "bench-internals"))]
pub(crate) mod uleb128;

mod apm;
mod autotune;
mod codec;
mod coder;
mod container;
mod entropy;
mod mixer;
mod models;

// `preprocessors` holds the Re-Pair tokenizer; the `bench-internals` feature widens the module (and
// `RepairTokenizer` within it, via `visibility::make`) to `pub` so `benches/` can reach it.
#[cfg(feature = "bench-internals")]
pub mod preprocessors;

#[cfg(not(feature = "bench-internals"))]
mod preprocessors;

// The `Transform` trait is widened to `pub` under `bench-internals` (via `visibility::make`) so
// `benches/` can drive the tokenizer through it, mirroring `preprocessors`/`uleb128`.
#[cfg(feature = "bench-internals")]
pub mod transform;

#[cfg(not(feature = "bench-internals"))]
mod transform;

pub use autotune::{AutoReport, DEFAULT_SAMPLE_BYTES, OperatingPoint, compress_auto};
pub use codec::{EncodeOptions, FeatureInfo, FeatureKind, Profile, features};
pub use container::{
    ModelScore, StageSize, compress, compress_owned, compress_owned_with, compress_owned_with_traced, compress_with,
    decompress,
};
