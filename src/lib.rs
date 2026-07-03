//! LZR Compression library.

#![allow(unused_features, reason = "Required for coverage_attribute feature.")]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

pub mod error;

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
#[cfg_attr(not(test), expect(dead_code, reason = "Development."))]
pub(crate) mod uleb128;
