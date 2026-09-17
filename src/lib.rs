//! LZR: a general-purpose compression library.
//!
//! The distinguishing goal of this crate is **random access** into compressed streams (seeking to
//! an arbitrary offset without decoding everything before it) and **tailing** (reading a stream
//! that is still being appended to). The wire format is unstable until 1.0.

#![allow(unused_features, reason = "Required for coverage_attribute feature.")]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

// The `bench-internals` feature widens these modules (and, via `visibility::make`
// inside them, their items) to `pub` so `benches/` can reach them. Off by default.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod adler32;
#[cfg(not(feature = "bench-internals"))]
pub(crate) mod adler32;

#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod uleb128;
#[cfg(not(feature = "bench-internals"))]
pub(crate) mod uleb128;
