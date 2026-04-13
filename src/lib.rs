//! LZ77-based compression library.

#![allow(unused_features)]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![allow(dead_code)]
pub(crate) mod adler32;
pub(crate) mod entropy;
pub(crate) mod lz77;
pub(crate) mod model;
pub(crate) mod uleb128;

#[cfg(any(test, feature = "bench-internals"))]
#[cfg_attr(coverage_nightly, coverage(off))]
#[allow(
    missing_docs,
    missing_copy_implementations,
    missing_debug_implementations,
    unreachable_pub,
    clippy::redundant_pub_crate,
    clippy::new_without_default,
    clippy::missing_const_for_fn
)]
pub mod bench;
