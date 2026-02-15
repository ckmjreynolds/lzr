//! LZ77-based compression library.

// Enable coverage attributes for nightly builds.
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
// False positive: deps used by the binary or integration tests, not the library.
#![allow(unused_crate_dependencies)]
