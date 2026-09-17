//! TODO

#![allow(unused_features, reason = "Required for coverage_attribute feature.")]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

#[cfg_attr(not(test), expect(dead_code, reason = "not wired into the codec yet"))]
pub(crate) mod adler32;
