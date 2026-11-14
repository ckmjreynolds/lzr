#![expect(missing_docs, reason = "Development.")]
#![cfg_attr(coverage_nightly, allow(unused_features, reason = "`expect` cannot catch this crate-level lint."))]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

// #[cfg_attr(not(test), expect(dead_code, reason = "not wired into the codec yet"))]
// pub(crate) mod adler32;
