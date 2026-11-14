#![expect(missing_docs, reason = "Development.")]
#![cfg_attr(coverage_nightly, allow(unused_features, reason = "`expect` cannot catch this crate-level lint."))]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

#[cfg_attr(not(any(test, feature = "bench")), expect(dead_code, reason = "not wired into the codec yet"))]
pub(crate) mod adler32;

/// Benchmark-only entry points into crate internals. Not part of the public API.
///
/// Wrapper functions rather than re-exports: `pub(crate)` items cannot be re-exported as `pub`, and making them
/// `pub` would trip `unreachable_pub` whenever the feature is off.
#[cfg(feature = "bench")]
#[doc(hidden)]
pub mod __bench {
    /// Adler-32 checksum of `data`.
    #[must_use]
    pub fn adler32(data: &[u8]) -> u32 {
        let mut adler = crate::adler32::Adler32::new();
        adler.update(data);
        adler.checksum()
    }
}
