//! Debug-only reserved-byte guard.
//!
//! Downstream preprocessors (case-fold, and later a dictionary) emit a handful
//! of marker bytes that the enwik corpora never contain — see the census in the
//! module docs. That absence is the contract those stages rely on instead of an
//! escape mechanism, so it must hold. This pass-through stage asserts it on the
//! raw input. It is added to the pipeline only in debug builds, so it costs the
//! submission nothing (no `L(D)`, no runtime); `build.sh`'s test run exercises
//! it and fails loudly if a reserved byte ever appears.

use super::Preprocessor;
use super::casefold::{UPPER_ONE, UPPER_TOGGLE};

/// Bytes reserved as preprocessor markers, absent from enwik8/enwik9. Sourced
/// from the stages that emit them (currently case-fold) so there is one
/// definition. All lie in the C0-control range free in both corpora.
const RESERVED: [u8; 2] = [UPPER_ONE, UPPER_TOGGLE];

/// Verifies the input contains no reserved marker byte; otherwise identity.
#[derive(Debug, Default)]
pub(crate) struct Guard;

impl Preprocessor for Guard {
    fn forward(&self, input: &[u8]) -> Vec<u8> {
        for &b in input {
            assert!(
                !RESERVED.contains(&b),
                "reserved marker byte {b:#04x} present in input"
            );
        }
        input.to_vec()
    }

    fn inverse(&self, input: &[u8]) -> Vec<u8> {
        input.to_vec()
    }
}
