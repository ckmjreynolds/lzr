//! The null (identity) preprocessor.

use super::Preprocessor;

/// Passes bytes through unchanged. The default pipeline's single stage and the
/// worked example for writing a real preprocessor.
#[derive(Debug, Default)]
pub(crate) struct Null;

impl Preprocessor for Null {
    fn forward(&self, input: &[u8]) -> Vec<u8> {
        input.to_vec()
    }

    fn inverse(&self, input: &[u8]) -> Vec<u8> {
        input.to_vec()
    }
}
