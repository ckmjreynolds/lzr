//! Null token model: an optional model slot that contributes nothing.
//!
//! Predicts a neutral bit (stretched logit `0`, i.e. probability ½) and ignores updates. It exists
//! so the entropy stage can carry an optional model with no effect on the output — a placeholder
//! that exercises the toggleable-model framework and a template for real models.

use super::{Context, TokenModel};

/// A model that always predicts ½ and never learns. When mixed in it is dead weight (the mixer's
/// weight on it stays near zero), so it leaves the coded output effectively unchanged.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct NullModel;

impl NullModel {
    /// A fresh null model.
    pub(crate) const fn new() -> Self {
        Self
    }
}

impl TokenModel for NullModel {
    fn predict(&mut self, _ctx: &Context, _hist: &[u8]) -> i32 {
        0
    }

    fn update(&mut self, _ctx: &Context, _hist: &[u8], _bit: u8) {}
}
