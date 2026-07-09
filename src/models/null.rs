//! Null model: the fallback predictor that contributes nothing.
//!
//! Predicts a neutral bit (stretched logit `0`, i.e. probability ½) and ignores updates. It is not a
//! user-selectable feature — the codec injects it (see [`crate::codec`]'s `model_builders`) when the
//! entropy stage is enabled but the profile selects no model, so entropy-with-no-models still codes
//! reversibly (without compressing). It also serves as a template for real models.

use super::{Context, Model};

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

impl Model for NullModel {
    fn predict(&mut self, _ctx: &Context, _hist: &[u8]) -> i32 {
        0
    }

    fn update(&mut self, _ctx: &Context, _hist: &[u8], _bit: u8) {}
}
