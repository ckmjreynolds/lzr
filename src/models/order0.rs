//! Order-0 token model: a direct `StateMap` over the 15-bit bit-tree.

use super::statemap::StateMap;
use super::{Context, SYMBOL_BITS, TokenModel};

/// Predicts each bit purely from its position in the current symbol's bit-tree
/// (the partial-symbol node `c0`), independent of prior symbols. A depth-15 tree
/// has `2^15 - 1` internal nodes, so a `2^15`-slot `StateMap` indexed by `c0`
/// covers every node with room to spare.
#[derive(Debug)]
pub(crate) struct Order0 {
    sm: StateMap,
}

impl Order0 {
    /// A fresh order-0 model.
    pub(crate) fn new() -> Self {
        Self {
            sm: StateMap::new(1 << SYMBOL_BITS),
        }
    }
}

impl TokenModel for Order0 {
    fn predict(&mut self, ctx: &Context) -> i32 {
        self.sm.predict(ctx.c0 as usize)
    }

    fn update(&mut self, ctx: &Context, bit: u8) {
        self.sm.update(ctx.c0 as usize, bit);
    }
}
