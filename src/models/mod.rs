//! Bit-prediction models and the shared [`Context`] they read.
//!
//! A model is queried once per bit ([`Model::predict`]) and then told the actual
//! bit ([`Model::update`]). Predictions are in the stretched (logit) domain so
//! the mixer can combine them directly.

pub(crate) mod context;

const RING_BITS: usize = 10; // last 1024 finalized bytes
const RING_SIZE: usize = 1 << RING_BITS;
const RING_MASK: usize = RING_SIZE - 1;

/// Mutable per-stream prediction context shared by every model.
#[derive(Debug)]
pub(crate) struct Context {
    ring: [u8; RING_SIZE],
    head: usize, // index where the next finalized byte is written
    /// Partial current byte: a leading-1 sentinel followed by the bits coded so far.
    pub(crate) c0: u32,
    /// Number of bits of the current byte already coded (`0..=7`).
    pub(crate) bpos: u8,
    /// The last four finalized bytes; the most recent is in the low 8 bits.
    /// Read by the order-1 and order-2 context models.
    pub(crate) c4: u32,
}

impl Context {
    /// New, empty context.
    pub(crate) const fn new() -> Self {
        Self {
            ring: [0; RING_SIZE],
            head: 0,
            c0: 1,
            bpos: 0,
            c4: 0,
        }
    }

    /// The finalized byte `i` positions back (`i` in `1..=1024`); `0` before
    /// that much history exists. Provided for future context models.
    #[allow(dead_code)]
    pub(crate) const fn byte_back(&self, i: usize) -> u8 {
        self.ring[self.head.wrapping_sub(i) & RING_MASK]
    }

    /// Append one freshly-coded bit to the partial current byte.
    pub(crate) fn push_bit(&mut self, bit: u8) {
        self.c0 = (self.c0 << 1) | u32::from(bit);
        self.bpos += 1;
    }

    /// Finalize the current byte once all 8 bits are in, and reset for the next.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn push_byte(&mut self) {
        let b = self.c0 as u8; // low 8 bits are the byte; the sentinel is bit 8
        self.ring[self.head & RING_MASK] = b;
        self.head = self.head.wrapping_add(1);
        self.c4 = (self.c4 << 8) | u32::from(b);
        self.c0 = 1;
        self.bpos = 0;
    }
}

/// A model that predicts the next bit from the [`Context`].
pub(crate) trait Model {
    /// Predict P(next bit == 1) in the stretched (logit) domain, clamped to
    /// roughly `[-2047, 2047]`. Called before the bit is known.
    fn predict(&mut self, ctx: &Context) -> i32;
    /// Observe the actual `bit`. `ctx` still reflects the pre-bit state.
    fn update(&mut self, ctx: &Context, bit: u8);
}
