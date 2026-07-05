//! The byte-level entropy stage: a context-mixing range coder wrapped as a [`Transform`].
//!
//! [`EntropyCoder`] is the final, optional pipeline stage. `forward` range-codes each byte of its
//! input as an 8-bit MSB-first bit-tree; `inverse` reverses it. Encode and decode build identical
//! fresh predictor state (the models are deterministic and the hashed-table sizes derive from the
//! framed byte count), so their predictions match bit-for-bit and the stream round-trips.
//!
//! This layer carries no container header — just a ULEB128 byte-count frame so the decoder knows
//! when to stop and how to size its tables. The self-describing container (magic, version, checksum)
//! lives in [`crate::container`].

use anyhow::{Context as _, Result, bail, ensure};

use crate::coder::{Decoder, Encoder};
use crate::mixer::Mixer;
use crate::models::{Context, SYMBOL_BITS, TokenModel};
use crate::transform::Transform;
use crate::uleb128;

/// Builds a model given the symbol (byte) count, so hashed tables size identically on encode and
/// decode. An [`EntropyCoder`]'s builders are the profile's enabled models (order-0, order-1, …), in
/// registry order; the coder requires at least one.
pub(crate) type ModelBuilder = fn(usize) -> Box<dyn TokenModel>;

/// Length of the history window handed to each model per byte: the most recent finalized bytes,
/// borrowed straight from the driver's buffer (no copy). Bounds the look-back the context models
/// (e.g. high-order [`crate::models::ordern::OrderN`]) can key on. The shipped order-0/order-1
/// models read 0–1 bytes, so this is headroom for higher-order and match models to come; it is a
/// slice bound only (no allocation, deterministic on both sides), so growing it later is free.
const WINDOW: usize = 1024;

/// Upper bound on decoded bytes per payload byte. The range coder spends a strictly positive number
/// of bits per symbol, so a genuine stream stays far under this; the cap only bounds the work a
/// corrupt or malicious count field can demand (a decompression-ratio ceiling). A production build
/// would expose a configurable absolute output limit instead.
const MAX_BYTES_PER_PAYLOAD_BYTE: usize = 4096;

/// The online predictor: every model's stretched prediction for the next bit, mixed into one
/// probability, plus the shared [`Context`] the models read.
struct Predictor {
    models: Vec<Box<dyn TokenModel>>,
    stretched: Vec<i32>,
    mixer: Mixer,
    ctx: Context,
}

impl Predictor {
    /// Builds the model set (hashed models are sized from `capacity` = the byte count, so encode and
    /// decode agree) from the profile's enabled `builders` (at least one).
    fn new(capacity: usize, builders: &[ModelBuilder]) -> Self {
        let models: Vec<Box<dyn TokenModel>> = builders.iter().map(|build| build(capacity)).collect();
        let n = models.len();
        Self {
            models,
            stretched: vec![0; n],
            mixer: Mixer::new(n, SYMBOL_BITS as usize),
            ctx: Context::new(),
        }
    }

    /// The mixed 12-bit probability that the next bit is 1. `hist` is the current byte's history
    /// window (the recent finalized bytes), the same for all 8 of its bits.
    #[expect(clippy::cast_sign_loss, reason = "`squash` returns a positive 12-bit probability")]
    fn predict(&mut self, hist: &[u8]) -> u32 {
        for (m, s) in self.models.iter_mut().zip(&mut self.stretched) {
            *s = m.predict(&self.ctx, hist);
        }
        self.mixer.mix(&self.stretched, usize::from(self.ctx.bpos)) as u32
    }

    /// Learn from the actual `bit` (mixer first, then models, all on the pre-bit context), then
    /// advance the context by one bit.
    fn commit(&mut self, hist: &[u8], bit: u8) {
        self.mixer.update(bit);
        for m in &mut self.models {
            m.update(&self.ctx, hist, bit);
        }
        self.ctx.push_bit(bit);
    }

    /// Close the current byte once all its bits are coded.
    const fn end_symbol(&mut self) {
        self.ctx.push_symbol();
    }
}

/// The byte-level entropy coder: every model the profile selected (order-0, order-1, …), mixed per
/// bit and range-coded. The final pipeline stage; toggled by the `entropy` feature. An empty
/// `builders` still codes reversibly (the mixer degrades to a constant ½), just without compressing;
/// [`crate::codec::Profile::validate`] rejects that useless-but-valid config at the CLI boundary.
pub(crate) struct EntropyCoder {
    builders: Vec<ModelBuilder>,
}

impl EntropyCoder {
    /// A coder over `builders` — every enabled model, in registry order.
    pub(crate) fn new(builders: Vec<ModelBuilder>) -> Self {
        Self {
            builders,
        }
    }
}

impl Transform for EntropyCoder {
    fn forward(&self, input: Vec<u8>) -> Vec<u8> {
        // Seed the encoder's buffer with the length header so `finish` returns
        // header + coded payload in one allocation (no second copy of the payload).
        let mut header = Vec::new();
        uleb128::encode_u64(input.len() as u64, &mut header);

        let mut pred = Predictor::new(input.len(), &self.builders);
        let mut enc = Encoder::with_prefix(header, input.len() + input.len() / 2 + 16);
        for (i, &byte) in input.iter().enumerate() {
            // The history window is a borrowed view into the already-encoded prefix — no copy.
            let hist = &input[i.saturating_sub(WINDOW)..i];
            for k in (0..SYMBOL_BITS).rev() {
                let bit = (byte >> k) & 1;
                let p = pred.predict(hist);
                enc.encode(bit, p);
                pred.commit(hist, bit);
            }
            pred.end_symbol();
        }
        enc.finish()
    }

    fn inverse(&self, input: Vec<u8>) -> Result<Vec<u8>> {
        let mut pos = 0;
        let count = uleb128::decode_u64(&input, &mut pos)?;
        let count = usize::try_from(count).context("byte count exceeds this platform's usize")?;

        let payload = &input[pos..];
        let limit = payload.len().saturating_mul(MAX_BYTES_PER_PAYLOAD_BYTE).saturating_add(1024);
        ensure!(count <= limit, "byte count {count} exceeds the {limit} a {}-byte payload can encode", payload.len());

        let mut pred = Predictor::new(count, &self.builders);
        let mut dec = Decoder::new(payload);
        // Cap the up-front reservation so a large (but in-limit) count cannot demand a huge
        // allocation before a single byte is decoded; the Vec grows as needed.
        let mut out = Vec::with_capacity(count.min(1 << 16));
        for i in 0..count {
            // A valid stream's decoder never reads past its payload (bar a small flush window). Once
            // it has, the remaining claimed bytes are only zero-padding artifacts of a corrupt count —
            // stop rather than churn through them. This bounds decode work to O(payload).
            if dec.consumed() > payload.len() + 8 {
                bail!("byte count {count} is inconsistent with the payload (exhausted at byte {i})");
            }
            // The history window is a borrowed view into the already-decoded prefix. Its borrow of
            // `out` ends (NLL) at the last read inside the bit loop, before the `out.push` below.
            let hist = &out[out.len().saturating_sub(WINDOW)..];
            let mut value = 0u8;
            for _ in 0..SYMBOL_BITS {
                let p = pred.predict(hist);
                let bit = dec.decode(p);
                pred.commit(hist, bit);
                value = (value << 1) | bit;
            }
            pred.end_symbol();
            out.push(value);
        }
        Ok(out)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::models::ordern::OrderN;

    /// An order-0-only coder — a minimal single-model entropy stage.
    fn coder() -> EntropyCoder {
        fn build_order0(capacity: usize) -> Box<dyn TokenModel> {
            Box::new(OrderN::new(0, capacity))
        }
        EntropyCoder::new(vec![build_order0])
    }

    proptest! {
        /// The entropy stage must round-trip arbitrary bytes — the full `0..=255` range exercises the
        /// symbol width and would break under a wrong sentinel mask in `Context::push_symbol`.
        #[test]
        fn bytes_roundtrip_full_range(raw in prop::collection::vec(any::<u8>(), 0..500)) {
            let c = coder();
            prop_assert_eq!(c.inverse(c.forward(raw.clone())).unwrap(), raw);
        }

        /// Decoding arbitrary bytes must never panic — only return Ok or Err.
        #[test]
        fn decode_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..2048)) {
            drop(coder().inverse(bytes));
        }
    }

    #[test]
    fn roundtrips_empty_and_single() {
        let c = coder();
        assert_eq!(c.inverse(c.forward(b"".to_vec())).unwrap(), b"");
        assert_eq!(c.inverse(c.forward(b"A".to_vec())).unwrap(), b"A");
    }

    #[test]
    fn compresses_repetitive_input() {
        let data = vec![b'a'; 10_000];
        let c = coder();
        let core = c.forward(data.clone());
        assert!(core.len() < data.len() / 10, "weak compression: {} bytes", core.len());
        assert_eq!(c.inverse(core).unwrap(), data);
    }

    #[test]
    fn rejects_absurd_byte_count() {
        // A tiny payload claiming a huge byte count must be rejected, not looped.
        let mut core = Vec::new();
        uleb128::encode_u64(u64::MAX, &mut core);
        core.extend_from_slice(&[0u8; 8]);
        assert!(coder().inverse(core).is_err());
    }
}
