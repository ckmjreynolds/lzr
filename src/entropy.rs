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

use std::cell::RefCell;
use std::sync::OnceLock;

use anyhow::{Context as _, Result, bail, ensure};

use crate::apm::Apm;
use crate::coder::{Decoder, Encoder};
use crate::mixer::{MIX_CONTEXTS, Mixer, TwoLayerMixer, squash};
use crate::models::{Context, Model, SYMBOL_BITS};
use crate::surprise::{BUCKETS, LEVELS, Surprise};
use crate::transform::{ModelTrace, Transform};
use crate::uleb128;

/// The 12-bit probability scale (`squash` returns `1..=4095`), matching the range coder.
const PROB_SCALE: i32 = 1 << 12;

/// Per-outcome log-loss cost table: `cost_lut()[p]` = `-log2(p / 4096)` bits — the ideal cost of coding
/// an outcome a model assigned 12-bit probability `p`. A LUT keeps the per-model scorecard accumulation
/// (once per model per bit) to a table lookup rather than a `log2` call, so it stays near-free.
fn cost_lut() -> &'static Vec<f64> {
    static LUT: OnceLock<Vec<f64>> = OnceLock::new();
    LUT.get_or_init(|| (0..PROB_SCALE).map(|p| -(f64::from(p.max(1)) / f64::from(PROB_SCALE)).log2()).collect())
}

/// Builds a model given the symbol (byte) count, so hashed tables size identically on encode and
/// decode. An [`EntropyCoder`]'s builders are the profile's enabled models (order-0, order-1, …), in
/// registry order; the coder requires at least one.
pub(crate) type ModelBuilder = fn(usize) -> Box<dyn Model>;

/// Length of the history window handed to each model per byte: the most recent finalized bytes,
/// borrowed straight from the driver's buffer (no copy). Bounds the look-back the byte-context models
/// (`crate::models::ordern::OrderN`, `crate::models::sparse::SparseModel`) can key on. It is a slice
/// bound only (no allocation, deterministic on both sides), so growing it later is free.
const WINDOW: usize = 1024;

/// Upper bound on decoded bytes per payload byte. The range coder spends a strictly positive number
/// of bits per symbol, so a genuine stream stays far under this; the cap only bounds the work a
/// corrupt or malicious count field can demand (a decompression-ratio ceiling). A production build
/// would expose a configurable absolute output limit instead.
const MAX_BYTES_PER_PAYLOAD_BYTE: usize = 4096;

/// Upper bound on output bytes to *pre-reserve* per payload byte. Pre-sizing the decode output to the
/// framed `count` (rather than growing from nothing) avoids the reallocation-doubling transient that,
/// at GB scale, spikes peak RSS well above the final size — the growing buffer would otherwise pass
/// through a ~1.5× allocate-and-copy while the model tables are resident. `with_capacity` only reserves
/// address space (pages fault in as bytes are written), so this never inflates RSS for a genuine
/// stream; the cap merely bounds the up-front allocation a corrupt-but-in-limit `count` can request.
/// Well above any real compression ratio (16× under [`MAX_BYTES_PER_PAYLOAD_BYTE`]), so genuine streams
/// still reserve their true size.
const MAX_RESERVE_PER_PAYLOAD_BYTE: usize = 256;

/// The entropy stage's refinement flags, read straight from the profile: which optional components
/// wrap the model mix. Each is independent of the others (the `sse`, `sse2`, `mix2`, `surprise`
/// registry flags); `sse2` and the surprise APM only take effect when `sse` is on.
#[derive(Debug, Clone, Copy, Default)]
#[expect(clippy::struct_excessive_bools, reason = "a plain set of independent profile flags, not a state machine")]
pub(crate) struct EntropyFlags {
    /// Apply the SSE/APM refinement (the `sse` feature).
    pub(crate) sse: bool,
    /// Chain the second order-1 APM (the `sse2` feature).
    pub(crate) sse2: bool,
    /// Use the two-layer mixing network in place of the single-layer mixer (the `mix2` flag).
    pub(crate) mix2: bool,
    /// Track coding surprise as a regime context (the `surprise` flag).
    pub(crate) surprise: bool,
}

/// The online predictor: every model's stretched prediction for the next bit, mixed into one
/// probability, plus the shared [`Context`] the models read.
struct Predictor {
    models: Vec<Box<dyn Model>>,
    stretched: Vec<i32>,
    mixer: Mixer,
    /// Optional two-layer mixing network (the `mix2` flag). When present it replaces `mixer` for the
    /// blend; `mixer` is left unused. A nonlinear generalisation of the single-layer mix.
    mixer2: Option<TwoLayerMixer>,
    /// Optional SSE stage refining the mixed probability (the `sse` feature); keyed on the bit-tree
    /// node. `None` leaves the mix unrefined.
    apm: Option<Apm>,
    /// Optional second SSE stage (the `sse2` feature), chained after `apm` and keyed on the previous
    /// byte (an order-1 context). `None` leaves `apm`'s output as the final probability.
    apm2: Option<Apm>,
    /// Optional coding-surprise regime tracker (the `surprise` flag): the recent final coding cost,
    /// quantized. Its fast level widens `mixer2`'s order-1 selector and keys `apm3`; its (fast, slow)
    /// bucket selects `mixer2`'s blend weights. `None` when the flag is off.
    surprise: Option<Surprise>,
    /// Optional third SSE stage (the `surprise` flag), chained after `apm2` and keyed on
    /// (fast surprise level × previous byte × bit-tree node).
    apm3: Option<Apm>,
    /// The final probability returned by the last `predict`, fed back to `surprise` on `commit`.
    last_p: i32,
    ctx: Context,
    /// Summed standalone log-loss (bits) per model, accumulated only when `track` is set (encode-side
    /// diagnostics). Divided by the byte count it gives each model's bits-per-byte-if-coded-alone.
    logloss: Vec<f64>,
    /// Whether to accumulate the per-model log-loss scorecard (encode only; skipped on decode).
    track: bool,
}

impl Predictor {
    /// Builds the model set (hashed models are sized from `capacity` = the byte count, so encode and
    /// decode agree) from the profile's enabled `builders` (at least one), wrapped per `flags`.
    /// `track` enables the encode-only per-model log-loss accumulation.
    fn new(capacity: usize, builders: &[ModelBuilder], flags: EntropyFlags, track: bool) -> Self {
        let EntropyFlags {
            sse: use_sse,
            sse2: use_sse2,
            mix2: use_mix2,
            surprise: use_surprise,
        } = flags;
        let models: Vec<Box<dyn Model>> = builders.iter().map(|build| build(capacity)).collect();
        let n = models.len();
        Self {
            models,
            stretched: vec![0; n],
            mixer: Mixer::new(n, SYMBOL_BITS as usize, MIX_CONTEXTS),
            mixer2: use_mix2.then(|| {
                // With `surprise` on, the order-1 sub-mixer selects by (previous byte × fast level) and
                // the blend layer by the (fast, slow) bucket; otherwise the plain order-1 / global blend.
                let (order1, blend) = if use_surprise {
                    (MIX_CONTEXTS * LEVELS, BUCKETS)
                } else {
                    (MIX_CONTEXTS, 1)
                };
                TwoLayerMixer::new(n, SYMBOL_BITS as usize, order1, blend, capacity)
            }),
            // One APM context per bit-tree node (`c0 & 0xff`), which uniquely encodes (bit position,
            // partial bits) — a minimal order-0 SSE.
            apm: use_sse.then(|| Apm::new(256)),
            // Order-1 SSE: one context per (previous byte × bit-tree node) = 256 × 256, chained after
            // `apm`. Requires `sse` to be meaningful (it refines `apm`'s output).
            apm2: (use_sse && use_sse2).then(|| Apm::new(256 * 256)),
            surprise: use_surprise.then(Surprise::new),
            apm3: (use_sse && use_surprise).then(|| Apm::new(LEVELS * 256 * 256)),
            last_p: PROB_SCALE / 2,
            ctx: Context::new(),
            logloss: vec![0.0; n],
            track,
        }
    }

    /// The per-model scorecard: each model's standalone bits-per-byte (its summed log-loss over
    /// `num_bytes`) and the mixer's average weight on it, paired with the aligned `names`.
    #[expect(clippy::cast_precision_loss, reason = "Diagnostic display; byte counts are well under 2^53.")]
    fn scores(&self, num_bytes: usize, names: &[&'static str]) -> Vec<ModelTrace> {
        let weights = self.mixer2.as_ref().map_or_else(|| self.mixer.input_weights(), TwoLayerMixer::input_weights);
        let denom = num_bytes.max(1) as f64;
        names
            .iter()
            .enumerate()
            .map(|(i, &name)| {
                (name, self.logloss.get(i).copied().unwrap_or(0.0) / denom, weights.get(i).copied().unwrap_or(0.0))
            })
            .collect()
    }

    /// The 12-bit probability that the next bit is 1: the models mixed, then (if enabled) refined by
    /// the APM. `hist` is the current byte's history window, the same for all 8 of its bits.
    #[expect(clippy::cast_sign_loss, reason = "the mix/APM both return a positive 12-bit probability")]
    fn predict(&mut self, hist: &[u8]) -> u32 {
        for (m, s) in self.models.iter_mut().zip(&mut self.stretched) {
            *s = m.predict(&self.ctx, hist);
        }
        // Select the mixer weight set by the previous byte (order-1 context), constant across all 8
        // bits of the current byte and identical on encode and decode.
        let sel = usize::from(hist.last().copied().unwrap_or(0));
        let bpos = usize::from(self.ctx.bpos);
        // Regime from the recent coding cost: the fast level and the (fast, slow) bucket (both 0 when
        // `surprise` is off, so the selectors below collapse to their plain forms).
        let (fl, sb) = self.surprise.as_ref().map_or((0, 0), |s| (s.fast_level(), s.bucket()));
        let p = if let Some(mixer2) = &mut self.mixer2 {
            // Layer-1 selectors: an order-1..6 progression via rolling hashes of the last 1..6 bytes,
            // so the blend can specialise by how deep a context is currently predictive.
            let p2 = usize::from(hist.iter().rev().nth(1).copied().unwrap_or(0));
            let p3 = usize::from(hist.iter().rev().nth(2).copied().unwrap_or(0));
            let p4 = usize::from(hist.iter().rev().nth(3).copied().unwrap_or(0));
            let o2 = sel.wrapping_mul(131).wrapping_add(p2);
            let o3 = o2.wrapping_mul(131).wrapping_add(p3);
            let o4 = o3.wrapping_mul(131).wrapping_add(p4);
            mixer2.mix(&self.stretched, [sel | (fl << 8), o2, o3, o4], sb, bpos)
        } else {
            self.mixer.mix(&self.stretched, sel, bpos)
        };
        let node = (self.ctx.c0 & 0xff) as usize;
        let p = self.apm.as_mut().map_or(p, |apm| apm.refine(p, node));
        // Chain the order-1 APM (previous byte × node), blending its refinement with its input so a
        // cold second stage cannot overcommit early. `None` when `sse2` is off.
        let p = self.apm2.as_mut().map_or(p, |apm2| (apm2.refine(p, (sel << 8) | node) * 3 + p) >> 2);
        // Chain the surprise-keyed APM (fast level × previous byte × node) the same way. `None` when
        // `surprise` is off.
        let p = self.apm3.as_mut().map_or(p, |apm3| (apm3.refine(p, (fl << 16) | (sel << 8) | node) * 3 + p) >> 2);
        self.last_p = p;
        p as u32
    }

    /// Learn from the actual `bit` (mixer and APM first, then models, all on the pre-bit context),
    /// then advance the context by one bit.
    #[expect(clippy::cast_sign_loss, reason = "`squash` and its complement are in 1..=4095, a valid index")]
    fn commit(&mut self, hist: &[u8], bit: u8) {
        if self.track {
            // Each model's standalone cost this bit: -log2 of the probability it gave the actual bit.
            let lut = cost_lut();
            for (loss, &s) in self.logloss.iter_mut().zip(&self.stretched) {
                let sq = squash(s);
                let p_correct = if bit == 1 {
                    sq
                } else {
                    PROB_SCALE - sq
                };
                *loss += lut[p_correct as usize];
            }
        }
        if let Some(mixer2) = &mut self.mixer2 {
            mixer2.update(bit);
        } else {
            self.mixer.update(bit);
        }
        if let Some(apm) = &mut self.apm {
            apm.update(bit);
        }
        if let Some(apm2) = &mut self.apm2 {
            apm2.update(bit);
        }
        if let Some(apm3) = &mut self.apm3 {
            apm3.update(bit);
        }
        if let Some(surprise) = &mut self.surprise {
            surprise.update(self.last_p, bit);
        }
        for m in &mut self.models {
            m.update(&self.ctx, hist, bit);
        }
        self.ctx.push_bit(bit);
    }

    /// Close the current byte once all its bits are coded, resetting the bit-tree for the next symbol.
    const fn end_symbol(&mut self) {
        self.ctx.push_symbol();
    }
}

/// The byte-level entropy coder: every model the profile selected (order-0, order-1, …), mixed per
/// bit and range-coded. The final pipeline stage; toggled by the `entropy` feature. When a profile
/// selects no model, [`crate::codec::Profile::model_builders`] injects the internal null model, so
/// `builders` is never empty via the codec; a directly-constructed empty `builders` would still code
/// reversibly (the mixer degrades to a constant ½), just without compressing.
pub(crate) struct EntropyCoder {
    builders: Vec<ModelBuilder>,
    /// The enabled models' names, aligned with `builders`, for the per-model scorecard.
    names: Vec<&'static str>,
    /// Which optional refinements wrap the mix (`sse`, `sse2`, `mix2`, `surprise`).
    flags: EntropyFlags,
    /// The per-model scorecard produced by the last `forward` (encode-only). Interior mutability
    /// because `Transform::forward` takes `&self`; read back via [`Transform::model_scores`].
    scores: RefCell<Vec<ModelTrace>>,
}

impl EntropyCoder {
    /// A coder over `builders` — every enabled model, in registry order — labelled by the aligned
    /// `names` (for the scorecard), wrapped by the refinements in `flags`.
    pub(crate) fn new(builders: Vec<ModelBuilder>, names: Vec<&'static str>, flags: EntropyFlags) -> Self {
        Self {
            builders,
            names,
            flags,
            scores: RefCell::new(Vec::new()),
        }
    }
}

impl Transform for EntropyCoder {
    fn forward(&self, input: Vec<u8>) -> Vec<u8> {
        // Seed the encoder's buffer with the length header so `finish` returns
        // header + coded payload in one allocation (no second copy of the payload).
        let mut header = Vec::new();
        uleb128::encode_u64(input.len() as u64, &mut header);

        let mut pred = Predictor::new(input.len(), &self.builders, self.flags, true);
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
        *self.scores.borrow_mut() = pred.scores(input.len(), &self.names);
        enc.finish()
    }

    fn inverse(&self, input: Vec<u8>) -> Result<Vec<u8>> {
        let mut pos = 0;
        let count = uleb128::decode_u64(&input, &mut pos)?;
        let count = usize::try_from(count).context("byte count exceeds this platform's usize")?;

        let payload = &input[pos..];
        let limit = payload.len().saturating_mul(MAX_BYTES_PER_PAYLOAD_BYTE).saturating_add(1024);
        ensure!(count <= limit, "byte count {count} exceeds the {limit} a {}-byte payload can encode", payload.len());

        let mut pred = Predictor::new(count, &self.builders, self.flags, false);
        let mut dec = Decoder::new(payload);
        // Pre-size the output to the framed count to avoid the doubling-reallocation transient (see
        // MAX_RESERVE_PER_PAYLOAD_BYTE), capped relative to the payload so a corrupt-but-in-limit count
        // cannot demand an absurd up-front allocation. `with_capacity` reserves address space only, so a
        // genuine stream's pages fault in as decoded — no RSS cost beyond the real output.
        let reserve = count.min(payload.len().saturating_mul(MAX_RESERVE_PER_PAYLOAD_BYTE).max(1 << 16));
        let mut out = Vec::with_capacity(reserve);
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

    fn model_scores(&self) -> Vec<ModelTrace> {
        self.scores.borrow().clone()
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
        fn build_order0(capacity: usize) -> Box<dyn Model> {
            Box::new(OrderN::new(0, capacity))
        }
        EntropyCoder::new(vec![build_order0], vec!["order0"], EntropyFlags::default())
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
