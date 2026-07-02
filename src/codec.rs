//! The encode/decode driver.
//!
//! One per-bit loop wires the preprocessor pipeline, the models, the mixer, and
//! the arithmetic coder. Encode and decode build identical fresh state so their
//! predictions match bit-for-bit. The only framing is a LEB128 varint length
//! prefix (the decoder must know how many bytes to emit); there is no header.

use crate::coder::{Decoder, Encoder};
use crate::mixer::{Apm, Mixer, NeuralMixer};
use crate::models::context::ContextModel;
use crate::models::indirect::IndirectModel;
#[cfg(feature = "arm")]
use crate::models::lstm::ArmModel;
use crate::models::match_model::MatchModel;
use crate::models::pretrained::PretrainedMlp;
#[cfg(feature = "ssm")]
use crate::models::ssm::SsmModel;
use crate::models::{AnyModel, Context, Model};
use crate::preprocessors::Pipeline;

/// The deterministic model set (no arm). Shared by the production [`models`] and
/// by the ablation tests, so an ablation measures its variant against the exact
/// shipped baseline. Adding/removing a deterministic model is one line here.
pub(crate) fn baseline_models(capacity: usize) -> Vec<AnyModel> {
    vec![
        ContextModel::new(0, capacity).into(),
        ContextModel::new(1, capacity).into(),
        ContextModel::new(2, capacity).into(),
        ContextModel::new(3, capacity).into(),
        ContextModel::new(4, capacity).into(),
        ContextModel::new(5, capacity).into(),
        ContextModel::new(6, capacity).into(),
        ContextModel::word(capacity).into(),
        ContextModel::sparse(0b101, capacity).into(), // bytes back 1 and 3 (skip 2)
        ContextModel::sparse(0b110, capacity).into(), // bytes back 2 and 3 (skip the last byte)
        ContextModel::sparse(0b1011, capacity).into(), // bytes back 1, 2 and 4 (skip 3)
        ContextModel::sparse(0b1100, capacity).into(), // bytes back 3 and 4 (skip 1, 2)
        ContextModel::sparse(0b10001, capacity).into(), // bytes back 1 and 5 (skip 2,3,4)
        ContextModel::number(capacity).into(),        // field-aware digit-run context
        MatchModel::new().into(),
        MatchModel::with_key(4).into(), // shorter-key match: faster acquisition
        // Higher-order matches (v7's order-scaling lever): acquire only from
        // longer repeats, decorrelated from key-8 the way key-8 is from key-4.
        // Marginals GROW with scale (coverage-driven): key-12 −0.0009 (8 MB) →
        // −0.0012 (20 MB); +key-16 → −0.0017 (20 MB). Appended AFTER the key-8/
        // key-4 pair so the mixer match-selector and the heads' match-state
        // deposits (both take the first match models) are untouched.
        MatchModel::with_key(12).into(),
        MatchModel::with_key(16).into(),
        // Indirect context models (paq ICM): predict from what historically
        // followed a context. Orders [1,2,3,4,6] — 5 and 8 add ~nothing.
        IndirectModel::new(1, capacity).into(),
        IndirectModel::new(2, capacity).into(),
        IndirectModel::new(3, capacity).into(),
        IndirectModel::new(4, capacity).into(),
        IndirectModel::new(6, capacity).into(),
        IndirectModel::word(capacity).into(),
    ]
}

/// Frozen pretrained byte-LM weights (`src/models/pretrained.rs`), trained offline
/// by `examples/pretrain.rs`. Embedded so encode and decode build the identical
/// predictor from the same bytes; the blob counts as L(D).
const PRETRAINED_NET: &[u8] = include_bytes!("../assets/pretrained_net.bin");

/// The pretrained net's online warming-head STACK: per-context online linear
/// readouts over the frozen embedding (a neural analog of the deterministic
/// context-model stack), each a separate mixer input. All share the one frozen
/// forward, so each head costs only its readout (~free); each warms WITH the data
/// (its marginal holds/grows at scale, unlike the frozen net). Ships no weights
/// (online → L(D)≈0). Full-stack enwik8 (10 MB) sweep, cumulative over no-head:
/// order-1 (prev byte) −0.0126; + word-context −0.0154; + order-2 (prev 2 bytes)
/// −0.0190; + sparse `byte_back(2,3)` −0.0202 — each added DECORRELATED context
/// keeps paying (diminishing: a 5th sparse head adds only −0.0006). lr≈4; each
/// head is a `HEAD_BITS`-wide table (~64 MB at 16; order-2's hash collisions there
/// are benign — the frozen embedding still separates colliding contexts). With
/// the dual-rate slow head (see `HEAD_SLOW_LR`) the stack is 5 heads ≈ 320 MB
/// (enwik9 RSS ~8.5 GB, under the 10 GB cap).
const HEAD_LR: f32 = 4.0;
const HEAD_BITS: u32 = 16;
/// A second, SLOW readout on the order-1 context (dual-rate): the fast head
/// (`HEAD_LR`) tracks recent structure, this slow one (lr 0.5) the stable
/// per-context structure the fast rate washes out — decorrelation by TIMESCALE.
/// As a 5th head it adds −0.0016 (10 MB), MORE than any new context at this depth
/// (a 5th distinct context added only −0.0004 to −0.0006); slow copies of the
/// other contexts overlap each other and add little more.
const HEAD_SLOW_LR: f32 = 0.5;
/// Learning rate for the match-derived heads (primary/secondary match, match×prev).
/// They fire only when a match is active — a sparser, more stationary distribution
/// than the byte-context heads — so a slower rate wins: a full-stack 10 MB sweep
/// put the match heads' optimum on a broad lr 1–2 plateau (−0.0242 vs −0.0234 at
/// lr 4), and the secondary-match head likewise gained −0.0007 moving 4→2.
const HEAD_MATCH_LR: f32 = 2.0;

/// The active model set: the deterministic baseline plus the frozen pretrained
/// net (shipped). The online-neural arm is appended only under `--features arm`
/// (opt-in, not shipped — see Cargo.toml).
fn models(capacity: usize) -> Vec<AnyModel> {
    let mut v = baseline_models(capacity);
    // `LZR_NO_NET` (offline ablation only) drops the pretrained net to isolate its
    // marginal / test additivity with the arm; unset (the default) keeps it shipped.
    if std::env::var("LZR_NO_NET").is_err() {
        let mut net = PretrainedMlp::from_blob_q8(PRETRAINED_NET);
        // The warming-head stack — each a separate mixer input that warms with the
        // data (ships no weights, L(D)≈0). Two axes: byte-context heads (order-1,
        // word, order-2, sparse byte_back(2,3), and a dual-rate slow order-1), and a
        // MATCH cluster (primary match state, the secondary/key-4 match, and the
        // primary match's predicted byte × the previous byte) — long-range LZP
        // context the frozen net's K-byte window lacks. The match cluster stacks:
        // each is decorrelated (different match model / different keying), full-stack
        // 20 MB marginals over the byte-context heads −0.0025 (m2) and −0.0014 (mc).
        // `LZR_NO_HEAD` drops the stack.
        if std::env::var("LZR_NO_HEAD").is_err() {
            net.push_head_ctx(HEAD_LR, 1, HEAD_BITS);
            net.push_head_word(HEAD_LR, HEAD_BITS);
            net.push_head_ctx(HEAD_LR, 2, HEAD_BITS);
            net.push_head_sparse(HEAD_LR, 0b110, HEAD_BITS);
            net.push_head_ctx(HEAD_SLOW_LR, 1, HEAD_BITS); // dual-rate: slow order-1
            net.push_head_match(HEAD_MATCH_LR, HEAD_BITS); // primary (key=8) match state
            net.push_head_match2(HEAD_MATCH_LR, HEAD_BITS); // secondary (key=4) match
            net.push_head_match_prev(HEAD_MATCH_LR, HEAD_BITS); // match pred × prev byte
            // Diversity sweep (2026-07-01): three more decorrelated axes, each an
            // independent −0.0006, gate-passed −0.0022 on full enwik8 and GROWING
            // with scale (−0.0018/−0.0020/−0.0022 at 8/40/100 MB). L(D)≈0.
            net.push_head_exp(HEAD_LR, 3, HEAD_BITS); // digit-field
            net.push_head_exp(HEAD_MATCH_LR, 6, HEAD_BITS); // secondary-match-pred × prev
            net.push_head_sparse(HEAD_LR, 0b1001, HEAD_BITS); // sparse bytes-back {1,4}
        }
        v.push(net.into());
    }
    #[cfg(feature = "arm")]
    v.push(ArmModel::arm().into());
    #[cfg(feature = "ssm")]
    v.push(SsmModel::arm().into());
    v
}

// Residual neural mixing stage (2026-06-24): a small online MLP that refines the
// linear mixer's logit, initialized to pass-through so it can only improve. The
// `h=64`, `lr=0.005`, single shared weight set (`nctx=1`) point was the 20 MB
// enwik8-slice optimum — marginal −0.0075 bpb and *growing* with data (it learns
// the ensemble's systematic miscalibration, which the linear log-odds mix cannot
// express); per-bit-position weight sets were a clean negative (online data
// efficiency favors sharing). Ships no weights → L(D)≈0; ~4× the per-bit cost,
// but the deterministic path's enwik9 ETA (~4 h) stays far under the time budget.
// Width re-tuned 64→96 once the warming-head stack added 4 strong mixer inputs:
// the richer input set wants more hidden capacity (10 MB −0.0004, 20 MB −0.0006,
// growing). 96 is the knee (128 adds only −0.0001). Overridable by `LZR_NMIXH`.
const NMIX_H: usize = 96;
const NMIX_NCTX: usize = 1;
const NMIX_LR: f32 = 0.005;

const APM_CTX: usize = 256; // SSE contexts: the partial-byte node `c0`
const APM_W: i32 = 2; // SSE blend: refined prob weighted `APM_W`/4 vs the mixer
// (equal blend; the stronger multi-mixer needs less SSE correction — sweep
// 1/2/3/4 = 1.5948/1.5929/1.5936/1.5971 on the 20 MB slice).

/// Shipped SSE blend weight, overridable by `LZR_APMW` for offline sweeps only.
fn apm_w() -> i32 {
    std::env::var("LZR_APMW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(APM_W)
}

/// Shipped residual-mixer hidden width, overridable by `LZR_NMIXH` for offline
/// sweeps only (re-tuning M1's capacity for the head-augmented input set).
fn nmix_h() -> usize {
    std::env::var("LZR_NMIXH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(NMIX_H)
}

/// Shipped residual-mixer learning rate, overridable by `LZR_NMIXLR` (offline).
fn nmix_lr() -> f32 {
    std::env::var("LZR_NMIXLR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(NMIX_LR)
}

/// The shared predictor state driven identically by both directions: encode and
/// decode differ only in where each bit comes from (read from the input vs.
/// decoded from the stream) and which coder consumes it.
struct CodecState {
    models: Vec<AnyModel>,
    mixer: Mixer,
    apm: Apm,
    apm_w: i32,
    ctx: Context,
    stretched: Vec<i32>,
    msel: [usize; 8], // mixer weight-set selectors, recomputed once per byte (bpos==0)
    // Each head-enabled net's warming-head logit occupies one extra `stretched`
    // slot beyond the models (starting at `models.len()`), so the mixer sees it as
    // an input distinct from the frozen logit. `n_heads` is how many such slots.
    n_heads: usize,
    // Residual neural mixing stage: `Some` in production (refines the linear
    // mixer's logit). Ablation tests set it to `None` to measure its marginal
    // against the linear-only path.
    nmix: Option<NeuralMixer>,
}

impl CodecState {
    fn new(capacity: usize) -> Self {
        Self::with_models(models(capacity), capacity)
    }

    fn with_models(models: Vec<AnyModel>, capacity: usize) -> Self {
        let n = models.len();
        // Each warming head adds one extra mixer input beyond the models. Reserve
        // those slots so the mixer/nmix are sized to see them.
        let n_heads: usize = models.iter().map(AnyModel::n_heads).sum();
        let inputs = n + n_heads;
        let stretched = vec![0i32; inputs];
        let mixer = Mixer::new(inputs);
        Self {
            models,
            mixer,
            apm: Apm::new(APM_CTX),
            apm_w: apm_w(),
            ctx: Context::with_capacity(capacity),
            stretched,
            msel: [0usize; 8],
            n_heads,
            nmix: Some(NeuralMixer::new(inputs, nmix_h(), NMIX_NCTX, nmix_lr())),
        }
    }

    /// Predict the next bit as P(bit == 1) in 12-bit probability form: mix the
    /// models, then refine through the SSE stage and blend (SSE-weighted).
    #[allow(clippy::cast_sign_loss)]
    fn predict(&mut self) -> u32 {
        // Every mixer selector (c1/c2/c3 from c4, word hash, match-length bucket,
        // digit-run/word/column positions) is byte-constant — none change until
        // push_byte. The model predict loop below never mutates them or the match
        // length, so recomputing the selector array once at bpos==0 and reusing it
        // for all 8 bits is identical to the per-bit computation, and drops 7/8 of
        // the find_map selector scans.
        if self.ctx.bpos == 0 {
            let c4 = self.ctx.c4;
            let match_sel = self.models.iter().find_map(AnyModel::selector).unwrap_or(0);
            // Deposit the match models' (length bucket, predicted byte) so the
            // frozen net's warming heads can condition on the long-range match. The
            // first (primary, key=8) feeds `match_*`; the second (key=4, faster
            // acquisition) feeds `match_*2` for experimental decorrelated heads.
            let ((l1, p1), (l2, p2)) = {
                let mut mk = self.models.iter().filter_map(|m| m.match_key(&self.ctx));
                (mk.next().unwrap_or((0, 0)), mk.next().unwrap_or((0, 0)))
            };
            self.ctx.match_len = l1;
            self.ctx.match_pb = p1;
            self.ctx.match_len2 = l2;
            self.ctx.match_pb2 = p2;
            self.msel = [
                (c4 & 0xff) as usize,                   // c1
                ((c4 >> 8) & 0xff) as usize,            // c2
                ((c4 >> 16) & 0xff) as usize,           // c3
                (self.ctx.word_hash & 0xff) as usize,   // current word
                match_sel,                              // match-length bucket
                usize::from(self.ctx.num_pos.min(15)),  // digit-run position
                usize::from(self.ctx.word_pos.min(15)), // word position
                usize::from(self.ctx.col.min(15)),      // column (line position)
            ];
        }
        for (m, s) in self.models.iter_mut().zip(&mut self.stretched) {
            *s = m.predict(&self.ctx);
        }
        // Each head's logit is an extra input past the models, in model/head order.
        if self.n_heads > 0 {
            let mut slot = self.models.len();
            for i in 0..self.models.len() {
                for hi in 0..self.models[i].n_heads() {
                    self.stretched[slot] = self.models[i].head_out(hi);
                    slot += 1;
                }
            }
        }
        let pm = self
            .mixer
            .mix(&self.stretched, &self.msel, usize::from(self.ctx.bpos));
        // Optional residual neural refinement of the linear mix (pass-through at
        // init); `base_p` is the linear mixer's `pm` when the probe is off.
        let base_p = match self.nmix.as_mut() {
            Some(nm) => nm.refine(pm, &self.stretched, usize::from(self.ctx.bpos)),
            None => pm,
        };
        let pa = self.apm.refine(base_p, (self.ctx.c0 & 0xff) as usize);
        ((base_p * (4 - self.apm_w) + pa * self.apm_w + 2) >> 2) as u32
    }

    /// Commit the actual `bit`: adapt the mixer, SSE, and models, advance context.
    fn commit(&mut self, bit: u8) {
        self.mixer.update(bit);
        if let Some(nm) = self.nmix.as_mut() {
            nm.update(bit);
        }
        self.apm.update(bit);
        for m in &mut self.models {
            m.update(&self.ctx, bit);
        }
        self.ctx.push_bit(bit);
    }

    /// Finalize the current symbol (byte) once its 8 bits are in.
    fn end_symbol(&mut self) {
        self.ctx.push_byte();
    }
}

#[allow(clippy::cast_possible_truncation)]
fn write_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn read_varint(input: &[u8]) -> (u64, usize) {
    let mut v = 0u64;
    let mut shift = 0;
    let mut i = 0;
    loop {
        let byte = input[i];
        v |= u64::from(byte & 0x7f) << shift;
        i += 1;
        if byte & 0x80 == 0 {
            return (v, i);
        }
        shift += 7;
    }
}

/// Arithmetic-code an already-preprocessed byte stream with no pipeline and no
/// progress — used by the offline preprocessor experiments.
#[cfg(test)]
fn code_stream(data: &[u8]) -> Vec<u8> {
    code_stream_inner(data, data.len(), false)
}

/// As [`code_stream`], but `orig_len` (the pre-pipeline length, for bpb scaling)
/// and a `progress` flag enable periodic stderr ETA / running-bpb lines on large
/// encodes — the binary writes its output only at the end, so this is the only
/// window into a multi-hour run. Progress never affects the coded output.
#[allow(clippy::cast_precision_loss)]
fn code_stream_inner(data: &[u8], orig_len: usize, progress: bool) -> Vec<u8> {
    let mut out = Vec::new();
    write_varint(&mut out, data.len() as u64);

    let mut state = CodecState::new(data.len());
    let mut enc = Encoder::with_capacity(data.len());
    let start = std::time::Instant::now();
    let report = progress && data.len() > 10_000_000;
    let step = (data.len() / 100).max(1);
    for (i, &byte) in data.iter().enumerate() {
        for k in (0..8).rev() {
            let bit = (byte >> k) & 1;
            let p = state.predict();
            enc.encode(bit, p);
            state.commit(bit);
        }
        state.end_symbol();
        if report && (i + 1) % step == 0 {
            let pos = i + 1;
            let elapsed = start.elapsed().as_secs_f64();
            let frac = pos as f64 / data.len() as f64;
            let rate = pos as f64 / elapsed;
            let eta_h = (data.len() - pos) as f64 / rate / 3600.0;
            let bpb = enc.output_len() as f64 * 8.0 / (frac * orig_len as f64);
            eprintln!(
                "[lzr] {:.1}% | {:.0}/{:.0} MB | {:.1} KB/s | ETA {eta_h:.1}h | bpb~{bpb:.4}",
                frac * 100.0,
                pos as f64 / 1e6,
                data.len() as f64 / 1e6,
                rate / 1e3,
            );
        }
    }

    out.extend_from_slice(&enc.finish());
    out
}

/// Like [`code_stream`] but with a caller-supplied model set (ablation only).
#[cfg(test)]
pub(crate) fn code_stream_models(models: Vec<AnyModel>, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    write_varint(&mut out, data.len() as u64);
    let mut state = CodecState::with_models(models, data.len());
    let mut enc = Encoder::with_capacity(data.len());
    for &byte in data {
        for k in (0..8).rev() {
            let bit = (byte >> k) & 1;
            let p = state.predict();
            enc.encode(bit, p);
            state.commit(bit);
        }
        state.end_symbol();
    }
    out.extend_from_slice(&enc.finish());
    out
}

/// Decode counterpart of [`code_stream_models`] (ablation/round-trip only):
/// decodes a stream produced by `code_stream_models` with the same model set.
/// Operates on the coded bytes directly (no pipeline inverse).
#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn decode_stream_models(models: Vec<AnyModel>, input: &[u8]) -> Vec<u8> {
    let (len, header) = read_varint(input);
    let len = len as usize;
    let mut state = CodecState::with_models(models, len);
    let mut dec = Decoder::new(&input[header..]);
    let mut data = Vec::with_capacity(len);
    for _ in 0..len {
        let mut byte = 0u8;
        for _ in 0..8 {
            let p = state.predict();
            let bit = dec.decode(p);
            state.commit(bit);
            byte = (byte << 1) | bit;
        }
        state.end_symbol();
        data.push(byte);
    }
    data
}

/// Compress `input` into the lzr byte stream. Takes the input by value so the
/// raw bytes (1 GB at enwik9) are freed once the pipeline has produced the
/// stream actually coded — the multi-hour coding phase then holds one copy,
/// not two, under the 10 GB judging cap.
pub(crate) fn encode(input: Vec<u8>) -> Vec<u8> {
    let orig_len = input.len();
    let data = Pipeline::default_pipeline().forward(&input);
    drop(input);
    code_stream_inner(&data, orig_len, true)
}

/// Decompress an lzr byte stream back into the original bytes.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn decode(input: &[u8]) -> Vec<u8> {
    let (len, header) = read_varint(input);
    let len = len as usize;

    let mut state = CodecState::new(len);
    let mut dec = Decoder::new(&input[header..]);
    let mut data = Vec::with_capacity(len);
    for _ in 0..len {
        let mut byte = 0u8;
        for _ in 0..8 {
            let p = state.predict();
            let bit = dec.decode(p);
            state.commit(bit);
            byte = (byte << 1) | bit;
        }
        state.end_symbol();
        data.push(byte);
    }

    Pipeline::default_pipeline().inverse(&data)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_precision_loss)]
    use super::*;

    #[test]
    fn roundtrip_text() {
        let data = b"Hello, context mixing! The quick brown fox. ".repeat(64);
        let coded = encode(data.clone());
        assert_eq!(decode(&coded), data);
    }

    #[test]
    fn roundtrip_empty() {
        assert_eq!(decode(&encode(Vec::new())), b"");
    }

    /// Round-trip the residual neural mixer inside the real codec on a small
    /// slice: encode and decode build identical state and run identical f32 ops,
    /// so the stage must be byte-exact in both directions.
    #[test]
    fn nmix_roundtrip() {
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let data = Pipeline::default_pipeline().forward(&e8[1_000_000..1_040_000]);
        let run = |encode: bool, input: &[u8]| -> Vec<u8> {
            let mut state = CodecState::new(data.len());
            state.nmix = Some(NeuralMixer::new(state.stretched.len(), 16, 1, 0.002));
            if encode {
                let mut enc = Encoder::with_capacity(input.len());
                for &byte in input {
                    for k in (0..8).rev() {
                        let bit = (byte >> k) & 1;
                        let p = state.predict();
                        enc.encode(bit, p);
                        state.commit(bit);
                    }
                    state.end_symbol();
                }
                enc.finish()
            } else {
                let mut dec = Decoder::new(input);
                let mut out = Vec::with_capacity(data.len());
                for _ in 0..data.len() {
                    let mut byte = 0u8;
                    for _ in 0..8 {
                        let p = state.predict();
                        let bit = dec.decode(p);
                        state.commit(bit);
                        byte = (byte << 1) | bit;
                    }
                    state.end_symbol();
                    out.push(byte);
                }
                out
            }
        };
        let coded = run(true, &data);
        let decoded = run(false, &coded);
        assert_eq!(
            decoded, data,
            "neural-mixer codec must round-trip byte-exact"
        );
    }

    /// Feasibility probe for the residual neural mixer ([`NeuralMixer`]). Encodes
    /// the same enwik8 slice through the real pipeline twice — linear mixer only,
    /// then with the neural refinement stage added — on identical cold-start, and
    /// reports the marginal bpb delta. The stage initializes to pass-through, so a
    /// nonpositive delta means the linear log-odds mix already captures the
    /// exploitable structure in the ensemble's logits. `LZR_LO`/`LZR_HI` slice
    /// (default the 20 MB `[1M..21M]` slice), `LZR_NMIXH` hidden width (default
    /// 32), `LZR_NMIXLR` learning rate (default 0.002). Run:
    /// `cargo test --release nmix_probe -- --ignored --nocapture`
    #[test]
    #[ignore = "neural-mixer probe: residual MLP refinement marginal on enwik8"]
    fn nmix_probe() {
        use std::time::Instant;
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(d)
        };
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let lo = env("LZR_LO", 1_000_000);
        let hi = env("LZR_HI", 21_000_000).min(e8.len());
        let orig = (hi - lo) as f64;
        let data = Pipeline::default_pipeline().forward(&e8[lo..hi]);
        let hm = env("LZR_NMIXH", 32);
        let lr: f32 = std::env::var("LZR_NMIXLR")
            .ok()
            .and_then(|x| x.parse().ok())
            .unwrap_or(0.002);

        // `None` = linear-only baseline; `Some((hm, lr))` = the residual neural mixer.
        let encode_len = |nmix: Option<(usize, f32)>| -> (usize, f64) {
            let t = Instant::now();
            let mut state = CodecState::new(data.len());
            state.nmix = nmix.map(|(hm, lr)| NeuralMixer::new(state.stretched.len(), hm, 1, lr));
            let mut enc = Encoder::with_capacity(data.len());
            for &byte in &data {
                for k in (0..8).rev() {
                    let bit = (byte >> k) & 1;
                    let p = state.predict();
                    enc.encode(bit, p);
                    state.commit(bit);
                }
                state.end_symbol();
            }
            (enc.finish().len(), t.elapsed().as_secs_f64())
        };

        let bpb = |len: usize| len as f64 * 8.0 / orig;
        let (base, base_s) = encode_len(None);
        println!(
            "linear mixer (baseline):       {:.4} bpb  ({base_s:.0}s)",
            bpb(base)
        );
        let (m1, m1_s) = encode_len(Some((hm, lr)));
        println!(
            "+ residual neural mixer h={hm} lr={lr}: {:.4} bpb  ({m1_s:.0}s)  marginal {:+.4}",
            bpb(m1),
            bpb(m1) - bpb(base)
        );
    }

    /// Exploration — a STACK of warming-readout heads over the frozen embedding,
    /// each a separate mixer input, measured over the full shipped stack (M1 + APM
    /// on) on the `[LZR_LO..LZR_HI]` slice (default 10 MB). `LZR_HEADS` is a
    /// comma-separated spec; each token is a context: `N` = byte-order N (`0` =
    /// bit-tree only), `w` = word, `sM` = sparse mask `M` (bit i → `byte_back(i+1)`),
    /// `m` = match state (length bucket, predicted byte — long-range LZP context).
    /// `LZR_HEAD` is the shared lr (default 4), `LZR_HEADBITS` the table size
    /// (default 16). Empty/unset spec = head-free baseline. Run:
    /// `LZR_HEADS=1,w,2 cargo test --release diversity_lab -- --ignored --nocapture`
    #[test]
    #[ignore = "explore: frozen-net warming-head stack marginal over the full stack"]
    #[allow(clippy::cast_precision_loss)]
    fn diversity_lab() {
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(d)
        };
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let lo = env("LZR_LO", 1_000_000);
        let hi = env("LZR_HI", 11_000_000).min(e8.len());
        let orig = (hi - lo) as f64;
        let data = Pipeline::default_pipeline().forward(&e8[lo..hi]);
        let cap = data.len();
        let lr: f32 = std::env::var("LZR_HEAD")
            .ok()
            .and_then(|x| x.parse().ok())
            .unwrap_or(4.0);
        let bits = std::env::var("LZR_HEADBITS")
            .ok()
            .and_then(|x| x.parse().ok())
            .unwrap_or(16u32);
        let spec = std::env::var("LZR_HEADS").unwrap_or_default();

        // A token may carry an optional per-head lr override as `CTX:LR` (e.g.
        // `w:0.5` for a slow word head — dual-rate stacking).
        let build = |spec: &str| -> Vec<AnyModel> {
            let mut v = baseline_models(cap);
            let mut net = PretrainedMlp::from_blob_q8(PRETRAINED_NET);
            for tok in spec.split(',').filter(|t| !t.is_empty()) {
                let (ctx, hlr) = match tok.split_once(':') {
                    Some((c, l)) => (c, l.parse().unwrap()),
                    None => (tok, lr),
                };
                if ctx == "w" {
                    net.push_head_word(hlr, bits);
                } else if ctx == "m" {
                    net.push_head_match(hlr, bits);
                } else if ctx == "m2" {
                    net.push_head_exp(hlr, 1, bits);
                } else if ctx == "col" {
                    net.push_head_exp(hlr, 2, bits);
                } else if ctx == "num" {
                    net.push_head_exp(hlr, 3, bits);
                } else if ctx == "mc" {
                    net.push_head_exp(hlr, 4, bits);
                } else if ctx == "mm" {
                    net.push_head_exp(hlr, 5, bits);
                } else if ctx == "m2c" {
                    net.push_head_exp(hlr, 6, bits);
                } else if let Some(m) = ctx.strip_prefix('s') {
                    net.push_head_sparse(hlr, m.parse().unwrap(), bits);
                } else {
                    net.push_head_ctx(hlr, ctx.parse().unwrap(), bits);
                }
            }
            v.push(net.into());
            v
        };
        let bpb = |m: Vec<AnyModel>| code_stream_models(m, &data).len() as f64 * 8.0 / orig;

        // Base spec (`LZR_HEADS_BASE`, default "" = headless) and the candidate
        // specs (`LZR_HEADS`, ';'-separated) are all encoded in THIS process on
        // THIS `data`, so every marginal is against the same reorder permutation.
        // Comparing across separate cargo runs is NOT clean — the reorder greedy
        // order depends on a per-process HashMap seed, shifting bpb ~±0.001.
        let base_spec = std::env::var("LZR_HEADS_BASE").unwrap_or_default();
        let base = bpb(build(&base_spec));
        println!("base heads=[{base_spec}] lr={lr} bits={bits}:   {base:.4} bpb");
        for cand in spec.split(';').map(str::trim).filter(|s| !s.is_empty()) {
            let v = bpb(build(cand));
            println!(
                "+ heads=[{cand}]:   {v:.4} bpb  marginal-over-base {:+.4}",
                v - base
            );
        }
    }

    /// Full-stack lab for higher-order match models (v7's order-scaling lever):
    /// measures the marginal of extra `MatchModel`s appended to the SHIPPED model
    /// set, on the `[LZR_LO..LZR_HI]` slice. `LZR_MKEYS` is a `;`-separated list of
    /// candidates, each a `,`-separated list of `key[@flat_bits]` specs (e.g.
    /// `LZR_MKEYS="16;12;16@26,12@26"`). Base and candidates run in this process.
    #[test]
    #[ignore = "explore: higher-order match models' marginal over the full stack"]
    fn match_order_lab() {
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(d)
        };
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let lo = env("LZR_LO", 1_000_000);
        let hi = env("LZR_HI", 9_000_000).min(e8.len());
        let orig = (hi - lo) as f64;
        let data = Pipeline::default_pipeline().forward(&e8[lo..hi]);
        let cap = data.len();
        // Token forms: `KEY[@BITS]` = plain flat-finder match (append);
        // `vKEY[@BUCKET_BITS[:VCAP]]` = verified multi-candidate match.
        // `LZR_MREPL=1` replaces the primary (key-8) match instead of appending.
        let repl = std::env::var("LZR_MREPL").is_ok();
        let build = |spec: &str| -> Vec<AnyModel> {
            let mut v = models(cap);
            for tok in spec.split(',').map(str::trim).filter(|t| !t.is_empty()) {
                #[allow(clippy::option_if_let_else)]
                let m = if let Some(rest) = tok.strip_prefix('v') {
                    let (key, cfg) = rest.split_once('@').unwrap_or((rest, "25:48"));
                    let (bits, vcap) = cfg.split_once(':').unwrap_or((cfg, "48"));
                    MatchModel::verified(
                        key.parse().unwrap(),
                        bits.parse().unwrap(),
                        vcap.parse().unwrap(),
                    )
                } else {
                    let (key, bits) = match tok.split_once('@') {
                        Some((k, b)) => (k.parse().unwrap(), b.parse().unwrap()),
                        None => (tok.parse().unwrap(), 27),
                    };
                    MatchModel::with_key_bits(key, bits)
                };
                if repl {
                    let i = v
                        .iter()
                        .position(|m| matches!(m, AnyModel::Match(_)))
                        .unwrap();
                    v[i] = m.into();
                } else {
                    v.push(m.into());
                }
            }
            v
        };
        let bpb = |m: Vec<AnyModel>| code_stream_models(m, &data).len() as f64 * 8.0 / orig;
        // `LZR_MBASE=<bpb>` substitutes a known base figure instead of re-encoding
        // it — sound across processes now that reorder is deterministic.
        let base = std::env::var("LZR_MBASE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| {
                let b = bpb(build(""));
                println!("base (shipped models):   {b:.4} bpb");
                b
            });
        let spec = std::env::var("LZR_MKEYS").unwrap_or_else(|_| "16".into());
        for cand in spec.split(';').map(str::trim).filter(|s| !s.is_empty()) {
            let v = bpb(build(cand));
            println!(
                "+ match[{cand}]:   {v:.4} bpb  marginal-over-base {:+.4}",
                v - base
            );
        }
    }

    // ---- preprocessor lab: runtime pipeline experiments on the linear mixer ----
    // M1 is off here (`nmix = None`): ~4x faster per encode, and directly comparable
    // to the journal's pre-M1 preprocessor numbers. Keepers get re-checked with the
    // full stack before shipping.

    /// Deterministic codec (linear mixer, M1 off) over already-preprocessed `data`
    /// with an explicit model set; returns the coded byte length.
    fn encode_linear_models(models: Vec<AnyModel>, data: &[u8]) -> usize {
        let mut state = CodecState::with_models(models, data.len());
        state.nmix = None;
        let mut enc = Encoder::with_capacity(data.len());
        for &byte in data {
            for k in (0..8).rev() {
                let bit = (byte >> k) & 1;
                let p = state.predict();
                enc.encode(bit, p);
                state.commit(bit);
            }
            state.end_symbol();
        }
        enc.finish().len()
    }

    /// As [`encode_linear_models`] with the shipped deterministic set. Shared by the
    /// prep-lab pipeline sweeps.
    fn encode_det_linear(data: &[u8]) -> usize {
        encode_linear_models(baseline_models(data.len()), data)
    }

    /// Exploration — marginal of candidate sparse-context masks on top of the
    /// shipped stack (linear mixer). `mask` bit `i` reads `byte_back(i+1)`. Run:
    /// `cargo test --release sparse_lab -- --ignored --nocapture`
    #[test]
    #[ignore = "explore: candidate sparse-context masks (marginal over the stack)"]
    #[allow(clippy::cast_precision_loss, clippy::unreadable_literal)]
    fn sparse_lab() {
        use crate::models::context::ContextModel;
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(d)
        };
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let lo = env("LZR_LO", 1_000_000);
        let hi = env("LZR_HI", 21_000_000).min(e8.len());
        let orig = (hi - lo) as f64;
        let data = Pipeline::default_pipeline().forward(&e8[lo..hi]);
        let base = encode_det_linear(&data);
        let base_bpb = base as f64 * 8.0 / orig;
        println!("baseline (shipped stack): {base_bpb:.4} bpb");
        // (mask, label) — untested patterns; shipped are {1,3}{2,3}{1,2,4}{3,4}{1,5}.
        let cands: [(u32, &str); 6] = [
            (0b1001, "{1,4}"),
            (0b1010, "{2,4}"),
            (0b10010, "{2,5}"),
            (0b100001, "{1,6}"),
            (0b10101, "{1,3,5}"),
            (0b100011, "{1,2,6}"),
        ];
        for (mask, label) in cands {
            let mut models = baseline_models(data.len());
            models.push(ContextModel::sparse(mask, data.len()).into());
            let coded = encode_linear_models(models, &data);
            let bpb = coded as f64 * 8.0 / orig;
            println!(
                "+ sparse {label:8} {bpb:.4} bpb  marginal {:+.4}",
                bpb - base_bpb
            );
        }
    }

    /// Offline probe — the break-exclusion idea (the encoder knows where every
    /// match breaks). Flag long-match breaks and let the primary match model
    /// abstain there (a stand-in for a side channel that tells the decoder "this
    /// match breaks here"), measure the GROSS coded-bit gain, then weigh it against
    /// the side channel's own entropy cost (one continue/break decision per
    /// active-long-match byte). Abstaining is a LOWER bound on the gain — true
    /// exclusion coding (renormalize without the predicted byte) would do better —
    /// so a negative net here is a real negative. Sweeps the length threshold T.
    /// Run: `cargo test --release match_break_lab -- --ignored --nocapture`
    #[test]
    #[ignore = "explore: break-exclusion gross gain vs side-channel cost"]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::suboptimal_flops
    )]
    fn match_break_lab() {
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(d)
        };
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let lo = env("LZR_LO", 1_000_000);
        let hi = env("LZR_HI", 9_000_000).min(e8.len());
        let orig = (hi - lo) as f64;
        let data = Pipeline::default_pipeline().forward(&e8[lo..hi]);
        let n = data.len();

        // Trace pass: drive a lone primary match model (key=8) over the
        // preprocessed stream, recording its pre-byte (length bucket, predicted
        // byte) at each byte. The match model's state depends only on the byte
        // history, so this is exactly what the codec's match model sees.
        let mut mm = MatchModel::new();
        let mut ctx = Context::with_capacity(n);
        let mut lp: Vec<(u32, u8)> = Vec::with_capacity(n);
        for &byte in &data {
            lp.push(mm.match_key(&ctx));
            for k in (0..8).rev() {
                let bit = (byte >> k) & 1;
                let _ = mm.predict(&ctx);
                ctx.push_bit(bit);
                mm.update(&ctx, bit);
            }
            ctx.push_byte();
        }

        let base = encode_det_linear(&data);
        let base_bpb = base as f64 * 8.0 / orig;
        println!(
            "baseline (det linear): {base_bpb:.4} bpb  ({base} B over {:.1} MB)",
            orig / 1e6
        );

        let binent = |q: f64| -> f64 {
            if q <= 0.0 || q >= 1.0 {
                0.0
            } else {
                -q * q.log2() - (1.0 - q) * (1.0 - q).log2()
            }
        };

        for t in [2u32, 4, 6, 8, 12, 16] {
            let mut flags = vec![false; n];
            let mut n_long = 0u64;
            let mut n_break = 0u64;
            for i in 0..n {
                let (len, pb) = lp[i];
                if len >= t {
                    n_long += 1;
                    if pb != data[i] {
                        flags[i] = true;
                        n_break += 1;
                    }
                }
            }
            let flags: std::rc::Rc<[bool]> = flags.into();
            let mut models = baseline_models(n);
            for m in &mut models {
                if let AnyModel::Match(mm) = m {
                    mm.set_abstain(flags.clone(), t);
                    break; // primary (key=8) only
                }
            }
            let treat = encode_linear_models(models, &data);
            let gross = base as f64 - treat as f64; // bytes saved
            let q = n_break as f64 / n_long.max(1) as f64;
            let sidech_bits = n_long as f64 * binent(q); // idealized adaptive coder
            let net = gross - sidech_bits / 8.0;
            println!(
                "T={t:2}  long {:>8} ({:>4.1}% bytes)  breaks {:>7} (q={:.4})  \
                 gross {:+.4}  sidech {:.4}  NET {:+.4} bpb",
                n_long,
                100.0 * n_long as f64 / n as f64,
                n_break,
                q,
                gross * 8.0 / orig,
                sidech_bits / orig,
                net * 8.0 / orig,
            );
        }
    }

    /// Offline: dump the preprocessed (post-pipeline) enwik8 — what every model,
    /// including a pretrained net, actually sees — for GPU pretraining (the burn
    /// `pretrain` example reads it). `LZR_PP_OUT` sets the path. Run:
    /// `LZR_PP_OUT=/tmp/lzr_pp.bin cargo test --release dump_preprocessed -- --ignored --nocapture`
    #[test]
    #[ignore = "offline: dump preprocessed enwik8 for net pretraining"]
    fn dump_preprocessed() {
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let out = std::env::var("LZR_PP_OUT").unwrap_or_else(|_| "/tmp/lzr_pp.bin".into());
        // `LZR_CF_ONLY=1` dumps casefold-only (no word dict) — easy data for a
        // training sanity check (a working net must beat unigram on it).
        let pp = if std::env::var("LZR_CF_ONLY").is_ok() {
            use crate::preprocessors::Preprocessor;
            crate::preprocessors::casefold::CaseFold.forward(&e8)
        } else {
            Pipeline::default_pipeline().forward(&e8)
        };
        std::fs::write(&out, &pp).unwrap();
        println!("wrote {} preprocessed bytes to {out}", pp.len());
    }

    /// Offline: dump a `casefold+word_dict` (NO reorder — a mid-corpus slice isn't
    /// whole `<page>` articles) preprocessed slice of enwik9, the held-out
    /// robustness probe for the Stage-0 backbone bake-off. `LZR_EVAL_OUT` path,
    /// `LZR_EVAL_LO`/`LZR_EVAL_HI` bounds (default 200M..210M, beyond enwik8). Run:
    /// `LZR_EVAL_OUT=/tmp/lzr_eval.bin cargo test --release dump_eval_slice -- --ignored --nocapture`
    #[test]
    #[ignore = "offline: dump a casefold+word_dict enwik9 slice for the bake-off"]
    fn dump_eval_slice() {
        use crate::preprocessors::Preprocessor;
        use crate::preprocessors::casefold::CaseFold;
        use crate::preprocessors::word_dict::WordDict;
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(d)
        };
        let Ok(e9) = std::fs::read("assets/enwik9") else {
            return;
        };
        let lo = env("LZR_EVAL_LO", 200_000_000).min(e9.len());
        let hi = env("LZR_EVAL_HI", 210_000_000).min(e9.len());
        let out = std::env::var("LZR_EVAL_OUT").unwrap_or_else(|_| "/tmp/lzr_eval.bin".into());
        let folded = CaseFold.forward(&e9[lo..hi]);
        let pp = WordDict::embedded().forward(&folded);
        std::fs::write(&out, &pp).unwrap();
        println!(
            "wrote {} preprocessed bytes ({lo}..{hi}) to {out}",
            pp.len()
        );
    }

    /// E1 — dict-N sweep: deterministic bpb (per original byte) and the post-dict
    /// stream length (the arm-speed proxy) as the word dictionary grows. Words are
    /// mined from full case-folded enwik8 at runtime (no rebuild); the
    /// `[LZR_LO..LZR_HI]` slice (default 20 MB) is encoded with each. Run:
    /// `cargo test --release dict_n_lab -- --ignored --nocapture`
    #[test]
    #[ignore = "prep-lab: dict-N sweep (bpb + stream length / speed proxy)"]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn dict_n_lab() {
        use crate::preprocessors::Preprocessor;
        use crate::preprocessors::casefold::CaseFold;
        use crate::preprocessors::word_dict::{WordDict, min_word_len};
        use std::collections::HashMap;
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(d)
        };
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let lo = env("LZR_LO", 1_000_000);
        let hi = env("LZR_HI", 21_000_000).min(e8.len());
        let orig = (hi - lo) as f64;

        let folded: &'static [u8] = Box::leak(CaseFold.forward(&e8).into_boxed_slice());
        let mut freq: HashMap<&'static [u8], u32> = HashMap::new();
        let mut i = 0;
        while i < folded.len() {
            if folded[i].is_ascii_lowercase() {
                let s = i;
                while i < folded.len() && folded[i].is_ascii_lowercase() {
                    i += 1;
                }
                *freq.entry(&folded[s..i]).or_insert(0) += 1;
            } else {
                i += 1;
            }
        }
        let mut sorted: Vec<(&'static [u8], u32)> = freq.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        let sorted: Vec<&'static [u8]> = sorted.into_iter().map(|(w, _)| w).collect();

        let folded_slice = CaseFold.forward(&e8[lo..hi]);
        let fold_len = folded_slice.len() as f64;
        println!("slice [{lo}..{hi}]  folded {} bytes", folded_slice.len());
        for &n in &[0usize, 2000, 4000, 8000, 16000, 32000] {
            let mut words: Vec<&'static [u8]> = Vec::new();
            for &w in &sorted {
                if words.len() >= n {
                    break;
                }
                if w.len() >= min_word_len(words.len()) {
                    words.push(w);
                }
            }
            let dict = WordDict::from_words(&words);
            let data = dict.forward(&folded_slice);
            let coded = encode_det_linear(&data);
            let bpb = coded as f64 * 8.0 / orig;
            let shrink = 100.0 * (1.0 - data.len() as f64 / fold_len);
            println!(
                "N={n:6}  bpb {bpb:.4}  stream {:9} ({shrink:4.1}% < fold)",
                data.len()
            );
        }
    }

    /// Dictionary-size sweep: no-dict vs static-N vs dynamic-frozen-N, on the
    /// net-free linear path, to find each curve's knee. The static dict is mined
    /// from the data with global frequency foresight and shipped (2× L(D)); the
    /// dynamic dict is built online — first-N distinct words in appearance order,
    /// then frozen (no ID reuse) — and ships nothing (1× L(C)). Defaults to FULL
    /// enwik8 (`LZR_LO=0 LZR_HI=100000000`): on a large slice the dynamic dict fills
    /// in the first ~1% and is effectively static for the rest, so warmup does not
    /// dominate the number (the failure mode on small slices). Round-trip asserted
    /// for the dynamic dict. Run (override the slice for a quick check):
    /// `LZR_LO=1000000 LZR_HI=9000000 cargo test --release online_dict_lab -- --ignored --nocapture`
    #[test]
    #[ignore = "prep-lab: online LRU dict vs static dict (bpb gap = price of generality)"]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn online_dict_lab() {
        use crate::preprocessors::Preprocessor;
        use crate::preprocessors::casefold::CaseFold;
        use crate::preprocessors::online_dict::OnlineDict;
        use crate::preprocessors::word_dict::{WordDict, min_word_len};
        use std::collections::HashMap;
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(d)
        };
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let lo = env("LZR_LO", 0);
        let hi = env("LZR_HI", 100_000_000).min(e8.len());
        let orig = (hi - lo) as f64;

        // Static-dict word list: global frequencies over all of case-folded enwik8.
        let folded: &'static [u8] = Box::leak(CaseFold.forward(&e8).into_boxed_slice());
        let mut freq: HashMap<&'static [u8], u32> = HashMap::new();
        let mut i = 0;
        while i < folded.len() {
            if folded[i].is_ascii_lowercase() {
                let s = i;
                while i < folded.len() && folded[i].is_ascii_lowercase() {
                    i += 1;
                }
                *freq.entry(&folded[s..i]).or_insert(0) += 1;
            } else {
                i += 1;
            }
        }
        let mut sorted: Vec<(&'static [u8], u32)> = freq.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        let sorted: Vec<&'static [u8]> = sorted.into_iter().map(|(w, _)| w).collect();

        let fold_slice: &'static [u8] = Box::leak(CaseFold.forward(&e8[lo..hi]).into_boxed_slice());
        let nodict_bpb = encode_det_linear(fold_slice) as f64 * 8.0 / orig;

        // Appearance-order distinct words (the dynamic dict's word *choice*). Pairing
        // these with the static machinery (stable codes from byte 0) isolates word
        // selection from online non-stationarity.
        let mut seen: std::collections::HashSet<&'static [u8]> = std::collections::HashSet::new();
        let mut appeared: Vec<&'static [u8]> = Vec::new();
        let mut i = 0;
        while i < fold_slice.len() {
            if fold_slice[i].is_ascii_lowercase() {
                let s = i;
                while i < fold_slice.len() && fold_slice[i].is_ascii_lowercase() {
                    i += 1;
                }
                let w = &fold_slice[s..i];
                if seen.insert(w) {
                    appeared.push(w);
                }
            } else {
                i += 1;
            }
        }
        println!(
            "slice [{lo}..{hi}]  folded {} bytes  |  no-dict {nodict_bpb:.4} bpb",
            fold_slice.len(),
        );
        println!("         static-freq      static-appear     dynamic-frozen    | select   online");
        // Pick the first N words from an ordered list, with the same length gating
        // the static dict uses (a word must be long enough for its code slot).
        let pick = |order: &[&'static [u8]], n: usize| -> Vec<&'static [u8]> {
            let mut w: Vec<&'static [u8]> = Vec::new();
            for &x in order {
                if w.len() >= n {
                    break;
                }
                if x.len() >= min_word_len(w.len()) {
                    w.push(x);
                }
            }
            w
        };
        for &n in &[1000usize, 2000, 4000, 8000, 16000, 32000] {
            let sf = encode_det_linear(&WordDict::from_words(&pick(&sorted, n)).forward(fold_slice))
                as f64
                * 8.0
                / orig;
            let sa =
                encode_det_linear(&WordDict::from_words(&pick(&appeared, n)).forward(fold_slice))
                    as f64
                    * 8.0
                    / orig;
            let dyn_ = OnlineDict::frozen(n).forward(fold_slice);
            let dy = encode_det_linear(&dyn_) as f64 * 8.0 / orig;
            assert_eq!(
                OnlineDict::frozen(n).inverse(&dyn_),
                fold_slice,
                "frozen dynamic dict must round-trip"
            );
            println!(
                "N={n:5}  {sf:7.4} ({:+.4})  {sa:7.4} ({:+.4})  {dy:7.4} ({:+.4})  | {:+.4}  {:+.4}",
                sf - nodict_bpb,
                sa - nodict_bpb,
                dy - nodict_bpb,
                sa - sf, // word-selection cost (appearance vs frequency, both static)
                dy - sa, // online non-stationarity cost (same words, online codes)
            );
        }
    }

    /// E2 — phrase dictionary: do boundary-crossing phrases earn dictionary code
    /// slots over plain words? Mines words (whole corpus) and phrases (n-grams with
    /// ≥1 non-letter, over a sample, freq-filtered), scores both by savings
    /// (freq × (len−1)), then compares, under one `LZR_NENT` code budget: A =
    /// words-only vs B = words+phrases competing for the same slots. Reports bpb
    /// (per original byte, linear mixer) + stream length (speed proxy), round-trip
    /// asserted. `LZR_PMAX` max phrase len, `LZR_PMIN` min phrase freq,
    /// `LZR_PSAMPLE` mining sample. Run:
    /// `cargo test --release phrase_lab -- --ignored --nocapture`
    #[test]
    #[ignore = "prep-lab: phrase dictionary (words+phrases) vs words-only"]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::too_many_lines
    )]
    fn phrase_lab() {
        use crate::preprocessors::Preprocessor;
        use crate::preprocessors::casefold::CaseFold;
        use crate::preprocessors::dictionary::GDict;
        use crate::preprocessors::word_dict::min_word_len;
        use std::collections::HashMap;
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(d)
        };
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let lo = env("LZR_LO", 1_000_000);
        let hi = env("LZR_HI", 21_000_000).min(e8.len());
        let orig = (hi - lo) as f64;
        let folded: &'static [u8] = Box::leak(CaseFold.forward(&e8).into_boxed_slice());

        // words: maximal a-z runs over the whole folded corpus.
        let mut wfreq: HashMap<&'static [u8], u32> = HashMap::new();
        let mut i = 0;
        while i < folded.len() {
            if folded[i].is_ascii_lowercase() {
                let s = i;
                while i < folded.len() && folded[i].is_ascii_lowercase() {
                    i += 1;
                }
                *wfreq.entry(&folded[s..i]).or_insert(0) += 1;
            } else {
                i += 1;
            }
        }
        // phrases: n-grams (2..=PMAX) with >=1 non-lowercase byte (complementary to
        // words), over a sample, kept if frequent enough.
        let pmax = env("LZR_PMAX", 6);
        let pminlen = env("LZR_PMINLEN", 3); // skip junk bigrams by default
        let pmin = env("LZR_PMIN", 50) as u32;
        let sample = env("LZR_PSAMPLE", 10_000_000).min(folded.len());
        let mut pfreq: HashMap<&'static [u8], u32> = HashMap::new();
        for n in pminlen..=pmax {
            for j in 0..sample.saturating_sub(n) {
                let g = &folded[j..j + n];
                if g.iter().all(u8::is_ascii_lowercase) {
                    continue;
                }
                *pfreq.entry(g).or_insert(0) += 1;
            }
        }
        // Frequency-order (word_dict's order) — beats savings-order for bpb (the
        // dict's value is canonicalizing the *most common* tokens, not removing the
        // most bytes).
        let score = |_s: &[u8], f: u32| u64::from(f);
        let mut words: Vec<(&'static [u8], u64)> =
            wfreq.iter().map(|(&w, &f)| (w, score(w, f))).collect();
        let mut phrases: Vec<(&'static [u8], u64)> = pfreq
            .iter()
            .filter(|&(_, &f)| f >= pmin)
            .map(|(&g, &f)| (g, score(g, f)))
            .collect();
        words.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        phrases.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

        let nent = env("LZR_NENT", 4000);
        let build = |cands: &[(&'static [u8], u64)], cap: usize| -> Vec<&'static [u8]> {
            let mut v: Vec<&'static [u8]> = Vec::new();
            for &(e, _) in cands {
                if v.len() >= cap {
                    break;
                }
                if e.len() >= min_word_len(v.len()) {
                    v.push(e);
                }
            }
            v
        };

        let folded_slice = CaseFold.forward(&e8[lo..hi]);
        let fold_len = folded_slice.len() as f64;
        let run = |label: &str, entries: &[&'static [u8]]| {
            let dict = GDict::new(entries);
            let data = dict.forward(&folded_slice);
            assert_eq!(dict.inverse(&data), folded_slice, "{label} must round-trip");
            let coded = encode_det_linear(&data);
            let bpb = coded as f64 * 8.0 / orig;
            let shrink = 100.0 * (1.0 - data.len() as f64 / fold_len);
            println!(
                "{label:30} entries {:6}  bpb {bpb:.4}  stream {:9} ({shrink:4.1}% < fold)",
                entries.len(),
                data.len()
            );
        };

        println!(
            "mined {} words, {} phrases (freq>={pmin}, len<= {pmax}, sample {sample})",
            words.len(),
            phrases.len()
        );
        let a = build(&words, nent);
        run("A words-only", &a);
        // B: words+phrases compete for one nent budget (freq-order).
        let mut merged = words.clone();
        merged.extend_from_slice(&phrases);
        merged.sort_by(|x, y| y.1.cmp(&x.1).then_with(|| x.0.cmp(y.0)));
        let b = build(&merged, nent);
        let nph = b
            .iter()
            .filter(|e| e.iter().any(|c| !c.is_ascii_lowercase()))
            .count();
        run(&format!("B merged-budget ({nph} ph)"), &b);
        // C: words(nent) + top phrases APPENDED (higher codes, don't displace
        // words) — the fairest "do phrases add on top of the word dict" test.
        let np = env("LZR_NPHRASE", 2000);
        let mut c = a.clone();
        for &(e, _) in &phrases {
            if c.len() >= nent + np {
                break;
            }
            if e.len() >= min_word_len(c.len()) {
                c.push(e);
            }
        }
        run(&format!("C words+{}ph-appended", c.len() - a.len()), &c);
    }

    /// E10 — word+trailing-space dictionary: absorb the post-word space into the
    /// code (still WHOLE-word matching, no sub-word fragmentation). Inter-word
    /// spaces are ~15% of the stream, so this is a clean speed lever; the bpb
    /// question is whether folding the (cheap but non-zero) space helps. Mines word
    /// and word+space freqs, freq-orders, compares A=words-only (≈ word_dict, a
    /// validation that a-z-run matching reproduces the baseline) vs B=words+wordspace
    /// under one `LZR_NENT` budget. Run:
    /// `cargo test --release wordspace_lab -- --ignored --nocapture`
    #[test]
    #[ignore = "prep-lab: word+trailing-space dictionary vs words-only"]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::similar_names,
        clippy::doc_markdown
    )]
    fn wordspace_lab() {
        use crate::preprocessors::Preprocessor;
        use crate::preprocessors::casefold::CaseFold;
        use crate::preprocessors::dictionary::WSDict;
        use crate::preprocessors::word_dict::min_word_len;
        use std::collections::HashMap;
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(d)
        };
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let lo = env("LZR_LO", 1_000_000);
        let hi = env("LZR_HI", 21_000_000).min(e8.len());
        let orig = (hi - lo) as f64;
        let folded: &'static [u8] = Box::leak(CaseFold.forward(&e8).into_boxed_slice());

        let mut wfreq: HashMap<&'static [u8], u32> = HashMap::new();
        let mut wsfreq: HashMap<&'static [u8], u32> = HashMap::new();
        let mut i = 0;
        while i < folded.len() {
            if folded[i].is_ascii_lowercase() {
                let s = i;
                while i < folded.len() && folded[i].is_ascii_lowercase() {
                    i += 1;
                }
                *wfreq.entry(&folded[s..i]).or_insert(0) += 1;
                if i < folded.len() && folded[i] == b' ' {
                    *wsfreq.entry(&folded[s..=i]).or_insert(0) += 1; // run + the space
                }
            } else {
                i += 1;
            }
        }
        let mut words: Vec<(&'static [u8], u32)> = wfreq.iter().map(|(&w, &f)| (w, f)).collect();
        let mut both: Vec<(&'static [u8], u32)> = words.clone();
        both.extend(wsfreq.iter().map(|(&w, &f)| (w, f)));
        let by_freq = |v: &mut Vec<(&'static [u8], u32)>| {
            v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        };
        by_freq(&mut words);
        by_freq(&mut both);

        let nent = env("LZR_NENT", 4000);
        let build = |cands: &[(&'static [u8], u32)]| -> Vec<&'static [u8]> {
            let mut v: Vec<&'static [u8]> = Vec::new();
            for &(e, _) in cands {
                if v.len() >= nent {
                    break;
                }
                if e.len() >= min_word_len(v.len()) {
                    v.push(e);
                }
            }
            v
        };

        let folded_slice = CaseFold.forward(&e8[lo..hi]);
        let fold_len = folded_slice.len() as f64;
        let run = |label: &str, entries: &[&'static [u8]]| {
            let dict = WSDict::new(entries);
            let data = dict.forward(&folded_slice);
            assert_eq!(dict.inverse(&data), folded_slice, "{label} must round-trip");
            let coded = encode_det_linear(&data);
            let bpb = coded as f64 * 8.0 / orig;
            let shrink = 100.0 * (1.0 - data.len() as f64 / fold_len);
            println!(
                "{label:28} entries {:6}  bpb {bpb:.4}  stream {:9} ({shrink:4.1}% < fold)",
                entries.len(),
                data.len()
            );
        };

        let a = build(&words);
        run("A words-only", &a);
        let b = build(&both);
        let nws = b.iter().filter(|e| e.last() == Some(&b' ')).count();
        run(&format!("B words+wordspace ({nws} ws)"), &b);
    }

    /// Exploration — frozen pretrained MLP byte-LM (GPU-trained by the `pretrain`
    /// example): its marginal over the shipped stack (linear mixer) and its `L(D)`.
    /// Loads the weight blob `LZR_NET` (default `/tmp/lzr_net.bin`); skips if
    /// absent. The net is trained on the corpus it compresses (the real Hutter
    /// setup), so the slice being in its training data is legitimate. Reports
    /// net-of-L(D) for f32 and ternary shipping. Run:
    /// `cargo test --release pretrained_lab -- --ignored --nocapture`
    #[test]
    #[ignore = "explore: frozen pretrained MLP marginal + L(D)"]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn pretrained_lab() {
        use crate::models::pretrained::PretrainedMlp;
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(d)
        };
        let net = std::env::var("LZR_NET").unwrap_or_else(|_| "/tmp/lzr_net.bin".into());
        let Ok(blob) = std::fs::read(&net) else {
            println!("no net blob at {net}; skipping");
            return;
        };
        let corpus = std::env::var("LZR_CORPUS").unwrap_or_else(|_| "assets/enwik8".into());
        let Ok(e8) = std::fs::read(&corpus) else {
            return;
        };
        let lo = env("LZR_LO", 1_000_000);
        let hi = env("LZR_HI", 11_000_000).min(e8.len());
        let orig = (hi - lo) as f64;
        let data = Pipeline::default_pipeline().forward(&e8[lo..hi]);
        // `LZR_QBITS` (default 32 = f32): quantize the big weight tensors to measure
        // the true low-bit marginal (not assume f32's survives) + the shipped L(D).
        let qbits = env("LZR_QBITS", 32) as u32;
        let base = encode_det_linear(&data);
        let base_bpb = base as f64 * 8.0 / orig;
        let mut pre = PretrainedMlp::from_blob(&blob);
        pre.quantize(qbits);
        let mut models = baseline_models(data.len());
        models.push(pre.into());
        let coded = encode_linear_models(models, &data);
        let bpb = coded as f64 * 8.0 / orig;
        let marginal = bpb - base_bpb;
        // Shipped-blob L(D): big tensors (emb,w1,w2) at `qbits`, biases at f32.
        let rd = |o: usize| u32::from_le_bytes(blob[o..o + 4].try_into().unwrap()) as usize;
        let (k, e, h, v) = (rd(0), rd(4), rd(8), rd(12));
        let big = v * e + k * e * h + h * v;
        let small = h + v;
        let shipped = if qbits >= 32 {
            blob.len()
        } else {
            16 + (big * qbits as usize).div_ceil(8) + small * 4
        };
        let ld = 16.0 * shipped as f64 / 1e9;
        println!(
            "baseline (linear) {base_bpb:.4}  +pretrained(q{qbits}) {bpb:.4}  marginal {marginal:+.4}"
        );
        println!(
            "net: {} params | shipped {shipped} B (q{qbits}) | L(D) {ld:.5}",
            big + small
        );
        println!("NET of L(D): {:+.4}", marginal + ld);
    }

    /// Authoritative pretrained-net marginal: the same frozen MLP measured over the
    /// **full production stack** (logistic multi-mixer + M1 residual mixer + APM)
    /// via [`code_stream_models`], not the linear panel `pretrained_lab` uses. The
    /// panel overstates an MLP's marginal over a linear mix 5–10× (see JOURNAL), so
    /// only this number gates a ship decision. Computes the production baseline
    /// once, then the f32 and 8-bit net marginals (with their `L(D)`) against it.
    /// Run held-out (the conservative gate) with `LZR_CORPUS=assets/enwik9
    /// LZR_LO=200000000 LZR_HI=210000000`; defaults to the in-distribution
    /// enwik8 `[1M..11M]` slice. `cargo test --release pretrained_e2e -- --ignored
    /// --nocapture`.
    #[test]
    #[ignore = "explore: frozen pretrained MLP marginal over the FULL production stack"]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn pretrained_e2e() {
        use crate::models::pretrained::PretrainedMlp;
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(d)
        };
        let net = std::env::var("LZR_NET").unwrap_or_else(|_| "/tmp/lzr_net.bin".into());
        let Ok(blob) = std::fs::read(&net) else {
            println!("no net blob at {net}; skipping");
            return;
        };
        let corpus = std::env::var("LZR_CORPUS").unwrap_or_else(|_| "assets/enwik8".into());
        let Ok(e8) = std::fs::read(&corpus) else {
            return;
        };
        let lo = env("LZR_LO", 1_000_000);
        let hi = env("LZR_HI", 11_000_000).min(e8.len());
        let orig = (hi - lo) as f64;
        let data = Pipeline::default_pipeline().forward(&e8[lo..hi]);
        // Full production stack (logistic mixer + M1 + APM), not the linear panel.
        let base = code_stream_models(baseline_models(data.len()), &data).len();
        let base_bpb = base as f64 * 8.0 / orig;
        println!(
            "baseline (production stack) {base_bpb:.4}  [{base} B over {} pp-bytes]",
            data.len()
        );
        let rd = |o: usize| u32::from_le_bytes(blob[o..o + 4].try_into().unwrap()) as usize;
        let (k, e, h, v) = (rd(0), rd(4), rd(8), rd(12));
        let big = v * e + k * e * h + h * v;
        let small = h + v;
        for qbits in [32u32, 8] {
            let mut pre = PretrainedMlp::from_blob(&blob);
            pre.quantize(qbits);
            let mut models = baseline_models(data.len());
            models.push(pre.into());
            let coded = code_stream_models(models, &data).len();
            let bpb = coded as f64 * 8.0 / orig;
            let marginal = bpb - base_bpb;
            let shipped = if qbits >= 32 {
                blob.len()
            } else {
                16 + (big * qbits as usize).div_ceil(8) + small * 4
            };
            let ld = 16.0 * shipped as f64 / 1e9;
            println!(
                "  +pretrained(q{qbits}) {bpb:.4}  marginal {marginal:+.4} | shipped {shipped} B | L(D) {ld:.5} | NET {:+.4}",
                marginal + ld
            );
        }
    }

    /// Idea 3 probe: deterministic-stack bpb of a raw byte file (`LZR_FILE`), for
    /// comparing original vs similarity-reordered article order. Same articles, same
    /// bytes → the bpb delta is the reorder's compression gain. The article
    /// permutation itself ships free (enwik order is page-id-monotonic, restored by
    /// sorting decoded articles on their `<id>`). Deterministic stack only (the
    /// reorder gain is match/context locality, the dominant mechanism — fast). Run:
    /// `LZR_FILE=/path cargo test --release reorder_lab -- --ignored --nocapture`
    #[test]
    #[ignore = "idea 3: deterministic bpb of a (re)ordered raw article file"]
    fn reorder_lab() {
        let Ok(path) = std::env::var("LZR_FILE") else {
            return;
        };
        let Ok(raw) = std::fs::read(&path) else {
            return;
        };
        let data = Pipeline::default_pipeline().forward(&raw);
        // LZR_FULLSTACK adds the frozen net + 8 warming heads (mirrors models()
        // minus the arm) — to check whether reordering, which the K=32 net was not
        // trained on, hurts the frozen net (it should not: reorder permutes whole
        // articles, leaving within-article 32-byte windows byte-identical).
        let mut models = baseline_models(data.len());
        let full = std::env::var("LZR_FULLSTACK").is_ok();
        if full {
            let mut net = PretrainedMlp::from_blob_q8(PRETRAINED_NET);
            net.push_head_ctx(HEAD_LR, 1, HEAD_BITS);
            net.push_head_word(HEAD_LR, HEAD_BITS);
            net.push_head_ctx(HEAD_LR, 2, HEAD_BITS);
            net.push_head_sparse(HEAD_LR, 0b110, HEAD_BITS);
            net.push_head_ctx(HEAD_SLOW_LR, 1, HEAD_BITS);
            net.push_head_match(HEAD_MATCH_LR, HEAD_BITS);
            net.push_head_match2(HEAD_MATCH_LR, HEAD_BITS);
            net.push_head_match_prev(HEAD_MATCH_LR, HEAD_BITS);
            models.push(net.into());
        }
        let coded = code_stream_models(models, &data).len();
        #[allow(clippy::cast_precision_loss)]
        let bpb = coded as f64 * 8.0 / raw.len() as f64;
        let tag = if full {
            "full-stack net+heads"
        } else {
            "deterministic"
        };
        println!(
            "{path}: {} raw -> {} folded -> {coded} coded = {bpb:.4} bpb ({tag})",
            raw.len(),
            data.len(),
        );
    }

    /// Idea 3 end-to-end: reorder an enwik8 window with the Rust [`Reorder`] stage
    /// (greedy TF-IDF similarity), assert it round-trips, and report deterministic
    /// bpb original vs reordered plus the ordering wall-time (projects enwik9
    /// feasibility — the ordering runs once at encode). `LZR_LO`/`LZR_HI` slice. Run:
    /// `cargo test --release reorder_e2e -- --ignored --nocapture`
    #[test]
    #[ignore = "idea 3: Rust reorder gain + round-trip + ordering time on an enwik8 window"]
    fn reorder_e2e() {
        use crate::preprocessors::Preprocessor;
        use crate::preprocessors::reorder::Reorder;
        use std::time::Instant;
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(d)
        };
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let lo = env("LZR_LO", 1_000_000);
        let hi = env("LZR_HI", 11_000_000).min(e8.len());
        let raw = &e8[lo..hi];
        let t = Instant::now();
        let reordered = Reorder.forward(raw);
        let order_s = t.elapsed().as_secs_f64();
        assert_eq!(Reorder.inverse(&reordered), raw, "reorder must round-trip");
        // LZR_NOENC: time the ordering + prove round-trip at scale, skip the encode.
        if std::env::var("LZR_NOENC").is_ok() {
            println!(
                "[{lo}..{hi}] {:.0} MB | round-trip OK | ordering {order_s:.1}s",
                (hi - lo) as f64 / 1e6,
            );
            return;
        }
        #[allow(clippy::cast_precision_loss)]
        let bpb = |bytes: &[u8]| {
            let d = Pipeline::default_pipeline().forward(bytes);
            code_stream_models(baseline_models(d.len()), &d).len() as f64 * 8.0 / raw.len() as f64
        };
        let bo = bpb(raw);
        let br = bpb(&reordered);
        println!(
            "[{lo}..{hi}] {:.0} MB | orig {bo:.4}  reorder {br:.4}  marginal {:+.4} | \
             ordering {order_s:.1}s",
            (hi - lo) as f64 / 1e6,
            br - bo,
        );
    }

    /// Tool: write `LZR_OUT` = `Reorder.forward(read(LZR_IN))` (idea 3) — a reordered
    /// enwik file to encode with the shipped binary. Asserts round-trip before
    /// writing. Run:
    /// `LZR_IN=assets/enwik9 LZR_OUT=/tmp/enwik9.reordered cargo test --release reorder_to_file -- --ignored --nocapture`
    #[test]
    #[ignore = "tool: reorder an enwik file (LZR_IN -> LZR_OUT)"]
    fn reorder_to_file() {
        use crate::preprocessors::Preprocessor;
        use crate::preprocessors::reorder::Reorder;
        let (Ok(inp), Ok(outp)) = (std::env::var("LZR_IN"), std::env::var("LZR_OUT")) else {
            println!("set LZR_IN and LZR_OUT");
            return;
        };
        let raw = std::fs::read(&inp).unwrap();
        let t = std::time::Instant::now();
        let re = Reorder.forward(&raw);
        let order_s = t.elapsed().as_secs_f64();
        assert_eq!(Reorder.inverse(&re), raw, "round-trip before writing");
        std::fs::write(&outp, &re).unwrap();
        println!(
            "reordered {inp} -> {outp} ({} bytes) | ordering+roundtrip {order_s:.1}s",
            re.len(),
        );
    }

    /// Composition check: does the opt-in LSTM arm (a recurrent *model*) still add
    /// on top of the residual neural *mixer* (M1), or do they overlap? Encodes one
    /// enwik8 slice through the 2×2 of {arm off/on} × {nmix off/on} and reports each
    /// marginal plus the additivity gap (combined minus the sum of the singles).
    /// Both are online (L(D)≈0). `LZR_LO`/`LZR_HI` slice (default `[1M..6M]`),
    /// `LZR_H` arm hidden width (default 128 for tractable runtime). Run:
    /// `cargo test --features arm --profile fast arm_nmix_compose -- --ignored --nocapture`
    #[cfg(feature = "arm")]
    #[test]
    #[ignore = "arm (recurrent model) × neural mixer (M1) composition on enwik8"]
    fn arm_nmix_compose() {
        use crate::models::lstm::ArmModel;
        use std::time::Instant;
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(d)
        };
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let lo = env("LZR_LO", 1_000_000);
        let hi = env("LZR_HI", 6_000_000).min(e8.len());
        let orig = (hi - lo) as f64;
        let data = Pipeline::default_pipeline().forward(&e8[lo..hi]);
        let h = env("LZR_H", 128);

        let run = |arm: bool, nmix: bool| -> (f64, f64) {
            let mut models = baseline_models(data.len());
            if arm {
                models.push(ArmModel::new(h).into());
            }
            let mut state = CodecState::with_models(models, data.len());
            if !nmix {
                state.nmix = None;
            }
            let t = Instant::now();
            let mut enc = Encoder::with_capacity(data.len());
            for &byte in &data {
                for k in (0..8).rev() {
                    let bit = (byte >> k) & 1;
                    let p = state.predict();
                    enc.encode(bit, p);
                    state.commit(bit);
                }
                state.end_symbol();
            }
            (
                enc.finish().len() as f64 * 8.0 / orig,
                t.elapsed().as_secs_f64(),
            )
        };

        let (lin, _) = run(false, false);
        let (nm, nm_s) = run(false, true);
        let (ar, ar_s) = run(true, false);
        let (both, both_s) = run(true, true);
        println!("linear only:      {lin:.4} bpb");
        println!(
            "+ nmix (M1):      {nm:.4} bpb  ({:+.4}, {nm_s:.0}s)",
            nm - lin
        );
        println!(
            "+ arm (h={h}):     {ar:.4} bpb  ({:+.4}, {ar_s:.0}s)",
            ar - lin
        );
        println!(
            "+ arm + nmix:     {both:.4} bpb  ({:+.4}, {both_s:.0}s)",
            both - lin
        );
        let sum = (nm - lin) + (ar - lin);
        println!(
            "additivity: singles-sum {sum:+.4} vs combined {:+.4}  (overlap {:+.4})",
            both - lin,
            (both - lin) - sum
        );
    }

    /// Error-threshold update-skipping on the arm (fx2-cmix's "skip weight updates
    /// when errors are below a threshold"): when a BPTT window was easy (high mean
    /// predicted prob of its coded bytes) skip its backward+Adam — saving the
    /// dominant per-byte cost over the predictable majority, so the same wall-clock
    /// buys a wider arm. Sweeps the skip τ over the FULL production stack
    /// (deterministic + frozen net + warming heads) + arm, reporting bpb, marginal
    /// over the arm-free stack, encode time, speedup vs no-skip, and (to stderr) the
    /// backward-skip rate. `LZR_LO`/`LZR_HI` slice (default 3 MB), `LZR_H` arm width
    /// (default 128). Run:
    /// `cargo test --features arm --release arm_skip_lab -- --ignored --nocapture`
    #[cfg(feature = "arm")]
    #[test]
    #[ignore = "arm error-threshold update-skipping: bpb vs throughput sweep"]
    fn arm_skip_lab() {
        use crate::models::lstm::ArmModel;
        use std::time::Instant;
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(d)
        };
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let lo = env("LZR_LO", 1_000_000);
        let hi = env("LZR_HI", 4_000_000).min(e8.len());
        let orig = (hi - lo) as f64;
        let data = Pipeline::default_pipeline().forward(&e8[lo..hi]);
        let h = env("LZR_H", 128);

        // Arm-free production stack: deterministic baseline + frozen net + the
        // shipped 8-head warming stack (mirrors `models()` minus the arm).
        let stack = || -> Vec<AnyModel> {
            let mut v = baseline_models(data.len());
            let mut net = PretrainedMlp::from_blob_q8(PRETRAINED_NET);
            net.push_head_ctx(HEAD_LR, 1, HEAD_BITS);
            net.push_head_word(HEAD_LR, HEAD_BITS);
            net.push_head_ctx(HEAD_LR, 2, HEAD_BITS);
            net.push_head_sparse(HEAD_LR, 0b110, HEAD_BITS);
            net.push_head_ctx(HEAD_SLOW_LR, 1, HEAD_BITS);
            net.push_head_match(HEAD_MATCH_LR, HEAD_BITS);
            net.push_head_match2(HEAD_MATCH_LR, HEAD_BITS);
            net.push_head_match_prev(HEAD_MATCH_LR, HEAD_BITS);
            v.push(net.into());
            v
        };

        let run = |arm_tau: Option<f32>| -> (f64, f64) {
            let mut models = stack();
            if let Some(tau) = arm_tau {
                models.push(ArmModel::with_skip(h, tau).into());
            }
            let mut state = CodecState::with_models(models, data.len());
            let t = Instant::now();
            let mut enc = Encoder::with_capacity(data.len());
            for &byte in &data {
                for k in (0..8).rev() {
                    let bit = (byte >> k) & 1;
                    let p = state.predict();
                    enc.encode(bit, p);
                    state.commit(bit);
                }
                state.end_symbol();
            }
            (
                enc.finish().len() as f64 * 8.0 / orig,
                t.elapsed().as_secs_f64(),
            )
        };

        let (off, off_s) = run(None);
        println!("arm-free stack:        {off:.4} bpb  ({off_s:.0}s)");
        let (b0, s0) = run(Some(0.0));
        println!(
            "arm h={h} (no skip):    {b0:.4} bpb  marginal {:+.4}  ({s0:.0}s, 1.00x)",
            b0 - off
        );
        for tau in [0.25f32, 0.4, 0.55, 0.7] {
            let (b, s) = run(Some(tau));
            println!(
                "arm skip τ={tau:.2}:       {b:.4} bpb  marginal {:+.4}  ({s:.0}s, {:.2}x speedup)",
                b - off,
                s0 / s,
            );
        }
    }

    /// Offline: the [`Lz`] preprocessor across a min-match sweep. Hash-chain
    /// matcher over the whole prior buffer, rep-distance cache, op-coded L/D
    /// widths. Reports stream shrinkage (the speed proxy) and the CM bpb delta per
    /// min-match — the question being whether re-coding already-cheap long matches
    /// as explicit (L,D) tokens costs bpb. Run on casefolded enwik8 (no word dict,
    /// generous to LZ — matches are longer). `LZR_LO`/`LZR_HI` slice; `LZR_MINS`
    /// overrides the sweep. Run:
    /// `cargo test --release lz_preprocess -- --ignored --nocapture`
    #[test]
    #[ignore = "offline: LZ-factoring preprocessor (stream shrink vs bpb)"]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::naive_bytecount
    )]
    fn lz_preprocess() {
        use crate::preprocessors::Preprocessor;
        use crate::preprocessors::casefold::CaseFold;
        use crate::preprocessors::lz::Lz;

        const ESC: u8 = 0x02; // census-free in enwik8 → never a casefolded literal

        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let lo = env("LZR_LO", 1_000_000);
        let hi = env("LZR_HI", 21_000_000).min(e8.len());
        let orig = (hi - lo) as f64;
        let folded = CaseFold.forward(&e8[lo..hi]);
        assert!(!folded.contains(&ESC), "ESC must be absent from literals");

        let base = code_stream(&folded).len();
        let base_bpb = base as f64 * 8.0 / orig;
        println!(
            "baseline (casefold only): {base_bpb:.4} bpb  ({} bytes in, {base} coded)",
            folded.len()
        );

        let mins: Vec<usize> = std::env::var("LZR_MINS")
            .unwrap_or_else(|_| "8,12,16,20,24,28,32,64,128".into())
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect();
        for minlen in mins {
            let lz = Lz::new(ESC, minlen);
            let fac = lz.forward(&folded);
            assert_eq!(lz.inverse(&fac), folded, "LZ factor must round-trip");
            let tokens = fac.iter().filter(|&&b| b == ESC).count();
            let coded = code_stream(&fac).len();
            let bpb = coded as f64 * 8.0 / orig;
            println!(
                "minlen={minlen:>3}: stream {:.1}% shorter ({tokens} tokens) | {bpb:.4} bpb ({:+.4})",
                100.0 * (folded.len() - fac.len()) as f64 / folded.len() as f64,
                bpb - base_bpb,
            );
        }
    }

    /// Harness sanity / reference baseline: encode an enwik8 slice (full
    /// pipeline) with the shipped deterministic model set and print bpb. The
    /// 20 MB slice [1M..21M] should land at ≈ 1.6695 (the journal's 14-model
    /// baseline). `LZR_LO`/`LZR_HI` override the slice. Run:
    /// `cargo test --release ablate_baseline -- --ignored --nocapture`
    #[test]
    #[ignore = "offline: reference baseline bpb on an enwik8 slice"]
    fn ablate_baseline() {
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let lo = env("LZR_LO", 1_000_000);
        let hi = env("LZR_HI", 21_000_000).min(e8.len());
        let orig = (hi - lo) as f64;
        let data = Pipeline::default_pipeline().forward(&e8[lo..hi]);
        let t0 = std::time::Instant::now();
        let n = code_stream_models(baseline_models(data.len()), &data).len();
        println!(
            "baseline: {:.4} bpb  ({} bytes, {:.0}s)",
            n as f64 * 8.0 / orig,
            n,
            t0.elapsed().as_secs_f64()
        );
    }

    #[test]
    fn roundtrip_enwik8_slice() {
        let Ok(bytes) = std::fs::read("assets/enwik8") else {
            return; // skip when the corpus is absent
        };
        // Small slice keeps this fast under the debug coverage build now that
        // the online LSTM arm is in the default model set; it still exercises
        // the full pipeline and round-trips byte-exact.
        let slice = &bytes[1_000_000..1_008_000];
        let coded = encode(slice.to_vec());
        assert_eq!(decode(&coded), slice);
        let bpb = coded.len() as f64 * 8.0 / slice.len() as f64;
        println!("full stack on enwik8 8 KB slice: {bpb:.4} bpb");
    }

    /// Indirect context models must round-trip byte-exact in the codec.
    #[test]
    fn indirect_roundtrip() {
        use crate::models::indirect::IndirectModel;
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let data = Pipeline::default_pipeline().forward(&e8[1_000_000..1_010_000]);
        let mk = || {
            let mut v = baseline_models(data.len());
            v.push(IndirectModel::new(2, data.len()).into());
            v.push(IndirectModel::new(4, data.len()).into());
            v.push(IndirectModel::new(6, data.len()).into());
            v
        };
        let coded = code_stream_models(mk(), &data);
        assert_eq!(decode_stream_models(mk(), &coded), data);
    }

    /// T1.1 sweep: baseline vs baseline + indirect models over several order
    /// sets, on an enwik8 slice. `LZR_LO`/`LZR_HI` (default 1M..21M), `LZR_HB`/
    /// `LZR_CB` (follower-hist / cell table bits, default the production caps).
    /// Run: `cargo test --release ablate_indirect -- --ignored --nocapture`
    #[test]
    #[ignore = "offline: indirect-context-model marginal on an enwik8 slice"]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn ablate_indirect() {
        use crate::models::indirect::IndirectModel;
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let lo = env("LZR_LO", 1_000_000);
        let hi = env("LZR_HI", 21_000_000).min(e8.len());
        let orig = (hi - lo) as f64;
        let data = Pipeline::default_pipeline().forward(&e8[lo..hi]);
        let cap = data.len();
        let hb = env("LZR_HB", 0);
        let cb = env("LZR_CB", 0);

        let base = code_stream_models(baseline_models(cap), &data).len();
        let base_bpb = base as f64 * 8.0 / orig;
        println!("baseline: {base_bpb:.4} bpb");

        let order_sets: &[&[usize]] = &[
            &[2, 3, 4, 6],
            &[1, 2, 3, 4, 6],
            &[1, 2, 3, 4, 5, 6],
            &[1, 2, 3, 4, 5, 6, 8],
        ];
        for set in order_sets {
            let mut v = baseline_models(cap);
            for &o in *set {
                let m = if hb > 0 && cb > 0 {
                    IndirectModel::with_bits(o, hb as u32, cb as u32)
                } else {
                    IndirectModel::new(o, cap)
                };
                v.push(m.into());
            }
            let n = code_stream_models(v, &data).len();
            let bpb = n as f64 * 8.0 / orig;
            println!("+ indirect {set:?}: {bpb:.4} bpb  ({:+.4})", bpb - base_bpb);
        }
    }

    /// Sweep additional sparse context patterns as marginals on the current
    /// baseline. Run: `cargo test --release ablate_sparse -- --ignored --nocapture`
    #[test]
    #[ignore = "offline: extra sparse-context patterns on an enwik8 slice"]
    #[allow(clippy::cast_precision_loss)]
    fn ablate_sparse() {
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let lo = env("LZR_LO", 1_000_000);
        let hi = env("LZR_HI", 21_000_000).min(e8.len());
        let orig = (hi - lo) as f64;
        let data = Pipeline::default_pipeline().forward(&e8[lo..hi]);
        let cap = data.len();
        let base = code_stream_models(baseline_models(cap), &data).len();
        let base_bpb = base as f64 * 8.0 / orig;
        println!("baseline: {base_bpb:.4} bpb");
        // (mask, label): bit i selects byte_back(i+1); avoid contiguous (= orders).
        let cands: &[(u32, &str)] = &[
            (0b1001, "{1,4}"),
            (0b1010, "{2,4}"),
            (0b1110, "{2,3,4}"),
            (0b1101, "{1,3,4}"),
            (0b10001, "{1,5}"),
        ];
        for &(mask, label) in cands {
            let mut v = baseline_models(cap);
            v.push(ContextModel::sparse(mask, cap).into());
            let n = code_stream_models(v, &data).len();
            let bpb = n as f64 * 8.0 / orig;
            println!("+ sparse {label}: {bpb:.4} bpb  ({:+.4})", bpb - base_bpb);
        }
    }

    fn hex(t: &[u8]) -> String {
        const H: &[u8; 16] = b"0123456789abcdef";
        let mut s = String::with_capacity(t.len() * 2);
        for &b in t {
            s.push(char::from(H[(b >> 4) as usize]));
            s.push(char::from(H[(b & 15) as usize]));
        }
        s
    }

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|j| u8::from_str_radix(&s[j..j + 2], 16).unwrap())
            .collect()
    }

    fn esc(t: &[u8]) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        for &b in t {
            if (0x20..0x7f).contains(&b) {
                s.push(char::from(b));
            } else {
                let _ = write!(s, "\\x{b:02x}");
            }
        }
        s
    }

    /// Stage 1 of dictionary selection: the top-1024 **arbitrary substrings** of
    /// case-folded enwik9 by `(non-overlapping occurrences)*(len-1)` — no lexical
    /// structure. A suffix array (offline dev-dep) + LCP (Kasai) + a
    /// largest-rectangle pass over the LCP array gives each distinct substring's
    /// occurrence interval; non-overlapping counting (what longest-match can take)
    /// then suppresses the O(L^2) self-overlap artifact of repeated-char runs.
    /// Written to `/tmp/dict_candidates.tsv` as `hex<TAB>score`. Run once:
    /// `cargo test --release dict_candidates_gen -- --ignored --nocapture`.
    #[test]
    #[ignore = "offline: generate dictionary candidates from enwik9"]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::too_many_lines,
        clippy::items_after_statements
    )]
    fn dict_candidates_gen() {
        use crate::preprocessors::Preprocessor;
        use crate::preprocessors::casefold::CaseFold;
        use std::collections::HashMap;
        use std::fmt::Write as _;

        const T: i64 = 10_000; // min freq*(len-1) of a collected substring

        let Ok(e9) = std::fs::read("assets/enwik9") else {
            return;
        };
        let folded = CaseFold.forward(&e9);
        drop(e9);
        let n = folded.len();
        if n == 0 {
            return;
        }

        // Suffix array, then LCP via Kasai (lcp[k] = LCP(sa[k-1], sa[k])).
        let sa: Vec<i32> = cdivsufsort::sort(&folded).into_parts().1;
        let mut rank = vec![0i32; n];
        for (i, &s) in sa.iter().enumerate() {
            rank[s as usize] = i as i32;
        }
        let mut lcp = vec![0i32; n];
        let mut h = 0usize;
        for i in 0..n {
            let r = rank[i] as usize;
            if r > 0 {
                let j = sa[r - 1] as usize;
                while i + h < n && j + h < n && folded[i + h] == folded[j + h] {
                    h += 1;
                }
                lcp[r] = h as i32;
                h = h.saturating_sub(1);
            } else {
                h = 0;
            }
        }
        drop(rank);

        // Largest-rectangle over the LCP histogram: popping bar `top` at SA
        // index `i` exposes a substring of length `height` shared by the suffix
        // interval `[prev, i-1]` — its (overlapping) occurrence count is `i-prev`.
        // Dedup by bytes here, keeping the full interval (max raw count).
        const LMAX: usize = 256; // length cap: an L(D) guard on embedded entries
        let mut best: HashMap<&[u8], (i64, usize, usize)> = HashMap::new(); // bytes -> (raw, lo, hi)
        let mut stack: Vec<usize> = Vec::new();
        for i in 1..=n {
            let cur = if i < n { i64::from(lcp[i]) } else { -1 };
            while let Some(&top) = stack.last() {
                let height = i64::from(lcp[top]) as usize;
                if i64::from(lcp[top]) <= cur {
                    break;
                }
                stack.pop();
                let prev = stack.last().copied().unwrap_or(0);
                let raw = (i - prev) as i64;
                if (2..=LMAX).contains(&height) && raw * (height as i64 - 1) >= T {
                    let pos = sa[top] as usize;
                    let e = best.entry(&folded[pos..pos + height]).or_insert((0, 0, 0));
                    if raw > e.0 {
                        *e = (raw, prev, i - 1);
                    }
                }
            }
            stack.push(i);
        }
        drop(stack);
        drop(lcp);

        // Rescore each distinct substring by NON-OVERLAPPING occurrences (what
        // longest-match can actually take). A substring self-overlaps only if it
        // has a proper border, so aperiodic ones keep their raw count for free;
        // bordered ones get greedy interval scheduling over their SA interval.
        fn has_border(s: &[u8]) -> bool {
            let mut lps = [0usize; 256];
            let mut k = 0;
            for i in 1..s.len() {
                while k > 0 && s[i] != s[k] {
                    k = lps[k - 1];
                }
                if s[i] == s[k] {
                    k += 1;
                }
                lps[i] = k;
            }
            lps[s.len() - 1] > 0
        }
        let mut ranked: Vec<(&[u8], i64)> = best
            .into_iter()
            .map(|(s, (raw, lo, hi))| {
                let len = s.len() as i64;
                let nonoverlap = if has_border(s) {
                    let mut pos: Vec<i32> = sa[lo..=hi].to_vec();
                    pos.sort_unstable();
                    let mut cnt = 0i64;
                    let mut last = i64::MIN;
                    for &p in &pos {
                        if i64::from(p) >= last + len {
                            cnt += 1;
                            last = i64::from(p);
                        }
                    }
                    cnt
                } else {
                    raw
                };
                (s, nonoverlap * (len - 1))
            })
            .collect();
        drop(sa);
        ranked.sort_by_key(|x| std::cmp::Reverse(x.1));
        ranked.truncate(1024);
        let mut out = String::new();
        for (s, score) in &ranked {
            let _ = writeln!(out, "{}\t{score}", hex(s));
        }
        std::fs::write("/tmp/dict_candidates.tsv", out).unwrap();
        println!("wrote {} candidates", ranked.len());
    }

    /// Stage 2: the **isolated** compressed-byte saving of each candidate —
    /// baseline (no dict) minus the full-enwik8 size with just that one token.
    /// Candidates don't compete for spans (whole-run match), so isolated savings
    /// rank true standalone power. Processes the index range `LZR_LO..LZR_HI`
    /// (env, default all) so several runs can cover the 256 in parallel:
    /// `LZR_LO=0 LZR_HI=64 cargo test --release dict_isolated -- --ignored --nocapture`.
    #[test]
    #[ignore = "offline: per-candidate isolated savings; set LZR_LO/LZR_HI"]
    #[allow(clippy::cast_possible_wrap)]
    fn dict_isolated() {
        use crate::preprocessors::Preprocessor;
        use crate::preprocessors::casefold::CaseFold;
        use crate::preprocessors::dictionary::Dictionary;

        let Ok(text) = std::fs::read_to_string("/tmp/dict_candidates.tsv") else {
            return;
        };
        let cands: Vec<Vec<u8>> = text
            .lines()
            .map(|l| unhex(l.split('\t').next().unwrap()))
            .collect();

        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let folded8 = CaseFold.forward(&e8);
        let orig = e8.len() as f64;

        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let lo = env("LZR_LO", 0);
        let hi = env("LZR_HI", cands.len()).min(cands.len());

        let baseline = code_stream(&folded8).len();
        for (idx, tok) in cands
            .iter()
            .enumerate()
            .skip(lo)
            .take(hi.saturating_sub(lo))
        {
            let dict = Dictionary::new(&[tok.as_slice()]);
            let bytes = code_stream(&dict.forward(&folded8)).len();
            let savings = baseline as i64 - bytes as i64;
            let bpb = bytes as f64 * 8.0 / orig;
            println!("{idx}\t{savings}\t{bpb:.4}\t{}", esc(tok));
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
    }

    /// Stage 3: overlap-aware greedy selection. Reads priority-ordered candidates
    /// (hex, one per line) from `/tmp/dict_greedy_in.tsv` — in practice the
    /// positive-isolated-savings candidates in savings order — and greedily adds
    /// each to the dictionary, keeping it only if it does not grow full-enwik8.
    /// Greedy (re-measure given what's accepted) is what correctly resolves the
    /// overlap between arbitrary substrings. Prints each decision and the final
    /// embeddable list (savings-descending). Run:
    /// `cargo test --release dict_greedy -- --ignored --nocapture`.
    #[test]
    #[ignore = "offline: overlap-aware greedy dictionary selection"]
    #[allow(clippy::cast_possible_wrap)]
    fn dict_greedy() {
        use crate::preprocessors::Preprocessor;
        use crate::preprocessors::casefold::CaseFold;
        use crate::preprocessors::dictionary::{Dictionary, code_pool};

        let Ok(text) = std::fs::read_to_string("/tmp/dict_greedy_in.tsv") else {
            return;
        };
        let cands: Vec<Vec<u8>> = text
            .lines()
            .map(|l| unhex(l.split('\t').next().unwrap()))
            .collect();

        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let folded8 = CaseFold.forward(&e8);
        let orig = e8.len() as f64;

        let encoded =
            |accepted: &[&[u8]]| code_stream(&Dictionary::new(accepted).forward(&folded8)).len();

        let max_codes = code_pool().len();
        let mut accepted: Vec<Vec<u8>> = Vec::new();
        let mut savings: Vec<i64> = Vec::new();
        let mut baseline = encoded(&[]);
        println!(
            "baseline_bpb={:.4} candidates={}",
            baseline as f64 * 8.0 / orig,
            cands.len()
        );

        for (rank, tok) in cands.iter().enumerate() {
            if accepted.len() >= max_codes {
                println!("code pool exhausted ({max_codes})");
                break;
            }
            let mut trial: Vec<&[u8]> = accepted.iter().map(Vec::as_slice).collect();
            trial.push(tok.as_slice());
            let bytes = encoded(&trial);
            let delta = baseline as i64 - bytes as i64;
            if bytes < baseline {
                println!(
                    "#{rank:3} ACCEPT \"{}\" saved={delta} bpb={:.4}",
                    esc(tok),
                    bytes as f64 * 8.0 / orig
                );
                accepted.push(tok.clone());
                savings.push(delta);
                baseline = bytes;
            } else {
                println!("#{rank:3} reject \"{}\" delta={delta}", esc(tok));
            }
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }

        let mut order: Vec<usize> = (0..accepted.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(savings[i]));
        println!(
            "=== final dictionary ({} entries, savings-descending) ===",
            accepted.len()
        );
        for &i in &order {
            println!("    b\"{}\", // +{}", esc(&accepted[i]), savings[i]);
        }
    }

    /// Offline diagnostic: a per-byte coding-cost map. Encodes the corpus through
    /// the real pipeline and records, for every coded byte, the ideal bits the
    /// coder spends on it (`log2(4096 / p_correct)` summed over its 8 bits) plus
    /// each model's *standalone* bits (its own squashed logit). Three artifacts
    /// under the `LZR_OUT` prefix (default `/tmp/cost`):
    ///   * `<prefix>_byte.f32`  — `[u64 LE count][count x f32 LE]`, bits/coded byte.
    ///   * `<prefix>_lines.tsv` — one row per line (split on LF, which survives
    ///     case-fold so lines map to originals): idx, coded offset, coded len,
    ///     total bits, bpb, per-model bits, escaped 100-byte snippet.
    ///   * `<prefix>_summary.txt` — overall bpb, per-model standalone bpb, and a
    ///     per-byte-value table (count, total bits, mean bits) sorted by total —
    ///     where the bulk of the bits go.
    ///
    /// Corpus is `LZR_CORPUS` (default `assets/enwik8`). Run:
    /// `cargo test --release cost_map -- --ignored --nocapture`
    #[test]
    #[ignore = "offline: per-byte coding-cost map"]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::too_many_lines
    )]
    fn cost_map() {
        use crate::mixer::squash;
        use std::fmt::Write as _;
        use std::io::{BufWriter, Write as _};

        let corpus = std::env::var("LZR_CORPUS").unwrap_or_else(|_| "assets/enwik8".into());
        let prefix = std::env::var("LZR_OUT").unwrap_or_else(|_| "/tmp/cost".into());
        let Ok(raw) = std::fs::read(&corpus) else {
            return;
        };
        let orig_len = raw.len();
        let data = Pipeline::default_pipeline().forward(&raw);
        drop(raw);
        let n = data.len();
        if n == 0 {
            return;
        }

        let mut state = CodecState::new(n);
        let nm = state.stretched.len();

        let mut byte_f =
            BufWriter::new(std::fs::File::create(format!("{prefix}_byte.f32")).unwrap());
        byte_f.write_all(&(n as u64).to_le_bytes()).unwrap();
        let mut lines =
            BufWriter::new(std::fs::File::create(format!("{prefix}_lines.tsv")).unwrap());
        write!(lines, "line\toff\tlen\tbits\tbpb").unwrap();
        for mi in 0..nm {
            write!(lines, "\tm{mi}").unwrap();
        }
        writeln!(lines, "\tsnippet").unwrap();

        let mut total_bits = 0f64;
        let mut model_bits = vec![0f64; nm];
        let mut val_cnt = vec![0u64; 256];
        let mut val_bits = vec![0f64; 256];

        let mut line_idx = 0u64;
        let mut line_start = 0usize;
        let mut line_bits = 0f64;
        let mut line_model = vec![0f64; nm];

        for (pos, &byte) in data.iter().enumerate() {
            let mut byte_bits = 0f64;
            for k in (0..8).rev() {
                let bit = (byte >> k) & 1;
                // mirror the coder's clamp (coder.rs): p is forced into 1..=4095.
                let p = f64::from(state.predict().clamp(1, 4095));
                let pc = if bit == 1 { p } else { 4096.0 - p };
                byte_bits += (4096.0 / pc).log2();
                for (mi, &s) in state.stretched.iter().enumerate() {
                    let pm = f64::from(squash(s));
                    let pmc = if bit == 1 { pm } else { 4096.0 - pm };
                    let b = (4096.0 / pmc).log2();
                    model_bits[mi] += b;
                    line_model[mi] += b;
                }
                state.commit(bit);
            }
            state.end_symbol();

            byte_f.write_all(&(byte_bits as f32).to_le_bytes()).unwrap();
            total_bits += byte_bits;
            val_cnt[byte as usize] += 1;
            val_bits[byte as usize] += byte_bits;
            line_bits += byte_bits;

            if byte == b'\n' || pos + 1 == n {
                let len = pos + 1 - line_start;
                write!(
                    lines,
                    "{line_idx}\t{line_start}\t{len}\t{line_bits:.3}\t{:.4}",
                    line_bits / len as f64
                )
                .unwrap();
                for v in &line_model {
                    write!(lines, "\t{v:.1}").unwrap();
                }
                let end = (line_start + 100).min(pos + 1);
                writeln!(lines, "\t{}", esc(&data[line_start..end])).unwrap();
                line_idx += 1;
                line_start = pos + 1;
                line_bits = 0.0;
                line_model.fill(0.0);
            }

            if pos > 0 && pos % 10_000_000 == 0 {
                eprintln!(
                    "  {pos}/{n} coded bytes, {:.4} bpb so far",
                    total_bits / (pos as f64 + 1.0)
                );
            }
        }
        byte_f.flush().unwrap();
        lines.flush().unwrap();

        let mut s = String::new();
        let _ = writeln!(
            s,
            "coded_bytes={n} original_bytes={orig_len} total_bits={total_bits:.0}"
        );
        let _ = writeln!(
            s,
            "bpb_coded={:.4} bpb_original={:.4}",
            total_bits / n as f64,
            total_bits / orig_len as f64
        );
        let _ = writeln!(s, "\n=== per-model standalone (bits, bpb_coded) ===");
        let mut mi_order: Vec<usize> = (0..nm).collect();
        mi_order.sort_by(|&a, &b| model_bits[b].total_cmp(&model_bits[a]));
        for mi in mi_order {
            let _ = writeln!(
                s,
                "m{mi}\t{:.0}\t{:.4}",
                model_bits[mi],
                model_bits[mi] / n as f64
            );
        }
        let _ = writeln!(
            s,
            "\n=== per-byte-value (byte, count, total_bits, mean) ==="
        );
        let mut bv: Vec<usize> = (0..256).filter(|&v| val_cnt[v] > 0).collect();
        bv.sort_by(|&a, &b| val_bits[b].total_cmp(&val_bits[a]));
        for v in bv {
            let _ = writeln!(
                s,
                "{}\t{}\t{:.0}\t{:.3}",
                esc(&[v as u8]),
                val_cnt[v],
                val_bits[v],
                val_bits[v] / val_cnt[v] as f64
            );
        }
        std::fs::write(format!("{prefix}_summary.txt"), &s).unwrap();
        print!("{s}");
    }

    /// Fair test of a SOTA-style word-canonicalizing dictionary (DRT-like). A
    /// real word list is mined from case-folded enwik8 by frequency and assigned
    /// codes from the 74 free post-fold bytes in three tiers — most-frequent
    /// words get 1-byte codes, then 2-byte (lead + index), then 3-byte (lead +
    /// two index bytes), each word taken only if its code is shorter than the
    /// word — so up to ~1M words fit (enough for the full ~44K SOTA list). The
    /// reversible transform replaces whole `a..=z` words with their codes; we
    /// then encode the *same* enwik8 slice with the current full stack,
    /// transformed vs. not. `L(D)` (the shipped word list) is reported so the
    /// net is visible. Params: `LZR_NS` (comma list of dictionary sizes, default
    /// `2000,44000`), `LZR_LO`/`LZR_HI` (slice, default full enwik8). The
    /// baseline is encoded once and shared across sizes. Run:
    /// `cargo test --release word_dict_test -- --ignored --nocapture`
    #[test]
    #[ignore = "offline: word-canonicalizing dictionary vs. current stack"]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::too_many_lines
    )]
    fn word_dict_test() {
        use crate::preprocessors::Preprocessor;
        use crate::preprocessors::casefold::CaseFold;
        use crate::preprocessors::dictionary::code_pool;
        use std::collections::HashMap;

        let corpus = std::env::var("LZR_CORPUS").unwrap_or_else(|_| "assets/enwik8".into());
        let Ok(e8) = std::fs::read(&corpus) else {
            return;
        };
        let e8_len = e8.len();
        let folded = CaseFold.forward(&e8);
        drop(e8);
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let ns: Vec<usize> = std::env::var("LZR_NS")
            .unwrap_or_else(|_| "2000,44000".into())
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect();
        let lo = env("LZR_LO", 0);
        let hi = env("LZR_HI", folded.len()).min(folded.len());

        // Word frequency over maximal a..=z runs of the whole folded corpus.
        let mut freq: HashMap<&[u8], u32> = HashMap::new();
        let mut i = 0;
        while i < folded.len() {
            if folded[i].is_ascii_lowercase() {
                let s = i;
                while i < folded.len() && folded[i].is_ascii_lowercase() {
                    i += 1;
                }
                *freq.entry(&folded[s..i]).or_insert(0) += 1;
            } else {
                i += 1;
            }
        }
        let mut words: Vec<(&[u8], u32)> = freq.into_iter().collect();
        words.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

        // Code tiers from the 74 free bytes: 24 single, 34 two-byte leads
        // (34*256), 16 three-byte leads (16*65536).
        let pool = code_pool();
        let singles = &pool[..24];
        let leads2 = &pool[24..58];
        let leads3 = &pool[58..];

        let slice = &folded[lo..hi];
        // bpb is per *original* byte; scale the folded-slice length back by the
        // fold ratio (exact for the full corpus, lo=0..folded.len()).
        let orig = (hi - lo) as f64 * e8_len as f64 / folded.len() as f64;
        let base = code_stream(slice).len();
        let base_bpb = base as f64 * 8.0 / orig;
        println!("baseline_bpb={base_bpb:.4} (slice {} bytes)", slice.len());

        for &nmax in &ns {
            let mut code_of: HashMap<&[u8], Vec<u8>> = HashMap::new();
            let mut is1 = [false; 256];
            let mut is2 = [false; 256];
            let mut is3 = [false; 256];
            let mut rev1: Vec<Option<&[u8]>> = vec![None; 256];
            let mut rev2: HashMap<u16, &[u8]> = HashMap::new();
            let mut rev3: HashMap<u32, &[u8]> = HashMap::new();
            let (mut s1, mut s2, mut s3) = (0usize, 0usize, 0usize);
            let mut dict_bytes = 0usize;
            for (w, _f) in &words {
                if code_of.len() >= nmax {
                    break;
                }
                if s1 < singles.len() && w.len() > 1 {
                    let c = singles[s1];
                    is1[c as usize] = true;
                    rev1[c as usize] = Some(w);
                    code_of.insert(w, vec![c]);
                    s1 += 1;
                } else if s2 < leads2.len() * 256 && w.len() > 2 {
                    let (lead, idx) = (leads2[s2 / 256], (s2 % 256) as u8);
                    is2[lead as usize] = true;
                    rev2.insert((u16::from(lead) << 8) | u16::from(idx), w);
                    code_of.insert(w, vec![lead, idx]);
                    s2 += 1;
                } else if s3 < leads3.len() * 65536 && w.len() > 3 {
                    let lead = leads3[s3 / 65536];
                    let (b1, b2) = (((s3 >> 8) & 0xff) as u8, (s3 & 0xff) as u8);
                    is3[lead as usize] = true;
                    rev3.insert(
                        (u32::from(lead) << 16) | (u32::from(b1) << 8) | u32::from(b2),
                        w,
                    );
                    code_of.insert(w, vec![lead, b1, b2]);
                    s3 += 1;
                } else if s1 >= singles.len()
                    && s2 >= leads2.len() * 256
                    && s3 >= leads3.len() * 65536
                {
                    break;
                } else {
                    continue;
                }
                dict_bytes += w.len();
            }

            let forward = |data: &[u8]| -> Vec<u8> {
                let mut out = Vec::with_capacity(data.len());
                let mut i = 0;
                while i < data.len() {
                    if data[i].is_ascii_lowercase() {
                        let s = i;
                        while i < data.len() && data[i].is_ascii_lowercase() {
                            i += 1;
                        }
                        if let Some(c) = code_of.get(&data[s..i]) {
                            out.extend_from_slice(c);
                        } else {
                            out.extend_from_slice(&data[s..i]);
                        }
                    } else {
                        out.push(data[i]);
                        i += 1;
                    }
                }
                out
            };
            let inverse = |data: &[u8]| -> Vec<u8> {
                let mut out = Vec::new();
                let mut i = 0;
                while i < data.len() {
                    let b = data[i];
                    if is1[b as usize] {
                        out.extend_from_slice(rev1[b as usize].unwrap());
                        i += 1;
                    } else if is2[b as usize] {
                        let key = (u16::from(b) << 8) | u16::from(data[i + 1]);
                        out.extend_from_slice(rev2[&key]);
                        i += 2;
                    } else if is3[b as usize] {
                        let key = (u32::from(b) << 16)
                            | (u32::from(data[i + 1]) << 8)
                            | u32::from(data[i + 2]);
                        out.extend_from_slice(rev3[&key]);
                        i += 3;
                    } else {
                        out.push(b);
                        i += 1;
                    }
                }
                out
            };

            let transformed = forward(slice);
            assert_eq!(
                inverse(&transformed),
                slice,
                "word transform must round-trip"
            );
            let treat = code_stream(&transformed).len();
            let treat_bpb = treat as f64 * 8.0 / orig;
            let ld_bpb = 16.0 * dict_bytes as f64 / 1e9; // enwik9 L(D): 2x, 8 bits, /1e9
            println!(
                "N={nmax} words_coded={} dict_bytes={dict_bytes} ({:.1}% shorter) treated_bpb={treat_bpb:.4} L(C)_delta={:.4} L(D)_enwik9={ld_bpb:.4} net={:.4}",
                code_of.len(),
                100.0 * (slice.len() - transformed.len()) as f64 / slice.len() as f64,
                treat_bpb - base_bpb,
                (treat_bpb - base_bpb) + ld_bpb
            );
        }
    }

    /// Word-model contribution ablation: encode an enwik8 slice with the shipped
    /// pipeline (case-fold + word dict) with and without the `word` model, plus
    /// the case-fold-only stream (no dict) as a reference. Used to confirm the
    /// `word` model still earns its place once the dictionary codes the frequent
    /// words (the `prev_word` model, which it did not, was dropped 2026-06-22).
    /// Params `LZR_LO`/`LZR_HI` (default 1M..21M). Run:
    /// `cargo test --release word_model_ablation -- --ignored --nocapture`
    #[test]
    #[ignore = "offline: word model contribution with the dict active"]
    #[allow(clippy::cast_precision_loss)]
    fn word_model_ablation() {
        use crate::preprocessors::Preprocessor;
        use crate::preprocessors::casefold::CaseFold;

        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let lo = env("LZR_LO", 1_000_000);
        let hi = env("LZR_HI", 21_000_000).min(e8.len());
        let slice = &e8[lo..hi];
        let orig = (hi - lo) as f64;

        let cap = slice.len();
        let mk = |word: bool| -> Vec<AnyModel> {
            let mut v: Vec<AnyModel> = (0..=6).map(|n| ContextModel::new(n, cap).into()).collect();
            if word {
                v.push(ContextModel::word(cap).into());
            }
            v.push(MatchModel::new().into());
            v
        };

        let with_dict = Pipeline::default_pipeline().forward(slice);
        let no_dict = CaseFold.forward(slice);
        for (label, stream) in [("with dict", &with_dict), ("no dict  ", &no_dict)] {
            for (name, w) in [("full   ", true), ("no_word", false)] {
                let bytes = code_stream_models(mk(w), stream).len();
                println!(
                    "{label}\t{name}\t{:.4} bpb\t{bytes} bytes",
                    bytes as f64 * 8.0 / orig
                );
            }
        }
    }
}
