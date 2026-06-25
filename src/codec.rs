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

/// The active model set. The online-neural arm is appended only under
/// `--features arm` (opt-in, not shipped — see Cargo.toml).
fn models(capacity: usize) -> Vec<AnyModel> {
    // `mut` is only used when the arm feature appends below.
    #[cfg_attr(not(feature = "arm"), allow(unused_mut))]
    let mut v = baseline_models(capacity);
    #[cfg(feature = "arm")]
    v.push(ArmModel::arm().into());
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
const NMIX_H: usize = 64;
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
        let stretched = vec![0i32; n];
        let mixer = Mixer::new(n);
        Self {
            models,
            mixer,
            apm: Apm::new(APM_CTX),
            apm_w: apm_w(),
            ctx: Context::with_capacity(capacity),
            stretched,
            msel: [0usize; 8],
            nmix: Some(NeuralMixer::new(n, NMIX_H, NMIX_NCTX, NMIX_LR)),
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

/// Compress `input` into the lzr byte stream.
pub(crate) fn encode(input: &[u8]) -> Vec<u8> {
    let data = Pipeline::default_pipeline().forward(input);
    code_stream_inner(&data, input.len(), true)
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
        let coded = encode(&data);
        assert_eq!(decode(&coded), data);
    }

    #[test]
    fn roundtrip_empty() {
        assert_eq!(decode(&encode(b"")), b"");
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
    #[allow(clippy::cast_precision_loss)]
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
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let lo = env("LZR_LO", 1_000_000);
        let hi = env("LZR_HI", 11_000_000).min(e8.len());
        let orig = (hi - lo) as f64;
        let data = Pipeline::default_pipeline().forward(&e8[lo..hi]);
        let base = encode_det_linear(&data);
        let base_bpb = base as f64 * 8.0 / orig;
        let mut models = baseline_models(data.len());
        models.push(PretrainedMlp::from_blob(&blob).into());
        let coded = encode_linear_models(models, &data);
        let bpb = coded as f64 * 8.0 / orig;
        let marginal = bpb - base_bpb;
        let params = (blob.len() - 16) / 4;
        let ld_f32 = 16.0 * blob.len() as f64 / 1e9;
        let ld_tern = 16.0 * (params as f64 * 2.0 / 8.0) / 1e9; // 2 bits/param
        println!("baseline (linear) {base_bpb:.4}  +pretrained {bpb:.4}  marginal {marginal:+.4}");
        println!(
            "net: {params} params, blob {} B | L(D): f32 {ld_f32:.5}, ternary {ld_tern:.5}",
            blob.len()
        );
        println!(
            "NET of L(D):  f32 {:+.4}   ternary {:+.4}",
            marginal + ld_f32,
            marginal + ld_tern
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
        let coded = encode(slice);
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
