//! The encode/decode driver.
//!
//! One per-bit loop wires the preprocessor pipeline, the models, the mixer, and
//! the arithmetic coder. Encode and decode build identical fresh state so their
//! predictions match bit-for-bit. The only framing is a LEB128 varint length
//! prefix (the decoder must know how many bytes to emit); there is no header.

use crate::coder::{Decoder, Encoder};
use crate::mixer::{Apm, Mixer};
use crate::models::context::ContextModel;
use crate::models::indirect::IndirectModel;
#[cfg(feature = "arm")]
use crate::models::lstm::ArmModel;
use crate::models::match_model::MatchModel;
use crate::models::{Context, Model};
use crate::preprocessors::Pipeline;

/// The deterministic model set (no arm). Shared by the production [`models`] and
/// by the ablation tests, so an ablation measures its variant against the exact
/// shipped baseline. Adding/removing a deterministic model is one line here.
pub(crate) fn baseline_models(capacity: usize) -> Vec<Box<dyn Model>> {
    vec![
        Box::new(ContextModel::new(0, capacity)),
        Box::new(ContextModel::new(1, capacity)),
        Box::new(ContextModel::new(2, capacity)),
        Box::new(ContextModel::new(3, capacity)),
        Box::new(ContextModel::new(4, capacity)),
        Box::new(ContextModel::new(5, capacity)),
        Box::new(ContextModel::new(6, capacity)),
        Box::new(ContextModel::word(capacity)),
        Box::new(ContextModel::sparse(0b101, capacity)), // bytes back 1 and 3 (skip 2)
        Box::new(ContextModel::sparse(0b110, capacity)), // bytes back 2 and 3 (skip the last byte)
        Box::new(ContextModel::sparse(0b1011, capacity)), // bytes back 1, 2 and 4 (skip 3)
        Box::new(ContextModel::sparse(0b1100, capacity)), // bytes back 3 and 4 (skip 1, 2)
        Box::new(ContextModel::sparse(0b10001, capacity)), // bytes back 1 and 5 (skip 2,3,4)
        Box::new(MatchModel::new()),
        Box::new(MatchModel::with_key(4)), // shorter-key match: faster acquisition
        // Indirect context models (paq ICM): predict from what historically
        // followed a context. Orders [1,2,3,4,6] — 5 and 8 add ~nothing.
        Box::new(IndirectModel::new(1, capacity)),
        Box::new(IndirectModel::new(2, capacity)),
        Box::new(IndirectModel::new(3, capacity)),
        Box::new(IndirectModel::new(4, capacity)),
        Box::new(IndirectModel::new(6, capacity)),
        Box::new(IndirectModel::word(capacity)),
    ]
}

/// The active model set. The online-neural arm is appended only under
/// `--features arm` (opt-in, not shipped — see Cargo.toml).
fn models(capacity: usize) -> Vec<Box<dyn Model>> {
    // `mut` is only used when the arm feature appends below.
    #[cfg_attr(not(feature = "arm"), allow(unused_mut))]
    let mut v = baseline_models(capacity);
    #[cfg(feature = "arm")]
    v.push(Box::new(ArmModel::arm()));
    v
}

const APM_CTX: usize = 256; // SSE contexts: the partial-byte node `c0`

/// The shared predictor state driven identically by both directions: encode and
/// decode differ only in where each bit comes from (read from the input vs.
/// decoded from the stream) and which coder consumes it.
struct CodecState {
    models: Vec<Box<dyn Model>>,
    mixer: Mixer,
    apm: Apm,
    ctx: Context,
    stretched: Vec<i32>,
}

impl CodecState {
    fn new(capacity: usize) -> Self {
        Self::with_models(models(capacity), capacity)
    }

    fn with_models(models: Vec<Box<dyn Model>>, capacity: usize) -> Self {
        let stretched = vec![0i32; models.len()];
        let mixer = Mixer::new(models.len());
        Self {
            models,
            mixer,
            apm: Apm::new(APM_CTX),
            ctx: Context::with_capacity(capacity),
            stretched,
        }
    }

    /// Predict the next bit as P(bit == 1) in 12-bit probability form: mix the
    /// models, then refine through the SSE stage and blend (SSE-weighted).
    #[allow(clippy::cast_sign_loss)]
    fn predict(&mut self) -> u32 {
        for (m, s) in self.models.iter_mut().zip(&mut self.stretched) {
            *s = m.predict(&self.ctx);
        }
        let c4 = self.ctx.c4;
        let match_sel = self.models.iter().find_map(|m| m.selector()).unwrap_or(0);
        let msel = [
            (c4 & 0xff) as usize,                 // c1
            ((c4 >> 8) & 0xff) as usize,          // c2
            ((c4 >> 16) & 0xff) as usize,         // c3
            (self.ctx.word_hash & 0xff) as usize, // current word
            match_sel,                            // match-length bucket
        ];
        let pm = self
            .mixer
            .mix(&self.stretched, &msel, usize::from(self.ctx.bpos));
        let pa = self.apm.refine(pm, (self.ctx.c0 & 0xff) as usize);
        ((pm + 3 * pa + 2) >> 2) as u32
    }

    /// Commit the actual `bit`: adapt the mixer, SSE, and models, advance context.
    fn commit(&mut self, bit: u8) {
        self.mixer.update(bit);
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
    let mut enc = Encoder::new();
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
pub(crate) fn code_stream_models(models: Vec<Box<dyn Model>>, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    write_varint(&mut out, data.len() as u64);
    let mut state = CodecState::with_models(models, data.len());
    let mut enc = Encoder::new();
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
pub(crate) fn decode_stream_models(models: Vec<Box<dyn Model>>, input: &[u8]) -> Vec<u8> {
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
            v.push(Box::new(IndirectModel::new(2, data.len())) as Box<dyn Model>);
            v.push(Box::new(IndirectModel::new(4, data.len())) as Box<dyn Model>);
            v.push(Box::new(IndirectModel::new(6, data.len())) as Box<dyn Model>);
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
                v.push(Box::new(m) as Box<dyn Model>);
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
            v.push(Box::new(ContextModel::sparse(mask, cap)) as Box<dyn Model>);
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
        let mk = |word: bool| -> Vec<Box<dyn Model>> {
            let mut v: Vec<Box<dyn Model>> = (0..=6)
                .map(|n| Box::new(ContextModel::new(n, cap)) as Box<dyn Model>)
                .collect();
            if word {
                v.push(Box::new(ContextModel::word(cap)));
            }
            v.push(Box::new(MatchModel::new()));
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
