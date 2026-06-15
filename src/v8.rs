//! v8 neural codec — pure-Rust BitNet transformer inference + arithmetic coder.
//!
//! This is the *submission-side* of the v8 neural arm: no `burn`, no threads.
//! It loads a weight blob packed by the training side (`examples/neural.rs`),
//! runs a scalar forward pass (LLVM auto-vectorized), and drives a multi-symbol
//! range coder. Compressor and decompressor share the identical per-position
//! forward, so the codec round-trips by construction — each invocation is
//! self-consistent regardless of float-order differences against the trainer.
//!
//! Weights are ternary BitNet (`{-1,0,1}` + per-tensor scale), shipped
//! trit-packed at 5 trits/byte ≈ 1.6 bits/weight (blob v4; v3's 2-bit packing
//! still reads) — including the tied token embedding. Only the positional table
//! and LayerNorm affine params stay f32; they are a small fraction of the count.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
#![allow(clippy::many_single_char_names, clippy::similar_names)]
// The hot kernels use `mul_add` (emits NEON `fmla.4s`); some non-hot scalar
// expressions don't, so `suboptimal_flops` is allowed file-wide rather than
// rewritten everywhere. Doc header names types (BitNet/LayerNorm/LZR8) in prose.
#![allow(clippy::suboptimal_flops, clippy::doc_markdown)]
#![allow(clippy::cast_lossless, clippy::needless_lifetimes)]
#![allow(clippy::missing_const_for_fn, clippy::needless_range_loop)]

// ---------------------------------------------------------------------------
// Carryless range coder (Subbotin style) — identical to the trainer's coder so
// the bpb measured here matches the bpb measured on the burn side.
// ---------------------------------------------------------------------------
const RC_TOP: u32 = 1 << 24;
const RC_BOT: u32 = 1 << 16;
const CDF_TOTAL: u32 = 1 << 16;

struct RangeEncoder {
    low: u32,
    range: u32,
    out: Vec<u8>,
}
impl RangeEncoder {
    fn new() -> Self {
        Self {
            low: 0,
            range: u32::MAX,
            out: Vec::new(),
        }
    }
    fn encode(&mut self, cum: u32, freq: u32, tot: u32) {
        let r = self.range / tot;
        self.low = self.low.wrapping_add(r * cum);
        self.range = r * freq;
        loop {
            if (self.low ^ self.low.wrapping_add(self.range)) < RC_TOP {
            } else if self.range < RC_BOT {
                self.range = self.low.wrapping_neg() & (RC_BOT - 1);
            } else {
                break;
            }
            self.out.push((self.low >> 24) as u8);
            self.low <<= 8;
            self.range <<= 8;
        }
    }
    fn finish(mut self) -> Vec<u8> {
        for _ in 0..4 {
            self.out.push((self.low >> 24) as u8);
            self.low <<= 8;
        }
        self.out
    }
}

struct RangeDecoder<'a> {
    low: u32,
    range: u32,
    code: u32,
    inp: &'a [u8],
    pos: usize,
}
impl<'a> RangeDecoder<'a> {
    fn new(inp: &'a [u8]) -> Self {
        let mut d = Self {
            low: 0,
            range: u32::MAX,
            code: 0,
            inp,
            pos: 0,
        };
        for _ in 0..4 {
            d.code = (d.code << 8) | d.next();
        }
        d
    }
    fn next(&mut self) -> u32 {
        let b = self.inp.get(self.pos).copied().unwrap_or(0) as u32;
        self.pos += 1;
        b
    }
    fn freq(&self, tot: u32) -> u32 {
        let r = self.range / tot;
        (self.code.wrapping_sub(self.low) / r).min(tot - 1)
    }
    fn update(&mut self, cum: u32, freq: u32, tot: u32) {
        let r = self.range / tot;
        self.low = self.low.wrapping_add(r * cum);
        self.range = r * freq;
        loop {
            if (self.low ^ self.low.wrapping_add(self.range)) < RC_TOP {
            } else if self.range < RC_BOT {
                self.range = self.low.wrapping_neg() & (RC_BOT - 1);
            } else {
                break;
            }
            self.code = (self.code << 8) | self.next();
            self.low <<= 8;
            self.range <<= 8;
        }
    }
}

/// Probability vector → strictly-increasing integer CDF summing to `CDF_TOTAL`,
/// every symbol floored at freq ≥ 1. Deterministic; identical on both sides.
fn probs_to_cdf(probs: &[f32]) -> Vec<u32> {
    let n = probs.len();
    let mut freqs = vec![1u32; n];
    let remaining = CDF_TOTAL - n as u32;
    let sum: f32 = probs.iter().sum::<f32>().max(1e-9);
    let mut used = 0u32;
    for i in 0..n {
        let add = ((probs[i] / sum) * remaining as f32) as u32;
        freqs[i] += add;
        used += add;
    }
    let leftover = remaining - used;
    let argmax = (0..n)
        .max_by(|&a, &b| probs[a].total_cmp(&probs[b]))
        .unwrap();
    freqs[argmax] += leftover;
    let mut cdf = vec![0u32; n + 1];
    for i in 0..n {
        cdf[i + 1] = cdf[i] + freqs[i];
    }
    cdf
}

// ---------------------------------------------------------------------------
// Packed weight blob.
//
// Layout (all little-endian; header is u32s, payload is f32 / packed-ternary):
//   magic = "LZR8" (0x385258_4C as LE bytes), version u32
//   vocab, d_model, n_layers, n_heads, ffn, ctx     (u32 each)
//   bpe      : merge list (`crate::bpe::Bpe::to_bytes`)
//   tok      : ternary linear [vocab, d]   (the tied embedding / unembedding)
//   pos      : ctx*d   f32
//   norm_f   : gamma[d] f32, beta[d] f32
//   header also carries n_experts (u32) after ctx.
//   blocks[n_layers], each:
//     norm1  : gamma[d] f32, beta[d] f32
//     wq,wk,wv,wo : each ternary linear (d x d)
//     norm2  : gamma[d] f32, beta[d] f32
//     router : f32 [n_experts, d]   (kept full precision; routing is sensitive)
//     experts[n_experts], each: ff1 ternary (ffn x d), ff2 ternary (d x ffn)
//   a ternary linear is: scale f32, then the packed trits — v3: ceil(out*in/4)
//   bytes at 2 bits/trit (0b00=-1, 0b01=0, 0b10=+1); v4: ceil(out*in/5) bytes,
//   5 trits/byte as little-endian base-3 digits (digit = trit + 1).
//
// Mixture-of-experts FFN: each token routes to the TOP_K of n_experts (by a
// softmax over the router logits, gates renormalized over the chosen experts),
// so per-token compute is k experts, not all of them — that decouples capacity
// (which ships in L(D)) from inference cost. `ffn` is the per-expert hidden size.
// The token embedding is ternary, tied (the same table is the unembedding).
// ---------------------------------------------------------------------------

pub(crate) const MAGIC: u32 = 0x3852_5A4C; // "LZR8" little-endian
// v3: ternary at 2 bits/weight. v4: trit-packed, 5 trits/byte (3^5 = 243 ≤ 256)
// ≈ 1.6 bits/weight — ~20% smaller ternary tables, ~0.035 bpb off L(D) at 43M
// params. The reader accepts both so v3 checkpoints stay measurable.
pub(crate) const VERSION: u32 = 4;
const TOP_K: usize = 2;
/// Trits per packed byte in the v4 format.
const TRITS_PER_BYTE: usize = 5;

#[derive(Clone, Copy, Debug)]
struct Config {
    vocab: usize,
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    ffn: usize, // per-expert hidden size
    ctx: usize,
    n_experts: usize,
}

struct TernLinear {
    scale: f32,
    // out*in ternary values {-1,0,1} as i8, row-major [out, in]. Activations are
    // int8-quantized too, so the matvec is an integer i8×i8→i32 dot that
    // auto-vectorizes to NEON `sdot` (and the AVX2/VNNI equivalents on x86). The
    // on-disk blob stays trit-packed (L(D) unaffected); this is the in-RAM copy.
    w: Vec<i8>,
    out_f: usize,
    in_f: usize,
}

struct LayerNorm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

struct Expert {
    ff1: TernLinear,
    ff2: TernLinear,
}

struct Block {
    norm1: LayerNorm,
    wq: TernLinear,
    wk: TernLinear,
    wv: TernLinear,
    wo: TernLinear,
    norm2: LayerNorm,
    router: Vec<f32>, // [n_experts, d]
    experts: Vec<Expert>,
}

pub(crate) struct Model {
    cfg: Config,
    bpe: crate::bpe::Bpe,
    // Tied token embedding [vocab, d]: row-gathered (widened i8→f32, d elements
    // per token) for the lookup, and run through the int8 `sdot` matvec as the
    // unembedding — which int8-quantizes the final activations, a deliberate
    // departure from the trainer's un-quantized unembed. Measured 2026-06-10 on
    // the 24M enwik9 checkpoint: Δ L(C) ≤ 0.0001 bpb across three 256 KB slices,
    // throughput 0.085 → 0.044 ms/byte (the unembed was ~85% of the forward).
    tok: TernLinear,
    pos: Vec<f32>, // ctx*d f32
    norm_f: LayerNorm,
    blocks: Vec<Block>,
}

// --- blob reader -----------------------------------------------------------

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
    version: u32,
}
impl Reader<'_> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn f32(&mut self) -> f32 {
        let v = f32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn f32s(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f32()).collect()
    }
    fn tern(&mut self, out_f: usize, in_f: usize) -> TernLinear {
        let scale = self.f32();
        let n = out_f * in_f;
        let mut w = Vec::with_capacity(n);
        if self.version == 3 {
            // 2 bits/trit: 0b00=-1, 0b01=0, 0b10=+1
            let nbytes = n.div_ceil(4);
            for bi in 0..nbytes {
                let byte = self.b[self.p + bi];
                for k in 0..4 {
                    if w.len() == n {
                        break;
                    }
                    let code = (byte >> (2 * k)) & 0b11;
                    w.push(match code {
                        0 => -1i8,
                        1 => 0,
                        _ => 1,
                    });
                }
            }
            self.p += nbytes;
        } else {
            // 5 trits/byte, base-3 little-endian digits, digit = trit + 1
            let nbytes = n.div_ceil(TRITS_PER_BYTE);
            for bi in 0..nbytes {
                let mut code = self.b[self.p + bi];
                for _ in 0..TRITS_PER_BYTE {
                    if w.len() == n {
                        break;
                    }
                    w.push((code % 3) as i8 - 1);
                    code /= 3;
                }
            }
            self.p += nbytes;
        }
        TernLinear {
            scale,
            w,
            out_f,
            in_f,
        }
    }
    fn ln(&mut self, d: usize) -> LayerNorm {
        LayerNorm {
            gamma: self.f32s(d),
            beta: self.f32s(d),
        }
    }
}

impl Model {
    pub(crate) fn from_blob(blob: &[u8]) -> Self {
        let mut r = Reader {
            b: blob,
            p: 0,
            version: 0,
        };
        assert_eq!(r.u32(), MAGIC, "v8 weight blob: bad magic");
        r.version = r.u32();
        assert!(
            r.version == 3 || r.version == VERSION,
            "v8 weight blob: unsupported version {}",
            r.version
        );
        let cfg = Config {
            vocab: r.u32() as usize,
            d_model: r.u32() as usize,
            n_layers: r.u32() as usize,
            n_heads: r.u32() as usize,
            ffn: r.u32() as usize,
            ctx: r.u32() as usize,
            n_experts: r.u32() as usize,
        };
        let (d, ffn) = (cfg.d_model, cfg.ffn);
        let (bpe, used) = crate::bpe::Bpe::from_bytes(&blob[r.p..]);
        r.p += used;
        let tok = r.tern(cfg.vocab, d);
        let pos = r.f32s(cfg.ctx * d);
        let norm_f = r.ln(d);
        let blocks = (0..cfg.n_layers)
            .map(|_| Block {
                norm1: r.ln(d),
                wq: r.tern(d, d),
                wk: r.tern(d, d),
                wv: r.tern(d, d),
                wo: r.tern(d, d),
                norm2: r.ln(d),
                router: r.f32s(cfg.n_experts * d),
                experts: (0..cfg.n_experts)
                    .map(|_| Expert {
                        ff1: r.tern(ffn, d),
                        ff2: r.tern(d, ffn),
                    })
                    .collect(),
            })
            .collect();
        Self {
            cfg,
            bpe,
            tok,
            pos,
            norm_f,
            blocks,
        }
    }

    /// `(vocab, d_model, n_layers, n_heads, ffn, ctx, n_experts)` for reporting.
    pub(crate) const fn dims(&self) -> (usize, usize, usize, usize, usize, usize, usize) {
        let c = self.cfg;
        (
            c.vocab,
            c.d_model,
            c.n_layers,
            c.n_heads,
            c.ffn,
            c.ctx,
            c.n_experts,
        )
    }

    /// `(total params, params active per token)`. MoE: only `TOP_K` experts run
    /// per token, so active ≪ total — that gap is the inference-speed win.
    pub(crate) const fn num_params(&self) -> (usize, usize) {
        let c = self.cfg;
        let (d, ffn) = (c.d_model, c.ffn);
        let attn = 2 * d + 4 * d * d + 2 * d; // norms + qkvo
        let router = c.n_experts * d;
        let expert = 2 * d * ffn;
        let common = c.vocab * d + c.ctx * d + 2 * d;
        let total = common + c.n_layers * (attn + router + c.n_experts * expert);
        let active = common + c.n_layers * (attn + router + TOP_K * expert);
        (total, active)
    }
}

// --- scalar math -----------------------------------------------------------

fn erf(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26, max abs error ~1.5e-7.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152) * t) + 1.421_413_7) * t - 0.284_496_73) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    sign * y
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// LayerNorm one row (`row` and `out` both length `d`).
fn layernorm_row(out: &mut [f32], row: &[f32], ln: &LayerNorm) {
    let d = row.len();
    let mean = row.iter().sum::<f32>() / d as f32;
    let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
    let inv = 1.0 / (var + 1e-5).sqrt();
    for i in 0..d {
        out[i] = (row[i] - mean) * inv * ln.gamma[i] + ln.beta[i];
    }
}

/// Test-only [t, d] wrapper over `layernorm_row` (the `predict` oracle uses it).
#[cfg(test)]
fn layernorm(x: &[f32], t: usize, d: usize, ln: &LayerNorm) -> Vec<f32> {
    let mut out = vec![0f32; t * d];
    for r in 0..t {
        layernorm_row(&mut out[r * d..(r + 1) * d], &x[r * d..(r + 1) * d], ln);
    }
    out
}

/// Auto-vectorizable f32 dot product. The `LANES`-wide accumulator array makes
/// the reduction associative-by-construction (each lane is an independent sum),
/// which is what lets LLVM emit SIMD FMAs — a single scalar accumulator cannot
/// vectorize without `-ffast-math`. `chunks_exact` keeps the hot loop
/// bounds-check-free. `a` and `b` must be the same length.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 16;
    let mut acc = [0f32; LANES];
    let mut ac = a.chunks_exact(LANES);
    let mut bc = b.chunks_exact(LANES);
    for (av, bv) in ac.by_ref().zip(bc.by_ref()) {
        for l in 0..LANES {
            acc[l] = av[l].mul_add(bv[l], acc[l]);
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (av, bv) in ac.remainder().iter().zip(bc.remainder()) {
        sum = av.mul_add(*bv, sum);
    }
    sum
}

/// Integer dot of int8 activations × ternary i8 weights → i32. Integer addition
/// is associative, so the multi-accumulator form is exact (not just `-ffast-math`
/// fast) and auto-vectorizes to NEON `sdot` (AVX2 `pmaddubsw` / VNNI `vpdpbusd`
/// on x86). No overflow: `|acc| ≤ in_f·127 ≪ i32::MAX` for any realistic `in_f`.
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    // Single-accumulator reduction: integer add IS associative, so LLVM may
    // both vectorize it and pattern-match the `i32 += sext(i8)·sext(i8)` idiom to
    // `sdot`. (A multi-accumulator array, needed for f32, instead blocks it.)
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i32::from(x) * i32::from(y))
        .sum()
}

/// `dst += a * src` over equal-length slices — the contiguous form lets the
/// attention value-mixing vectorize (the strided gather form does not).
fn saxpy(dst: &mut [f32], a: f32, src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d = a.mul_add(*s, *d);
    }
}

/// BitLinear one row: int8-quantize `row` (codes into `xq` scratch, len ≥ in_f)
/// and integer-matvec against the ternary weights into `out` (len out_f).
/// `out[o] = (Σ xq·w) · s · scale` reproduces the trainer's dequantized math.
fn bitlinear_row(out: &mut [f32], row: &[f32], lin: &TernLinear, xq: &mut [i8]) {
    let in_f = lin.in_f;
    let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
    let s = (amax / 127.0).max(1e-5);
    for i in 0..in_f {
        xq[i] = (row[i] / s).round().clamp(-127.0, 127.0) as i8;
    }
    let f = s * lin.scale;
    for o in 0..lin.out_f {
        out[o] = dot_i8(&xq[..in_f], &lin.w[o * in_f..o * in_f + in_f]) as f32 * f;
    }
}

/// Test-only [t, *] wrapper over `bitlinear_row` (the `predict` oracle uses it).
#[cfg(test)]
fn bitlinear(x: &[f32], t: usize, lin: &TernLinear) -> Vec<f32> {
    let (in_f, out_f) = (lin.in_f, lin.out_f);
    let mut out = vec![0f32; t * out_f];
    let mut xq = vec![0i8; in_f];
    for r in 0..t {
        bitlinear_row(
            &mut out[r * out_f..(r + 1) * out_f],
            &x[r * in_f..(r + 1) * in_f],
            lin,
            &mut xq,
        );
    }
    out
}

/// Indices of the two largest elements (`a` ≥ `b`), deterministic on ties.
fn top2(v: &[f32]) -> (usize, usize) {
    let (mut a, mut b) = if v[0] >= v[1] { (0, 1) } else { (1, 0) };
    for i in 2..v.len() {
        if v[i] > v[a] {
            b = a;
            a = i;
        } else if v[i] > v[b] {
            b = i;
        }
    }
    (a, b)
}

fn softmax_inplace(v: &mut [f32]) {
    let m = v.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut s = 0f32;
    for x in v.iter_mut() {
        *x = (*x - m).exp();
        s += *x;
    }
    let inv = 1.0 / s;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Incremental decode state: per-layer K and V for the tokens currently in the
/// context window, row-major `[len, d_model]`. Past tokens' K/V are finalized by
/// causality, so they never change once written — that is what the cache
/// exploits to turn the per-byte forward from O(t²) into O(t).
pub(crate) struct Cache {
    k: Vec<Vec<f32>>, // k[layer]: len*d
    v: Vec<Vec<f32>>, // v[layer]: len*d
    len: usize,
}

/// Per-step scratch buffers, allocated once and reused across every byte —
/// `step` would otherwise heap-allocate ~70 short-lived `Vec`s per byte. All
/// are length `d_model` except `ff`/`xq` (`ffn`), `scores` (`ctx`), `logits`
/// (`vocab`).
pub(crate) struct Scratch {
    x: Vec<f32>,
    nrm: Vec<f32>, // layernorm output (norm1 / norm2 / final, used then consumed)
    q: Vec<f32>,
    kbuf: Vec<f32>,
    vbuf: Vec<f32>,
    ctx: Vec<f32>,
    tmp: Vec<f32>,     // attn-out / ff-out
    ff: Vec<f32>,      // expert ff1 hidden (ffn)
    xq: Vec<i8>,       // int8 activation-quant scratch (sized to max in_f)
    rlogits: Vec<f32>, // router logits (n_experts)
    eacc: Vec<f32>,    // weighted expert-output accumulator (d)
    scores: Vec<f32>,
    logits: Vec<f32>,
    // diagnostic: top-2 routing tally [layer][expert], cheap enough to keep hot
    expert_counts: Vec<Vec<u64>>,
}

impl Model {
    pub(crate) fn new_cache(&self) -> Cache {
        let cap = self.cfg.ctx * self.cfg.d_model;
        Cache {
            k: (0..self.cfg.n_layers)
                .map(|_| Vec::with_capacity(cap))
                .collect(),
            v: (0..self.cfg.n_layers)
                .map(|_| Vec::with_capacity(cap))
                .collect(),
            len: 0,
        }
    }

    pub(crate) fn new_scratch(&self) -> Scratch {
        let (d, ffn) = (self.cfg.d_model, self.cfg.ffn);
        Scratch {
            x: vec![0f32; d],
            nrm: vec![0f32; d],
            q: vec![0f32; d],
            kbuf: vec![0f32; d],
            vbuf: vec![0f32; d],
            ctx: vec![0f32; d],
            tmp: vec![0f32; d],
            ff: vec![0f32; ffn],
            xq: vec![0i8; ffn.max(d)],
            rlogits: vec![0f32; self.cfg.n_experts],
            eacc: vec![0f32; d],
            scores: vec![0f32; self.cfg.ctx],
            logits: vec![0f32; self.cfg.vocab],
            expert_counts: vec![vec![0u64; self.cfg.n_experts]; self.cfg.n_layers],
        }
    }

    /// Process one token through the stack, appending its K/V to `cache`, and
    /// write the next-token distribution into `s.logits`. Exact (within float
    /// tolerance) to a from-scratch forward over the same context.
    ///
    /// Learned absolute positions cannot slide without invalidating every cached
    /// token's K/V, so when the window fills (`len == ctx`) the cache is reset and
    /// the incoming token becomes position 0 — a fresh block, matching how the
    /// model trained (contiguous windows from position 0). The just-fed token
    /// seeds the new block, so there is no synthetic boundary token.
    pub(crate) fn step(&self, cache: &mut Cache, s: &mut Scratch, token: i32) {
        let cfg = self.cfg;
        let (d, h) = (cfg.d_model, cfg.n_heads);
        let dk = d / h;
        if cache.len == cfg.ctx {
            for kk in &mut cache.k {
                kk.clear();
            }
            for vv in &mut cache.v {
                vv.clear();
            }
            cache.len = 0;
        }
        let pos = cache.len;
        let id = token as usize;
        let trow = &self.tok.w[id * d..id * d + d];
        for i in 0..d {
            s.x[i] = self.tok.scale * f32::from(trow[i]) + self.pos[pos * d + i];
        }
        let scale = 1.0 / (dk as f32).sqrt();
        for (l, blk) in self.blocks.iter().enumerate() {
            // attention: project the new token, append to cache, attend to all
            layernorm_row(&mut s.nrm, &s.x, &blk.norm1);
            bitlinear_row(&mut s.q, &s.nrm, &blk.wq, &mut s.xq);
            bitlinear_row(&mut s.kbuf, &s.nrm, &blk.wk, &mut s.xq);
            bitlinear_row(&mut s.vbuf, &s.nrm, &blk.wv, &mut s.xq);
            cache.k[l].extend_from_slice(&s.kbuf);
            cache.v[l].extend_from_slice(&s.vbuf);
            let clen = cache.len + 1; // cached positions, including the new token
            s.ctx.iter_mut().for_each(|c| *c = 0.0);
            let scores = &mut s.scores[..clen];
            for head in 0..h {
                let off = head * dk;
                for (p, sc) in scores.iter_mut().enumerate() {
                    let kp = &cache.k[l][p * d + off..p * d + off + dk];
                    *sc = dot(&s.q[off..off + dk], kp) * scale;
                }
                softmax_inplace(scores);
                let dst = &mut s.ctx[off..off + dk];
                for (p, &w) in scores.iter().enumerate() {
                    saxpy(dst, w, &cache.v[l][p * d + off..p * d + off + dk]);
                }
            }
            bitlinear_row(&mut s.tmp, &s.ctx, &blk.wo, &mut s.xq);
            for i in 0..d {
                s.x[i] += s.tmp[i];
            }
            // MoE feed-forward: route the token to its TOP_K experts
            layernorm_row(&mut s.nrm, &s.x, &blk.norm2);
            let ne = self.cfg.n_experts;
            for (e, lg) in s.rlogits[..ne].iter_mut().enumerate() {
                *lg = dot(&s.nrm, &blk.router[e * d..e * d + d]);
            }
            softmax_inplace(&mut s.rlogits[..ne]);
            let (ea, eb) = top2(&s.rlogits[..ne]);
            s.expert_counts[l][ea] += 1;
            s.expert_counts[l][eb] += 1;
            let denom = s.rlogits[ea] + s.rlogits[eb];
            s.eacc.iter_mut().for_each(|c| *c = 0.0);
            for (e, g) in [(ea, s.rlogits[ea] / denom), (eb, s.rlogits[eb] / denom)] {
                let exp = &blk.experts[e];
                bitlinear_row(&mut s.ff, &s.nrm, &exp.ff1, &mut s.xq);
                for v in &mut s.ff {
                    *v = gelu(*v);
                }
                bitlinear_row(&mut s.tmp, &s.ff, &exp.ff2, &mut s.xq);
                for i in 0..d {
                    s.eacc[i] += g * s.tmp[i];
                }
            }
            for i in 0..d {
                s.x[i] += s.eacc[i];
            }
        }
        cache.len += 1;
        layernorm_row(&mut s.nrm, &s.x, &self.norm_f);
        bitlinear_row(&mut s.logits, &s.nrm, &self.tok, &mut s.xq);
        softmax_inplace(&mut s.logits);
    }

    /// Next-byte probability distribution given a context of token ids.
    /// Full O(t²) recompute — the reference the cached `step` is checked against.
    #[cfg(test)]
    fn predict(&self, ids: &[i32]) -> Vec<f32> {
        let cfg = self.cfg;
        let (d, t, h) = (cfg.d_model, ids.len(), cfg.n_heads);
        let dk = d / h;
        // embed + positional
        let mut x = vec![0f32; t * d];
        for r in 0..t {
            let id = ids[r] as usize;
            let trow = &self.tok.w[id * d..id * d + d];
            for i in 0..d {
                x[r * d + i] = self.tok.scale * f32::from(trow[i]) + self.pos[r * d + i];
            }
        }
        for blk in &self.blocks {
            // --- attention ---
            let hn = layernorm(&x, t, d, &blk.norm1);
            let q = bitlinear(&hn, t, &blk.wq);
            let k = bitlinear(&hn, t, &blk.wk);
            let v = bitlinear(&hn, t, &blk.wv);
            let mut ctx = vec![0f32; t * d];
            let scale = 1.0 / (dk as f32).sqrt();
            for head in 0..h {
                let off = head * dk;
                for r1 in 0..t {
                    let mut scores = vec![0f32; r1 + 1];
                    for (r2, sc) in scores.iter_mut().enumerate() {
                        let qh = &q[r1 * d + off..r1 * d + off + dk];
                        let kh = &k[r2 * d + off..r2 * d + off + dk];
                        *sc = dot(qh, kh) * scale;
                    }
                    softmax_inplace(&mut scores);
                    let dst = &mut ctx[r1 * d + off..r1 * d + off + dk];
                    for (r2, &w) in scores.iter().enumerate() {
                        saxpy(dst, w, &v[r2 * d + off..r2 * d + off + dk]);
                    }
                }
            }
            let o = bitlinear(&ctx, t, &blk.wo);
            for i in 0..t * d {
                x[i] += o[i];
            }
            // --- MoE feed-forward (per token) ---
            let (ne, ffn) = (self.cfg.n_experts, self.cfg.ffn);
            let mut hn2 = vec![0f32; d];
            let mut ff = vec![0f32; ffn];
            let mut tmp = vec![0f32; d];
            let mut xq = vec![0i8; ffn.max(d)];
            let mut rlogits = vec![0f32; ne];
            for r in 0..t {
                layernorm_row(&mut hn2, &x[r * d..r * d + d], &blk.norm2);
                for (e, lg) in rlogits.iter_mut().enumerate() {
                    *lg = dot(&hn2, &blk.router[e * d..e * d + d]);
                }
                softmax_inplace(&mut rlogits);
                let (ea, eb) = top2(&rlogits);
                let denom = rlogits[ea] + rlogits[eb];
                for (e, g) in [(ea, rlogits[ea] / denom), (eb, rlogits[eb] / denom)] {
                    let exp = &blk.experts[e];
                    bitlinear_row(&mut ff, &hn2, &exp.ff1, &mut xq);
                    for v in &mut ff {
                        *v = gelu(*v);
                    }
                    bitlinear_row(&mut tmp, &ff, &exp.ff2, &mut xq);
                    for i in 0..d {
                        x[r * d + i] += g * tmp[i];
                    }
                }
            }
        }
        let xf = layernorm(&x, t, d, &self.norm_f);
        // tied unembedding for the last position only
        let last = &xf[(t - 1) * d..t * d];
        let mut logits = vec![0f32; cfg.vocab];
        let mut xq = vec![0i8; d];
        bitlinear_row(&mut logits, last, &self.tok, &mut xq);
        softmax_inplace(&mut logits);
        logits
    }

    /// Compress `bytes`: BPE-tokenize, then code each token id through the
    /// BOS-seeded incremental forward. Decode runs the identical forward, so
    /// the codec round-trips by construction.
    ///
    /// Tokenization runs in byte chunks (`TOK_CHUNK`) rather than over the whole
    /// input at once: the tokenizer's working structures are O(chunk), so a
    /// 1 GB input stays well under the memory budget instead of needing tens of
    /// GB for one linked list. This is safe for the codec — the decoder expands
    /// whatever token ids the encoder emitted, so chunking only the *encoder's*
    /// tokenization cannot affect decodability; the sole cost is that a token
    /// straddling a chunk seam is split (a few sub-optimal bytes per chunk). The
    /// range coder and KV-cache stream continuously across chunks, so in-window
    /// context is unaffected.
    pub(crate) fn compress(&self, bytes: &[u8], bos: i32) -> Vec<u8> {
        const TOK_CHUNK: usize = 8 << 20; // 8 MiB
        self.compress_chunked(bytes, bos, TOK_CHUNK)
    }

    fn compress_chunked(&self, bytes: &[u8], bos: i32, chunk: usize) -> Vec<u8> {
        let mut enc = RangeEncoder::new();
        let mut cache = self.new_cache();
        let mut s = self.new_scratch();
        self.step(&mut cache, &mut s, bos);
        let mut off = 0;
        while off < bytes.len() {
            let end = (off + chunk).min(bytes.len());
            for id in self.bpe.encode(&bytes[off..end]) {
                let cdf = probs_to_cdf(&s.logits);
                let id = id as usize;
                enc.encode(cdf[id], cdf[id + 1] - cdf[id], CDF_TOTAL);
                self.step(&mut cache, &mut s, id as i32);
            }
            off = end;
        }
        enc.finish()
    }

    /// Decompress to exactly `n` bytes. Decodes token ids and expands each to
    /// its byte span until the byte count is reached — token boundaries align
    /// with `n` because the encoder tokenized exactly those `n` bytes.
    pub(crate) fn decompress(&self, archive: &[u8], n: usize, bos: i32) -> Vec<u8> {
        let mut dec = RangeDecoder::new(archive);
        let mut out = Vec::with_capacity(n);
        let mut cache = self.new_cache();
        let mut s = self.new_scratch();
        self.step(&mut cache, &mut s, bos);
        while out.len() < n {
            let cdf = probs_to_cdf(&s.logits);
            let f = dec.freq(CDF_TOTAL);
            let mut sym = cdf.partition_point(|&c| c <= f) - 1;
            if sym + 1 >= cdf.len() {
                sym = cdf.len() - 2;
            }
            dec.update(cdf[sym], cdf[sym + 1] - cdf[sym], CDF_TOTAL);
            out.extend_from_slice(self.bpe.expand(sym as u32));
            self.step(&mut cache, &mut s, sym as i32);
        }
        out.truncate(n);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Vec<u8> {
        "the quick brown fox the lazy dog the the the quick fox jumps over "
            .repeat(60)
            .into_bytes()
    }

    /// Assemble a valid v2 blob for `bpe` with deterministic pseudo-random
    /// ternary/f32 weights — a random-init model that round-trips (round-trip
    /// is weight-independent), exercising reader + tokenizer + forward + coder
    /// without the GPU trainer.
    #[allow(clippy::too_many_arguments)]
    fn build_blob(
        bpe: &crate::bpe::Bpe,
        d: usize,
        layers: usize,
        heads: usize,
        ffn: usize,
        ctx: usize,
        n_experts: usize,
        seed: u32,
    ) -> Vec<u8> {
        let vocab = bpe.vocab_size();
        let mut out = Vec::new();
        for v in [
            MAGIC,
            VERSION,
            vocab as u32,
            d as u32,
            layers as u32,
            heads as u32,
            ffn as u32,
            ctx as u32,
            n_experts as u32,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&bpe.to_bytes());
        let mut s = seed;
        let mut nf = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) - 0.5
        };
        let push_f = |out: &mut Vec<u8>, n: usize, f: &mut dyn FnMut() -> f32| {
            for _ in 0..n {
                out.extend_from_slice(&f().to_le_bytes());
            }
        };
        let push_tern = |out: &mut Vec<u8>, of: usize, inf: usize, f: &mut dyn FnMut() -> f32| {
            out.extend_from_slice(&0.3f32.to_le_bytes()); // scale
            let n = of * inf;
            for chunk in 0..n.div_ceil(TRITS_PER_BYTE) {
                let mut byte = 0u8;
                let mut pow = 1u8;
                for k in 0..TRITS_PER_BYTE {
                    if chunk * TRITS_PER_BYTE + k >= n {
                        break;
                    }
                    let digit = (((f() + 0.5) * 3.0) as u8).min(2);
                    byte += digit * pow;
                    pow = pow.saturating_mul(3);
                }
                out.push(byte);
            }
        };
        push_tern(&mut out, vocab, d, &mut nf); // ternary tok embedding
        push_f(&mut out, ctx * d, &mut nf); // f32 pos
        push_f(&mut out, d, &mut nf); // norm_f gamma
        push_f(&mut out, d, &mut nf); // norm_f beta
        for _ in 0..layers {
            push_f(&mut out, d, &mut nf);
            push_f(&mut out, d, &mut nf); // norm1
            for _ in 0..4 {
                push_tern(&mut out, d, d, &mut nf); // wq,wk,wv,wo
            }
            push_f(&mut out, d, &mut nf);
            push_f(&mut out, d, &mut nf); // norm2
            push_f(&mut out, n_experts * d, &mut nf); // router (f32)
            for _ in 0..n_experts {
                push_tern(&mut out, ffn, d, &mut nf); // expert ff1
                push_tern(&mut out, d, ffn, &mut nf); // expert ff2
            }
        }
        out
    }

    fn tiny_blob() -> (Vec<u8>, usize) {
        let bpe = crate::bpe::Bpe::learn(&corpus(), 288);
        let vocab = bpe.vocab_size();
        (build_blob(&bpe, 8, 2, 2, 16, 32, 4, 0x1234_5678), vocab)
    }

    /// v3 (2-bit) blobs must keep loading — live-run checkpoints are v3 until
    /// the trainer relaunches. Builds the same tiny model in both formats and
    /// asserts identical weights and identical archives.
    #[test]
    fn reads_v3_blobs() {
        let (blob_v4, _) = tiny_blob();
        let model_v4 = Model::from_blob(&blob_v4);
        // rebuild the same model as a v3 blob: header, bpe, each section
        // re-packed at 2 bits/trit from the parsed weights
        let bpe = crate::bpe::Bpe::learn(&corpus(), 288);
        let mut out = Vec::new();
        let c = model_v4.cfg;
        for v in [
            MAGIC,
            3u32,
            c.vocab as u32,
            c.d_model as u32,
            c.n_layers as u32,
            c.n_heads as u32,
            c.ffn as u32,
            c.ctx as u32,
            c.n_experts as u32,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&bpe.to_bytes());
        let pack2 = |out: &mut Vec<u8>, lin: &TernLinear| {
            out.extend_from_slice(&lin.scale.to_le_bytes());
            for chunk in lin.w.chunks(4) {
                let mut byte = 0u8;
                for (k, &t) in chunk.iter().enumerate() {
                    byte |= ((t + 1) as u8) << (2 * k);
                }
                out.push(byte);
            }
        };
        let push_f = |out: &mut Vec<u8>, vals: &[f32]| {
            for v in vals {
                out.extend_from_slice(&v.to_le_bytes());
            }
        };
        pack2(&mut out, &model_v4.tok);
        push_f(&mut out, &model_v4.pos);
        push_f(&mut out, &model_v4.norm_f.gamma);
        push_f(&mut out, &model_v4.norm_f.beta);
        for blk in &model_v4.blocks {
            push_f(&mut out, &blk.norm1.gamma);
            push_f(&mut out, &blk.norm1.beta);
            for lin in [&blk.wq, &blk.wk, &blk.wv, &blk.wo] {
                pack2(&mut out, lin);
            }
            push_f(&mut out, &blk.norm2.gamma);
            push_f(&mut out, &blk.norm2.beta);
            push_f(&mut out, &blk.router);
            for exp in &blk.experts {
                pack2(&mut out, &exp.ff1);
                pack2(&mut out, &exp.ff2);
            }
        }
        let model_v3 = Model::from_blob(&out);
        assert_eq!(model_v3.tok.w, model_v4.tok.w, "v3/v4 weights must agree");
        let msg = corpus()[..200].to_vec();
        assert_eq!(
            model_v3.compress(&msg, 0),
            model_v4.compress(&msg, 0),
            "v3 and v4 readings of the same model must produce identical archives"
        );
    }

    #[test]
    fn roundtrips_tiny_model() {
        let model = Model::from_blob(&tiny_blob().0);
        let msg = corpus()[..400].to_vec();
        let arc = model.compress(&msg, 0);
        let dec = model.decompress(&arc, msg.len(), 0);
        assert_eq!(dec, msg, "v8 codec must round-trip");
    }

    #[test]
    fn chunked_tokenization_roundtrips_across_seams() {
        // Tokenizing in small chunks must still decode byte-for-byte (the decoder
        // expands whatever ids the encoder emitted) — exercise many chunk seams
        // with a tiny chunk size over a multi-token corpus.
        let model = Model::from_blob(&tiny_blob().0);
        let msg = corpus()[..1000].to_vec();
        for chunk in [3usize, 7, 16, 64] {
            let arc = model.compress_chunked(&msg, 0, chunk);
            let dec = model.decompress(&arc, msg.len(), 0);
            assert_eq!(dec, msg, "chunked compress (chunk={chunk}) must round-trip");
        }
        // a chunk ≥ len equals the single-shot path
        let whole = model.compress_chunked(&msg, 0, msg.len());
        assert_eq!(model.compress(&msg, 0), whole);
    }

    #[test]
    fn roundtrips_arbitrary_bytes() {
        // every byte value is a base token, so non-corpus input still round-trips
        let model = Model::from_blob(&tiny_blob().0);
        let msg: Vec<u8> = (0..=255u8).chain((0..=255u8).rev()).collect();
        let arc = model.compress(&msg, 0);
        let dec = model.decompress(&arc, msg.len(), 0);
        assert_eq!(dec, msg);
    }

    #[test]
    fn cached_step_matches_full_recompute() {
        // Within the context window, incremental `step` must equal the O(t²)
        // reference `predict` to float tolerance — that is the cache's contract.
        let (blob, vocab) = tiny_blob();
        let model = Model::from_blob(&blob);
        let toks: Vec<i32> = (0..11).map(|i| (i * 7 % vocab) as i32).collect(); // < ctx=32
        let mut cache = model.new_cache();
        let mut s = model.new_scratch();
        for end in 1..=toks.len() {
            model.step(&mut cache, &mut s, toks[end - 1]);
            let reference = model.predict(&toks[..end]);
            for (a, b) in s.logits.iter().zip(&reference) {
                assert!((a - b).abs() < 1e-5, "cache diverged: {a} vs {b}");
            }
        }
    }

    /// Offline router-health probe against a checkpoint blob on disk (default
    /// `/tmp/v8_step4k.bin`, override via `V8_PROBE_BLOB`): forward an enwik9
    /// slice and print per-layer expert-usage entropy. Healthy top-2-of-32
    /// routing should read ≳3.5 bits; ~1 bit means the router collapsed.
    /// Run: `cargo test --release router_health -- --ignored --nocapture`
    #[test]
    #[ignore = "offline diagnostic: needs a checkpoint blob and enwik9 on disk"]
    fn router_health_probe() {
        let path = std::env::var("V8_PROBE_BLOB").unwrap_or_else(|_| "/tmp/v8_step4k.bin".into());
        let blob = std::fs::read(&path).expect("probe blob");
        let data = std::fs::read("assets/enwik9").expect("enwik9");
        let model = Model::from_blob(&blob);
        let off = 1 << 20;
        let ids = model.bpe.encode(&data[off..off + 64 * 1024]);
        let mut cache = model.new_cache();
        let mut s = model.new_scratch();
        for &id in &ids {
            model.step(&mut cache, &mut s, id as i32);
        }
        for (l, counts) in s.expert_counts.iter().enumerate() {
            let tot: u64 = counts.iter().sum();
            let h: f64 = counts
                .iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let p = c as f64 / tot as f64;
                    -p * p.log2()
                })
                .sum();
            let mut sorted = counts.clone();
            sorted.sort_unstable_by(|a, b| b.cmp(a));
            println!(
                "layer {l}: usage entropy {h:.2} bits (uniform {:.2}), top4 {:?} of {tot}",
                (model.cfg.n_experts as f64).log2(),
                &sorted[..4.min(sorted.len())]
            );
        }
    }

    /// Offline context-use probe against a checkpoint blob (same selection as
    /// `router_health_probe`): per-position-in-window cross-entropy over an
    /// enwik9 slice. Flat across positions ⟹ the model is using no context
    /// (unigram regime); falling with position ⟹ in-context circuits forming.
    /// Run: `cargo test --release position_ce -- --ignored --nocapture`
    #[test]
    #[ignore = "offline diagnostic: needs a checkpoint blob and enwik9 on disk"]
    fn position_ce_probe() {
        let path = std::env::var("V8_PROBE_BLOB").unwrap_or_else(|_| "/tmp/v8_step4k.bin".into());
        let blob = std::fs::read(&path).expect("probe blob");
        let data = std::fs::read("assets/enwik9").expect("enwik9");
        let model = Model::from_blob(&blob);
        let off = 1 << 20;
        let ids = model.bpe.encode(&data[off..off + 256 * 1024]);
        let mut cache = model.new_cache();
        let mut s = model.new_scratch();
        let nb = 8;
        let bucket_w = model.cfg.ctx / nb;
        let (mut bits, mut count) = (vec![0f64; nb], vec![0u64; nb]);
        model.step(&mut cache, &mut s, 0);
        for &id in &ids {
            // s.logits predicts `id` given the cache.len tokens currently held
            let b = ((cache.len - 1) / bucket_w).min(nb - 1);
            bits[b] -= f64::from(s.logits[id as usize].max(1e-12)).log2();
            count[b] += 1;
            model.step(&mut cache, &mut s, id as i32);
        }
        for i in 0..nb {
            println!(
                "ctx pos [{:3}..{:3}): {:.3} bits/token over {}",
                i * bucket_w,
                (i + 1) * bucket_w,
                bits[i] / count[i] as f64,
                count[i]
            );
        }
    }

    /// Reconstruct a copy of `m` keeping only the top-`k` experts per layer (by
    /// the `ranking`), with the matching router rows — a faithful E_k variant of
    /// a model trained at E32. Mutates `m` in place; no shipped-code hook.
    fn prune_experts(m: &mut Model, k: usize, ranking: &[Vec<usize>]) {
        let d = m.cfg.d_model;
        for (l, blk) in m.blocks.iter_mut().enumerate() {
            let mut keep: Vec<usize> = ranking[l].iter().copied().take(k).collect();
            keep.sort_unstable();
            let old_e = std::mem::take(&mut blk.experts);
            blk.experts = old_e
                .into_iter()
                .enumerate()
                .filter(|(i, _)| keep.binary_search(i).is_ok())
                .map(|(_, e)| e)
                .collect();
            let old_r = std::mem::take(&mut blk.router);
            let mut nr = Vec::with_capacity(k * d);
            for &i in &keep {
                nr.extend_from_slice(&old_r[i * d..i * d + d]);
            }
            blk.router = nr;
        }
        m.cfg.n_experts = k;
    }

    /// Expert-count ablation on a trained checkpoint: drop to the top-K experts
    /// per layer (ranked by usage on a calibration slice) and measure L(C) on
    /// held slices. The reported L(C) is an UPPER bound on a from-scratch E_K
    /// (ablation routes worse than a model trained with K experts), so a net
    /// improvement here is conservative. L(D) is the analytic trit-packed
    /// estimate (0.0157 + total_M·0.00328, fit to the measured 0.1599 at 43.6M).
    /// Run: `cargo test --release expert_ablation -- --ignored --nocapture`
    #[test]
    #[ignore = "offline: needs assets/v8weights.bin (or V8_PROBE_BLOB) and enwik9"]
    fn expert_ablation() {
        use std::fmt::Write as _;
        let path = std::env::var("V8_PROBE_BLOB").unwrap_or_else(|_| "assets/v8weights.bin".into());
        let blob = std::fs::read(&path).expect("probe blob");
        let data = std::fs::read("assets/enwik9").expect("enwik9");
        let full = Model::from_blob(&blob);
        let (_, _, n_layers, _, _, _, n_experts) = full.dims();

        // calibration: per-layer expert usage on a slice → ranking (most-used first)
        let cal = &data[1 << 20..(1 << 20) + 256 * 1024];
        let mut cache = full.new_cache();
        let mut s = full.new_scratch();
        for id in full.bpe.encode(cal) {
            full.step(&mut cache, &mut s, id as i32);
        }
        let ranking: Vec<Vec<usize>> = (0..n_layers)
            .map(|l| {
                let mut idx: Vec<usize> = (0..n_experts).collect();
                idx.sort_by(|&a, &b| s.expert_counts[l][b].cmp(&s.expert_counts[l][a]));
                idx
            })
            .collect();

        let meas: [(&str, usize); 2] = [("@1MB", 1 << 20), ("@250MB", 250_000_000)];
        println!("\nexpert ablation (top-K by usage; L(C) is an UPPER bound vs from-scratch E_K):");
        for k in [n_experts, 24, 16, 12, 8, 4] {
            if k > n_experts {
                continue;
            }
            let mut m = Model::from_blob(&blob);
            prune_experts(&mut m, k, &ranking);
            let total = m.num_params().0;
            let ld = 0.0157 + (total as f64 / 1e6) * 0.003_28;
            let mut line = format!(
                "  K={k:2}  total {:5.1}M  L(D)~{ld:.4} |",
                total as f64 / 1e6
            );
            for (tag, off) in meas {
                let slice = &data[off..off + 256 * 1024];
                let arc = m.compress(slice, 0);
                let lc = 8.0 * arc.len() as f64 / slice.len() as f64;
                let _ = write!(line, "  {tag} L(C) {lc:.4} net {:.4}", lc + ld);
            }
            println!("{line}");
        }
    }

    /// Integration check on real data: a (reproducible) random point ≥ 1 MB from
    /// either end of enwik8, online-BPE up to that point, then the CPU codec
    /// encodes→decodes 1000 tokens of the following held-out bytes byte-for-byte.
    /// Skips cleanly if `assets/enwik8` is absent (e.g. fresh CI checkout).
    #[test]
    fn enwik8_random_point_roundtrip() {
        let Ok(data) = std::fs::read("assets/enwik8") else {
            eprintln!("skip enwik8_random_point_roundtrip: assets/enwik8 absent");
            return;
        };
        let mb = 1 << 20;
        assert!(data.len() > 4 * mb);
        // reproducible "random" point in [1 MB, len-1 MB]
        let mut state = 0x9E37_79B9u32;
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let p = mb + (state as usize) % (data.len() - 2 * mb);

        let win = 256 * 1024; // bounded online-BPE window up to the point
        let bpe = crate::bpe::Bpe::learn(&data[p - win..p], 2048);
        let after = bpe.encode(&data[p..p + 16 * 1024]);
        assert!(after.len() >= 1000);
        let toks = &after[..1000];
        let mut bytes = Vec::new();
        for &t in toks {
            bytes.extend_from_slice(bpe.expand(t));
        }

        // random-init model over this fresh vocab (round-trip is weight-independent)
        let blob = build_blob(&bpe, 64, 2, 4, 48, 256, 24, 0xABCD_1234);
        let model = Model::from_blob(&blob);
        let arc = model.compress(&bytes, 0);
        let dec = model.decompress(&arc, bytes.len(), 0);
        assert_eq!(
            dec, bytes,
            "CPU codec must round-trip 1000 tokens from a random enwik8 point"
        );
    }
}
