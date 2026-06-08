//! v8 neural codec — pure-Rust BitNet transformer inference + arithmetic coder.
//!
//! This is the *submission-side* of the v8 neural arm: no `burn`, no threads.
//! It loads a weight blob packed by the training side (`examples/neural.rs`),
//! runs a scalar forward pass (LLVM auto-vectorized), and drives a multi-symbol
//! range coder. Compressor and decompressor share the identical per-position
//! forward, so the codec round-trips by construction — each invocation is
//! self-consistent regardless of float-order differences against the trainer.
//!
//! Weights are ternary BitNet (`{-1,0,1}` + per-tensor scale), so they ship at
//! ~2 bits/weight (2-bit packing here; 1.6 b/w trit-packing is the next
//! tightening). Embeddings, positional table, and LayerNorm affine params stay
//! f32 — they are a small fraction of the parameter count.

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
//   blocks[n_layers], each:
//     norm1  : gamma[d] f32, beta[d] f32
//     wq,wk,wv,wo : each ternary linear (d x d)
//     norm2  : gamma[d] f32, beta[d] f32
//     ff1    : ternary linear (ffn x d)
//     ff2    : ternary linear (d x ffn)
//   a ternary linear is: scale f32, then ceil(out*in/4) packed bytes
//   (2 bits/weight: 0b00=-1, 0b01=0, 0b10=+1).
//
// The token embedding is ternary (a vocab×d table is the dominant parameter
// count once the BPE vocab is large); pos and the LayerNorm affines stay f32
// (small). The embedding is tied: the same ternary table is the unembedding.
// ---------------------------------------------------------------------------

pub(crate) const MAGIC: u32 = 0x3852_5A4C; // "LZR8" little-endian
pub(crate) const VERSION: u32 = 2;

#[derive(Clone, Copy, Debug)]
struct Config {
    vocab: usize,
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    ffn: usize,
    ctx: usize,
}

struct TernLinear {
    scale: f32,
    // out*in ternary values {-1,0,1} as f32, row-major [out, in]. f32 (not i8)
    // so the dot-product inner loop is a pure-f32 reduction the compiler can
    // auto-vectorize — costs 4× RAM, but the on-disk blob stays 2-bit (L(D)
    // unchanged), and it is unpacked here once at load.
    w: Vec<f32>,
    out_f: usize,
    in_f: usize,
}

struct LayerNorm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

struct Block {
    norm1: LayerNorm,
    wq: TernLinear,
    wk: TernLinear,
    wv: TernLinear,
    wo: TernLinear,
    norm2: LayerNorm,
    ff1: TernLinear,
    ff2: TernLinear,
}

pub(crate) struct Model {
    cfg: Config,
    bpe: crate::bpe::Bpe,
    tok: TernLinear, // [vocab, d] ternary; tied embedding + unembedding
    pos: Vec<f32>,   // ctx*d f32
    norm_f: LayerNorm,
    blocks: Vec<Block>,
}

// --- blob reader -----------------------------------------------------------

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
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
        let nbytes = n.div_ceil(4);
        let mut w = Vec::with_capacity(n);
        for bi in 0..nbytes {
            let byte = self.b[self.p + bi];
            for k in 0..4 {
                if w.len() == n {
                    break;
                }
                let code = (byte >> (2 * k)) & 0b11;
                w.push(match code {
                    0 => -1.0f32,
                    1 => 0.0,
                    _ => 1.0,
                });
            }
        }
        self.p += nbytes;
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
        let mut r = Reader { b: blob, p: 0 };
        assert_eq!(r.u32(), MAGIC, "v8 weight blob: bad magic");
        assert_eq!(r.u32(), VERSION, "v8 weight blob: bad version");
        let cfg = Config {
            vocab: r.u32() as usize,
            d_model: r.u32() as usize,
            n_layers: r.u32() as usize,
            n_heads: r.u32() as usize,
            ffn: r.u32() as usize,
            ctx: r.u32() as usize,
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
                ff1: r.tern(ffn, d),
                ff2: r.tern(d, ffn),
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

/// `dst += a * src` over equal-length slices — the contiguous form lets the
/// attention value-mixing vectorize (the strided gather form does not).
fn saxpy(dst: &mut [f32], a: f32, src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d = a.mul_add(*s, *d);
    }
}

/// BitLinear one row: int8-quantize `row` (into `xq` scratch, len ≥ in_f) and
/// matvec against the ternary weights into `out` (len out_f). Replicates the
/// trainer's inference math.
fn bitlinear_row(out: &mut [f32], row: &[f32], lin: &TernLinear, xq: &mut [f32]) {
    let in_f = lin.in_f;
    let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
    let s = (amax / 127.0).max(1e-5);
    for i in 0..in_f {
        xq[i] = (row[i] / s).round().clamp(-127.0, 127.0) * s;
    }
    for o in 0..lin.out_f {
        out[o] = dot(&xq[..in_f], &lin.w[o * in_f..o * in_f + in_f]) * lin.scale;
    }
}

/// Test-only [t, *] wrapper over `bitlinear_row` (the `predict` oracle uses it).
#[cfg(test)]
fn bitlinear(x: &[f32], t: usize, lin: &TernLinear) -> Vec<f32> {
    let (in_f, out_f) = (lin.in_f, lin.out_f);
    let mut out = vec![0f32; t * out_f];
    let mut xq = vec![0f32; in_f];
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
    tmp: Vec<f32>, // attn-out / ff-out
    ff: Vec<f32>,  // ff1 hidden
    xq: Vec<f32>,  // activation-quant scratch (sized to max in_f = ffn)
    scores: Vec<f32>,
    logits: Vec<f32>,
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
            xq: vec![0f32; ffn],
            scores: vec![0f32; self.cfg.ctx],
            logits: vec![0f32; self.cfg.vocab],
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
            s.x[i] = self.tok.scale * trow[i] + self.pos[pos * d + i];
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
            // feed-forward
            layernorm_row(&mut s.nrm, &s.x, &blk.norm2);
            bitlinear_row(&mut s.ff, &s.nrm, &blk.ff1, &mut s.xq);
            for v in &mut s.ff {
                *v = gelu(*v);
            }
            bitlinear_row(&mut s.tmp, &s.ff, &blk.ff2, &mut s.xq);
            for i in 0..d {
                s.x[i] += s.tmp[i];
            }
        }
        cache.len += 1;
        layernorm_row(&mut s.nrm, &s.x, &self.norm_f);
        for (vix, lg) in s.logits.iter_mut().enumerate() {
            *lg = self.tok.scale * dot(&s.nrm, &self.tok.w[vix * d..vix * d + d]);
        }
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
                x[r * d + i] = self.tok.scale * trow[i] + self.pos[r * d + i];
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
            // --- feed-forward ---
            let hn2 = layernorm(&x, t, d, &blk.norm2);
            let mut f = bitlinear(&hn2, t, &blk.ff1);
            for v in &mut f {
                *v = gelu(*v);
            }
            let f2 = bitlinear(&f, t, &blk.ff2);
            for i in 0..t * d {
                x[i] += f2[i];
            }
        }
        let xf = layernorm(&x, t, d, &self.norm_f);
        // tied unembedding for the last position only
        let last = &xf[(t - 1) * d..t * d];
        let mut logits = vec![0f32; cfg.vocab];
        for (vix, lg) in logits.iter_mut().enumerate() {
            *lg = self.tok.scale * dot(last, &self.tok.w[vix * d..vix * d + d]);
        }
        softmax_inplace(&mut logits);
        logits
    }

    /// Compress `bytes`: BPE-tokenize, then code each token id through the
    /// BOS-seeded incremental forward. Decode runs the identical forward, so
    /// the codec round-trips by construction.
    pub(crate) fn compress(&self, bytes: &[u8], bos: i32) -> Vec<u8> {
        let ids = self.bpe.encode(bytes);
        let mut enc = RangeEncoder::new();
        let mut cache = self.new_cache();
        let mut s = self.new_scratch();
        self.step(&mut cache, &mut s, bos);
        for &id in &ids {
            let cdf = probs_to_cdf(&s.logits);
            let id = id as usize;
            enc.encode(cdf[id], cdf[id + 1] - cdf[id], CDF_TOTAL);
            self.step(&mut cache, &mut s, id as i32);
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

    /// Build a minimal valid v2 blob (real tiny BPE + tiny model with
    /// deterministic pseudo-random ternary/f32 weights) — exercises the reader,
    /// tokenizer, forward, and coder end-to-end without the GPU trainer.
    fn tiny_blob() -> (Vec<u8>, usize) {
        let bpe = crate::bpe::Bpe::learn(&corpus(), 288);
        let vocab = bpe.vocab_size();
        let (d, layers, heads, ffn, ctx) = (8usize, 2usize, 2usize, 16usize, 32usize);
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
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&bpe.to_bytes());
        let mut s = 0x1234_5678u32;
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
            for chunk in 0..n.div_ceil(4) {
                let mut byte = 0u8;
                for k in 0..4 {
                    if chunk * 4 + k >= n {
                        break;
                    }
                    let code = (((f() + 0.5) * 3.0) as u8).min(2);
                    byte |= code << (2 * k);
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
            push_tern(&mut out, ffn, d, &mut nf); // ff1
            push_tern(&mut out, d, ffn, &mut nf); // ff2
        }
        (out, vocab)
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
}
