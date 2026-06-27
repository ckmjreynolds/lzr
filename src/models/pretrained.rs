//! Frozen pretrained MLP byte-LM, mixed in as one more [`Model`].
//!
//! The weights are trained offline on GPU (the `pretrain` example, `neural`
//! feature) and embedded as a compact blob; here a hand-rolled scalar forward
//! runs a `K`-byte-context MLP once per byte to produce a next-byte distribution,
//! and each bit is predicted by marginalizing that distribution over the partial
//! byte (`c0`) — the same bit-tree contract the arm and context models use. The
//! net is **frozen** (no online update), so its only cost is the forward and the
//! shipped weights (`L(D)`); it earns its place by decorrelating from the
//! deterministic stack and the online arm.

use crate::models::{Context, Model};

/// Blob layout: `[K,E,H,V : u32 LE]` then `emb[V*E]`, `w1[K*E*H]`, `b1[H]`,
/// `w2[H*V]`, `b2[V]`, all f32 LE, row-major (matching the burn export).
#[derive(Debug)]
pub(crate) struct PretrainedMlp {
    k: usize,
    e: usize,
    h: usize,
    v: usize,
    emb: Vec<f32>,    // [v*e]
    w1: Vec<f32>,     // [h*(k*e)] — transposed from the blob's [k*e, h] at load
    b1: Vec<f32>,     // [h]
    w2: Vec<f32>,     // [v*h] — transposed from the blob's [h, v] at load
    b2: Vec<f32>,     // [v]
    cumsum: Vec<f32>, // [v+1] prefix sums of the current next-byte distribution
    x: Vec<f32>,      // [k*e] flattened context embedding
    hid: Vec<f32>,    // [h]
    // Online warming-readout heads (see `Head`): each is a per-context linear
    // readout over the frozen embedding `hid`, contributed as a SEPARATE mixer
    // input. The frozen net's edge erodes as the online stack warms (it is
    // front-loaded); these heads re-learn the hidden→bit mapping online, so they
    // warm WITH the data. Empty (default) = frozen-only.
    heads: Vec<Head>,
}

/// Transpose a `[rows, cols]` row-major matrix to `[cols, rows]` row-major. Done
/// once at load so the per-byte matvec reads each weight row contiguously. The
/// blob stores `w1`/`w2` in the matvec's strided orientation; contiguous rows are
/// neutral while the ~768 KB of weights stay cache-resident (small slices) but
/// avoid cache misses at enwik9 scale, where the deterministic hash tables evict
/// them — and it is the orientation a future vectorized dot would need.
fn transpose(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; src.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    out
}

/// `bias + sum w[i]*x[i]` with 16 independent accumulators, so the `mul_add` chain
/// is throughput-bound (auto-vectorizes to NEON `fmla`) instead of latency-bound
/// on a single dependent accumulator — the per-byte forward's dominant cost. The
/// 16-way reassociation differs from a sequential sum by ~1e-6 (f32), immaterial
/// to the prediction (`recompute_matches_naive_approx` bounds it).
#[allow(clippy::needless_range_loop)]
fn dot(w: &[f32], x: &[f32], bias: f32) -> f32 {
    let mut acc = [0.0f32; 16];
    let mut wi = w.chunks_exact(16);
    let mut xi = x.chunks_exact(16);
    for (wc, xc) in wi.by_ref().zip(xi.by_ref()) {
        for l in 0..16 {
            acc[l] = wc[l].mul_add(xc[l], acc[l]);
        }
    }
    let mut s = bias + acc.iter().sum::<f32>();
    for (wr, xr) in wi.remainder().iter().zip(xi.remainder()) {
        s = wr.mul_add(*xr, s);
    }
    s
}

const HEAD_HASH: u64 = 0x9E37_79B9_7F4A_7C15;

/// One online warming-readout head: for each context (the node) it learns, by
/// online logistic SGD, a linear readout over the frozen net's hidden embedding
/// `hid`, emitted as a residual on the frozen logit. A stack of heads on distinct
/// contexts (prev-byte, word, …) forms a neural analog of the deterministic
/// context-model stack — all sharing the one frozen forward, so each costs only
/// its readout. Zero-init → contributes nothing until it learns; ships no weights.
#[derive(Debug)]
struct Head {
    weights: Vec<f32>, // [nodes * h]
    h: usize,
    lr: f32,
    ctx_bytes: usize, // prev finalized bytes in the node (0 = bit-tree only), unless `word`/`mask`
    word: bool,       // node hashes the word hash instead of prev bytes
    mask: u32,        // if nonzero, node hashes the sparse byte set (bit i → byte_back(i+1))
    bits: u32,        // hashed-node table size (bits); unused when ctx_bytes == 0 && !word
    node: usize,      // node of the last eval (for the learn step)
    p: f32,           // this head's squashed P(bit==1) at the last eval
    out: i32,         // this head's logit (extra mixer input) at the last eval
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
impl Head {
    fn new(h: usize, lr: f32, ctx_bytes: usize, word: bool, mask: u32, bits: u32) -> Self {
        let nodes = if !word && mask == 0 && ctx_bytes == 0 {
            256
        } else {
            1usize << bits
        };
        Self {
            weights: vec![0.0; nodes * h],
            h,
            lr,
            ctx_bytes,
            word,
            mask,
            bits,
            node: 0,
            p: 0.0,
            out: 0,
        }
    }

    fn node(&self, ctx: &Context) -> usize {
        let c0 = u64::from(ctx.c0 & 0xff);
        if self.word {
            let key = (ctx.word_hash << 8) | c0;
            return (key.wrapping_mul(HEAD_HASH) >> (64 - self.bits)) as usize;
        }
        if self.mask != 0 {
            let mut cv = u64::from(self.mask);
            let mut m = self.mask;
            while m != 0 {
                let i = m.trailing_zeros() as usize + 1;
                cv = (cv << 8) | u64::from(ctx.byte_back(i));
                m &= m - 1;
            }
            let key = (cv << 8) | c0;
            return (key.wrapping_mul(HEAD_HASH) >> (64 - self.bits)) as usize;
        }
        if self.ctx_bytes == 0 {
            return c0 as usize;
        }
        let mut cv = 0u64;
        for i in 1..=self.ctx_bytes {
            cv = (cv << 8) | u64::from(ctx.byte_back(i));
        }
        let key = (cv << 8) | c0;
        (key.wrapping_mul(HEAD_HASH) >> (64 - self.bits)) as usize
    }

    /// Residual readout = `weights[node] · hid` added to the frozen logit; cached
    /// for the mixer (`out`) and the learn step (`node`, `p`).
    fn eval(&mut self, ctx: &Context, hid: &[f32], frozen: i32) {
        let node = self.node(ctx);
        let hl = dot(&self.weights[node * self.h..(node + 1) * self.h], hid, 0.0);
        let combined = (frozen + hl as i32).clamp(-2047, 2047);
        self.node = node;
        self.p = crate::mixer::squash(combined) as f32 / 4096.0;
        self.out = combined;
    }

    /// One online logistic SGD step on this head's own prediction (the frozen
    /// logit is a fixed offset baked into `p`).
    fn learn(&mut self, hid: &[f32], bit: u8) {
        let g = self.lr * (f32::from(bit) - self.p);
        let row = &mut self.weights[self.node * self.h..(self.node + 1) * self.h];
        for (w, &x) in row.iter_mut().zip(hid) {
            *w = g.mul_add(x, *w);
        }
    }
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::suboptimal_flops
)]
impl PretrainedMlp {
    /// Parse an f32 weight blob (see the layout note above). Test-only: production
    /// ships the int8 blob via `from_blob_q8`; the f32 path drives the offline
    /// marginal/quantization labs.
    #[cfg(test)]
    pub(crate) fn from_blob(blob: &[u8]) -> Self {
        let u32_at =
            |o: usize| -> usize { u32::from_le_bytes(blob[o..o + 4].try_into().unwrap()) as usize };
        let (k, e, h, v) = (u32_at(0), u32_at(4), u32_at(8), u32_at(12));
        let mut off = 16usize;
        let mut take = |n: usize| -> Vec<f32> {
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                out.push(f32::from_le_bytes(blob[off..off + 4].try_into().unwrap()));
                off += 4;
            }
            out
        };
        let emb = take(v * e);
        let w1 = take(k * e * h);
        let b1 = take(h);
        let w2 = take(h * v);
        let b2 = take(v);
        let w1 = transpose(&w1, k * e, h); // [k*e, h] -> [h, k*e] (contiguous matvec)
        let w2 = transpose(&w2, h, v); // [h, v] -> [v, h]
        let mut m = Self {
            k,
            e,
            h,
            v,
            emb,
            w1,
            b1,
            w2,
            b2,
            cumsum: vec![0.0; v + 1],
            x: vec![0.0; k * e],
            hid: vec![0.0; h],
            heads: Vec::new(),
        };
        m.recompute(&Context::with_capacity(0));
        m
    }

    /// Load an int8-quantized blob: `[K,E,H,V : u32]`, then per-tensor scales
    /// (`emb`,`w1`,`w2` : f32), then `i8` `emb`/`w1`/`w2`, then `f32` `b1`/`b2`.
    /// Dequantizes to f32 in RAM (the on-disk `i8` is what sets `L(D)`); reproduces
    /// `quantize(8)` exactly, so the shipped net equals the measured 8-bit one.
    pub(crate) fn from_blob_q8(blob: &[u8]) -> Self {
        let u32_at =
            |o: usize| -> usize { u32::from_le_bytes(blob[o..o + 4].try_into().unwrap()) as usize };
        let f32_at = |o: usize| f32::from_le_bytes(blob[o..o + 4].try_into().unwrap());
        let (k, e, h, v) = (u32_at(0), u32_at(4), u32_at(8), u32_at(12));
        let (es, w1s, w2s) = (f32_at(16), f32_at(20), f32_at(24));
        let mut off = 28usize;
        let mut deq_i8 = |n: usize, scale: f32| -> Vec<f32> {
            let out: Vec<f32> = blob[off..off + n]
                .iter()
                .map(|&b| f32::from(i8::from_le_bytes([b])) * scale)
                .collect();
            off += n;
            out
        };
        let emb = deq_i8(v * e, es);
        let w1 = deq_i8(k * e * h, w1s);
        let w2 = deq_i8(h * v, w2s);
        let w1 = transpose(&w1, k * e, h); // [k*e, h] -> [h, k*e] (contiguous matvec)
        let w2 = transpose(&w2, h, v); // [h, v] -> [v, h]
        let bias_off = 28 + v * e + k * e * h + h * v;
        let read_f32 = |start: usize, n: usize| -> Vec<f32> {
            (0..n).map(|i| f32_at(start + i * 4)).collect()
        };
        let b1 = read_f32(bias_off, h);
        let b2 = read_f32(bias_off + h * 4, v);
        let mut m = Self {
            k,
            e,
            h,
            v,
            emb,
            w1,
            b1,
            w2,
            b2,
            cumsum: vec![0.0; v + 1],
            x: vec![0.0; k * e],
            hid: vec![0.0; h],
            heads: Vec::new(),
        };
        m.recompute(&Context::with_capacity(0));
        m
    }

    /// In-place symmetric quantization of the weight tensors to `bits`, dequantized
    /// back to f32 — to estimate the L(C) cost of shipping low-bit weights (the
    /// L(D) saving is computed separately). `bits >= 32` is a no-op. `bits == 2` is
    /// ternary (`{-1,0,1}` × per-tensor abs-mean).
    #[cfg(test)]
    pub(crate) fn quantize(&mut self, bits: u32) {
        if bits >= 32 {
            return;
        }
        let q = |w: &mut [f32]| {
            if w.is_empty() {
                return;
            }
            if bits == 2 {
                let scale = w.iter().map(|x| x.abs()).sum::<f32>() / w.len() as f32;
                if scale > 0.0 {
                    for x in w.iter_mut() {
                        *x = (*x / scale).round().clamp(-1.0, 1.0) * scale;
                    }
                }
            } else {
                let lvl = f32::from((1u16 << (bits - 1)) - 1); // 8-bit -> 127
                let scale = w.iter().map(|x| x.abs()).fold(0.0f32, f32::max) / lvl;
                if scale > 0.0 {
                    for x in w.iter_mut() {
                        *x = (*x / scale).round().clamp(-lvl, lvl) * scale;
                    }
                }
            }
        };
        q(&mut self.emb);
        q(&mut self.w1);
        q(&mut self.w2);
        self.recompute(&Context::with_capacity(0));
    }

    /// Add a per-context warming head: the node hashes the last `ctx_bytes`
    /// finalized bytes (with the bit-tree position) into a `bits`-wide table, so the
    /// head learns a distinct online readout over the frozen embedding per context.
    /// `ctx_bytes == 0` is the plain per-bit-tree-node head (256 nodes, direct).
    pub(crate) fn push_head_ctx(&mut self, lr: f32, ctx_bytes: usize, bits: u32) {
        self.heads
            .push(Head::new(self.h, lr, ctx_bytes, false, 0, bits));
    }

    /// Add a word-context warming head: the node hashes the current word hash with
    /// the bit-tree position (a decorrelation axis distinct from the byte context).
    pub(crate) fn push_head_word(&mut self, lr: f32, bits: u32) {
        self.heads.push(Head::new(self.h, lr, 0, true, 0, bits));
    }

    /// Add a sparse-context warming head: the node hashes the byte set selected by
    /// `mask` (bit i → `byte_back(i+1)`) — a skip/gap context decorrelated from the
    /// contiguous orders.
    pub(crate) fn push_head_sparse(&mut self, lr: f32, mask: u32, bits: u32) {
        self.heads.push(Head::new(self.h, lr, 0, false, mask, bits));
    }

    /// Number of warming heads (each contributes one extra mixer input).
    pub(crate) fn n_heads(&self) -> usize {
        self.heads.len()
    }

    /// The `i`-th head's logit for this bit (an extra mixer input). Valid after
    /// [`Model::predict`] ran for the bit.
    pub(crate) fn head_out(&self, i: usize) -> i32 {
        self.heads[i].out
    }

    /// Forward the MLP over the last `k` finalized bytes and refresh the
    /// next-byte distribution's prefix sums. Bytes before history start are 0.
    fn recompute(&mut self, ctx: &Context) {
        // x = concat of the k byte-embeddings, oldest first (matching training:
        // window data[p-K..p], so the most recent finalized byte is last).
        for j in 0..self.k {
            let back = self.k - j; // j=0 -> oldest (k back), j=k-1 -> 1 back
            let b = usize::from(ctx.byte_back(back));
            let src = &self.emb[b * self.e..(b + 1) * self.e];
            self.x[j * self.e..(j + 1) * self.e].copy_from_slice(src);
        }
        // hid = tanh(x @ w1 + b1); w1 stored transposed [h, k*e] so each unit's
        // weights are contiguous over x for the vectorized `dot`. (tanh, not relu:
        // burn's activation::relu detached the autodiff graph in training.)
        let ke = self.k * self.e;
        for hi in 0..self.h {
            let row = &self.w1[hi * ke..hi * ke + ke];
            self.hid[hi] = dot(row, &self.x, self.b1[hi]).tanh();
        }
        // logits = hid @ w2 + b2; w2 stored transposed [v, h] (contiguous over hid).
        let mut maxl = f32::MIN;
        // reuse cumsum[1..=v] as the logits scratch, then prefix-sum in place.
        for vi in 0..self.v {
            let row = &self.w2[vi * self.h..vi * self.h + self.h];
            let acc = dot(row, &self.hid, self.b2[vi]);
            self.cumsum[vi + 1] = acc;
            if acc > maxl {
                maxl = acc;
            }
        }
        let mut sum = 0.0f32;
        for vi in 0..self.v {
            let p = (self.cumsum[vi + 1] - maxl).exp();
            self.cumsum[vi + 1] = p;
            sum += p;
        }
        let inv = 1.0 / sum;
        self.cumsum[0] = 0.0;
        for vi in 0..self.v {
            self.cumsum[vi + 1] = self.cumsum[vi] + self.cumsum[vi + 1] * inv;
        }
    }
}

impl Model for PretrainedMlp {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn predict(&mut self, ctx: &Context) -> i32 {
        if ctx.bpos == 0 {
            self.recompute(ctx);
        }
        let bpos = ctx.bpos as usize;
        let prefix = (ctx.c0 - (1 << bpos)) as usize; // top `bpos` bits of the byte
        let shift = 8 - bpos;
        let lo = prefix << shift;
        let half = 1usize << (shift - 1);
        let p0 = self.cumsum[lo + half] - self.cumsum[lo];
        let p1 = self.cumsum[lo + (1 << shift)] - self.cumsum[lo + half];
        let pr = p1 / (p0 + p1 + 1e-12);
        let q = ((pr * 4096.0) as i32).clamp(1, 4095);
        let frozen = crate::mixer::stretch(q);
        // Each warming head evaluates a residual readout over the shared `hid` and
        // caches its logit as a SEPARATE mixer input (the codec reads them via
        // `head_out`); the model's own predict return stays the clean frozen logit.
        let Self { heads, hid, .. } = self;
        for head in heads.iter_mut() {
            head.eval(ctx, hid, frozen);
        }
        frozen
    }

    fn update(&mut self, _ctx: &Context, bit: u8) {
        // The frozen net learns nothing (recompute is driven from predict at the
        // byte boundary). Each warming head does one online logistic SGD step.
        let Self { heads, hid, .. } = self;
        for head in heads.iter_mut() {
            head.learn(hid, bit);
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::suboptimal_flops
)]
mod tests {
    use super::*;

    /// Quantize an f32 net blob to the int8 ship format — the same per-tensor
    /// symmetric quantization as `quantize(8)`: `[K,E,H,V u32]`, then
    /// `emb`/`w1`/`w2` scales (f32), then `i8` `emb`/`w1`/`w2`, then `f32`
    /// `b1`/`b2`. Produces what `from_blob_q8` reads.
    pub(crate) fn f32_blob_to_q8(f32_blob: &[u8]) -> Vec<u8> {
        let u32_at = |o: usize| u32::from_le_bytes(f32_blob[o..o + 4].try_into().unwrap()) as usize;
        let (k, e, h, v) = (u32_at(0), u32_at(4), u32_at(8), u32_at(12));
        let mut off = 16usize;
        let mut take = |n: usize| -> Vec<f32> {
            let out: Vec<f32> = (0..n)
                .map(|i| {
                    f32::from_le_bytes(f32_blob[off + i * 4..off + i * 4 + 4].try_into().unwrap())
                })
                .collect();
            off += n * 4;
            out
        };
        let emb = take(v * e);
        let w1 = take(k * e * h);
        let b1 = take(h);
        let w2 = take(h * v);
        let b2 = take(v);
        let q8 = |w: &[f32]| -> (f32, Vec<i8>) {
            let m = w.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
            let scale = if m > 0.0 { m / 127.0 } else { 1.0 };
            let q = w
                .iter()
                .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8)
                .collect();
            (scale, q)
        };
        let (es, eq) = q8(&emb);
        let (w1s, w1q) = q8(&w1);
        let (w2s, w2q) = q8(&w2);
        let mut out = Vec::new();
        for d in [k as u32, e as u32, h as u32, v as u32] {
            out.extend_from_slice(&d.to_le_bytes());
        }
        for s in [es, w1s, w2s] {
            out.extend_from_slice(&s.to_le_bytes());
        }
        for &q in eq.iter().chain(&w1q).chain(&w2q) {
            out.push(q as u8);
        }
        for &x in b1.iter().chain(&b2) {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    fn synth_f32_blob(k: usize, e: usize, h: usize, v: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for d in [k, e, h, v] {
            out.extend_from_slice(&(d as u32).to_le_bytes());
        }
        let n = v * e + k * e * h + h + h * v + v;
        let mut s = 0x1234_5678u32;
        for _ in 0..n {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let x = (s >> 16) as f32 / f32::from(u16::MAX) * 2.0 - 1.0;
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    /// The int8 ship path (`f32_blob_to_q8` → `from_blob_q8`) must reproduce the
    /// measured 8-bit net (`from_blob` → `quantize(8)`) bit-for-bit, so the
    /// validated 8-bit marginal transfers to the shipped binary.
    #[test]
    fn q8_blob_reproduces_quantize8() {
        let blob = synth_f32_blob(2, 2, 3, 4);
        let mut a = PretrainedMlp::from_blob(&blob);
        a.quantize(8);
        let b = PretrainedMlp::from_blob_q8(&f32_blob_to_q8(&blob));
        assert_eq!(a.emb, b.emb, "emb");
        assert_eq!(a.w1, b.w1, "w1");
        assert_eq!(a.w2, b.w2, "w2");
        assert_eq!(a.b1, b.b1, "b1");
        assert_eq!(a.b2, b.b2, "b2");
    }

    /// The transposed + vectorized matvec must compute what the naive strided
    /// `[k*e,h]`/`[h,v]` single-accumulator matvec would, up to f32 reassociation
    /// (the 16-lane `dot` reorders the sum by ~1e-6). Round-trip tests can't catch
    /// a transpose/index typo (a wrong-but-consistent net still round-trips, just
    /// predicts garbage and silently loses the marginal), so this is the real guard.
    #[test]
    fn recompute_matches_naive_approx() {
        let (k, e, h, v) = (3usize, 4usize, 5usize, 7usize);
        let blob = synth_f32_blob(k, e, h, v);
        let net = PretrainedMlp::from_blob(&blob); // recompute ran on the empty ctx
        // Naive strided forward on the same empty context (all-zero history ->
        // x = emb[0..e] repeated k times), from blob-order weights.
        let mut off = 16usize;
        let mut take = |n: usize| -> Vec<f32> {
            let o: Vec<f32> = (0..n)
                .map(|i| f32::from_le_bytes(blob[off + i * 4..off + i * 4 + 4].try_into().unwrap()))
                .collect();
            off += n * 4;
            o
        };
        let emb = take(v * e);
        let w1 = take(k * e * h);
        let b1 = take(h);
        let w2 = take(h * v);
        let b2 = take(v);
        let ke = k * e;
        let mut x = vec![0.0f32; ke];
        for j in 0..k {
            x[j * e..(j + 1) * e].copy_from_slice(&emb[0..e]);
        }
        let mut hid = vec![0.0f32; h];
        for hi in 0..h {
            let mut acc = b1[hi];
            for xi in 0..ke {
                acc = w1[xi * h + hi].mul_add(x[xi], acc);
            }
            hid[hi] = acc.tanh();
        }
        let mut cumsum = vec![0.0f32; v + 1];
        let mut maxl = f32::MIN;
        for vi in 0..v {
            let mut acc = b2[vi];
            for hi in 0..h {
                acc = w2[hi * v + vi].mul_add(hid[hi], acc);
            }
            cumsum[vi + 1] = acc;
            if acc > maxl {
                maxl = acc;
            }
        }
        let mut sum = 0.0f32;
        for vi in 0..v {
            let p = (cumsum[vi + 1] - maxl).exp();
            cumsum[vi + 1] = p;
            sum += p;
        }
        let inv = 1.0 / sum;
        for vi in 0..v {
            cumsum[vi + 1] = cumsum[vi] + cumsum[vi + 1] * inv;
        }
        let max_err = net
            .cumsum
            .iter()
            .zip(&cumsum)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-4,
            "vectorized recompute must match the naive strided matvec within f32 \
             reassociation (max abs cumsum error {max_err:e})"
        );
    }

    /// One-shot tool: convert the f32 net (`LZR_NET`, default `/tmp/lzr_net.bin`)
    /// to the int8 ship asset (`LZR_Q8_OUT`, default `assets/pretrained_net.bin`).
    /// `LZR_NET=/tmp/lzr_net.bin cargo test --release dump_q8_asset -- --ignored`.
    #[test]
    #[ignore = "tool: write assets/pretrained_net.bin (int8) from an f32 net blob"]
    fn dump_q8_asset() {
        let src = std::env::var("LZR_NET").unwrap_or_else(|_| "/tmp/lzr_net.bin".into());
        let dst =
            std::env::var("LZR_Q8_OUT").unwrap_or_else(|_| "assets/pretrained_net.bin".into());
        let Ok(f32_blob) = std::fs::read(&src) else {
            println!("no f32 blob at {src}; skipping");
            return;
        };
        let q8 = f32_blob_to_q8(&f32_blob);
        std::fs::write(&dst, &q8).unwrap();
        println!(
            "wrote {} int8 bytes to {dst} (from {} f32 bytes {src})",
            q8.len(),
            f32_blob.len()
        );
    }
}
