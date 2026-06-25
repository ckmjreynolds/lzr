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
    w1: Vec<f32>,     // [(k*e)*h]
    b1: Vec<f32>,     // [h]
    w2: Vec<f32>,     // [h*v]
    b2: Vec<f32>,     // [v]
    cumsum: Vec<f32>, // [v+1] prefix sums of the current next-byte distribution
    x: Vec<f32>,      // [k*e] flattened context embedding
    hid: Vec<f32>,    // [h]
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::suboptimal_flops
)]
impl PretrainedMlp {
    /// Parse a weight blob (see the layout note above).
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
        };
        m.recompute(&Context::with_capacity(0));
        m
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
        // hid = tanh(x @ w1 + b1); w1 is [k*e, h] row-major. (tanh, not relu: the
        // burn `activation::relu` detached the autodiff graph in training.)
        for hi in 0..self.h {
            let mut acc = self.b1[hi];
            for (xi, &xv) in self.x.iter().enumerate() {
                acc = self.w1[xi * self.h + hi].mul_add(xv, acc);
            }
            self.hid[hi] = acc.tanh();
        }
        // logits = hid @ w2 + b2; w2 is [h, v] row-major. Softmax -> cumsum.
        let mut maxl = f32::MIN;
        // reuse cumsum[1..=v] as the logits scratch, then prefix-sum in place.
        for vi in 0..self.v {
            let mut acc = self.b2[vi];
            for (hi, &hv) in self.hid.iter().enumerate() {
                acc = self.w2[hi * self.v + vi].mul_add(hv, acc);
            }
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
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
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
        crate::mixer::stretch(q)
    }

    fn update(&mut self, _ctx: &Context, _bit: u8) {
        // Frozen: recompute is driven from `predict` at the byte boundary, when
        // the finalized history (read via byte_back) already includes the byte
        // just completed. Nothing to learn.
    }
}
