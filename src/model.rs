//! Pure-Rust byte-level `BitNet` transformer inference.
//!
//! One `step(token)` call advances the sliding-window state with the new
//! token and returns logits predicting the next byte. All per-step buffers
//! are pre-allocated in [`Scratch`] — the hot loop performs no heap
//! allocation.
//!
//! Architecture (see [`crate::arch`]): pre-LN, MHA with `RoPE` on Q and K,
//! `SwiGLU` MLP, tied embeddings. K / V are stored pre-RoPE in the cache and
//! re-rotated on every read, so positions always stay within the trained
//! range as the window slides.

use crate::arch::{
    CONTEXT_LEN, D_FF, D_MODEL, HEAD_DIM, N_HEADS, N_LAYERS, RMS_EPS, VOCAB, rope_cos_sin_tables,
};
use crate::bitnet::{matvec_ternary, quantize_activations};
use crate::kv_cache::KvCache;
use crate::weights::{TernaryMatrix, Weights};

/// Pre-allocated per-step scratch buffers.
#[derive(Debug)]
pub(crate) struct Scratch {
    x: Vec<f32>,
    residual: Vec<f32>,
    normed: Vec<f32>,
    xq: Vec<i8>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    q_rot: Vec<f32>,
    k_rot: Vec<f32>,
    attn_out: Vec<f32>,
    attn_scores: Vec<f32>,
    mlp_hidden: Vec<f32>,
    mlp_gate: Vec<f32>,
    mlp_up: Vec<f32>,
    xq_ff: Vec<i8>,
    logits: [f32; VOCAB],
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
}

impl Default for Scratch {
    fn default() -> Self {
        Self::new()
    }
}

impl Scratch {
    pub(crate) fn new() -> Self {
        let (rope_cos, rope_sin) = rope_cos_sin_tables();
        Self {
            x: vec![0.0; D_MODEL],
            residual: vec![0.0; D_MODEL],
            normed: vec![0.0; D_MODEL],
            xq: vec![0; D_MODEL],
            q: vec![0.0; D_MODEL],
            k: vec![0.0; D_MODEL],
            v: vec![0.0; D_MODEL],
            q_rot: vec![0.0; D_MODEL],
            k_rot: vec![0.0; D_MODEL],
            attn_out: vec![0.0; D_MODEL],
            attn_scores: vec![0.0; CONTEXT_LEN],
            mlp_hidden: vec![0.0; D_FF],
            mlp_gate: vec![0.0; D_FF],
            mlp_up: vec![0.0; D_FF],
            xq_ff: vec![0; D_FF],
            logits: [0.0; VOCAB],
            rope_cos,
            rope_sin,
        }
    }
}

/// In-place half-dim `RoPE` on one head of length `HEAD_DIM`.
///
/// Single-char identifiers (`v`, `a`, `b`, `c`, `s`) follow the standard
/// 2×2 rotation-matrix notation: the body is `(a, b) ← (a·c − b·s, a·s + b·c)`.
#[allow(clippy::many_single_char_names)]
fn apply_rope_head(v: &mut [f32], pos: usize, cos_tab: &[f32], sin_tab: &[f32]) {
    debug_assert_eq!(v.len(), HEAD_DIM);
    let half = HEAD_DIM / 2;
    let row = pos * half;
    for j in 0..half {
        let c = cos_tab[row + j];
        let s = sin_tab[row + j];
        let a = v[j];
        let b = v[j + half];
        v[j] = a.mul_add(c, -(b * s));
        v[j + half] = a.mul_add(s, b * c);
    }
}

fn apply_rope_multihead(vec: &mut [f32], pos: usize, cos_tab: &[f32], sin_tab: &[f32]) {
    debug_assert_eq!(vec.len(), N_HEADS * HEAD_DIM);
    for h in 0..N_HEADS {
        let start = h * HEAD_DIM;
        apply_rope_head(&mut vec[start..start + HEAD_DIM], pos, cos_tab, sin_tab);
    }
}

// `n` is at most `D_MODEL` (256 in the shipped model) so `usize -> f32` is
// exact. The `ss += v * v` accumulator uses `mul_add` for one-ulp accuracy
// (RMSNorm is sensitive to drift with long inner sums).
#[allow(clippy::cast_precision_loss)]
fn rmsnorm(x: &[f32], weight: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), out.len());
    let n = x.len();
    let mut ss = 0.0_f32;
    for &v in x {
        ss = v.mul_add(v, ss);
    }
    let scale = (ss / n as f32 + RMS_EPS).sqrt().recip();
    for i in 0..n {
        out[i] = x[i] * scale * weight[i];
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    // SiLU / Swish: x * sigmoid(x) = x / (1 + e^-x)
    x / (1.0 + (-x).exp())
}

fn matvec(mat: &TernaryMatrix<'_>, x: &[f32], xq_buf: &mut [i8], out: &mut [f32]) {
    let x_scale = quantize_activations(x, xq_buf);
    matvec_ternary(mat.packed, mat.scale, xq_buf, x_scale, out);
}

/// Byte-level `BitNet` transformer — owns weights, KV cache, and scratch.
#[derive(Debug)]
pub(crate) struct ByteTransformer {
    weights: Weights,
    kv: KvCache,
    scratch: Scratch,
}

impl ByteTransformer {
    pub(crate) fn new(weights: Weights) -> Self {
        Self {
            weights,
            kv: KvCache::new(),
            scratch: Scratch::new(),
        }
    }

    /// Reset inference state (clears KV cache, leaves weights/scratch alive).
    pub(crate) fn reset(&mut self) {
        self.kv.reset();
    }

    /// Advance state with `token` and return logits predicting the next byte.
    pub(crate) fn step(&mut self, token: u8) -> &[f32; VOCAB] {
        let scratch = &mut self.scratch;
        // Embedding lookup.
        let tok_emb = self.weights.tok_emb();
        let start = (token as usize) * D_MODEL;
        scratch.x.copy_from_slice(&tok_emb[start..start + D_MODEL]);

        for li in 0..N_LAYERS {
            let layer = self.weights.layer(li);

            // ---- Attention ----
            scratch.residual.copy_from_slice(&scratch.x);
            rmsnorm(&scratch.x, layer.attn_norm, &mut scratch.normed);

            matvec(&layer.q, &scratch.normed, &mut scratch.xq, &mut scratch.q);
            matvec(&layer.k, &scratch.normed, &mut scratch.xq, &mut scratch.k);
            matvec(&layer.v, &scratch.normed, &mut scratch.xq, &mut scratch.v);

            // Push pre-RoPE K, V to the ring buffer.
            self.kv.layers[li].push(&scratch.k, &scratch.v);
            let filled = self.kv.layers[li].len();
            let q_pos = filled - 1;

            // Rotate Q at the newest position.
            scratch.q_rot.copy_from_slice(&scratch.q);
            apply_rope_multihead(
                &mut scratch.q_rot,
                q_pos,
                &scratch.rope_cos,
                &scratch.rope_sin,
            );

            // Attention: for each head, per-cache-entry score then weighted V sum.
            compute_attention(
                &scratch.q_rot,
                &self.kv.layers[li],
                &scratch.rope_cos,
                &scratch.rope_sin,
                &mut scratch.k_rot,
                &mut scratch.attn_scores,
                &mut scratch.attn_out,
            );

            // Output projection + residual.
            matvec(&layer.o, &scratch.attn_out, &mut scratch.xq, &mut scratch.x);
            for (x, r) in scratch.x.iter_mut().zip(scratch.residual.iter()) {
                *x += r;
            }

            // ---- MLP (SwiGLU) ----
            scratch.residual.copy_from_slice(&scratch.x);
            rmsnorm(&scratch.x, layer.mlp_norm, &mut scratch.normed);

            matvec(
                &layer.w1,
                &scratch.normed,
                &mut scratch.xq,
                &mut scratch.mlp_gate,
            );
            matvec(
                &layer.w3,
                &scratch.normed,
                &mut scratch.xq,
                &mut scratch.mlp_up,
            );
            for ((h, &g), &u) in scratch
                .mlp_hidden
                .iter_mut()
                .zip(scratch.mlp_gate.iter())
                .zip(scratch.mlp_up.iter())
            {
                *h = silu(g) * u;
            }

            matvec(
                &layer.w2,
                &scratch.mlp_hidden,
                &mut scratch.xq_ff,
                &mut scratch.x,
            );
            for (x, r) in scratch.x.iter_mut().zip(scratch.residual.iter()) {
                *x += r;
            }
        }

        // Tied-embedding output head: logit[i] = x · tok_emb[i].
        for (i, logit) in scratch.logits.iter_mut().enumerate() {
            let row = &tok_emb[i * D_MODEL..(i + 1) * D_MODEL];
            *logit = scratch.x.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
        }

        &scratch.logits
    }
}

/// Multi-head attention over a (pre-RoPE) sliding window.
///
/// `q` is already rotated for the current query position; each cached K is
/// rotated in place (via scratch `k_rot_buf`) at its logical position during
/// the scan.
///
/// **Do not "optimize" this by pre-rotating K at push time.** The sliding
/// window relabels every cached entry's logical position whenever a new
/// token pushes the oldest one out, so a K pushed at logical position 199
/// becomes position 198 the next step, 197 the step after, and so on. `RoPE`
/// position is baked into K by a multiplicative angle; to keep attention
/// scores correct we must apply the rotation with the *current* logical
/// position, which means re-rotating at read time. (`CONTEXT_LEN` rotations
/// per step per layer is the cost we pay for the sliding window.)
///
/// `HEAD_DIM` is a compile-time constant (64 by default), so `usize -> f32`
/// for the attention scale is exact.
#[allow(clippy::cast_precision_loss)]
fn compute_attention(
    q: &[f32],
    layer: &crate::kv_cache::LayerCache,
    rope_cos: &[f32],
    rope_sin: &[f32],
    k_rot_buf: &mut [f32],
    attn_scores: &mut [f32],
    out: &mut [f32],
) {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Per-head weighted value sum. Accumulate per-head in an f32 buffer.
    for h in 0..N_HEADS {
        let head_off = h * HEAD_DIM;
        let q_head = &q[head_off..head_off + HEAD_DIM];

        // Score pass: fill attn_scores[i] = (q · rope(k_i, pos=i)) * scale.
        let mut max_score = f32::NEG_INFINITY;
        let mut count: usize = 0;
        for (logical_pos, k_slot, _v_slot) in layer.entries() {
            let k_head_pre = &k_slot[head_off..head_off + HEAD_DIM];
            k_rot_buf[head_off..head_off + HEAD_DIM].copy_from_slice(k_head_pre);
            apply_rope_head(
                &mut k_rot_buf[head_off..head_off + HEAD_DIM],
                logical_pos,
                rope_cos,
                rope_sin,
            );
            let k_head = &k_rot_buf[head_off..head_off + HEAD_DIM];
            let dot: f32 = q_head.iter().zip(k_head).map(|(a, b)| a * b).sum();
            let s = dot * scale;
            attn_scores[count] = s;
            if s > max_score {
                max_score = s;
            }
            count += 1;
        }

        // Softmax over attn_scores[..count].
        let mut sum = 0.0_f32;
        for s in &mut attn_scores[..count] {
            *s = (*s - max_score).exp();
            sum += *s;
        }
        let inv_sum = sum.recip();
        for s in &mut attn_scores[..count] {
            *s *= inv_sum;
        }

        // Weighted sum of V_i into out[head_off..head_off + HEAD_DIM].
        let out_head = &mut out[head_off..head_off + HEAD_DIM];
        for slot in out_head.iter_mut() {
            *slot = 0.0;
        }
        for (idx, (_pos, _k, v_slot)) in layer.entries().enumerate() {
            let v_head = &v_slot[head_off..head_off + HEAD_DIM];
            let w = attn_scores[idx];
            for (o, &v) in out_head.iter_mut().zip(v_head.iter()) {
                *o = w.mul_add(v, *o);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::Weights;

    fn zero_weights() -> Weights {
        let buf = vec![0u8; crate::arch::PACKED_WEIGHTS_LEN];
        Weights::from_bytes(&buf).unwrap()
    }

    #[test]
    fn step_with_zero_weights_runs_without_panic() {
        let mut model = ByteTransformer::new(zero_weights());
        // All-zero weights: embedding returns zero, attention is uniform over
        // one cached slot (the first token), MLP returns zero, logits zero.
        // We don't assert specific values — just that the forward runs.
        let _ = model.step(b'a');
        let _ = model.step(b'b');
        let _ = model.step(b'c');
    }

    #[test]
    fn reset_clears_cache() {
        let mut model = ByteTransformer::new(zero_weights());
        for i in 0..10u8 {
            let _ = model.step(i);
        }
        assert_eq!(model.kv.layers[0].len(), 10);
        model.reset();
        assert_eq!(model.kv.layers[0].len(), 0);
    }

    #[test]
    fn rmsnorm_basic() {
        let x = [3.0_f32, 4.0, 0.0, 0.0];
        let w = [1.0_f32; 4];
        let mut out = [0.0_f32; 4];
        rmsnorm(&x, &w, &mut out);
        // ss = 25, rms = sqrt(25/4 + eps) ≈ 2.5, out[0] ≈ 3/2.5 = 1.2
        assert!((out[0] - 1.2).abs() < 1e-3);
        assert!((out[1] - 1.6).abs() < 1e-3);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn rope_is_identity_at_position_zero() {
        let (cos, sin) = rope_cos_sin_tables();
        let mut v: Vec<f32> = (0..HEAD_DIM).map(|i| i as f32).collect();
        let orig = v.clone();
        apply_rope_head(&mut v, 0, &cos, &sin);
        for (a, b) in v.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
