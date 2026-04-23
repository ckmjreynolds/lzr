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
use crate::bitnet::{
    build_lut, lut_scratch_bytes, matvec_ternary, matvec_ternary_lut, quantize_activations,
};
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
    /// Arch-preferred LUT scratch, sized for the widest activation vector
    /// (`max(D_MODEL, D_FF)`). Zero-length on architectures without a LUT
    /// kernel. `build_lut` writes only the valid entries per group; callers
    /// must keep the bytes zero-initialized (they are zero-initialized once
    /// at `Scratch::new` and the written entries are deterministic per
    /// activation so no explicit re-zero is needed).
    lut: Vec<i8>,
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
            lut: vec![0i8; lut_scratch_bytes(D_MODEL.max(D_FF))],
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

/// Quantize activations once and return the (xq buffer, scale) so multiple
/// matmuls with the same input can share the work.
fn quantize_into<'a>(x: &[f32], xq_buf: &'a mut [i8]) -> (&'a [i8], f32) {
    let scale = quantize_activations(x, xq_buf);
    (xq_buf, scale)
}

/// Run a matvec against `mat`, picking the arch-preferred LUT kernel when
/// the Weights loader populated one and falling back to the `I2_S` direct
/// path otherwise. `lut` must already have been filled from `x_q` via
/// `build_lut` — sharing that build across matvecs that consume the same
/// activation (Q/K/V, w1/w3) is the point of hoisting it out here.
fn matvec_prequant(mat: &TernaryMatrix<'_>, x_q: &[i8], lut: &[i8], x_scale: f32, out: &mut [f32]) {
    if mat.lut_packed.is_empty() {
        matvec_ternary(mat.packed, mat.scale, x_q, x_scale, out);
    } else {
        matvec_ternary_lut(mat.lut_packed, mat.scale, lut, x_scale, x_q.len(), out);
    }
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
    ///
    /// The `(xq_*, xs_*)` binding pairs pass pre-quantized inputs to the
    /// matvec kernels so that Q/K/V share one quant, as do w1/w3. clippy's
    /// `similar_names` warns on these by default but the pairing reads more
    /// clearly than unique-per-call names.
    #[allow(clippy::similar_names)]
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

            // Quantize the norm output once; Q / K / V all consume it.
            // Build the LUT once, reuse across the three matvecs.
            let (xq_norm, xs_norm) = quantize_into(&scratch.normed, &mut scratch.xq);
            let lut_len = lut_scratch_bytes(xq_norm.len());
            build_lut(xq_norm, &mut scratch.lut[..lut_len]);
            let lut = &scratch.lut[..lut_len];
            matvec_prequant(&layer.q, xq_norm, lut, xs_norm, &mut scratch.q);
            matvec_prequant(&layer.k, xq_norm, lut, xs_norm, &mut scratch.k);
            matvec_prequant(&layer.v, xq_norm, lut, xs_norm, &mut scratch.v);

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
            let (xq_attn, xs_attn) = quantize_into(&scratch.attn_out, &mut scratch.xq);
            let lut_len = lut_scratch_bytes(xq_attn.len());
            build_lut(xq_attn, &mut scratch.lut[..lut_len]);
            let lut = &scratch.lut[..lut_len];
            matvec_prequant(&layer.o, xq_attn, lut, xs_attn, &mut scratch.x);
            for (x, r) in scratch.x.iter_mut().zip(scratch.residual.iter()) {
                *x += r;
            }

            // ---- MLP (SwiGLU) ----
            scratch.residual.copy_from_slice(&scratch.x);
            rmsnorm(&scratch.x, layer.mlp_norm, &mut scratch.normed);

            // Quantize norm output once; w1 and w3 share it. One LUT build.
            let (xq_norm, xs_norm) = quantize_into(&scratch.normed, &mut scratch.xq);
            let lut_len = lut_scratch_bytes(xq_norm.len());
            build_lut(xq_norm, &mut scratch.lut[..lut_len]);
            let lut = &scratch.lut[..lut_len];
            matvec_prequant(&layer.w1, xq_norm, lut, xs_norm, &mut scratch.mlp_gate);
            matvec_prequant(&layer.w3, xq_norm, lut, xs_norm, &mut scratch.mlp_up);

            for ((h, &g), &u) in scratch
                .mlp_hidden
                .iter_mut()
                .zip(scratch.mlp_gate.iter())
                .zip(scratch.mlp_up.iter())
            {
                *h = silu(g) * u;
            }

            let (xq_hid, xs_hid) = quantize_into(&scratch.mlp_hidden, &mut scratch.xq_ff);
            let lut_len = lut_scratch_bytes(xq_hid.len());
            build_lut(xq_hid, &mut scratch.lut[..lut_len]);
            let lut = &scratch.lut[..lut_len];
            matvec_prequant(&layer.w2, xq_hid, lut, xs_hid, &mut scratch.x);
            for (x, r) in scratch.x.iter_mut().zip(scratch.residual.iter()) {
                *x += r;
            }
        }

        // Tied-embedding output head: logit[i] = x · tok_emb[i]. D_MODEL
        // is divisible by 16 so the SIMD dot kicks in.
        for (i, logit) in scratch.logits.iter_mut().enumerate() {
            let row = &tok_emb[i * D_MODEL..(i + 1) * D_MODEL];
            *logit = dot_f32(&scratch.x, row);
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

    for h in 0..N_HEADS {
        let head_off = h * HEAD_DIM;
        let q_head = &q[head_off..head_off + HEAD_DIM];

        // Score pass: fill attn_scores[i] = (q · rope(k_i, pos=i)) * scale.
        let mut max_score = f32::NEG_INFINITY;
        let mut count: usize = 0;
        for (logical_pos, k_slot, _v_slot) in layer.entries() {
            let k_head_pre = &k_slot[head_off..head_off + HEAD_DIM];
            let k_head_rot = &mut k_rot_buf[head_off..head_off + HEAD_DIM];
            k_head_rot.copy_from_slice(k_head_pre);
            rope_head_inplace(k_head_rot, logical_pos, rope_cos, rope_sin);
            let dot = dot_f32(q_head, k_head_rot);
            let s = dot * scale;
            attn_scores[count] = s;
            if s > max_score {
                max_score = s;
            }
            count += 1;
        }

        // Softmax over attn_scores[..count]. `f32::exp` on Apple's libm is
        // fast enough (~1 ns); a polynomial NEON-SIMD exp was tried and
        // gave no measurable speedup at this model size while trading
        // ~1e-6 relative error for the gain that wasn't there.
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
            axpy_head(out_head, w, v_head);
        }
    }
}

// --- Hot-path helpers -------------------------------------------------------
//
// Each of these has a NEON path (for aarch64) and a scalar fallback. The
// inner loops run `~CONTEXT_LEN * N_HEADS * N_LAYERS` times per encoded /
// decoded byte — this is where most of the inference wall-clock lives
// outside the matvec kernels.

/// Apply half-dim `RoPE` to one head in place.
#[inline]
fn rope_head_inplace(v: &mut [f32], pos: usize, cos_tab: &[f32], sin_tab: &[f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of the aarch64 baseline ABI.
        #[allow(unsafe_code)]
        unsafe {
            neon_rope_head(v, pos, cos_tab, sin_tab);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    apply_rope_head(v, pos, cos_tab, sin_tab);
}

/// Dot product of two f32 slices. Length must be divisible by 16; both
/// `HEAD_DIM = 64` and `D_MODEL = 256` satisfy this.
#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len() % 16, 0);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        #[allow(unsafe_code)]
        unsafe {
            return neon_dot_f32(a, b);
        }
    }
    #[allow(unreachable_code)]
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `out += w * v`, element-wise, 64 f32 lanes.
#[inline]
fn axpy_head(out: &mut [f32], w: f32, v: &[f32]) {
    debug_assert_eq!(out.len(), HEAD_DIM);
    debug_assert_eq!(v.len(), HEAD_DIM);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        #[allow(unsafe_code)]
        unsafe {
            neon_axpy_head(out, w, v);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for (o, &x) in out.iter_mut().zip(v.iter()) {
            *o = w.mul_add(x, *o);
        }
    }
}

/// NEON: half-dim `RoPE` on one head in place.
///
/// Rotates pairs `(v[j], v[j + HEAD_DIM/2])` by the per-j angle at
/// `position = pos`, 4 pairs per iteration. `HEAD_DIM` is a const so the
/// loop-trip count is known at compile time; LLVM can unroll and schedule
/// the FMA dependency chain across lanes.
// Single-letter names follow the 2×2 rotation-matrix convention: `(a, b)
// are the two halves of the head, `(c, s)` are cos / sin.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, clippy::many_single_char_names, clippy::similar_names)]
unsafe fn neon_rope_head(v: &mut [f32], pos: usize, cos_tab: &[f32], sin_tab: &[f32]) {
    use std::arch::aarch64::{vfmaq_f32, vld1q_f32, vmulq_f32, vnegq_f32, vst1q_f32};
    debug_assert_eq!(v.len(), HEAD_DIM);
    let half = HEAD_DIM / 2;
    let row = pos * half;
    // SAFETY: callers guarantee slice lengths.
    unsafe {
        let mut j = 0;
        while j + 4 <= half {
            let c = vld1q_f32(cos_tab.as_ptr().add(row + j));
            let s = vld1q_f32(sin_tab.as_ptr().add(row + j));
            let a = vld1q_f32(v.as_ptr().add(j));
            let b = vld1q_f32(v.as_ptr().add(j + half));
            // new_a = a*c - b*s      = fma(-b*s, something)  — simpler to do:
            //   new_a = (a * c) + (-b) * s via vfmaq
            let neg_b = vnegq_f32(b);
            let new_a = vfmaq_f32(vmulq_f32(a, c), neg_b, s);
            // new_b = a*s + b*c
            let new_b = vfmaq_f32(vmulq_f32(b, c), a, s);
            vst1q_f32(v.as_mut_ptr().add(j), new_a);
            vst1q_f32(v.as_mut_ptr().add(j + half), new_b);
            j += 4;
        }
        // Tail path — scalar — for any leftover lanes. With HEAD_DIM = 64
        // and half = 32 (divisible by 4) this is dead code but kept for
        // robustness if HEAD_DIM ever changes.
        while j < half {
            let c = *cos_tab.as_ptr().add(row + j);
            let s = *sin_tab.as_ptr().add(row + j);
            let a = v[j];
            let b = v[j + half];
            v[j] = a.mul_add(c, -(b * s));
            v[j + half] = a.mul_add(s, b * c);
            j += 1;
        }
    }
}

/// NEON: f32 dot product for any length divisible by 16. 4-way unrolled
/// FMA with 4 independent accumulators to hide the FMA pipeline latency.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn neon_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len() % 16, 0);
    // SAFETY: callers guarantee lengths; NEON is baseline on aarch64.
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut off = 0;
        while off + 16 <= a.len() {
            let a0 = vld1q_f32(a.as_ptr().add(off));
            let a1 = vld1q_f32(a.as_ptr().add(off + 4));
            let a2 = vld1q_f32(a.as_ptr().add(off + 8));
            let a3 = vld1q_f32(a.as_ptr().add(off + 12));
            let b0 = vld1q_f32(b.as_ptr().add(off));
            let b1 = vld1q_f32(b.as_ptr().add(off + 4));
            let b2 = vld1q_f32(b.as_ptr().add(off + 8));
            let b3 = vld1q_f32(b.as_ptr().add(off + 12));
            acc0 = vfmaq_f32(acc0, a0, b0);
            acc1 = vfmaq_f32(acc1, a1, b1);
            acc2 = vfmaq_f32(acc2, a2, b2);
            acc3 = vfmaq_f32(acc3, a3, b3);
            off += 16;
        }
        let s01 = vaddq_f32(acc0, acc1);
        let s23 = vaddq_f32(acc2, acc3);
        vaddvq_f32(vaddq_f32(s01, s23))
    }
}

/// NEON: `out += w * v` over 64 f32 lanes. Unrolled 4× for ILP.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn neon_axpy_head(out: &mut [f32], w: f32, v: &[f32]) {
    use std::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};
    debug_assert_eq!(out.len(), HEAD_DIM);
    debug_assert_eq!(v.len(), HEAD_DIM);
    // SAFETY: callers guarantee lengths; NEON is baseline on aarch64.
    unsafe {
        let wv = vdupq_n_f32(w);
        let mut off = 0;
        while off + 16 <= HEAD_DIM {
            let o0 = vld1q_f32(out.as_ptr().add(off));
            let o1 = vld1q_f32(out.as_ptr().add(off + 4));
            let o2 = vld1q_f32(out.as_ptr().add(off + 8));
            let o3 = vld1q_f32(out.as_ptr().add(off + 12));
            let v0 = vld1q_f32(v.as_ptr().add(off));
            let v1 = vld1q_f32(v.as_ptr().add(off + 4));
            let v2 = vld1q_f32(v.as_ptr().add(off + 8));
            let v3 = vld1q_f32(v.as_ptr().add(off + 12));
            vst1q_f32(out.as_mut_ptr().add(off), vfmaq_f32(o0, wv, v0));
            vst1q_f32(out.as_mut_ptr().add(off + 4), vfmaq_f32(o1, wv, v1));
            vst1q_f32(out.as_mut_ptr().add(off + 8), vfmaq_f32(o2, wv, v2));
            vst1q_f32(out.as_mut_ptr().add(off + 12), vfmaq_f32(o3, wv, v3));
            off += 16;
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
