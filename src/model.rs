//! RWKV v4 byte-level inference.
//!
//! One linear-time-mix + one channel-mix per block, sharing ternary matvec
//! kernels with the retired transformer path. Per-layer state is five
//! `D_MODEL`-sized vectors; there is no KV cache and no context-length
//! bound — each `step` consumes one token in `O(1)` work independent of
//! stream length.
//!
//! RWKV reference: <https://github.com/BlinkDL/RWKV-LM> (v4 `RavenNeo`
//! variant). The WKV numerical-stability trick (`pp` log-space tracking)
//! matches the reference implementation's `time_mixing` function.

use crate::arch::{D_FF, D_MODEL, N_LAYERS, RMS_EPS, VOCAB};
use crate::bitnet::{
    build_lut, dequantize_i2s_row, lut_scratch_bytes, matvec_ternary, matvec_ternary_lut,
    quantize_activations,
};
use crate::tokenizer::Token;
use crate::weights::{TernaryMatrix, Weights};

/// Per-layer RWKV state. Five `D_MODEL`-sized f32 vectors.
#[derive(Debug)]
pub(crate) struct LayerState {
    /// Previous post-norm input to the time-mix block (token shift source).
    x_tm: Vec<f32>,
    /// Previous post-norm input to the channel-mix block (token shift source).
    x_cm: Vec<f32>,
    /// WKV numerator running sum (in log-scale via `pp`).
    aa: Vec<f32>,
    /// WKV denominator running sum.
    bb: Vec<f32>,
    /// Log-scale tracker for the running-max trick that keeps `aa` / `bb`
    /// in representable range regardless of stream length.
    pp: Vec<f32>,
}

impl LayerState {
    fn new() -> Self {
        Self {
            x_tm: vec![0.0; D_MODEL],
            x_cm: vec![0.0; D_MODEL],
            aa: vec![0.0; D_MODEL],
            bb: vec![0.0; D_MODEL],
            pp: vec![f32::NEG_INFINITY; D_MODEL],
        }
    }

    fn reset(&mut self) {
        for v in &mut self.x_tm {
            *v = 0.0;
        }
        for v in &mut self.x_cm {
            *v = 0.0;
        }
        for v in &mut self.aa {
            *v = 0.0;
        }
        for v in &mut self.bb {
            *v = 0.0;
        }
        for v in &mut self.pp {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Pre-allocated per-step scratch buffers.
#[derive(Debug)]
pub(crate) struct Scratch {
    x: Vec<f32>,
    residual: Vec<f32>,
    normed: Vec<f32>,
    /// Token-shift lerp output used per matvec. Three copies so xr/xk/xv
    /// are alive at once during the time-mix block.
    xr: Vec<f32>,
    xk: Vec<f32>,
    xv: Vec<f32>,
    /// `D_MODEL` quantization scratch.
    xq: Vec<i8>,
    /// `D_FF` quantization scratch (channel-mix `V` input).
    xq_ff: Vec<i8>,
    /// Arch-preferred LUT scratch, sized for the widest activation vector
    /// (`max(D_MODEL, D_FF)`). Zero-length on architectures without a LUT
    /// kernel. `build_lut` writes only the valid entries per group; callers
    /// must keep the bytes zero-initialized (they are zero-initialized once
    /// at `Scratch::new` and the written entries are deterministic per
    /// activation so no explicit re-zero is needed).
    lut: Vec<i8>,
    /// Time-mix intermediates.
    r: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    wkv: Vec<f32>,
    rwkv: Vec<f32>,
    /// Channel-mix intermediates.
    cm_k: Vec<f32>,
    cm_r: Vec<f32>,
    logits: [f32; VOCAB],
}

impl Default for Scratch {
    fn default() -> Self {
        Self::new()
    }
}

impl Scratch {
    pub(crate) fn new() -> Self {
        Self {
            x: vec![0.0; D_MODEL],
            residual: vec![0.0; D_MODEL],
            normed: vec![0.0; D_MODEL],
            xr: vec![0.0; D_MODEL],
            xk: vec![0.0; D_MODEL],
            xv: vec![0.0; D_MODEL],
            xq: vec![0; D_MODEL],
            xq_ff: vec![0; D_FF],
            lut: vec![0i8; lut_scratch_bytes(D_MODEL.max(D_FF))],
            r: vec![0.0; D_MODEL],
            k: vec![0.0; D_MODEL],
            v: vec![0.0; D_MODEL],
            wkv: vec![0.0; D_MODEL],
            rwkv: vec![0.0; D_MODEL],
            cm_k: vec![0.0; D_FF],
            cm_r: vec![0.0; D_MODEL],
            logits: [0.0; VOCAB],
        }
    }
}

#[inline]
fn rmsnorm(x: &[f32], weight: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), out.len());
    #[allow(clippy::cast_precision_loss)]
    let n = x.len() as f32;
    let mut ss = 0.0_f32;
    for &v in x {
        ss = v.mul_add(v, ss);
    }
    let scale = (ss / n + RMS_EPS).sqrt().recip();
    for i in 0..x.len() {
        out[i] = x[i] * scale * weight[i];
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `out[i] = cur[i] * mix[i] + prev[i] * (1 - mix[i])`. RWKV token-shift.
#[inline]
fn lerp(cur: &[f32], prev: &[f32], mix: &[f32], out: &mut [f32]) {
    debug_assert_eq!(cur.len(), prev.len());
    debug_assert_eq!(cur.len(), mix.len());
    debug_assert_eq!(cur.len(), out.len());
    for i in 0..cur.len() {
        out[i] = cur[i].mul_add(mix[i], prev[i] * (1.0 - mix[i]));
    }
}

fn matvec_prequant(mat: &TernaryMatrix<'_>, x_q: &[i8], lut: &[i8], x_scale: f32, out: &mut [f32]) {
    if mat.lut_packed.is_empty() {
        matvec_ternary(mat.packed, mat.scale, x_q, x_scale, out);
    } else {
        matvec_ternary_lut(mat.lut_packed, mat.scale, lut, x_scale, x_q.len(), out);
    }
}

/// Quantize activations into `xq_buf` and build the arch-preferred LUT.
/// Returns `(x_q_slice, x_scale)`.
fn quantize_and_build_lut<'a>(
    x: &[f32],
    xq_buf: &'a mut [i8],
    lut_buf: &mut [i8],
) -> (&'a [i8], f32) {
    let scale = quantize_activations(x, xq_buf);
    let lut_len = lut_scratch_bytes(xq_buf.len());
    build_lut(xq_buf, &mut lut_buf[..lut_len]);
    (xq_buf, scale)
}

/// RWKV v4 byte-level model.
#[derive(Debug)]
pub(crate) struct ByteTransformer {
    weights: Weights,
    state: [LayerState; N_LAYERS],
    scratch: Scratch,
}

impl ByteTransformer {
    pub(crate) fn new(weights: Weights) -> Self {
        Self {
            weights,
            state: core::array::from_fn(|_| LayerState::new()),
            scratch: Scratch::new(),
        }
    }

    /// Clear all per-layer state. Call at the start of every encode / decode
    /// session; RWKV carries stream state across tokens otherwise.
    pub(crate) fn reset(&mut self) {
        for s in &mut self.state {
            s.reset();
        }
    }

    /// Consume one token, advance state, return the logits predicting the
    /// next token.
    // The step function reads top-to-bottom in the same order as RWKV v4's
    // reference implementation (embed → ln0 → per-layer time-mix +
    // channel-mix → ln_f → tied LM head); splitting it into helpers would
    // obscure the dataflow.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn step(&mut self, token: Token) -> &[f32; VOCAB] {
        let scratch = &mut self.scratch;

        // Embedding + ln0. Dequantize one row of the ternary `tok_emb`
        // into `scratch.x` via per-row scale.
        let emb = self.weights.tok_emb_mat();
        let row_packed_off = (token as usize) * (D_MODEL / 4);
        let row_packed = &emb.packed[row_packed_off..row_packed_off + (D_MODEL / 4)];
        let row_scale = emb.scale[token as usize];
        dequantize_i2s_row(row_packed, row_scale, D_MODEL, &mut scratch.x);
        let ln0_w = self.weights.ln0();
        let mut tmp = [0.0_f32; D_MODEL];
        rmsnorm(&scratch.x, ln0_w, &mut tmp);
        scratch.x.copy_from_slice(&tmp);

        for li in 0..N_LAYERS {
            let layer = self.weights.layer(li);
            let state = &mut self.state[li];

            // ---- Time-mix ----
            scratch.residual.copy_from_slice(&scratch.x);
            rmsnorm(&scratch.x, layer.tm_norm, &mut scratch.normed);

            // Three token-shift lerps: xr, xk, xv.
            lerp(
                &scratch.normed,
                &state.x_tm,
                layer.time_mix_r,
                &mut scratch.xr,
            );
            lerp(
                &scratch.normed,
                &state.x_tm,
                layer.time_mix_k,
                &mut scratch.xk,
            );
            lerp(
                &scratch.normed,
                &state.x_tm,
                layer.time_mix_v,
                &mut scratch.xv,
            );
            // Shift: previous post-norm x becomes the current one for next step.
            state.x_tm.copy_from_slice(&scratch.normed);

            // Receptance: r = sigmoid(R · xr)
            {
                let (x_q, x_scale) =
                    quantize_and_build_lut(&scratch.xr, &mut scratch.xq, &mut scratch.lut);
                let lut_len = lut_scratch_bytes(x_q.len());
                let lut = &scratch.lut[..lut_len];
                matvec_prequant(&layer.tm_r, x_q, lut, x_scale, &mut scratch.r);
            }
            for v in &mut scratch.r {
                *v = sigmoid(*v);
            }

            // Key: k = K · xk
            {
                let (x_q, x_scale) =
                    quantize_and_build_lut(&scratch.xk, &mut scratch.xq, &mut scratch.lut);
                let lut_len = lut_scratch_bytes(x_q.len());
                let lut = &scratch.lut[..lut_len];
                matvec_prequant(&layer.tm_k, x_q, lut, x_scale, &mut scratch.k);
            }

            // Value: v = V · xv
            {
                let (x_q, x_scale) =
                    quantize_and_build_lut(&scratch.xv, &mut scratch.xq, &mut scratch.lut);
                let lut_len = lut_scratch_bytes(x_q.len());
                let lut = &scratch.lut[..lut_len];
                matvec_prequant(&layer.tm_v, x_q, lut, x_scale, &mut scratch.v);
            }

            // WKV with the running-max pp trick:
            //   ww = time_first + k
            //   qq = max(pp, ww)
            //   e1 = exp(pp - qq); e2 = exp(ww - qq)
            //   wkv = (e1 * aa + e2 * v) / (e1 * bb + e2)
            //
            //   ww' = pp + time_decay
            //   qq' = max(ww', k)
            //   e1' = exp(ww' - qq'); e2' = exp(k - qq')
            //   aa = e1' * aa + e2' * v
            //   bb = e1' * bb + e2'
            //   pp = qq'
            for i in 0..D_MODEL {
                let ki = scratch.k[i];
                let vi = scratch.v[i];
                let aa = state.aa[i];
                let bb = state.bb[i];
                let pp = state.pp[i];

                let ww = layer.time_first[i] + ki;
                let qq = pp.max(ww);
                let e1 = (pp - qq).exp();
                let e2 = (ww - qq).exp();
                let num = e1.mul_add(aa, e2 * vi);
                let den = e1.mul_add(bb, e2);
                scratch.wkv[i] = num / den;

                let ww2 = pp + layer.time_decay[i];
                let qq2 = ww2.max(ki);
                let e1b = (ww2 - qq2).exp();
                let e2b = (ki - qq2).exp();
                state.aa[i] = e1b.mul_add(aa, e2b * vi);
                state.bb[i] = e1b.mul_add(bb, e2b);
                state.pp[i] = qq2;
            }

            // rwkv = r * wkv; output = O · rwkv; residual.
            for i in 0..D_MODEL {
                scratch.rwkv[i] = scratch.r[i] * scratch.wkv[i];
            }
            {
                let (x_q, x_scale) =
                    quantize_and_build_lut(&scratch.rwkv, &mut scratch.xq, &mut scratch.lut);
                let lut_len = lut_scratch_bytes(x_q.len());
                let lut = &scratch.lut[..lut_len];
                matvec_prequant(&layer.tm_o, x_q, lut, x_scale, &mut scratch.x);
            }
            for (x, r) in scratch.x.iter_mut().zip(scratch.residual.iter()) {
                *x += r;
            }

            // ---- Channel-mix ----
            scratch.residual.copy_from_slice(&scratch.x);
            rmsnorm(&scratch.x, layer.cm_norm, &mut scratch.normed);

            lerp(
                &scratch.normed,
                &state.x_cm,
                layer.channel_mix_k,
                &mut scratch.xk,
            );
            lerp(
                &scratch.normed,
                &state.x_cm,
                layer.channel_mix_r,
                &mut scratch.xr,
            );
            state.x_cm.copy_from_slice(&scratch.normed);

            // Receptance: r = sigmoid(R · xr)
            {
                let (x_q, x_scale) =
                    quantize_and_build_lut(&scratch.xr, &mut scratch.xq, &mut scratch.lut);
                let lut_len = lut_scratch_bytes(x_q.len());
                let lut = &scratch.lut[..lut_len];
                matvec_prequant(&layer.cm_r, x_q, lut, x_scale, &mut scratch.cm_r);
            }
            for v in &mut scratch.cm_r {
                *v = sigmoid(*v);
            }

            // Key: k = K · xk (D_MODEL -> D_FF), squared-ReLU.
            {
                let (x_q, x_scale) =
                    quantize_and_build_lut(&scratch.xk, &mut scratch.xq, &mut scratch.lut);
                let lut_len = lut_scratch_bytes(x_q.len());
                let lut = &scratch.lut[..lut_len];
                matvec_prequant(&layer.cm_k, x_q, lut, x_scale, &mut scratch.cm_k);
            }
            for v in &mut scratch.cm_k {
                let vv = v.max(0.0);
                *v = vv * vv;
            }

            // Value: v = V · cm_k (D_FF -> D_MODEL); output = r * v; residual.
            {
                let (x_q, x_scale) =
                    quantize_and_build_lut(&scratch.cm_k, &mut scratch.xq_ff, &mut scratch.lut);
                let lut_len = lut_scratch_bytes(x_q.len());
                let lut = &scratch.lut[..lut_len];
                matvec_prequant(&layer.cm_v, x_q, lut, x_scale, &mut scratch.x);
            }
            for i in 0..D_MODEL {
                scratch.x[i] = scratch.cm_r[i].mul_add(scratch.x[i], scratch.residual[i]);
            }
        }

        // Final ln_f.
        let ln_f_w = self.weights.ln_f();
        let mut tmp = [0.0_f32; D_MODEL];
        rmsnorm(&scratch.x, ln_f_w, &mut tmp);
        scratch.x.copy_from_slice(&tmp);

        // Tied-embedding output head: logit[i] = x · tok_emb[i]. The
        // ternary matvec runs through the same kernel as the per-layer
        // weights — `tok_emb_mat()` returns a TernaryMatrix and
        // `matvec_prequant` dispatches to LUT or scalar.
        {
            let (x_q, x_scale) =
                quantize_and_build_lut(&scratch.x, &mut scratch.xq, &mut scratch.lut);
            let lut_len = lut_scratch_bytes(x_q.len());
            let lut = &scratch.lut[..lut_len];
            matvec_prequant(&emb, x_q, lut, x_scale, &mut scratch.logits);
        }

        &scratch.logits
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
        let _ = model.step(u16::from(b'a'));
        let _ = model.step(u16::from(b'b'));
        let _ = model.step(u16::from(b'c'));
    }

    #[test]
    fn reset_clears_state() {
        let mut model = ByteTransformer::new(zero_weights());
        for i in 0..10u16 {
            let _ = model.step(i);
        }
        model.reset();
        for s in &model.state {
            assert!(s.x_tm.iter().all(|v| *v == 0.0));
            assert!(s.x_cm.iter().all(|v| *v == 0.0));
            assert!(s.aa.iter().all(|v| *v == 0.0));
            assert!(s.bb.iter().all(|v| *v == 0.0));
            assert!(s.pp.iter().all(|v| *v == f32::NEG_INFINITY));
        }
    }

    #[test]
    fn rmsnorm_basic() {
        let x = [3.0_f32, 4.0, 0.0, 0.0];
        let w = [1.0_f32; 4];
        let mut out = [0.0_f32; 4];
        rmsnorm(&x, &w, &mut out);
        assert!((out[0] - 1.2).abs() < 1e-3);
        assert!((out[1] - 1.6).abs() < 1e-3);
    }

    #[test]
    fn lerp_mixes_correctly() {
        let cur = [1.0_f32, 2.0, 3.0];
        let prev = [0.0_f32, 0.0, 0.0];
        let mix = [0.5_f32, 0.25, 1.0];
        let mut out = [0.0_f32; 3];
        lerp(&cur, &prev, &mix, &mut out);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
        assert!((out[2] - 3.0).abs() < 1e-6);
    }
}
