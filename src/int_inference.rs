//! Integer-only inference primitives — Phase 50A.
//!
//! Builds on top of the existing f32 q-dq scheme by keeping a packed
//! int8 representation alongside the f32 weights and providing a
//! dynamic per-token activation quantizer plus an int8 × int8 → int32
//! → f32 GEMM. The motivation is twofold:
//!
//! - **Reproducibility across backends**: an integer GEMM with an
//!   integer accumulator produces the same result regardless of
//!   reduction order, which is the prerequisite for a GPU/CPU
//!   backend swap that ships interoperable archives (Phase 50C).
//! - **Smaller activations on GPU**: int8 activations halve memory
//!   bandwidth vs f32, which is the throughput-binding term on
//!   Apple-Silicon Metal for our tiny per-step matmul sizes.
//!
//! The CPU path here is intentionally simple scalar code that LLVM
//! can auto-vectorize; matches the project's "trust the compiler
//! first" policy. SIMD intrinsics may be added later if profiling
//! warrants it.

#![allow(dead_code)]

/// Per-output-channel packed int8 weight tensor stored row-major
/// (rows = output channels, cols = input dim). One f32 scale per row.
/// Reconstruction: `w_f32[r, c] ≈ w_i8[r, c] * scales[r]`.
///
/// When the `gpu-inference` feature is on, the tensor also carries a
/// lazily-initialized `OnceLock<GpuTensor>` (Phase 50C Day 3). The
/// first time the dispatcher routes a matmul on this tensor to the
/// GPU, the weight bytes and per-row scales are uploaded into Metal
/// device buffers; every subsequent dispatch reuses them. This
/// eliminates the per-call buffer-allocation overhead that dominated
/// the Day-2 GPU timing.
#[derive(Debug)]
pub(crate) struct IntTensor {
    pub(crate) data: Vec<i8>,
    pub(crate) scales: Vec<f32>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    #[cfg(feature = "gpu-inference")]
    pub(crate) gpu: std::sync::OnceLock<crate::gpu::GpuTensor>,
}

impl IntTensor {
    /// Pack an f32 row-major weight matrix `[rows, cols]` into int8
    /// using per-output-channel (per-row) symmetric quantization.
    /// `cap_bits` is the number of significant bits to use (e.g. 8 for
    /// full int8, 5 for an int5-stored-in-int8 cell). The packed
    /// values still occupy a full `i8` cell each; `cap_bits < 8`
    /// merely matches the Phase-49 `mixed*` precision envelope so
    /// L(C) parity is achievable.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub(crate) fn pack_per_channel_sym(
        weights: &[f32],
        rows: usize,
        cols: usize,
        cap_bits: u32,
    ) -> Self {
        assert_eq!(weights.len(), rows * cols, "shape mismatch");
        assert!((1..=8).contains(&cap_bits), "cap_bits must be 1..=8");
        let qmax = (1i32 << (cap_bits - 1)) - 1; // e.g. 127 for 8 bits
        let mut data = vec![0i8; rows * cols];
        let mut scales = vec![0f32; rows];
        for r in 0..rows {
            let row = &weights[r * cols..(r + 1) * cols];
            let mut max_abs = 0f32;
            for &v in row {
                let a = v.abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
            let scale = if max_abs > 0.0 {
                max_abs / qmax as f32
            } else {
                1.0
            };
            scales[r] = scale;
            let inv = 1.0 / scale;
            let out_row = &mut data[r * cols..(r + 1) * cols];
            for c in 0..cols {
                let q = (row[c] * inv).round().clamp(-qmax as f32, qmax as f32) as i32;
                out_row[c] = q as i8;
            }
        }
        Self {
            data,
            scales,
            rows,
            cols,
            #[cfg(feature = "gpu-inference")]
            gpu: std::sync::OnceLock::new(),
        }
    }
}

/// Per-token activation quantizer: max-abs symmetric, scale-only.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub(crate) fn quantize_act_i8(x: &[f32], out: &mut Vec<i8>) -> f32 {
    out.clear();
    out.reserve(x.len());
    let mut max_abs = 0f32;
    for &v in x {
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    let qmax = 127i32;
    let scale = if max_abs > 0.0 {
        max_abs / qmax as f32
    } else {
        1.0
    };
    let inv = 1.0 / scale;
    for &v in x {
        let q = (v * inv).round().clamp(-127.0, 127.0) as i32;
        out.push(q as i8);
    }
    scale
}

/// `y[n] = sum_k x_i8[k] * w_i8[n, k]`   (i32 accumulator)
/// `y_f32[n] = y_i32[n] * x_scale * w.scales[n]`
///
/// Shapes:
/// - `x_i8`: \[k\]
/// - `w`: rows = n, cols = k
/// - `out`: \[n\], **overwritten**
#[allow(clippy::cast_precision_loss, clippy::needless_range_loop)]
pub(crate) fn matmul_i8_i8_per_channel(x_i8: &[i8], x_scale: f32, w: &IntTensor, out: &mut [f32]) {
    let k = w.cols;
    let n = w.rows;
    assert_eq!(x_i8.len(), k, "activation length mismatch");
    assert_eq!(out.len(), n, "out length mismatch");
    for row in 0..n {
        let w_row = &w.data[row * k..(row + 1) * k];
        let mut acc: i32 = 0;
        for c in 0..k {
            acc += i32::from(x_i8[c]) * i32::from(w_row[c]);
        }
        out[row] = (acc as f32) * x_scale * w.scales[row];
    }
}

/// Env-var-controlled toggle for routing matmuls through the Metal
/// GPU backend (only available with the `gpu-inference` feature). The
/// env var is read once on first call.
#[cfg(feature = "gpu-inference")]
fn gpu_backend_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("LZR_GPU_BACKEND")
            .ok()
            .as_deref()
            .is_some_and(|s| s == "1" || s.eq_ignore_ascii_case("true"))
    })
}

/// Day-4 Phase 2b: batched int8 matmul dispatcher.
///
/// Processes M rows of activation through the same weight tensor in
/// one GPU dispatch (or M CPU dispatches in the CPU path). Same
/// bit-equivalence contract as `matmul_dispatch`: every output
/// element is identical to what M sequential `matmul_dispatch` calls
/// would produce.
///
/// `x_i8` is `[M, K]` row-major. `x_scales[m]` is the per-row
/// activation scale (one per token). `out` is `[M, N]` row-major.
/// On the GPU path, the weight tensor's `OnceLock<GpuTensor>` is
/// uploaded lazily on first use just like the single-row path.
///
/// **Small-M fast path:** below `GPU_BATCH_MIN_M`, even on the GPU
/// backend we fall back to the per-row CPU kernel. Apple Metal's
/// ~250 µs dispatch floor dominates a tiny GEMM, while CPU NEON int8
/// finishes the same work in 1 µs. The threshold is set so the
/// per-expert `MoE` FFN path (~8 tokens/expert on average) stays on
/// CPU until Phase 3 introduces a proper expert-batched kernel.
pub(crate) fn batched_matmul_dispatch(
    x_i8: &[i8],
    x_scales: &[f32],
    w: &IntTensor,
    m: usize,
    out: &mut [f32],
) {
    let n = w.rows;
    let k = w.cols;
    debug_assert_eq!(x_i8.len(), m * k);
    debug_assert_eq!(x_scales.len(), m);
    debug_assert_eq!(out.len(), m * n);
    #[cfg(feature = "gpu-inference")]
    if gpu_backend_enabled() && m >= GPU_BATCH_MIN_M {
        let gpu = crate::gpu::gpu().expect("GPU init");
        let w_gpu = w.gpu.get_or_init(|| gpu.upload_tensor(&w.data, &w.scales));
        gpu.batched_matmul_i8(x_i8, x_scales, w_gpu, m, n, k, out)
            .expect("GPU batched matmul");
        return;
    }
    // CPU fallback: loop the per-row kernel.
    for r in 0..m {
        matmul_i8_i8_per_channel(
            &x_i8[r * k..(r + 1) * k],
            x_scales[r],
            w,
            &mut out[r * n..(r + 1) * n],
        );
    }
}

/// Minimum batch size for the GPU batched matmul to win.  Below this,
/// CPU NEON int8 GEMM is faster than the Metal dispatch overhead. See
/// the `gpu::tests::batched_matmul_wall_bench` numbers — at M=16 the
/// GPU batched kernel is ~15× per-row but the absolute wall is
/// dominated by command-buffer overhead until M is large enough that
/// the kernel compute can amortize it.
#[cfg(feature = "gpu-inference")]
const GPU_BATCH_MIN_M: usize = 32;
#[cfg(not(feature = "gpu-inference"))]
#[allow(dead_code)]
const GPU_BATCH_MIN_M: usize = 32;

/// Dispatcher that picks the CPU or GPU backend for one int8 matmul.
/// Always CPU when the `gpu-inference` feature is off (i.e., the
/// submission build).
///
/// On the GPU path (Day 3 onward) the weight tensor's buffers are
/// lazily uploaded on the first call and cached on the `IntTensor`
/// itself via `OnceLock`. The activation buffer is still allocated
/// per call (it changes every step) but the dominant cost — a
/// per-call upload of the entire weight matrix, up to 2 MB for the
/// vocab projection — is paid exactly once per tensor.
pub(crate) fn matmul_dispatch(x_i8: &[i8], x_scale: f32, w: &IntTensor, out: &mut [f32]) {
    #[cfg(feature = "gpu-inference")]
    if gpu_backend_enabled() {
        let gpu = crate::gpu::gpu().expect("GPU init");
        let w_gpu = w.gpu.get_or_init(|| gpu.upload_tensor(&w.data, &w.scales));
        gpu.matmul_with_buffers(x_i8, x_scale, w_gpu, w.rows, w.cols, out)
            .expect("GPU matmul");
        return;
    }
    matmul_i8_i8_per_channel(x_i8, x_scale, w, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::many_single_char_names, clippy::suboptimal_flops)]
    fn matmul_f32_ref(x: &[f32], w: &[f32], out: &mut [f32], k: usize, n: usize) {
        for row in 0..n {
            let mut s = 0f32;
            for c in 0..k {
                s += x[c] * w[row * k + c];
            }
            out[row] = s;
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn int_matmul_matches_f32_within_tolerance() {
        let k = 128;
        let n = 256;
        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.03).sin()).collect();
        let w: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.013).cos()).collect();

        let mut y_ref = vec![0f32; n];
        matmul_f32_ref(&x, &w, &mut y_ref, k, n);

        let w_pack = IntTensor::pack_per_channel_sym(&w, n, k, 8);
        let mut x_i8 = Vec::new();
        let x_scale = quantize_act_i8(&x, &mut x_i8);
        let mut y_int = vec![0f32; n];
        matmul_i8_i8_per_channel(&x_i8, x_scale, &w_pack, &mut y_int);

        // Mixed criterion: per-output we accept either small absolute
        // error (relative to the output scale) or small relative error.
        // This is the standard way to bench int8 inference because tiny
        // outputs near zero blow up the relative metric.
        let y_max = y_ref.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let abs_floor = 0.01 * y_max;
        let mut max_rel_large = 0f32;
        for i in 0..n {
            let abs = (y_int[i] - y_ref[i]).abs();
            if y_ref[i].abs() > abs_floor {
                let rel = abs / y_ref[i].abs();
                if rel > max_rel_large {
                    max_rel_large = rel;
                }
            }
        }
        assert!(
            max_rel_large < 0.05,
            "max relative error on non-tiny outputs {max_rel_large:.4}"
        );
    }

    #[test]
    fn pack_zero_row_handled() {
        let w = vec![0.0f32; 64];
        let p = IntTensor::pack_per_channel_sym(&w, 4, 16, 8);
        for &s in &p.scales {
            assert!(s.is_finite());
        }
    }
}
