//! Apple Silicon Metal GPU backend for the int8 inference path —
//! Phase 50C. Gated behind the `gpu-inference` cargo feature so the
//! submission binary stays Metal-free.
//!
//! Day-1+2 scope (legacy):
//!   - Global `Gpu` singleton owning a Metal device, command queue,
//!     library, and compiled int8-GEMM pipeline state.
//!   - `Gpu::matmul_i8_per_channel(x_i8, x_scale, w_i8, w_scales, out)`
//!     allocating every buffer per call. Kept for the test gate.
//!
//! Day-3 scope (added here):
//!   - `GpuTensor { data_buf, scales_buf }` — Metal device buffers
//!     holding a pre-uploaded weight tensor. Lives behind a
//!     `OnceLock` on `IntTensor` so first GPU dispatch uploads, all
//!     subsequent dispatches reuse.
//!   - `Gpu::upload_tensor(data, scales) -> GpuTensor`.
//!   - `Gpu::matmul_with_buffers(x_i8, x_scale, w_gpu, n, k, out)`
//!     same kernel, but the weight (n*k) and per-row-scale buffers
//!     are read straight from the pre-uploaded `GpuTensor`. Only the
//!     activation, output, and params buffers are allocated per call.
//!
//! Bit-exactness vs CPU is the acceptance gate. The integer
//! accumulator is the same i32 sum regardless of reduction order,
//! and the per-row dequant `(acc as f32) * x_scale * w_scales[row]`
//! is the same three-multiply expression. The pre-uploaded path
//! reads from the identical bytes the per-call path uploads, so the
//! kernel output must be bit-for-bit identical.
//!
//! Not in scope today: batched encode (one shader dispatch over the
//! full token sequence), GPU-side activation quantization, embedding
//! lookup, KV cache. Those land in Day-4.

#![allow(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::suboptimal_flops
)]

use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow};
use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize,
};

const SHADER_SRC: &str = r"
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint k;
    uint n;
    float x_scale;
};

kernel void matmul_i8_per_channel(
    device const char* x_i8 [[buffer(0)]],
    device const char* w_i8 [[buffer(1)]],
    device const float* w_scales [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant Params& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    int acc = 0;
    for (uint c = 0; c < p.k; c++) {
        acc += int(x_i8[c]) * int(w_i8[gid * p.k + c]);
    }
    out[gid] = float(acc) * p.x_scale * w_scales[gid];
}

// Day-4 Phase 2a: batched per-token int8 matmul.
//   x_i8 [M, K], x_scales [M] (one per token row)
//   w_i8 [N, K] row-major, w_scales [N] (one per output channel)
//   out  [M, N]
// One thread per (m, n) output. Same i32 accumulator + same f32
// multiply order as the per-token kernel, so out[m, n] is bit-for-bit
// identical to the per-token kernel called M times.
struct BatchedParams {
    uint m;
    uint k;
    uint n;
};

kernel void batched_matmul_i8_per_channel(
    device const char*  x_i8     [[buffer(0)]],
    device const float* x_scales [[buffer(1)]],
    device const char*  w_i8     [[buffer(2)]],
    device const float* w_scales [[buffer(3)]],
    device float*       out      [[buffer(4)]],
    constant BatchedParams& p    [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.x;  // m
    uint col = gid.y;  // n
    if (row >= p.m || col >= p.n) return;
    int acc = 0;
    device const char* x_row = x_i8 + row * p.k;
    device const char* w_row = w_i8 + col * p.k;
    for (uint c = 0; c < p.k; c++) {
        acc += int(x_row[c]) * int(w_row[c]);
    }
    out[row * p.n + col] = float(acc) * x_scales[row] * w_scales[col];
}
";

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Params {
    k: u32,
    n: u32,
    x_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct BatchedParams {
    m: u32,
    k: u32,
    n: u32,
}

#[derive(Debug)]
pub(crate) struct Gpu {
    device: Device,
    queue: CommandQueue,
    matmul_pipeline: ComputePipelineState,
    batched_matmul_pipeline: ComputePipelineState,
}

// `Device`, `CommandQueue`, etc. wrap Objective-C pointers that the
// Metal framework guarantees safe to share across threads (the
// runtime uses CommandQueue internally for thread-safe submission).
// We hold them in a global OnceLock and only mutate via `device` /
// `queue` APIs that are themselves thread-safe per Apple's docs.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

/// A weight tensor's payload pre-uploaded into Metal device memory.
/// Created once per `IntTensor` on first GPU dispatch via
/// `Gpu::upload_tensor` and held in a `OnceLock` on the owning
/// `IntTensor`. Reused by every subsequent `matmul_with_buffers`
/// call, eliminating the per-call buffer-allocation overhead that
/// dominated Day-2 timings.
#[derive(Debug)]
pub(crate) struct GpuTensor {
    pub(crate) data_buf: Buffer,
    pub(crate) scales_buf: Buffer,
}

// Same rationale as `Gpu` above: Metal `Buffer` wraps an Objective-C
// pointer that the runtime allows to be shared across threads. We
// only ever read these buffers from the GPU kernel after upload, so
// CPU-side aliasing concerns don't apply.
unsafe impl Send for GpuTensor {}
unsafe impl Sync for GpuTensor {}

fn instance() -> Result<&'static Gpu> {
    static G: OnceLock<Result<Gpu, String>> = OnceLock::new();
    let cell = G.get_or_init(|| Gpu::init().map_err(|e| format!("{e:#}")));
    match cell {
        Ok(g) => Ok(g),
        Err(s) => Err(anyhow!("GPU init failed: {s}")),
    }
}

impl Gpu {
    fn init() -> Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| anyhow!("no system-default Metal device (run on Apple Silicon Mac)"))?;
        let queue = device.new_command_queue();
        let library: Library = device
            .new_library_with_source(SHADER_SRC, &metal::CompileOptions::new())
            .map_err(|e| anyhow!("Metal shader compile failed: {e}"))?;
        let function = library
            .get_function("matmul_i8_per_channel", None)
            .map_err(|e| anyhow!("missing kernel function: {e}"))?;
        let matmul_pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| anyhow!("pipeline creation failed: {e}"))?;
        let batched_function = library
            .get_function("batched_matmul_i8_per_channel", None)
            .map_err(|e| anyhow!("missing batched kernel function: {e}"))?;
        let batched_matmul_pipeline = device
            .new_compute_pipeline_state_with_function(&batched_function)
            .map_err(|e| anyhow!("batched pipeline creation failed: {e}"))?;
        eprintln!(
            "[gpu] init OK: device={} max_threads_per_threadgroup={}",
            device.name(),
            matmul_pipeline.max_total_threads_per_threadgroup()
        );
        Ok(Self {
            device,
            queue,
            matmul_pipeline,
            batched_matmul_pipeline,
        })
    }

    fn buffer_from_slice<T: Copy>(&self, data: &[T]) -> Buffer {
        let bytes = size_of_val(data);
        let ptr = data.as_ptr().cast::<std::ffi::c_void>();
        self.device
            .new_buffer_with_data(ptr, bytes as u64, MTLResourceOptions::StorageModeShared)
    }

    fn empty_buffer(&self, bytes: usize) -> Buffer {
        self.device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
    }

    /// `y[n] = sum_k x_i8[k] * w_i8[n, k]`   (int accumulator)
    /// then `y[n] = y[n] * x_scale * w_scales[n]`.
    ///
    /// Mirrors `int_inference::matmul_i8_i8_per_channel`. Output is
    /// written into `out` (which must be `vocab` or `n` length).
    ///
    /// Day 1+2 API kept for the bit-exactness regression test in this
    /// module. The production path (Day 3) uses
    /// `matmul_with_buffers` with pre-uploaded weight buffers.
    #[allow(dead_code)]
    pub(crate) fn matmul_i8_per_channel(
        &self,
        x_i8: &[i8],
        x_scale: f32,
        w_i8: &[i8],
        w_scales: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let k = x_i8.len();
        let n = out.len();
        if w_scales.len() != n {
            return Err(anyhow!("w_scales len {} != n {n}", w_scales.len()));
        }
        if w_i8.len() != n * k {
            return Err(anyhow!(
                "w_i8 len {} != n*k = {} * {} = {}",
                w_i8.len(),
                n,
                k,
                n * k
            ));
        }

        let x_buf = self.buffer_from_slice(x_i8);
        let w_buf = self.buffer_from_slice(w_i8);
        let ws_buf = self.buffer_from_slice(w_scales);
        let out_buf = self.empty_buffer(size_of::<f32>().saturating_mul(n));
        let params = Params {
            k: u32::try_from(k).context("k overflow")?,
            n: u32::try_from(n).context("n overflow")?,
            x_scale,
        };
        let params_buf = self.buffer_from_slice(&[params]);

        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.matmul_pipeline);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&w_buf), 0);
        enc.set_buffer(2, Some(&ws_buf), 0);
        enc.set_buffer(3, Some(&out_buf), 0);
        enc.set_buffer(4, Some(&params_buf), 0);

        let threads_per_grid = MTLSize::new(n as u64, 1, 1);
        let max_tg = self.matmul_pipeline.max_total_threads_per_threadgroup();
        let tg_width = max_tg.min(n as u64).max(1);
        let threads_per_tg = MTLSize::new(tg_width, 1, 1);
        enc.dispatch_threads(threads_per_grid, threads_per_tg);
        enc.end_encoding();

        cb.commit();
        cb.wait_until_completed();

        // SAFETY: shared-storage Metal buffer is host-readable.
        let out_ptr = out_buf.contents().cast::<f32>();
        let slice = unsafe { std::slice::from_raw_parts(out_ptr, n) };
        out.copy_from_slice(slice);
        Ok(())
    }

    /// Upload an `(i8 data, f32 scales)` pair into shared-storage
    /// Metal buffers once. Called by the dispatcher on first GPU use
    /// of a given weight tensor; the result is cached in the owning
    /// `IntTensor`'s `OnceLock<GpuTensor>` and reused for the rest of
    /// the program's lifetime.
    pub(crate) fn upload_tensor(&self, data: &[i8], scales: &[f32]) -> GpuTensor {
        let data_buf = self.buffer_from_slice(data);
        let scales_buf = self.buffer_from_slice(scales);
        GpuTensor {
            data_buf,
            scales_buf,
        }
    }

    /// Same kernel as `matmul_i8_per_channel`, but the weight bytes
    /// and per-row scales are read from a pre-uploaded `GpuTensor`
    /// instead of being copied in on every call. The activation,
    /// output, and params buffers are still allocated per call (the
    /// activation changes every step and the output is tiny).
    pub(crate) fn matmul_with_buffers(
        &self,
        x_i8: &[i8],
        x_scale: f32,
        w_gpu: &GpuTensor,
        n: usize,
        k: usize,
        out: &mut [f32],
    ) -> Result<()> {
        if x_i8.len() != k {
            return Err(anyhow!("x_i8 len {} != k {k}", x_i8.len()));
        }
        if out.len() != n {
            return Err(anyhow!("out len {} != n {n}", out.len()));
        }

        let x_buf = self.buffer_from_slice(x_i8);
        let out_buf = self.empty_buffer(size_of::<f32>().saturating_mul(n));
        let params = Params {
            k: u32::try_from(k).context("k overflow")?,
            n: u32::try_from(n).context("n overflow")?,
            x_scale,
        };
        let params_buf = self.buffer_from_slice(&[params]);

        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.matmul_pipeline);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&w_gpu.data_buf), 0);
        enc.set_buffer(2, Some(&w_gpu.scales_buf), 0);
        enc.set_buffer(3, Some(&out_buf), 0);
        enc.set_buffer(4, Some(&params_buf), 0);

        let threads_per_grid = MTLSize::new(n as u64, 1, 1);
        let max_tg = self.matmul_pipeline.max_total_threads_per_threadgroup();
        let tg_width = max_tg.min(n as u64).max(1);
        let threads_per_tg = MTLSize::new(tg_width, 1, 1);
        enc.dispatch_threads(threads_per_grid, threads_per_tg);
        enc.end_encoding();

        cb.commit();
        cb.wait_until_completed();

        // SAFETY: shared-storage Metal buffer is host-readable.
        let out_ptr = out_buf.contents().cast::<f32>();
        let slice = unsafe { std::slice::from_raw_parts(out_ptr, n) };
        out.copy_from_slice(slice);
        Ok(())
    }

    /// Day-4 Phase 2a: batched int8 matmul. Processes M rows of
    /// activation against the same `n × k` weight tensor in a single
    /// 2D-dispatch.
    ///
    /// Inputs:
    /// - `x_i8`: `[M, K]` int8 activation, row-major. Length `M * K`.
    /// - `x_scales`: `[M]` per-row activation scales (one per token).
    /// - `w_gpu`: pre-uploaded weight tensor `(rows = N, cols = K)`.
    /// - `m`, `n`, `k`: dimensions.
    /// - `out`: `[M, N]` f32 output, row-major. Length `M * N`. Overwritten.
    ///
    /// Same i32 accumulator and same `acc * x_scale * w_scale` triple
    /// product as `matmul_with_buffers`, so every output element is
    /// bit-for-bit identical to a per-row call of the single-row
    /// kernel on the same inputs.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_matmul_i8(
        &self,
        x_i8: &[i8],
        x_scales: &[f32],
        w_gpu: &GpuTensor,
        m: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
    ) -> Result<()> {
        if x_i8.len() != m.saturating_mul(k) {
            return Err(anyhow!("x_i8 len {} != m*k = {}*{}", x_i8.len(), m, k));
        }
        if x_scales.len() != m {
            return Err(anyhow!("x_scales len {} != m {m}", x_scales.len()));
        }
        if out.len() != m.saturating_mul(n) {
            return Err(anyhow!("out len {} != m*n = {}*{}", out.len(), m, n));
        }
        let x_buf = self.buffer_from_slice(x_i8);
        let x_scales_buf = self.buffer_from_slice(x_scales);
        let out_buf = self.empty_buffer(size_of::<f32>().saturating_mul(m * n));
        let params = BatchedParams {
            m: u32::try_from(m).context("m overflow")?,
            k: u32::try_from(k).context("k overflow")?,
            n: u32::try_from(n).context("n overflow")?,
        };
        let params_buf = self.buffer_from_slice(&[params]);

        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.batched_matmul_pipeline);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&x_scales_buf), 0);
        enc.set_buffer(2, Some(&w_gpu.data_buf), 0);
        enc.set_buffer(3, Some(&w_gpu.scales_buf), 0);
        enc.set_buffer(4, Some(&out_buf), 0);
        enc.set_buffer(5, Some(&params_buf), 0);

        let grid = MTLSize::new(m as u64, n as u64, 1);
        // 2D threadgroup: pick a reasonable rectangle. Apple GPUs run
        // 32-thread SIMD groups; (16 × 16) gives 256 threads which
        // is well within max_total_threads_per_threadgroup (1024 on
        // M3 Pro) and keeps occupancy high.
        let max_tg = self
            .batched_matmul_pipeline
            .max_total_threads_per_threadgroup();
        let tg_x = 16u64.min(m as u64).max(1);
        let tg_y = 16u64.min(n as u64).max(1);
        let tg_x = tg_x.min(max_tg);
        let tg_y = (max_tg / tg_x).min(tg_y).max(1);
        let tg = MTLSize::new(tg_x, tg_y, 1);
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        // SAFETY: shared-storage Metal buffer is host-readable.
        let out_ptr = out_buf.contents().cast::<f32>();
        let slice = unsafe { std::slice::from_raw_parts(out_ptr, m * n) };
        out.copy_from_slice(slice);
        Ok(())
    }
}

/// Public accessor for the lazily-initialized singleton.
pub(crate) fn gpu() -> Result<&'static Gpu> {
    instance()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::int_inference::{IntTensor, matmul_i8_i8_per_channel, quantize_act_i8};

    #[test]
    fn gpu_matmul_matches_cpu_exactly() {
        // Synthetic test: random-ish weights and activation, both
        // backends must agree to the last bit.
        let k = 128;
        let n = 384;
        let x_f: Vec<f32> = (0..k).map(|i| (i as f32 * 0.027 - 1.3).sin()).collect();
        let w_f: Vec<f32> = (0..n * k)
            .map(|i| (i as f32 * 0.011 + 0.4).cos() - 0.1)
            .collect();
        let w = IntTensor::pack_per_channel_sym(&w_f, n, k, 8);
        let mut x_i8 = Vec::new();
        let x_scale = quantize_act_i8(&x_f, &mut x_i8);
        let mut y_cpu = vec![0f32; n];
        matmul_i8_i8_per_channel(&x_i8, x_scale, &w, &mut y_cpu);

        let mut y_gpu = vec![0f32; n];
        let g = gpu().expect("Metal device init");
        g.matmul_i8_per_channel(&x_i8, x_scale, &w.data, &w.scales, &mut y_gpu)
            .expect("gpu matmul OK");

        let mut max_diff = 0_f32;
        let mut exact_matches = 0_usize;
        for i in 0..n {
            let d = (y_gpu[i] - y_cpu[i]).abs();
            if d > max_diff {
                max_diff = d;
            }
            if y_gpu[i].to_bits() == y_cpu[i].to_bits() {
                exact_matches += 1;
            }
        }
        let y_max = y_cpu.iter().map(|v| v.abs()).fold(0_f32, f32::max);
        eprintln!(
            "[gpu test] max_abs_diff={max_diff:e}  y_max={y_max:e}  exact_match={exact_matches}/{n}"
        );
        // We allow tiny f32 rounding differences if the final
        // multiplication order differs by hardware path. The integer
        // accumulator part is exact, so any difference comes from the
        // x_scale * w_scale * acc rounding.
        assert!(max_diff <= 1e-3 * y_max, "GPU vs CPU drift too large");
    }

    /// Diagnostic: measure CPU vs GPU per-dispatch latency at the
    /// six matmul shapes that show up in the `MoE` forward pass. Run
    /// with:
    /// ```text
    /// cargo test --features gpu-inference --release \
    ///     gpu::tests::dispatch_latency_bench -- --nocapture --ignored
    /// ```
    /// `--ignored` because we don't want this in the regular suite.
    #[test]
    #[ignore = "diagnostic — run on demand with --ignored, not in CI"]
    fn dispatch_latency_bench() {
        use crate::int_inference::{IntTensor, matmul_i8_i8_per_channel, quantize_act_i8};
        use std::time::Instant;
        let shapes: &[(&str, usize, usize)] = &[
            ("qkv", 3 * 128, 128),
            ("proj", 128, 128),
            ("router", 32, 128),
            ("fc1", 512, 128),
            ("fc2", 128, 512),
            ("vocab", 16384, 128),
        ];
        let g = gpu().expect("metal init");
        let header = format!(
            "\n{:>8}  {:>5}  {:>5}  {:>10}  {:>10}  {:>10}  match",
            "shape", "n", "k", "cpu (µs)", "gpu (µs)", "gpu/cpu"
        );
        eprintln!("{header}");
        let sep = "-".repeat(65);
        eprintln!("{sep}");
        for (name, n, k) in shapes {
            let n = *n;
            let k = *k;
            let w_f: Vec<f32> = (0..n * k)
                .map(|i| ((i as f32) * 0.013 - 0.2).sin())
                .collect();
            let x_f: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.027 + 0.4).cos()).collect();
            let w = IntTensor::pack_per_channel_sym(&w_f, n, k, 8);
            let mut x_i8 = Vec::new();
            let x_scale = quantize_act_i8(&x_f, &mut x_i8);
            let mut out_cpu = vec![0f32; n];
            let mut out_gpu = vec![0f32; n];
            let iters: u32 = 200;
            let t0 = Instant::now();
            for _ in 0..iters {
                matmul_i8_i8_per_channel(&x_i8, x_scale, &w, &mut out_cpu);
            }
            let cpu_us = (t0.elapsed().as_secs_f64() * 1e6) / f64::from(iters);
            let w_gpu = g.upload_tensor(&w.data, &w.scales);
            g.matmul_with_buffers(&x_i8, x_scale, &w_gpu, n, k, &mut out_gpu)
                .expect("gpu warmup");
            let t0 = Instant::now();
            for _ in 0..iters {
                g.matmul_with_buffers(&x_i8, x_scale, &w_gpu, n, k, &mut out_gpu)
                    .expect("gpu");
            }
            let gpu_us = (t0.elapsed().as_secs_f64() * 1e6) / f64::from(iters);
            let mut max_d = 0_f32;
            for i in 0..n {
                max_d = max_d.max((out_cpu[i] - out_gpu[i]).abs());
            }
            let cpu_max = out_cpu.iter().fold(0_f32, |a, &b| a.max(b.abs()));
            let ratio = gpu_us / cpu_us;
            let ok = if max_d <= 1e-3 * cpu_max {
                "OK"
            } else {
                "DRIFT"
            };
            eprintln!(
                "{name:>8}  {n:>5}  {k:>5}  {cpu_us:>10.1}  {gpu_us:>10.1}  {ratio:>9.2}x  {ok}"
            );
        }
        eprintln!(
            "\nIf gpu (µs) >> cpu (µs), per-dispatch sync dominates — Day-4 batched encode needed.\nIf gpu (µs) <  cpu (µs), kernel itself wins — sync model is the bottleneck."
        );
    }

    /// Day-4 Phase 2a diagnostic: wall comparison of the batched
    /// kernel vs M sequential per-row dispatches at production matmul
    /// shapes for varying M. Confirms the kernel actually wins the
    /// dispatch-amortization battle.
    /// ```text
    /// cargo test --features gpu-inference --release \
    ///   gpu::tests::batched_matmul_wall_bench -- --nocapture --ignored
    /// ```
    #[test]
    #[ignore = "diagnostic — run on demand"]
    fn batched_matmul_wall_bench() {
        use std::time::Instant;
        let shapes: &[(&str, usize, usize)] = &[
            ("qkv", 3 * 128, 128),
            ("proj", 128, 128),
            ("router", 32, 128),
            ("fc1", 512, 128),
            ("fc2", 128, 512),
            ("vocab", 16384, 128),
        ];
        let ms: &[usize] = &[16, 64, 256];
        let g = gpu().expect("metal init");
        let header = format!(
            "\n{:>8}  {:>5}  {:>5}  {:>4}  {:>12}  {:>12}  {:>10}",
            "shape", "n", "k", "M", "per-row (ms)", "batched (ms)", "speedup"
        );
        eprintln!("{header}");
        eprintln!("{}", "-".repeat(75));
        for (name, n, k) in shapes {
            let n = *n;
            let k = *k;
            let w_f: Vec<f32> = (0..n * k)
                .map(|i| ((i as f32) * 0.013 - 0.2).sin())
                .collect();
            let w = IntTensor::pack_per_channel_sym(&w_f, n, k, 8);
            let w_gpu = g.upload_tensor(&w.data, &w.scales);
            for &m in ms {
                let x_f: Vec<f32> = (0..m * k)
                    .map(|i| ((i as f32) * 0.027 + 0.4).cos())
                    .collect();
                let mut x_i8 = vec![0i8; m * k];
                let mut x_scales = vec![0f32; m];
                let mut tmp = Vec::new();
                for r in 0..m {
                    x_scales[r] = quantize_act_i8(&x_f[r * k..(r + 1) * k], &mut tmp);
                    x_i8[r * k..(r + 1) * k].copy_from_slice(&tmp);
                }
                let mut out_seq = vec![0f32; m * n];
                let mut out_bat = vec![0f32; m * n];
                let iters: u32 = 30;
                // warmup
                g.matmul_with_buffers(&x_i8[..k], x_scales[0], &w_gpu, n, k, &mut out_seq[..n])
                    .expect("warm");
                let t0 = Instant::now();
                for _ in 0..iters {
                    for r in 0..m {
                        g.matmul_with_buffers(
                            &x_i8[r * k..(r + 1) * k],
                            x_scales[r],
                            &w_gpu,
                            n,
                            k,
                            &mut out_seq[r * n..(r + 1) * n],
                        )
                        .expect("seq");
                    }
                }
                let per_row_ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(iters);
                g.batched_matmul_i8(&x_i8, &x_scales, &w_gpu, m, n, k, &mut out_bat)
                    .expect("warm batched");
                let t0 = Instant::now();
                for _ in 0..iters {
                    g.batched_matmul_i8(&x_i8, &x_scales, &w_gpu, m, n, k, &mut out_bat)
                        .expect("bat");
                }
                let batched_ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(iters);
                let speedup = per_row_ms / batched_ms;
                eprintln!(
                    "{name:>8}  {n:>5}  {k:>5}  {m:>4}  {per_row_ms:>12.2}  {batched_ms:>12.2}  {speedup:>9.2}x"
                );
            }
        }
    }

    /// Day-4 Phase 2a acceptance: batched int8 matmul on M rows of
    /// activation must produce the same `[M, N]` output that M
    /// per-row calls of `matmul_with_buffers` would.
    #[test]
    fn batched_matmul_i8_matches_per_row() {
        let m = 31; // odd to expose row-handling bugs
        let k = 96;
        let n = 256;
        let x_f: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32) * 0.0173 - 0.7).sin())
            .collect();
        let w_f: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.011 + 0.4).cos())
            .collect();
        let w = IntTensor::pack_per_channel_sym(&w_f, n, k, 8);

        // Per-row quantize.
        let mut x_i8 = vec![0i8; m * k];
        let mut x_scales = vec![0f32; m];
        let mut tmp = Vec::new();
        for r in 0..m {
            x_scales[r] = quantize_act_i8(&x_f[r * k..(r + 1) * k], &mut tmp);
            x_i8[r * k..(r + 1) * k].copy_from_slice(&tmp);
        }

        let g = gpu().expect("metal init");
        let w_gpu = g.upload_tensor(&w.data, &w.scales);

        // Reference: M per-row dispatches.
        let mut ref_out = vec![0f32; m * n];
        for r in 0..m {
            g.matmul_with_buffers(
                &x_i8[r * k..(r + 1) * k],
                x_scales[r],
                &w_gpu,
                n,
                k,
                &mut ref_out[r * n..(r + 1) * n],
            )
            .expect("per-row matmul");
        }

        // Batched: one dispatch over (M × N) outputs.
        let mut batched = vec![0f32; m * n];
        g.batched_matmul_i8(&x_i8, &x_scales, &w_gpu, m, n, k, &mut batched)
            .expect("batched matmul");

        let mut exact = 0usize;
        let mut max_d = 0_f32;
        for i in 0..m * n {
            if batched[i].to_bits() == ref_out[i].to_bits() {
                exact += 1;
            }
            max_d = max_d.max((batched[i] - ref_out[i]).abs());
        }
        let total = m * n;
        eprintln!(
            "[batched matmul test] {exact}/{total} bit-exact vs per-row, max_abs_diff={max_d:e}"
        );
        assert_eq!(
            exact, total,
            "batched kernel must bit-match per-row kernel — max_abs_diff={max_d:e}"
        );
    }

    /// Day-3 path: drive the dispatcher with pre-uploaded weight
    /// buffers via `matmul_with_buffers`. Result must be bit-for-bit
    /// identical to the Day-1+2 per-call path because both kernels
    /// read the same i8 bytes and apply the same dequant.
    #[test]
    fn gpu_matmul_with_buffers_matches_per_call() {
        let k = 96;
        let n = 256;
        let x_f: Vec<f32> = (0..k).map(|i| (i as f32 * 0.041 + 0.7).cos()).collect();
        let w_f: Vec<f32> = (0..n * k)
            .map(|i| (i as f32 * 0.013 - 0.2).sin() * 0.9)
            .collect();
        let w = IntTensor::pack_per_channel_sym(&w_f, n, k, 8);
        let mut x_i8 = Vec::new();
        let x_scale = quantize_act_i8(&x_f, &mut x_i8);

        let g = gpu().expect("Metal device init");
        let mut y_a = vec![0f32; n];
        g.matmul_i8_per_channel(&x_i8, x_scale, &w.data, &w.scales, &mut y_a)
            .expect("per-call ok");

        // Upload once, dispatch twice — second dispatch hits the
        // cached buffers without re-uploading.
        let w_gpu = g.upload_tensor(&w.data, &w.scales);
        let mut y_b = vec![0f32; n];
        let mut y_c = vec![0f32; n];
        g.matmul_with_buffers(&x_i8, x_scale, &w_gpu, n, k, &mut y_b)
            .expect("with-buffers ok");
        g.matmul_with_buffers(&x_i8, x_scale, &w_gpu, n, k, &mut y_c)
            .expect("with-buffers (reuse) ok");

        let mut exact_ab = 0;
        let mut exact_bc = 0;
        for i in 0..n {
            if y_a[i].to_bits() == y_b[i].to_bits() {
                exact_ab += 1;
            }
            if y_b[i].to_bits() == y_c[i].to_bits() {
                exact_bc += 1;
            }
        }
        eprintln!(
            "[gpu day3 test] per-call vs with-buffers: {exact_ab}/{n} exact; \
             repeated dispatch: {exact_bc}/{n} exact"
        );
        assert_eq!(exact_ab, n, "Day-3 path must bit-match Day-2 path");
        assert_eq!(exact_bc, n, "Repeated GPU dispatch must be deterministic");
    }
}
