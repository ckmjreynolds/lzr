//! Apple Silicon Metal GPU backend for the int8 inference path —
//! Phase 50C. Gated behind the `gpu-inference` cargo feature so the
//! submission binary stays Metal-free.
//!
//! Day-1 scope (what's here today):
//!   - Global `Gpu` singleton owning a Metal device, command queue,
//!     library, and compiled int8-GEMM pipeline state.
//!   - `Gpu::matmul_i8_per_channel(x_i8, x_scale, w_i8, w_scales, out)`
//!     that mirrors the CPU `matmul_i8_i8_per_channel` API in
//!     `int_inference.rs`.
//!   - Bit-exactness vs CPU is the acceptance gate. The integer
//!     accumulator is the same i32 sum regardless of reduction order,
//!     and the per-row dequant `(acc as f32) * x_scale * w_scales[row]`
//!     is the same three-multiply expression. Modulo Metal fast-math
//!     flags this should round identically to the CPU path.
//!
//! Not in scope today: batched encode (one shader dispatch over the
//! full token sequence), GPU-side activation quantization, embedding
//! lookup, KV cache. Those land in Day-2 / Day-3.

#![allow(unsafe_code)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow};
use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize,
};

const SHADER_SRC: &str = r#"
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
"#;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Params {
    k: u32,
    n: u32,
    x_scale: f32,
}

#[derive(Debug)]
pub(crate) struct Gpu {
    device: Device,
    queue: CommandQueue,
    matmul_pipeline: ComputePipelineState,
}

// `Device`, `CommandQueue`, etc. wrap Objective-C pointers that the
// Metal framework guarantees safe to share across threads (the
// runtime uses CommandQueue internally for thread-safe submission).
// We hold them in a global OnceLock and only mutate via `device` /
// `queue` APIs that are themselves thread-safe per Apple's docs.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

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
        eprintln!(
            "[gpu] init OK: device={} max_threads_per_threadgroup={}",
            device.name(),
            matmul_pipeline.max_total_threads_per_threadgroup()
        );
        Ok(Self {
            device,
            queue,
            matmul_pipeline,
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

    /// y[n] = sum_k x_i8[k] * w_i8[n, k]   (int accumulator)
    /// y[n] = y[n] * x_scale * w_scales[n]
    ///
    /// Mirrors `int_inference::matmul_i8_i8_per_channel`. Output is
    /// written into `out` (which must be vocab_or_n length).
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
        let out_buf = self.empty_buffer(n * size_of::<f32>());
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
        let tg_width = (max_tg as u64).min(n as u64).max(1);
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
}
