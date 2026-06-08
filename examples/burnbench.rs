//! Minimal burn 0.21 bring-up + GPU-vs-CPU matmul benchmark.
//! Verifies (a) burn compiles, (b) the wgpu/Metal GPU and ndarray CPU backends
//! both run, (c) GPU is faster than CPU on a representative op (goal 4).
//!
//! Run: cargo run --release --example burnbench --features neural
#![recursion_limit = "256"]

use std::time::Instant;

use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};

fn bench<B: Backend>(device: &B::Device, n: usize, iters: usize) -> f64 {
    let a = Tensor::<B, 2>::random([n, n], Distribution::Default, device);
    let b = Tensor::<B, 2>::random([n, n], Distribution::Default, device);
    // Warm up and force the (async) GPU queue to flush before timing.
    let mut c = a.matmul(b.clone());
    let _ = c.to_data();
    let t = Instant::now();
    for _ in 0..iters {
        c = c.matmul(b.clone());
    }
    let _ = c.to_data(); // force completion of all queued matmuls
    let dt = t.elapsed().as_secs_f64();
    dt * 1000.0 / iters as f64
}

fn main() {
    let n = 1024;
    let iters = 50;

    type Gpu = burn::backend::Wgpu;
    type Cpu = burn::backend::NdArray;

    let gpu_dev = burn::backend::wgpu::WgpuDevice::default();
    let cpu_dev = burn::backend::ndarray::NdArrayDevice::default();

    let gpu_ms = bench::<Gpu>(&gpu_dev, n, iters);
    println!("GPU (wgpu/Metal): {gpu_ms:.3} ms/matmul ({n}x{n})");

    let cpu_ms = bench::<Cpu>(&cpu_dev, n, iters);
    println!("CPU (ndarray):    {cpu_ms:.3} ms/matmul ({n}x{n})");

    println!("GPU speedup vs CPU: {:.2}x", cpu_ms / gpu_ms);
}
