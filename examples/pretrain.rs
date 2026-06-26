//! Offline GPU pretraining of a small byte-LM via `burn` (the `neural` feature).
//!
//! Trains a `K`-byte-context MLP next-byte predictor on the preprocessed
//! (post-pipeline) enwik8 the codec's models actually see, then exports a compact
//! weight blob (`[K,E,H,V:u32]` then `emb,w1,b1,w2,b2` f32 LE, row-major) that the
//! submission build embeds and runs on the CPU (`src/models/pretrained.rs`). Never
//! in the submission build. Training the net on the corpus it compresses is the
//! correct Hutter setup — the weights ship as `L(D)` and the decompressor
//! reconstructs from them.
//!
//! Run: `LZR_PP_DATA=/tmp/lzr_pp.bin LZR_NET_OUT=/tmp/lzr_net.bin \
//!       cargo run --example pretrain --features neural`
//!
//! STATUS (2026-06-24): FIXED. The net was stuck at the unigram because the
//! `[batch*K, E] -> [batch, K*E]` flatten reshape has a broken backward in burn
//! 0.21 (it moved the params but produced non-descent gradients — the
//! `LZR_OVERFIT=1` test couldn't even memorize one batch). Replacing the
//! concat-flatten with a per-position weight `w1[K,E,H]` and a batched matmul
//! summed over positions (no `[batch,K*E]` ever formed) fixes it: it now overfits
//! a batch to ~0 and trains normally. `w1[K,E,H]` is row-major-identical to
//! `[K*E,H]`, so the exported blob and the CPU forward (`pretrained.rs`) are
//! unchanged. Whether a frozen net on the word-dict'd stream earns its L(D) is the
//! open question `pretrained_lab` measures.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::module::{Module, Param};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor, TensorData};

const E: usize = 32; // byte embedding dim
const H: usize = 256; // hidden width
const V: usize = 256; // vocab (bytes)

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    emb: Param<Tensor<B, 2>>, // [V, E]
    w1: Param<Tensor<B, 3>>,  // [K, E, H] (per-position; row-major == [K*E, H])
    b1: Param<Tensor<B, 1>>,  // [H]
    w2: Param<Tensor<B, 2>>,  // [H, V]
    b2: Param<Tensor<B, 1>>,  // [V]
}

impl<B: Backend> Mlp<B> {
    fn new(k: usize, device: &B::Device) -> Self {
        let rnd2 = |a: usize, b: usize, std: f64| {
            Param::from_tensor(Tensor::random(
                [a, b],
                Distribution::Normal(0.0, std),
                device,
            ))
        };
        // Embeddings at unit scale; the per-position w1 [K,E,H] has effective
        // fan-in K*E (summed over positions).
        Self {
            emb: rnd2(V, E, 1.0),
            w1: Param::from_tensor(Tensor::random(
                [k, E, H],
                Distribution::Normal(0.0, (2.0 / (k * E) as f64).sqrt()),
                device,
            )),
            b1: Param::from_tensor(Tensor::zeros([H], device)),
            w2: rnd2(H, V, (2.0 / H as f64).sqrt()),
            b2: Param::from_tensor(Tensor::zeros([V], device)),
        }
    }

    /// `ctx`: `[batch, K]` byte indices → logits `[batch, V]`.
    fn forward(&self, ctx: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let [batch, k] = ctx.dims();
        let flat = ctx.reshape([batch * k]);
        // [batch,K,E] -> [K,batch,E] @ w1[K,E,H] (batched over positions) -> sum_K.
        // Avoids the [batch,K*E] flatten, whose backward is broken in burn 0.21.
        let g3 = self
            .emb
            .val()
            .select(0, flat)
            .reshape([batch, k, E])
            .swap_dims(0, 1); // [K, batch, E]
        let pre = g3
            .matmul(self.w1.val()) // [K, batch, H]
            .sum_dim(0)
            .reshape([batch, H])
            .add(self.b1.val().reshape([1, H]));
        let hid = pre.tanh();
        hid.matmul(self.w2.val()).add(self.b2.val().reshape([1, V]))
    }
}

fn vec2<B: Backend>(p: &Param<Tensor<B, 2>>) -> Vec<f32> {
    p.val().into_data().to_vec::<f32>().unwrap()
}
fn vec3<B: Backend>(p: &Param<Tensor<B, 3>>) -> Vec<f32> {
    p.val().into_data().to_vec::<f32>().unwrap()
}
fn vec1<B: Backend>(p: &Param<Tensor<B, 1>>) -> Vec<f32> {
    p.val().into_data().to_vec::<f32>().unwrap()
}

fn main() {
    type AB = Autodiff<Wgpu>;
    let device = WgpuDevice::default();
    let data_path = std::env::var("LZR_PP_DATA").unwrap_or_else(|_| "/tmp/lzr_pp.bin".into());
    let out_path = std::env::var("LZR_NET_OUT").unwrap_or_else(|_| "/tmp/lzr_net.bin".into());
    let steps: usize = std::env::var("LZR_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);
    let k: usize = std::env::var("LZR_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16); // context bytes
    let batch: usize = 512;
    let lr: f64 = std::env::var("LZR_LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-3);
    // Opt-in cosine lr decay to 0 over the run (LZR_COSINE): unlocks descent past
    // the fixed-lr plateau. Off by default so the trainer stays reproducible.
    let cosine = std::env::var("LZR_COSINE").is_ok();
    let data = std::fs::read(&data_path).expect("preprocessed training data");
    let span = data.len() - k - 1;
    println!(
        "pretrain MLP K={k} E={E} H={H} V={V} (~{} params) | data {} B | {steps} steps batch {batch}",
        V * E + k * E * H + H + H * V + V,
        data.len()
    );

    let mut model = Mlp::<AB>::new(k, &device);
    let mut optim = AdamConfig::new().init();
    let mut rng = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (rng >> 33) as usize
    };

    // One-shot diagnostic: does the gathered context `g` actually vary with the
    // input? (≈0 batch-variance ⇒ the embedding gather is input-independent.)
    {
        let mut cids: Vec<i32> = Vec::new();
        for _ in 0..batch {
            let p = k + (next() % span);
            for j in 0..k {
                cids.push(i32::from(data[p - k + j]));
            }
        }
        let c = Tensor::<AB, 2, Int>::from_data(TensorData::new(cids, [batch, k]), &device);
        let flat = c.clone().reshape([batch * k]);
        let g = model.emb.val().select(0, flat).reshape([batch, k * E]);
        let var = g.var(0).mean().into_data().to_vec::<f32>().unwrap()[0];
        let cvar = c
            .reshape([batch * k])
            .float()
            .var(0)
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        println!("DIAG: g batch-variance {var:.5} | ctx-index variance {cvar:.1}");
    }

    // `LZR_OVERFIT=1`: train on ONE fixed batch every step. A working net must
    // drive the loss toward 0 (memorize 512 samples); if it sticks at the unigram,
    // the forward/backward is broken, not the data/lr.
    let overfit = std::env::var("LZR_OVERFIT").is_ok();
    let fixed = if overfit {
        let mut ci: Vec<i32> = Vec::with_capacity(batch * k);
        let mut ti: Vec<i32> = Vec::with_capacity(batch);
        for _ in 0..batch {
            let p = k + (next() % span);
            for j in 0..k {
                ci.push(i32::from(data[p - k + j]));
            }
            ti.push(i32::from(data[p]));
        }
        Some((
            Tensor::<AB, 2, Int>::from_data(TensorData::new(ci, [batch, k]), &device),
            Tensor::<AB, 1, Int>::from_data(TensorData::new(ti, [batch]), &device),
        ))
    } else {
        None
    };

    for step in 0..steps {
        let (ctx, tgt) = if let Some((c, t)) = &fixed {
            (c.clone(), t.clone())
        } else {
            let mut ctx_ids: Vec<i32> = Vec::with_capacity(batch * k);
            let mut tgt_ids: Vec<i32> = Vec::with_capacity(batch);
            for _ in 0..batch {
                let p = k + (next() % span);
                for j in 0..k {
                    ctx_ids.push(i32::from(data[p - k + j]));
                }
                tgt_ids.push(i32::from(data[p]));
            }
            (
                Tensor::<AB, 2, Int>::from_data(TensorData::new(ctx_ids, [batch, k]), &device),
                Tensor::<AB, 1, Int>::from_data(TensorData::new(tgt_ids, [batch]), &device),
            )
        };
        let logits = model.forward(ctx);
        let loss = CrossEntropyLossConfig::new()
            .init(&device)
            .forward(logits, tgt);
        if step % 2000 == 0 {
            let nats = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            let bpb = f64::from(nats) / std::f64::consts::LN_2;
            // L1 norms — which params are actually moving from their init?
            let n2 = |p: &Param<Tensor<AB, 2>>| {
                p.val().abs().sum().into_data().to_vec::<f32>().unwrap()[0]
            };
            let n3 = |p: &Param<Tensor<AB, 3>>| {
                p.val().abs().sum().into_data().to_vec::<f32>().unwrap()[0]
            };
            let n1 = |p: &Param<Tensor<AB, 1>>| {
                p.val().abs().sum().into_data().to_vec::<f32>().unwrap()[0]
            };
            println!(
                "step {step:7} loss {nats:.4} ({bpb:.4} bpb) | |emb| {:.0} |w1| {:.0} |w2| {:.0} |b1| {:.1} |b2| {:.1}",
                n2(&model.emb),
                n3(&model.w1),
                n2(&model.w2),
                n1(&model.b1),
                n1(&model.b2)
            );
        }
        let grads = loss.backward();
        let gp = GradientsParams::from_grads(grads, &model);
        let lr_t = if cosine {
            lr * 0.5 * (1.0 + (std::f64::consts::PI * step as f64 / steps as f64).cos())
        } else {
            lr
        };
        model = optim.step(lr_t, model, gp);
    }

    let mut blob = Vec::new();
    for d in [k as u32, E as u32, H as u32, V as u32] {
        blob.extend_from_slice(&d.to_le_bytes());
    }
    for x in vec2(&model.emb)
        .into_iter()
        .chain(vec3(&model.w1))
        .chain(vec1(&model.b1))
        .chain(vec2(&model.w2))
        .chain(vec1(&model.b2))
    {
        blob.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(&out_path, &blob).unwrap();
    println!("wrote {} weight bytes to {out_path}", blob.len());
}
