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
//! STATUS (2026-06-24): the pipeline is end-to-end (compiles, trains on the GPU,
//! the embedding gather is verified input-dependent, all params update), but this
//! MLP currently **collapses to the unigram** — on casefold data it converges to
//! 3.40 nats vs the 3.4278-nat unigram, i.e. the context path contributes ~nothing
//! despite training. A "collapse to the marginal" pathology that resisted init /
//! activation / lr fixes; needs careful debugging (overfit-a-batch, grad checks).
//! Independently, a *frozen* simple net on the word-dict'd stream is expected to be
//! low-value (the dict removes the learnable word-structure; the deterministic
//! stack + online arm already cover it), so this is kept as a scaffold, not shipped.

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::module::{Module, Param};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor, TensorData};

const K: usize = 16; // context bytes
const E: usize = 32; // byte embedding dim
const H: usize = 256; // hidden width
const V: usize = 256; // vocab (bytes)

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    emb: Param<Tensor<B, 2>>, // [V, E]
    w1: Param<Tensor<B, 2>>,  // [K*E, H]
    b1: Param<Tensor<B, 1>>,  // [H]
    w2: Param<Tensor<B, 2>>,  // [H, V]
    b2: Param<Tensor<B, 1>>,  // [V]
}

impl<B: Backend> Mlp<B> {
    fn new(device: &B::Device) -> Self {
        let rnd2 = |a: usize, b: usize, std: f64| {
            Param::from_tensor(Tensor::random(
                [a, b],
                Distribution::Normal(0.0, std),
                device,
            ))
        };
        // Embeddings at unit scale (std 0.1 left the hidden activations tiny, so
        // the context path could not drive the logits and only the unigram bias
        // was learned); He init for the relu layers.
        Self {
            emb: rnd2(V, E, 1.0),
            w1: rnd2(K * E, H, (2.0 / (K * E) as f64).sqrt()),
            b1: Param::from_tensor(Tensor::zeros([H], device)),
            w2: rnd2(H, V, (2.0 / H as f64).sqrt()),
            b2: Param::from_tensor(Tensor::zeros([V], device)),
        }
    }

    /// `ctx`: `[batch, K]` byte indices → logits `[batch, V]`.
    fn forward(&self, ctx: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let [batch, k] = ctx.dims();
        let flat = ctx.reshape([batch * k]);
        let g = self.emb.val().select(0, flat).reshape([batch, k * E]);
        let pre = g.matmul(self.w1.val()).add(self.b1.val().reshape([1, H]));
        let hid = pre.tanh();
        hid.matmul(self.w2.val()).add(self.b2.val().reshape([1, V]))
    }
}

fn vec2<B: Backend>(p: &Param<Tensor<B, 2>>) -> Vec<f32> {
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
    let batch: usize = 512;
    let lr: f64 = std::env::var("LZR_LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-3);
    let data = std::fs::read(&data_path).expect("preprocessed training data");
    let span = data.len() - K - 1;
    println!(
        "pretrain MLP K={K} E={E} H={H} V={V} (~{} params) | data {} B | {steps} steps batch {batch}",
        V * E + K * E * H + H + H * V + V,
        data.len()
    );

    let mut model = Mlp::<AB>::new(&device);
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
            let p = K + (next() % span);
            for j in 0..K {
                cids.push(i32::from(data[p - K + j]));
            }
        }
        let c = Tensor::<AB, 2, Int>::from_data(TensorData::new(cids, [batch, K]), &device);
        let flat = c.clone().reshape([batch * K]);
        let g = model.emb.val().select(0, flat).reshape([batch, K * E]);
        let var = g.var(0).mean().into_data().to_vec::<f32>().unwrap()[0];
        let cvar = c
            .reshape([batch * K])
            .float()
            .var(0)
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0];
        println!("DIAG: g batch-variance {var:.5} | ctx-index variance {cvar:.1}");
    }

    let t0 = std::time::Instant::now();
    for step in 0..steps {
        let mut ctx_ids: Vec<i32> = Vec::with_capacity(batch * K);
        let mut tgt_ids: Vec<i32> = Vec::with_capacity(batch);
        for _ in 0..batch {
            let p = K + (next() % span);
            for j in 0..K {
                ctx_ids.push(i32::from(data[p - K + j]));
            }
            tgt_ids.push(i32::from(data[p]));
        }
        let ctx = Tensor::<AB, 2, Int>::from_data(TensorData::new(ctx_ids, [batch, K]), &device);
        let tgt = Tensor::<AB, 1, Int>::from_data(TensorData::new(tgt_ids, [batch]), &device);
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
            let n1 = |p: &Param<Tensor<AB, 1>>| {
                p.val().abs().sum().into_data().to_vec::<f32>().unwrap()[0]
            };
            println!(
                "step {step:7} loss {nats:.4} ({bpb:.4} bpb) | |emb| {:.0} |w1| {:.0} |w2| {:.0} |b1| {:.1} |b2| {:.1}",
                n2(&model.emb),
                n2(&model.w1),
                n2(&model.w2),
                n1(&model.b1),
                n1(&model.b2)
            );
        }
        let grads = loss.backward();
        let gp = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, gp);
    }

    let mut blob = Vec::new();
    for d in [K as u32, E as u32, H as u32, V as u32] {
        blob.extend_from_slice(&d.to_le_bytes());
    }
    for x in vec2(&model.emb)
        .into_iter()
        .chain(vec2(&model.w1))
        .chain(vec1(&model.b1))
        .chain(vec2(&model.w2))
        .chain(vec1(&model.b2))
    {
        blob.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(&out_path, &blob).unwrap();
    println!("wrote {} weight bytes to {out_path}", blob.len());
}
