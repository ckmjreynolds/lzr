//! Stage-0 backbone bake-off (the `neural`/burn GPU kill-gate for the
//! bpb-per-FLOP moonshot — see the approved plan).
//!
//! Question it answers: at a *fixed* MACs/byte budget, does a diagonal-SSM or
//! RWKV-style backbone extract more bits than the LSTM arm, in the codec's actual
//! regime — **cold-start, online, single-pass, stateful truncated-BPTT**?
//!
//! It does NOT write any CPU backward. burn autodiff provides the backward; we
//! only measure standalone online single-pass bpb per architecture. The regime is
//! faithful to the shipped arm (one Adam step per BPTT window, state carried and
//! detached across windows, each token seen exactly once), but batched over `B`
//! contiguous shards of the stream for GPU throughput — which makes the *absolute*
//! bpb a touch optimistic vs the true B=1 codec, but the **relative** ranking
//! (the greenlight bar) is what we read, and that is batch-invariant.
//!
//! Data: the post-pipeline byte stream the codec's models see (dump it with the
//! in-crate `dump_preprocessed` test → `LZR_PP_DATA`). bpb is per post-pipeline
//! byte (a fixed denominator across archs — the comparison is relative).
//!
//! Run (compare LSTM vs SSM on a 20 MB post-pipeline slice):
//!   LZR_PP_OUT=/tmp/lzr_pp.bin cargo test --release dump_preprocessed -- --ignored --nocapture
//!   LZR_PP_DATA=/tmp/lzr_pp.bin LZR_BYTES=20000000 \
//!     cargo run --release --example bakeoff --features neural

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::many_single_char_names
)]

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::module::{AutodiffModule, Module, Param};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{sigmoid, silu, tanh};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor, TensorData};
use std::time::Instant;

type AB = Autodiff<Wgpu>;

const V: usize = 256; // byte vocab

// ───────────────────────── backbone trait ─────────────────────────

/// A causal sequence backbone for the bake-off. Owns its embedding and output
/// projection; `forward_chunk` runs one BPTT window (carrying recurrent state)
/// and returns per-position logits `[batch, w, V]`. State is a `Vec` of 2-D
/// tensors so the training loop is generic over the architecture (LSTM carries
/// `[h, c]`, the SSM carries one state tensor per layer).
trait Backbone<B: Backend>: Module<B> {
    fn init_state(&self, batch: usize, device: &B::Device) -> Vec<Tensor<B, 2>>;
    fn forward_chunk(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: Vec<Tensor<B, 2>>,
    ) -> (Tensor<B, 3>, Vec<Tensor<B, 2>>);
    /// Forward MACs per token (the FLOP-budget control knob; backward ≈ 2× for
    /// all archs, so matching forward MACs matches the online compute budget).
    fn macs_per_token(&self) -> u64;
    fn n_params(&self) -> u64;
    fn label(&self) -> String;
}

fn rnd<B: Backend>(shape: [usize; 2], std: f64, device: &B::Device) -> Param<Tensor<B, 2>> {
    Param::from_tensor(Tensor::random(
        shape,
        Distribution::Normal(0.0, std),
        device,
    ))
}
fn zeros1<B: Backend>(n: usize, device: &B::Device) -> Param<Tensor<B, 1>> {
    Param::from_tensor(Tensor::zeros([n], device))
}

// ───────────────────────── LSTM baseline ─────────────────────────

/// Single-layer LSTM reproducing the shipped arm (gate layout `[i|f|g|o]`).
#[derive(Module, Debug)]
struct LstmNet<B: Backend> {
    emb: Param<Tensor<B, 2>>,  // [V, E]
    wx: Param<Tensor<B, 2>>,   // [E, 4H]
    wh: Param<Tensor<B, 2>>,   // [H, 4H]
    bias: Param<Tensor<B, 1>>, // [4H]
    wo: Param<Tensor<B, 2>>,   // [H, V]
    bo: Param<Tensor<B, 1>>,   // [V]
}

impl<B: Backend> LstmNet<B> {
    fn new(e: usize, h: usize, device: &B::Device) -> Self {
        Self {
            emb: rnd([V, e], 1.0, device),
            wx: rnd([e, 4 * h], (1.0 / e as f64).sqrt(), device),
            wh: rnd([h, 4 * h], (1.0 / h as f64).sqrt(), device),
            bias: zeros1(4 * h, device),
            wo: rnd([h, V], (1.0 / h as f64).sqrt(), device),
            bo: zeros1(V, device),
        }
    }
    fn dims(&self) -> (usize, usize) {
        let e = self.emb.val().dims()[1];
        let h = self.wh.val().dims()[0];
        (e, h)
    }
}

impl<B: Backend> Backbone<B> for LstmNet<B> {
    fn init_state(&self, batch: usize, device: &B::Device) -> Vec<Tensor<B, 2>> {
        let (_, h) = self.dims();
        vec![
            Tensor::zeros([batch, h], device),
            Tensor::zeros([batch, h], device),
        ]
    }

    fn forward_chunk(
        &self,
        tokens: Tensor<B, 2, Int>,
        mut state: Vec<Tensor<B, 2>>,
    ) -> (Tensor<B, 3>, Vec<Tensor<B, 2>>) {
        let [batch, w] = tokens.dims();
        let (e, h) = self.dims();
        // Embed the whole chunk and pre-project the input gates once.
        let emb = self
            .emb
            .val()
            .select(0, tokens.reshape([batch * w]))
            .reshape([batch, w, e]); // [B,W,E]
        let xg = emb
            .reshape([batch * w, e])
            .matmul(self.wx.val())
            .reshape([batch, w, 4 * h]); // [B,W,4H]

        let mut c = state.pop().unwrap();
        let mut hid = state.pop().unwrap();
        let mut outs: Vec<Tensor<B, 2>> = Vec::with_capacity(w);
        for t in 0..w {
            let xg_t = xg
                .clone()
                .slice([0..batch, t..t + 1, 0..4 * h])
                .reshape([batch, 4 * h]);
            let a = xg_t
                .add(hid.clone().matmul(self.wh.val()))
                .add(self.bias.val().reshape([1, 4 * h]));
            let i = sigmoid(a.clone().slice([0..batch, 0..h]));
            let f = sigmoid(a.clone().slice([0..batch, h..2 * h]));
            let g = tanh(a.clone().slice([0..batch, 2 * h..3 * h]));
            let o = sigmoid(a.slice([0..batch, 3 * h..4 * h]));
            c = f.mul(c).add(i.mul(g));
            hid = o.mul(tanh(c.clone()));
            outs.push(hid.clone());
        }
        let hseq = Tensor::stack::<3>(outs, 1); // [B,W,H]
        let logits = hseq
            .reshape([batch * w, h])
            .matmul(self.wo.val())
            .add(self.bo.val().reshape([1, V]))
            .reshape([batch, w, V]);
        (logits, vec![hid, c])
    }

    fn macs_per_token(&self) -> u64 {
        let (e, h) = self.dims();
        (4 * h * e + 4 * h * h + V * h) as u64
    }
    fn n_params(&self) -> u64 {
        let (e, h) = self.dims();
        (V * e + e * 4 * h + h * 4 * h + 4 * h + h * V + V) as u64
    }
    fn label(&self) -> String {
        let (e, h) = self.dims();
        format!("LSTM  e={e} h={h}")
    }
}

// ───────────────────────── diagonal SSM (LRU-style) ─────────────────────────

/// One diagonal-SSM layer: project input, run a per-channel element-wise leaky
/// recurrence `s = a⊙s + (1-a)⊙u` (a = sigmoid(decay), the cheap CPU-friendly
/// scan), then a `D×D` channel-mix + SiLU + residual. FLOPs live in the
/// projections/mix (information-carrying), not a dense recurrent matrix.
#[derive(Module, Debug)]
struct SsmLayer<B: Backend> {
    win: Param<Tensor<B, 2>>,   // [D, D]  input projection
    decay: Param<Tensor<B, 1>>, // [D]     recurrence decay logit
    wmix: Param<Tensor<B, 2>>,  // [D, D]  channel mix
    bmix: Param<Tensor<B, 1>>,  // [D]
}

impl<B: Backend> SsmLayer<B> {
    fn new(d: usize, device: &B::Device) -> Self {
        Self {
            win: rnd([d, d], (1.0 / d as f64).sqrt(), device),
            // init decay logit ≈ 2.0 → a ≈ 0.88 (retain memory at start).
            decay: Param::from_tensor(Tensor::full([d], 2.0, device)),
            wmix: rnd([d, d], (1.0 / d as f64).sqrt(), device),
            bmix: zeros1(d, device),
        }
    }
    /// Run the layer over a chunk `[B,W,D]`, carrying state `[B,D]`.
    fn run(&self, x: Tensor<B, 3>, mut s: Tensor<B, 2>) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let [batch, w, d] = x.dims();
        let u = x
            .clone()
            .reshape([batch * w, d])
            .matmul(self.win.val())
            .reshape([batch, w, d]); // [B,W,D]
        let a = sigmoid(self.decay.val().reshape([1, d])); // [1,D]
        let one_minus_a = a.clone().mul_scalar(-1.0).add_scalar(1.0);
        let mut outs: Vec<Tensor<B, 2>> = Vec::with_capacity(w);
        for t in 0..w {
            let u_t = u
                .clone()
                .slice([0..batch, t..t + 1, 0..d])
                .reshape([batch, d]);
            s = a.clone().mul(s).add(one_minus_a.clone().mul(u_t));
            outs.push(s.clone());
        }
        let sseq = Tensor::stack::<3>(outs, 1); // [B,W,D]
        let mixed = sseq
            .reshape([batch * w, d])
            .matmul(self.wmix.val())
            .add(self.bmix.val().reshape([1, d]))
            .reshape([batch, w, d]);
        let out = x.add(silu(mixed)); // residual
        (out, s)
    }
}

/// Stacked diagonal-SSM next-byte model.
#[derive(Module, Debug)]
struct SsmNet<B: Backend> {
    emb: Param<Tensor<B, 2>>, // [V, D]
    layers: Vec<SsmLayer<B>>,
    wo: Param<Tensor<B, 2>>, // [D, V]
    bo: Param<Tensor<B, 1>>, // [V]
}

impl<B: Backend> SsmNet<B> {
    fn new(d: usize, n_layers: usize, device: &B::Device) -> Self {
        Self {
            emb: rnd([V, d], 1.0, device),
            layers: (0..n_layers).map(|_| SsmLayer::new(d, device)).collect(),
            wo: rnd([d, V], (1.0 / d as f64).sqrt(), device),
            bo: zeros1(V, device),
        }
    }
    fn d(&self) -> usize {
        self.emb.val().dims()[1]
    }
}

impl<B: Backend> Backbone<B> for SsmNet<B> {
    fn init_state(&self, batch: usize, device: &B::Device) -> Vec<Tensor<B, 2>> {
        let d = self.d();
        self.layers
            .iter()
            .map(|_| Tensor::zeros([batch, d], device))
            .collect()
    }

    fn forward_chunk(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: Vec<Tensor<B, 2>>,
    ) -> (Tensor<B, 3>, Vec<Tensor<B, 2>>) {
        let [batch, w] = tokens.dims();
        let d = self.d();
        let mut x = self
            .emb
            .val()
            .select(0, tokens.reshape([batch * w]))
            .reshape([batch, w, d]); // [B,W,D]
        let mut new_state = Vec::with_capacity(self.layers.len());
        for (layer, s) in self.layers.iter().zip(state) {
            let (out, s2) = layer.run(x, s);
            x = out;
            new_state.push(s2);
        }
        let logits = x
            .reshape([batch * w, d])
            .matmul(self.wo.val())
            .add(self.bo.val().reshape([1, V]))
            .reshape([batch, w, V]);
        (logits, new_state)
    }

    fn macs_per_token(&self) -> u64 {
        let d = self.d();
        let per_layer = (2 * d * d) as u64; // win + wmix (recurrence is element-wise, ~free)
        per_layer * self.layers.len() as u64 + (V * d) as u64
    }
    fn n_params(&self) -> u64 {
        let d = self.d();
        let per_layer = (d * d + d + d * d + d) as u64;
        (V * d) as u64 + per_layer * self.layers.len() as u64 + (d * V + V) as u64
    }
    fn label(&self) -> String {
        format!("SSM   d={} layers={}", self.d(), self.layers.len())
    }
}

// ───────────────────────── selective SSM (Mamba-lite) ─────────────────────────

/// One *selective* diagonal-SSM layer: the decay `a_t`, input `u_t`, and output
/// gate `g_t` are all input-dependent (the Mamba innovation), so the recurrence
/// `s = a_t⊙s + (1-a_t)⊙u_t`, `y = g_t⊙s` can route/forget per token — the
/// mechanism the bpb-per-FLOP thesis rests on. Still an element-wise scan
/// (CPU-friendly, clean reverse-scan backward).
#[derive(Module, Debug)]
struct SelLayer<B: Backend> {
    win: Param<Tensor<B, 2>>,  // [D,D]
    wa: Param<Tensor<B, 2>>,   // [D,D]  decay logits
    ba: Param<Tensor<B, 1>>,   // [D]
    wg: Param<Tensor<B, 2>>,   // [D,D]  output gate
    wmix: Param<Tensor<B, 2>>, // [D,D]
    bmix: Param<Tensor<B, 1>>, // [D]
}

impl<B: Backend> SelLayer<B> {
    fn new(d: usize, device: &B::Device) -> Self {
        let s = (1.0 / d as f64).sqrt();
        Self {
            win: rnd([d, d], s, device),
            wa: rnd([d, d], s, device),
            ba: Param::from_tensor(Tensor::full([d], 2.0, device)), // a≈0.88 at init
            wg: rnd([d, d], s, device),
            wmix: rnd([d, d], s, device),
            bmix: zeros1(d, device),
        }
    }
    fn run(&self, x: Tensor<B, 3>, mut s: Tensor<B, 2>) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let [batch, w, d] = x.dims();
        let flat = x.clone().reshape([batch * w, d]);
        let u = flat.clone().matmul(self.win.val()).reshape([batch, w, d]);
        let a = sigmoid(
            flat.clone()
                .matmul(self.wa.val())
                .add(self.ba.val().reshape([1, d])),
        )
        .reshape([batch, w, d]);
        let g = sigmoid(flat.matmul(self.wg.val())).reshape([batch, w, d]);
        let mut outs: Vec<Tensor<B, 2>> = Vec::with_capacity(w);
        for t in 0..w {
            let a_t = a
                .clone()
                .slice([0..batch, t..t + 1, 0..d])
                .reshape([batch, d]);
            let u_t = u
                .clone()
                .slice([0..batch, t..t + 1, 0..d])
                .reshape([batch, d]);
            let g_t = g
                .clone()
                .slice([0..batch, t..t + 1, 0..d])
                .reshape([batch, d]);
            let keep = a_t.clone().mul_scalar(-1.0).add_scalar(1.0);
            s = a_t.mul(s).add(keep.mul(u_t));
            outs.push(g_t.mul(s.clone()));
        }
        let yseq = Tensor::stack::<3>(outs, 1); // [B,W,D]
        let mixed = yseq
            .reshape([batch * w, d])
            .matmul(self.wmix.val())
            .add(self.bmix.val().reshape([1, d]))
            .reshape([batch, w, d]);
        (x.add(mixed), s) // residual
    }
}

#[derive(Module, Debug)]
struct SelNet<B: Backend> {
    emb: Param<Tensor<B, 2>>, // [V,D]
    layers: Vec<SelLayer<B>>,
    wo: Param<Tensor<B, 2>>, // [D,V]
    bo: Param<Tensor<B, 1>>, // [V]
}

impl<B: Backend> SelNet<B> {
    fn new(d: usize, n_layers: usize, vocab: usize, device: &B::Device) -> Self {
        // Embedding init std. Default 1.0 (what the byte-level greenlight used);
        // LZR_EMBSTD lowers it — the unnormalized SEL residual stream blows up at
        // larger vocab when rare-token embeddings sit at the std-1.0 init scale.
        let emb_std = std::env::var("LZR_EMBSTD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        Self {
            emb: rnd([vocab, d], emb_std, device),
            layers: (0..n_layers).map(|_| SelLayer::new(d, device)).collect(),
            wo: rnd([d, vocab], (1.0 / d as f64).sqrt(), device),
            bo: zeros1(vocab, device),
        }
    }
    fn d(&self) -> usize {
        self.emb.val().dims()[1]
    }
    fn vocab(&self) -> usize {
        self.emb.val().dims()[0]
    }
}

impl<B: Backend> Backbone<B> for SelNet<B> {
    fn init_state(&self, batch: usize, device: &B::Device) -> Vec<Tensor<B, 2>> {
        let d = self.d();
        self.layers
            .iter()
            .map(|_| Tensor::zeros([batch, d], device))
            .collect()
    }
    fn forward_chunk(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: Vec<Tensor<B, 2>>,
    ) -> (Tensor<B, 3>, Vec<Tensor<B, 2>>) {
        let [batch, w] = tokens.dims();
        let d = self.d();
        let mut x = self
            .emb
            .val()
            .select(0, tokens.reshape([batch * w]))
            .reshape([batch, w, d]);
        let mut new_state = Vec::with_capacity(self.layers.len());
        for (layer, s) in self.layers.iter().zip(state) {
            let (out, s2) = layer.run(x, s);
            x = out;
            new_state.push(s2);
        }
        let vocab = self.vocab();
        let logits = x
            .reshape([batch * w, d])
            .matmul(self.wo.val())
            .add(self.bo.val().reshape([1, vocab]))
            .reshape([batch, w, vocab]);
        (logits, new_state)
    }
    fn macs_per_token(&self) -> u64 {
        let (d, v) = (self.d(), self.vocab());
        (4 * d * d) as u64 * self.layers.len() as u64 + (v * d) as u64 // win+wa+wg+wmix
    }
    fn n_params(&self) -> u64 {
        let (d, v) = (self.d(), self.vocab());
        let per = (4 * d * d + 2 * d) as u64;
        (v * d) as u64 + per * self.layers.len() as u64 + (d * v + v) as u64
    }
    fn label(&self) -> String {
        format!(
            "SEL   d={} layers={} vocab={}",
            self.d(),
            self.layers.len(),
            self.vocab()
        )
    }
}

// ───────────────────────── true Mamba (state dim N) ─────────────────────────

/// One Mamba-style selective-SSM layer with a real-diagonal state of dim `N` per
/// channel (the capacity mechanism the N=1 `SelLayer` lacks). Input-dependent
/// `Δ`, `B`, `C` (selective); `A` a learned negative-diagonal `[D,N]` parameter.
/// Scan: `s[B,D,N] = exp(Δ⊙A)⊙s + (Δ⊙B)⊙u`, `y[B,D] = Σ_N C⊙s`, SiLU-gated,
/// residual. Still an element-wise scan over time (CPU-friendly reverse scan).
#[derive(Module, Debug)]
struct MambaLayer<B: Backend> {
    norm: Param<Tensor<B, 1>>,  // [D]  pre-norm RMSNorm scale
    win: Param<Tensor<B, 2>>,   // [D,D]  input u
    wdt: Param<Tensor<B, 2>>,   // [D,D]  Δ (timestep) logits
    bdt: Param<Tensor<B, 1>>,   // [D]
    wb: Param<Tensor<B, 2>>,    // [D,N]  selective B
    wc: Param<Tensor<B, 2>>,    // [D,N]  selective C
    a_log: Param<Tensor<B, 2>>, // [D,N]  A = -exp(a_log)
    wz: Param<Tensor<B, 2>>,    // [D,D]  SiLU gate
    wmix: Param<Tensor<B, 2>>,  // [D,D]
    bmix: Param<Tensor<B, 1>>,  // [D]
}

impl<B: Backend> MambaLayer<B> {
    fn new(d: usize, n: usize, device: &B::Device) -> Self {
        let s = (1.0 / d as f64).sqrt();
        Self {
            norm: Param::from_tensor(Tensor::ones([d], device)),
            win: rnd([d, d], s, device),
            wdt: rnd([d, d], s, device),
            // dt_bias init: softplus(bias) sets initial Δ. Standard Mamba uses very
            // negative (slow warm); the single-pass online regime may want larger.
            // LZR_DTBIAS sweeps it.
            bdt: Param::from_tensor(Tensor::full(
                [d],
                std::env::var("LZR_DTBIAS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(-4.0_f32),
                device,
            )),
            wb: rnd([d, n], s, device),
            wc: rnd([d, n], s, device),
            // A = -exp(a_log); init a_log small so A ∈ ~[-2,-0.5].
            a_log: Param::from_tensor(Tensor::random(
                [d, n],
                Distribution::Normal(0.0, 0.3),
                device,
            )),
            wz: rnd([d, d], s, device),
            wmix: rnd([d, d], s, device),
            bmix: zeros1(d, device),
        }
    }
    fn run(&self, x: Tensor<B, 3>, mut s: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        use burn::tensor::activation::softplus;
        let [batch, w, d] = x.dims();
        let n = self.wb.val().dims()[1];
        // Pre-norm RMSNorm: xn = x / rms(x) * norm_scale.
        let ms = x.clone().powf_scalar(2.0).mean_dim(2); // [B,W,1]
        let xn = x
            .clone()
            .div(ms.add_scalar(1e-5).sqrt())
            .mul(self.norm.val().reshape([1, 1, d]));
        let flat = xn.reshape([batch * w, d]);
        let u = flat
            .clone()
            .matmul(self.win.val())
            .reshape([batch, w, d, 1]); // [B,W,D,1]
        let dt = softplus(
            flat.clone()
                .matmul(self.wdt.val())
                .add(self.bdt.val().reshape([1, d])),
            1.0,
        )
        .reshape([batch, w, d, 1]); // [B,W,D,1]
        let bb = flat.clone().matmul(self.wb.val()).reshape([batch, w, 1, n]); // [B,W,1,N]
        let cc = flat.clone().matmul(self.wc.val()).reshape([batch, w, 1, n]); // [B,W,1,N]
        let z = silu(flat.matmul(self.wz.val())).reshape([batch, w, d]); // gate
        // A = -exp(a_log)  [1,1,D,N]
        let a = self
            .a_log
            .val()
            .exp()
            .mul_scalar(-1.0)
            .reshape([1, 1, d, n]);
        let da = dt.clone().mul(a).exp(); // exp(Δ⊙A) [B,W,D,N]
        let dbu = dt.mul(bb).mul(u); // (Δ⊙B)⊙u [B,W,D,N]
        let mut outs: Vec<Tensor<B, 2>> = Vec::with_capacity(w);
        for t in 0..w {
            let da_t = da
                .clone()
                .slice([0..batch, t..t + 1, 0..d, 0..n])
                .reshape([batch, d, n]);
            let dbu_t = dbu
                .clone()
                .slice([0..batch, t..t + 1, 0..d, 0..n])
                .reshape([batch, d, n]);
            s = da_t.mul(s).add(dbu_t); // [B,D,N]
            let c_t = cc
                .clone()
                .slice([0..batch, t..t + 1, 0..1, 0..n])
                .reshape([batch, 1, n]);
            let y_t = c_t.mul(s.clone()).sum_dim(2).reshape([batch, d]); // Σ_N
            outs.push(y_t);
        }
        let yseq = Tensor::stack::<3>(outs, 1).mul(z); // [B,W,D] gated
        let mixed = yseq
            .reshape([batch * w, d])
            .matmul(self.wmix.val())
            .add(self.bmix.val().reshape([1, d]))
            .reshape([batch, w, d]);
        (x.add(mixed), s)
    }
}

#[derive(Module, Debug)]
struct MambaNet<B: Backend> {
    emb: Param<Tensor<B, 2>>, // [V,D]
    layers: Vec<MambaLayer<B>>,
    wo: Param<Tensor<B, 2>>, // [D,V]
    bo: Param<Tensor<B, 1>>, // [V]
    n: usize,
}

impl<B: Backend> MambaNet<B> {
    fn new(d: usize, n: usize, n_layers: usize, device: &B::Device) -> Self {
        Self {
            emb: rnd([V, d], 1.0, device),
            layers: (0..n_layers)
                .map(|_| MambaLayer::new(d, n, device))
                .collect(),
            wo: rnd([d, V], (1.0 / d as f64).sqrt(), device),
            bo: zeros1(V, device),
            n,
        }
    }
    fn d(&self) -> usize {
        self.emb.val().dims()[1]
    }
}

impl<B: Backend> Backbone<B> for MambaNet<B> {
    fn init_state(&self, batch: usize, device: &B::Device) -> Vec<Tensor<B, 2>> {
        // State is [B,D,N]; we carry it flattened to [B, D*N] in the generic Vec
        // and reshape inside forward_chunk.
        let d = self.d();
        self.layers
            .iter()
            .map(|_| Tensor::zeros([batch, d * self.n], device))
            .collect()
    }
    fn forward_chunk(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: Vec<Tensor<B, 2>>,
    ) -> (Tensor<B, 3>, Vec<Tensor<B, 2>>) {
        let [batch, w] = tokens.dims();
        let d = self.d();
        let n = self.n;
        let mut x = self
            .emb
            .val()
            .select(0, tokens.reshape([batch * w]))
            .reshape([batch, w, d]);
        let mut new_state = Vec::with_capacity(self.layers.len());
        for (layer, s) in self.layers.iter().zip(state) {
            let s3 = s.reshape([batch, d, n]);
            let (out, s2) = layer.run(x, s3);
            x = out;
            new_state.push(s2.reshape([batch, d * n]));
        }
        let logits = x
            .reshape([batch * w, d])
            .matmul(self.wo.val())
            .add(self.bo.val().reshape([1, V]))
            .reshape([batch, w, V]);
        (logits, new_state)
    }
    fn macs_per_token(&self) -> u64 {
        let d = self.d();
        let n = self.n;
        // win + wdt + wz + wmix (4 D²) + wb + wc (2 D·N) + C·s readout (D·N)
        ((4 * d * d + 3 * d * n) * self.layers.len() + V * d) as u64
    }
    fn n_params(&self) -> u64 {
        let d = self.d();
        let n = self.n;
        let per = 4 * d * d + 3 * d + 3 * d * n; // win,wdt,wz,wmix + norm,bdt,bmix + wb,wc,a_log
        ((V * d) + per * self.layers.len() + d * V + V) as u64
    }
    fn label(&self) -> String {
        format!(
            "MAMBA d={} n={} layers={}",
            self.d(),
            self.n,
            self.layers.len()
        )
    }
}

// ───────────────────────── training loop ─────────────────────────

struct Cfg {
    bytes: usize,  // post-pipeline bytes to stream
    batch: usize,  // parallel shards
    window: usize, // truncated-BPTT window
    lr: f64,
}

/// Cold-start online single-pass over `data`, batched into `cfg.batch`
/// contiguous shards. Returns bpb per post-pipeline byte. One Adam step per
/// window; state detached across windows.
fn run<M: Backbone<AB> + AutodiffModule<AB>>(
    mut model: M,
    data: &[u8],
    cfg: &Cfg,
    device: &WgpuDevice,
) -> (f64, f64) {
    let n = cfg.bytes.min(data.len());
    let shard = n / cfg.batch;
    let label = model.label();
    let macs = model.macs_per_token();
    let params = model.n_params();
    println!(
        "  [{label}] params {params} | {:.0} K MACs/byte | shard {shard} B × {} | win {}",
        macs as f64 / 1000.0,
        cfg.batch,
        cfg.window
    );

    // Gradient-norm clipping stabilizes the unnormalized SEL at larger vocab/width
    // (the byte path was marginally stable; the sharper 1024-way softmax diverges
    // without it). Applied to both run() and run_bpe() so comparisons stay fair.
    let mut optim = AdamConfig::new()
        .init()
        .with_grad_clipping(GradientClippingConfig::Norm(1.0).init());
    let mut state = model.init_state(cfg.batch, device);
    let mut total_nats = 0.0f64;
    let mut total_positions = 0u64;
    let w = cfg.window;
    // Warmed-regime ("tail") bpb over the final quartile — decouples the cold-start
    // the cumulative bpb folds in. The bpb-per-FLOP thesis predicts the SSM/RWKV
    // edge GROWS as the recurrence warms, so the tail is the more telling number.
    let total_steps = (shard.saturating_sub(w + 1)) / w;
    let tail_start = total_steps * 3 / 4;
    let mut tail_nats = 0.0f64;
    let mut tail_positions = 0u64;
    let t0 = Instant::now();
    let mut s = 0usize;
    let mut step = 0usize;
    while s + w + 1 <= shard {
        // Gather [batch, w] inputs and targets from the shards.
        let mut inp: Vec<i32> = Vec::with_capacity(cfg.batch * w);
        let mut tgt: Vec<i32> = Vec::with_capacity(cfg.batch * w);
        for b in 0..cfg.batch {
            let base = b * shard + s;
            for j in 0..w {
                inp.push(i32::from(data[base + j]));
                tgt.push(i32::from(data[base + j + 1]));
            }
        }
        let inp_t = Tensor::<AB, 2, Int>::from_data(TensorData::new(inp, [cfg.batch, w]), device);
        let tgt_t = Tensor::<AB, 1, Int>::from_data(TensorData::new(tgt, [cfg.batch * w]), device);

        let (logits, new_state) = model.forward_chunk(inp_t, state);
        let loss = CrossEntropyLossConfig::new()
            .init(device)
            .forward(logits.reshape([cfg.batch * w, V]), tgt_t);

        let nats = f64::from(loss.clone().into_data().to_vec::<f32>().unwrap()[0]);
        let chunk_pos = (cfg.batch * w) as f64;
        total_nats += nats * chunk_pos;
        total_positions += cfg.batch as u64 * w as u64;
        if step >= tail_start {
            tail_nats += nats * chunk_pos;
            tail_positions += cfg.batch as u64 * w as u64;
        }

        let grads = loss.backward();
        let gp = GradientsParams::from_grads(grads, &model);
        model = optim.step(cfg.lr, model, gp);
        state = new_state.into_iter().map(Tensor::detach).collect();

        if step % 200 == 0 {
            let bpb = total_nats / std::f64::consts::LN_2 / total_positions as f64;
            let pos_s = total_positions as f64 / t0.elapsed().as_secs_f64();
            println!(
                "    step {step:6} | {:5.1}% | running bpb {bpb:.4} | {:.0} pos/s",
                100.0 * (s + w) as f64 / shard as f64,
                pos_s
            );
        }
        s += w;
        step += 1;
    }
    let bpb = total_nats / std::f64::consts::LN_2 / total_positions as f64;
    let tail_bpb = if tail_positions > 0 {
        tail_nats / std::f64::consts::LN_2 / tail_positions as f64
    } else {
        bpb
    };
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "  [{label}] DONE  bpb {bpb:.4} | tail-25% {tail_bpb:.4}  ({secs:.0}s, {total_positions} positions)\n"
    );
    (bpb, tail_bpb)
}

// ───────────────────────── BPE (the bpb-per-FLOP token lever) ─────────────────────────

/// Greedy BPE: learn merges on `data` (bytes → tokens up to `target_vocab`).
/// New token id = 256 + merge index. Trained on a sample (merges generalize).
fn bpe_train(data: &[u8], target_vocab: usize) -> Vec<(u32, u32)> {
    use std::collections::HashMap;
    let mut seq: Vec<u32> = data.iter().map(|&b| u32::from(b)).collect();
    let mut merges = Vec::new();
    for m in 0..target_vocab.saturating_sub(256) {
        let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
        for win in seq.windows(2) {
            *counts.entry((win[0], win[1])).or_insert(0) += 1;
        }
        let Some((&pair, &cnt)) = counts.iter().max_by_key(|&(_, &c)| c) else {
            break;
        };
        if cnt < 2 {
            break;
        }
        let new_id = 256 + m as u32;
        merges.push(pair);
        let mut out = Vec::with_capacity(seq.len());
        let mut i = 0;
        while i < seq.len() {
            if i + 1 < seq.len() && seq[i] == pair.0 && seq[i + 1] == pair.1 {
                out.push(new_id);
                i += 2;
            } else {
                out.push(seq[i]);
                i += 1;
            }
        }
        seq = out;
    }
    merges
}

/// Tokenize `data` with learned `merges` (applied in learned order). Returns the
/// token-id stream and the byte count each token covers (for per-byte bpb).
fn bpe_tokenize(data: &[u8], merges: &[(u32, u32)]) -> (Vec<u32>, Vec<u32>) {
    let mut seq: Vec<u32> = data.iter().map(|&b| u32::from(b)).collect();
    let mut lens: Vec<u32> = vec![1; seq.len()];
    for (m, &pair) in merges.iter().enumerate() {
        let new_id = 256 + m as u32;
        let mut out = Vec::with_capacity(seq.len());
        let mut ol = Vec::with_capacity(seq.len());
        let mut i = 0;
        while i < seq.len() {
            if i + 1 < seq.len() && seq[i] == pair.0 && seq[i + 1] == pair.1 {
                out.push(new_id);
                ol.push(lens[i] + lens[i + 1]);
                i += 2;
            } else {
                out.push(seq[i]);
                ol.push(lens[i]);
                i += 1;
            }
        }
        seq = out;
        lens = ol;
    }
    (seq, lens)
}

/// Like `run`, but over a BPE token stream — bpb is per ORIGINAL BYTE (sum of the
/// predicted tokens' byte lengths), so it is directly comparable to the byte-level
/// archs. SEL-only (the settled winner). Vocab=256 / 0 merges reproduces `run`'s
/// byte-level bpb (the sanity gate — validated: 2.931 vs run()'s 2.933).
///
/// FINDING (2026-06-30): real-vocab (1024/2048) training DIVERGES (bpb → 1e17+,
/// forward overflow) regardless of gradient clipping (Norm 1.0) or embedding-init
/// scale (1.0 / 0.1). The unnormalized SEL residual stream is only conditionally
/// stable — fine on the byte stream, unstable on the BPE token stream. Evaluating
/// BPE needs a normalized SEL (pre-norm RMSNorm, as in the Mamba block here, which
/// never diverged) + byte-baseline re-validation. Deferred; the byte-level SEL is
/// the stable, committed Stage-0 winner.
fn run_bpe(
    mut model: SelNet<AB>,
    tokens: &[u32],
    byte_lens: &[u32],
    cfg: &Cfg,
    device: &WgpuDevice,
) -> (f64, f64) {
    let vocab = model.vocab();
    let n = tokens.len();
    let shard = n / cfg.batch;
    let w = cfg.window;
    let total_byte: u64 = byte_lens.iter().map(|&l| u64::from(l)).sum();
    let avg_bpt = total_byte as f64 / n as f64;
    let macs_byte = model.macs_per_token() as f64 / avg_bpt;
    let label = model.label();
    println!(
        "  [{label}] {:.0} K MACs/byte ({:.0} K/tok ÷ {avg_bpt:.2} B/tok) | {n} tokens, shard {shard} × {} | win {w}",
        macs_byte / 1000.0,
        model.macs_per_token() as f64 / 1000.0,
        cfg.batch
    );

    // Gradient-norm clipping stabilizes the unnormalized SEL at larger vocab/width
    // (the byte path was marginally stable; the sharper 1024-way softmax diverges
    // without it). Applied to both run() and run_bpe() so comparisons stay fair.
    let mut optim = AdamConfig::new()
        .init()
        .with_grad_clipping(GradientClippingConfig::Norm(1.0).init());
    let mut state = model.init_state(cfg.batch, device);
    let (mut tot_nats, mut tot_bytes) = (0.0f64, 0.0f64);
    let total_steps = (shard.saturating_sub(w + 1)) / w;
    let tail_start = total_steps * 3 / 4;
    let (mut tail_nats, mut tail_bytes) = (0.0f64, 0.0f64);
    let t0 = Instant::now();
    let (mut s, mut step) = (0usize, 0usize);
    while s + w + 1 <= shard {
        let mut inp: Vec<i32> = Vec::with_capacity(cfg.batch * w);
        let mut tgt: Vec<i32> = Vec::with_capacity(cfg.batch * w);
        let mut chunk_bytes = 0.0f64;
        for b in 0..cfg.batch {
            let base = b * shard + s;
            for j in 0..w {
                inp.push(tokens[base + j] as i32);
                tgt.push(tokens[base + j + 1] as i32);
                chunk_bytes += f64::from(byte_lens[base + j + 1]);
            }
        }
        let inp_t = Tensor::<AB, 2, Int>::from_data(TensorData::new(inp, [cfg.batch, w]), device);
        let tgt_t = Tensor::<AB, 1, Int>::from_data(TensorData::new(tgt, [cfg.batch * w]), device);
        let (logits, new_state) = model.forward_chunk(inp_t, state);
        let loss = CrossEntropyLossConfig::new()
            .init(device)
            .forward(logits.reshape([cfg.batch * w, vocab]), tgt_t);
        let nats = f64::from(loss.clone().into_data().to_vec::<f32>().unwrap()[0]);
        let chunk_nats = nats * (cfg.batch * w) as f64;
        tot_nats += chunk_nats;
        tot_bytes += chunk_bytes;
        if step >= tail_start {
            tail_nats += chunk_nats;
            tail_bytes += chunk_bytes;
        }
        let grads = loss.backward();
        let gp = GradientsParams::from_grads(grads, &model);
        model = optim.step(cfg.lr, model, gp);
        state = new_state.into_iter().map(Tensor::detach).collect();
        if step % 200 == 0 {
            println!(
                "    step {step:6} | {:5.1}% | running bpb {:.4}",
                100.0 * (s + w) as f64 / shard as f64,
                tot_nats / std::f64::consts::LN_2 / tot_bytes
            );
        }
        s += w;
        step += 1;
    }
    let bpb = tot_nats / std::f64::consts::LN_2 / tot_bytes;
    let tail = if tail_bytes > 0.0 {
        tail_nats / std::f64::consts::LN_2 / tail_bytes
    } else {
        bpb
    };
    println!(
        "  [{label}] DONE  bpb {bpb:.4} | tail-25% {tail:.4}  ({:.0}s)\n",
        t0.elapsed().as_secs_f64()
    );
    (bpb, tail)
}

/// Stage-1 forward-equivalence gate: a hand-written SCALAR CPU forward of the SEL
/// architecture must reproduce burn's forward for the same (random) weights and
/// input — the math the codec's online arm will run. Processing one timestep
/// through all layers (carrying per-layer state) equals burn's layer-over-sequence
/// pass, and matches the codec's one-byte-at-a-time use. Prints max abs/rel logit
/// diff; wgpu f32 vs scalar f32 differ ~1e-3/op so a small residual is expected.
fn verify_sel_cpu(device: &WgpuDevice) {
    let d = 16usize;
    let nl = 3usize;
    let w = 24usize;
    let net = SelNet::<AB>::new(d, nl, V, device);

    let toks: Vec<i32> = (0..w).map(|t| ((t * 37 + 11) % V) as i32).collect();
    let inp = Tensor::<AB, 2, Int>::from_data(TensorData::new(toks.clone(), [1, w]), device);
    let (logits, _) = net.forward_chunk(inp, net.init_state(1, device));
    let burn_logits = logits.into_data().to_vec::<f32>().unwrap(); // [w*V] row-major

    let v2 = |p: &Param<Tensor<AB, 2>>| p.val().into_data().to_vec::<f32>().unwrap();
    let v1 = |p: &Param<Tensor<AB, 1>>| p.val().into_data().to_vec::<f32>().unwrap();
    let emb = v2(&net.emb);
    let wo = v2(&net.wo);
    let bo = v1(&net.bo);
    struct L {
        win: Vec<f32>,
        wa: Vec<f32>,
        ba: Vec<f32>,
        wg: Vec<f32>,
        wmix: Vec<f32>,
        bmix: Vec<f32>,
    }
    let layers: Vec<L> = net
        .layers
        .iter()
        .map(|l| L {
            win: v2(&l.win),
            wa: v2(&l.wa),
            ba: v1(&l.ba),
            wg: v2(&l.wg),
            wmix: v2(&l.wmix),
            bmix: v1(&l.bmix),
        })
        .collect();

    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    // out[j] = sum_k x[k] * w[k*d+j]  (burn matmul [1,d]@[d,d] convention)
    let matvec = |x: &[f32], wt: &[f32]| -> Vec<f32> {
        let mut o = vec![0.0f32; d];
        for k in 0..d {
            let xk = x[k];
            let base = k * d;
            for j in 0..d {
                o[j] += xk * wt[base + j];
            }
        }
        o
    };

    let mut s: Vec<Vec<f32>> = vec![vec![0.0f32; d]; nl];
    let mut cpu_logits = vec![0.0f32; w * V];
    for (t, &tok) in toks.iter().enumerate() {
        let tok = tok as usize;
        let mut x: Vec<f32> = emb[tok * d..tok * d + d].to_vec();
        for (li, l) in layers.iter().enumerate() {
            let u = matvec(&x, &l.win);
            let a: Vec<f32> = matvec(&x, &l.wa)
                .iter()
                .zip(&l.ba)
                .map(|(v, b)| sig(v + b))
                .collect();
            let g: Vec<f32> = matvec(&x, &l.wg).iter().map(|v| sig(*v)).collect();
            for j in 0..d {
                s[li][j] = a[j] * s[li][j] + (1.0 - a[j]) * u[j];
            }
            let y: Vec<f32> = (0..d).map(|j| g[j] * s[li][j]).collect();
            let mixed = matvec(&y, &l.wmix);
            for j in 0..d {
                x[j] += mixed[j] + l.bmix[j];
            }
        }
        for v in 0..V {
            let mut acc = bo[v];
            for k in 0..d {
                acc += x[k] * wo[k * V + v];
            }
            cpu_logits[t * V + v] = acc;
        }
    }

    let (mut max_abs, mut max_rel) = (0.0f32, 0.0f32);
    for (c, b) in cpu_logits.iter().zip(&burn_logits) {
        let diff = (c - b).abs();
        max_abs = max_abs.max(diff);
        max_rel = max_rel.max(diff / b.abs().max(1.0));
    }
    println!(
        "SEL CPU-vs-burn forward equivalence: max_abs {max_abs:.3e}  max_rel {max_rel:.3e}  (d={d} layers={nl} w={w})"
    );
    println!(
        "{}",
        if max_rel < 1e-2 {
            "  PASS — scalar CPU forward matches burn."
        } else {
            "  FAIL — forward mismatch, investigate."
        }
    );
}

fn main() {
    let device = WgpuDevice::default();
    if std::env::var("LZR_VERIFY").is_ok() {
        verify_sel_cpu(&device);
        return;
    }
    let env_us = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let env_f = |k: &str, d: f64| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };

    let data_path = std::env::var("LZR_PP_DATA").unwrap_or_else(|_| "/tmp/lzr_pp.bin".into());
    let data = std::fs::read(&data_path).expect("post-pipeline stream (run dump_preprocessed)");
    let cfg = Cfg {
        bytes: env_us("LZR_BYTES", 20_000_000),
        batch: env_us("LZR_BATCH", 32),
        window: env_us("LZR_WIN", 32),
        lr: env_f("LZR_LR", 1e-3),
    };
    let arch = std::env::var("LZR_ARCH").unwrap_or_else(|_| "lstm,ssm".into());
    println!(
        "bake-off | {} B data, streaming {} B | batch {} win {} lr {}\n",
        data.len(),
        cfg.bytes.min(data.len()),
        cfg.batch,
        cfg.window,
        cfg.lr
    );

    let mut results: Vec<(String, f64, f64)> = Vec::new();
    for a in arch.split(',') {
        let (lbl, (bpb, tail)) = match a.trim() {
            "lstm" => {
                let m = LstmNet::<AB>::new(env_us("LZR_E", 64), env_us("LZR_H", 192), &device);
                (m.label(), run(m, &data, &cfg, &device))
            }
            "ssm" => {
                let m = SsmNet::<AB>::new(env_us("LZR_D", 256), env_us("LZR_LAYERS", 2), &device);
                (m.label(), run(m, &data, &cfg, &device))
            }
            "sel" => {
                let m =
                    SelNet::<AB>::new(env_us("LZR_SD", 160), env_us("LZR_SLAYERS", 2), V, &device);
                (m.label(), run(m, &data, &cfg, &device))
            }
            "mamba" => {
                let m = MambaNet::<AB>::new(
                    env_us("LZR_MD", 112),
                    env_us("LZR_N", 8),
                    env_us("LZR_MLAYERS", 4),
                    &device,
                );
                (m.label(), run(m, &data, &cfg, &device))
            }
            // SEL over a BPE token stream: the bpb-per-FLOP token lever. LZR_VOCAB
            // (0 merges → vocab 256 = byte-level identity, the sanity gate).
            "bpe" => {
                let nslice = cfg.bytes.min(data.len());
                let target = env_us("LZR_VOCAB", 1024);
                let sample = (nslice / 4).clamp(1, 4_000_000).min(nslice);
                println!("  training BPE→vocab {target} on {sample} B sample…");
                let merges = bpe_train(&data[..sample], target);
                let (tokens, lens) = bpe_tokenize(&data[..nslice], &merges);
                let vocab = 256 + merges.len();
                let m = SelNet::<AB>::new(
                    env_us("LZR_SD", 112),
                    env_us("LZR_SLAYERS", 4),
                    vocab,
                    &device,
                );
                (m.label(), run_bpe(m, &tokens, &lens, &cfg, &device))
            }
            other => {
                println!("(unknown arch '{other}' — skipping)");
                continue;
            }
        };
        results.push((lbl, bpb, tail));
    }

    println!("=== SUMMARY (bpb per post-pipeline byte, lower better) ===");
    println!("  {:24} {:>10} {:>10}", "arch", "cum-bpb", "tail-25%");
    for (lbl, bpb, tail) in &results {
        println!("  {lbl:24} {bpb:10.4} {tail:10.4}");
    }
    if results.len() >= 2 {
        let (base_lbl, base_cum, base_tail) = &results[0];
        for (lbl, bpb, tail) in &results[1..] {
            println!(
                "  {lbl:24} cum {:+.1}% | tail {:+.1}%  vs {base_lbl}",
                100.0 * (bpb - base_cum) / base_cum,
                100.0 * (tail - base_tail) / base_tail
            );
        }
    }
}
