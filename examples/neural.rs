//! v8 neural codec skeleton (self-contained example).
//! Stage A: a multi-symbol range coder, verified by an inline round-trip test.
//! Run: cargo run --release --example neural --features neural
#![recursion_limit = "256"]
#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    missing_docs,
    dead_code
)]
#![allow(elided_lifetimes_in_paths)]

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::module::{AutodiffModule, Module};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::backend::AutodiffBackend;

// The submission tokenizer, shared verbatim so train and decode tokenize
// identically. Compiled into this example crate via `#[path]`.
#[path = "../src/bpe.rs"]
mod bpe;

// ---------------------------------------------------------------------------
// Carryless range coder (Subbotin style). Multi-symbol via cumulative-freq CDF.
// CDF is a slice of length alphabet+1: cdf[0]=0, cdf[alphabet]=TOTAL_FREQ,
// strictly increasing. Symbol s has cum=cdf[s], freq=cdf[s+1]-cdf[s].
// ---------------------------------------------------------------------------
const RC_TOP: u32 = 1 << 24;
const RC_BOT: u32 = 1 << 16;

struct RangeEncoder {
    low: u32,
    range: u32,
    out: Vec<u8>,
}
impl RangeEncoder {
    fn new() -> Self {
        Self {
            low: 0,
            range: u32::MAX,
            out: Vec::new(),
        }
    }
    fn encode(&mut self, cum: u32, freq: u32, tot: u32) {
        let r = self.range / tot;
        self.low = self.low.wrapping_add(r * cum);
        self.range = r * freq;
        loop {
            if (self.low ^ self.low.wrapping_add(self.range)) < RC_TOP {
                // top byte settled
            } else if self.range < RC_BOT {
                self.range = self.low.wrapping_neg() & (RC_BOT - 1);
            } else {
                break;
            }
            self.out.push((self.low >> 24) as u8);
            self.low <<= 8;
            self.range <<= 8;
        }
    }
    fn finish(mut self) -> Vec<u8> {
        for _ in 0..4 {
            self.out.push((self.low >> 24) as u8);
            self.low <<= 8;
        }
        self.out
    }
}

struct RangeDecoder<'a> {
    low: u32,
    range: u32,
    code: u32,
    inp: &'a [u8],
    pos: usize,
}
impl<'a> RangeDecoder<'a> {
    fn new(inp: &'a [u8]) -> Self {
        let mut d = Self {
            low: 0,
            range: u32::MAX,
            code: 0,
            inp,
            pos: 0,
        };
        for _ in 0..4 {
            d.code = (d.code << 8) | d.next();
        }
        d
    }
    fn next(&mut self) -> u32 {
        let b = self.inp.get(self.pos).copied().unwrap_or(0) as u32;
        self.pos += 1;
        b
    }
    /// Current frequency target in [0, tot).
    fn freq(&self, tot: u32) -> u32 {
        let r = self.range / tot;
        (self.code.wrapping_sub(self.low) / r).min(tot - 1)
    }
    fn update(&mut self, cum: u32, freq: u32, tot: u32) {
        let r = self.range / tot;
        self.low = self.low.wrapping_add(r * cum);
        self.range = r * freq;
        loop {
            if (self.low ^ self.low.wrapping_add(self.range)) < RC_TOP {
            } else if self.range < RC_BOT {
                self.range = self.low.wrapping_neg() & (RC_BOT - 1);
            } else {
                break;
            }
            self.code = (self.code << 8) | self.next();
            self.low <<= 8;
            self.range <<= 8;
        }
    }
}

const CDF_TOTAL: u32 = 1 << 16; // freq resolution per step

/// Convert a probability vector to a strictly-increasing integer CDF summing to
/// CDF_TOTAL, with every symbol floored at freq >= 1 (no zero-mass → coder-safe).
/// Deterministic: identical on encode and decode.
fn probs_to_cdf(probs: &[f32]) -> Vec<u32> {
    let n = probs.len();
    let mut freqs = vec![1u32; n]; // floor everyone at 1
    let remaining = CDF_TOTAL - n as u32; // distribute the rest by probability
    let sum: f32 = probs.iter().sum::<f32>().max(1e-9);
    let mut used = 0u32;
    for i in 0..n {
        let add = ((probs[i] / sum) * remaining as f32) as u32;
        freqs[i] += add;
        used += add;
    }
    // dump rounding remainder onto the argmax (deterministic)
    let leftover = remaining - used;
    let argmax = (0..n)
        .max_by(|&a, &b| probs[a].total_cmp(&probs[b]))
        .unwrap();
    freqs[argmax] += leftover;
    // build CDF
    let mut cdf = vec![0u32; n + 1];
    for i in 0..n {
        cdf[i + 1] = cdf[i] + freqs[i];
    }
    debug_assert_eq!(cdf[n], CDF_TOTAL);
    cdf
}

fn ac_encode_symbol(enc: &mut RangeEncoder, cdf: &[u32], sym: usize) {
    enc.encode(cdf[sym], cdf[sym + 1] - cdf[sym], CDF_TOTAL);
}
fn ac_decode_symbol(dec: &mut RangeDecoder, cdf: &[u32]) -> usize {
    let f = dec.freq(CDF_TOTAL);
    // rightmost i with cdf[i] <= f
    let mut sym = cdf.partition_point(|&c| c <= f) - 1;
    if sym + 1 >= cdf.len() {
        sym = cdf.len() - 2;
    }
    dec.update(cdf[sym], cdf[sym + 1] - cdf[sym], CDF_TOTAL);
    sym
}

// ---------------------------------------------------------------------------
// Decoder-only Transformer (byte/token-level), tied embeddings, learned pos.
// fp first; BitNet QAT linear is a localized swap once round-trip works.
// ---------------------------------------------------------------------------
// BitNet b1.58 QAT linear: ternary absmean weights + per-token int8 activations,
// straight-through estimator via detach(). Forward == quantized; grad == identity
// to the latent fp weight. detach() is a no-op off-autodiff, so the same forward
// runs for training (Autodiff) and inference. No bias (BitNet removes biases).
#[derive(Module, Debug)]
struct BitLinear<B: Backend> {
    weight: burn::module::Param<Tensor<B, 2>>, // [out, in] latent fp
    bit: bool,                                 // false = plain fp linear (A/B toggle)
}

impl<B: Backend> BitLinear<B> {
    fn new(in_f: usize, out_f: usize, bit: bool, device: &B::Device) -> Self {
        let std = (1.0 / in_f as f64).sqrt();
        let weight = Tensor::random([out_f, in_f], Distribution::Normal(0.0, std), device);
        Self {
            weight: burn::module::Param::from_tensor(weight),
            bit,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let w = self.weight.val(); // [out, in]
        if !self.bit {
            return x.matmul(w.transpose().unsqueeze_dim(0));
        }
        // ternary weight (per-tensor absmean) with STE
        let gamma = w.clone().abs().mean().reshape([1, 1]).add_scalar(1e-5);
        let w_t = w.clone().div(gamma.clone()).round().clamp(-1.0, 1.0);
        let w_q = w_t.mul(gamma);
        let w_used = w.clone() + (w_q - w).detach();
        // per-token int8 activation (absmax) with STE
        let s = x.clone().abs().max_dim(2).div_scalar(127.0).clamp_min(1e-5); // [b,t,1]
        let x_q = x.clone().div(s.clone()).round().clamp(-127.0, 127.0).mul(s);
        let x_used = x.clone() + (x_q - x).detach();
        x_used.matmul(w_used.transpose().unsqueeze_dim(0))
    }
}

// Hand-built causal multi-head attention with BitLinear Q/K/V/O projections.
// Replaces burn's MHA so (a) every projection weight is accessible for packing,
// (b) the whole model is ternary.
fn causal_mask<B: Backend>(t: usize, device: &B::Device) -> Tensor<B, 4> {
    let mut m = vec![0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            m[i * t + j] = -1e9;
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(m, [t, t]), device).reshape([1, 1, t, t])
}

#[derive(Module, Debug)]
struct Expert<B: Backend> {
    ff1: BitLinear<B>,
    ff2: BitLinear<B>,
}

#[derive(Module, Debug)]
struct Block<B: Backend> {
    norm1: LayerNorm<B>,
    wq: BitLinear<B>,
    wk: BitLinear<B>,
    wv: BitLinear<B>,
    wo: BitLinear<B>,
    norm2: LayerNorm<B>,
    router: burn::module::Param<Tensor<B, 2>>, // [n_experts, d], full precision
    experts: Vec<Expert<B>>,
    n_heads: usize,
}

impl<B: Backend> Block<B> {
    fn attn(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, d] = x.dims();
        let dk = d / self.n_heads;
        let to_heads = |z: Tensor<B, 3>| z.reshape([b, t, self.n_heads, dk]).swap_dims(1, 2);
        let q = to_heads(self.wq.forward(x.clone())); // [b,h,t,dk]
        let k = to_heads(self.wk.forward(x.clone()));
        let v = to_heads(self.wv.forward(x.clone()));
        let scores = q.matmul(k.swap_dims(2, 3)).div_scalar((dk as f64).sqrt()); // [b,h,t,t]
        let scores = scores + causal_mask::<B>(t, &x.device());
        let ctx = softmax(scores, 3).matmul(v); // [b,h,t,dk]
        let ctx = ctx.swap_dims(1, 2).reshape([b, t, d]);
        self.wo.forward(ctx)
    }

    /// Top-2 mixture-of-experts FFN. Training computes all experts densely and
    /// masks to the top-2 (simple + correct; the codec does the sparse gather).
    /// Gate = softmax over router logits, renormalized over the chosen experts.
    fn moe(&self, n2: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, d] = n2.dims();
        let rlogits = n2
            .clone()
            .matmul(self.router.val().transpose().unsqueeze_dim(0)); // [b,t,ne]
        let gate = softmax(rlogits, 2);
        // 2nd-largest gate (topk is unimplemented for autodiff): take the max,
        // mask it out, take the max again. Then keep experts ≥ that threshold.
        let max1 = gate.clone().max_dim(2); // [b,t,1]
        let is_max = gate.clone().equal(max1); // [b,t,ne] bool
        let without_max = gate.clone().mask_fill(is_max, f32::NEG_INFINITY);
        let kth = without_max.max_dim(2); // [b,t,1] = 2nd-largest gate
        let mask = gate.clone().greater_equal(kth).float(); // [b,t,ne]
        let masked = gate * mask;
        let denom = masked.clone().sum_dim(2); // [b,t,1]
        let gnorm = masked / denom;
        let mut out = Tensor::<B, 3>::zeros([b, t, d], &n2.device());
        for (e, exp) in self.experts.iter().enumerate() {
            let oe = exp.ff2.forward(gelu(exp.ff1.forward(n2.clone()))); // [b,t,d]
            let ge = gnorm.clone().slice([0..b, 0..t, e..e + 1]); // [b,t,1]
            out = out + oe * ge;
        }
        out
    }
}

/// Ternary absmean quantization with a straight-through estimator (matches
/// `BitLinear`'s weight path). Used for the tied token embedding so it ships at
/// ~2 b/w like every other weight; `detach` is a no-op off-autodiff.
fn ternary_ste<B: Backend>(w: Tensor<B, 2>) -> Tensor<B, 2> {
    let gamma = w.clone().abs().mean().reshape([1, 1]).add_scalar(1e-5);
    let w_t = w.clone().div(gamma.clone()).round().clamp(-1.0, 1.0);
    let w_q = w_t.mul(gamma);
    w.clone() + (w_q - w).detach()
}

#[derive(Module, Debug)]
struct Gpt<B: Backend> {
    tok_w: burn::module::Param<Tensor<B, 2>>, // [vocab, d] latent; ternary when bit
    pos: Embedding<B>,
    blocks: Vec<Block<B>>,
    norm_f: LayerNorm<B>,
    bit: bool,
}

#[derive(Config, Debug)]
struct GptConfig {
    vocab: usize,
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    ffn: usize, // per-expert hidden size
    ctx: usize,
    n_experts: usize,
    #[config(default = false)]
    bit: bool,
}

impl GptConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Gpt<B> {
        let d = self.d_model;
        let blocks = (0..self.n_layers)
            .map(|_| Block {
                norm1: LayerNormConfig::new(d).init(device),
                wq: BitLinear::new(d, d, self.bit, device),
                wk: BitLinear::new(d, d, self.bit, device),
                wv: BitLinear::new(d, d, self.bit, device),
                wo: BitLinear::new(d, d, self.bit, device),
                norm2: LayerNormConfig::new(d).init(device),
                router: burn::module::Param::from_tensor(Tensor::random(
                    [self.n_experts, d],
                    Distribution::Normal(0.0, 0.02),
                    device,
                )),
                experts: (0..self.n_experts)
                    .map(|_| Expert {
                        ff1: BitLinear::new(d, self.ffn, self.bit, device),
                        ff2: BitLinear::new(self.ffn, d, self.bit, device),
                    })
                    .collect(),
                n_heads: self.n_heads,
            })
            .collect();
        let tok = Tensor::random(
            [self.vocab, self.d_model],
            Distribution::Normal(0.0, 0.02),
            device,
        );
        Gpt {
            tok_w: burn::module::Param::from_tensor(tok),
            pos: EmbeddingConfig::new(self.ctx, self.d_model).init(device),
            blocks,
            norm_f: LayerNormConfig::new(self.d_model).init(device),
            bit: self.bit,
        }
    }
}

impl<B: Backend> Gpt<B> {
    /// tokens [batch, seq] Int -> logits [batch, seq, vocab].
    fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, t] = tokens.dims();
        let d = self.tok_w.val().dims()[1];
        let device = tokens.device();
        // tied, optionally-ternary embedding table
        let tok = self.tok_w.val();
        let wq = if self.bit { ternary_ste(tok) } else { tok };
        // embedding lookup: gather rows, [b*t, d] -> [b, t, d]
        let flat = tokens.reshape([b * t]);
        let mut x = wq.clone().select(0, flat).reshape([b, t, d]);
        let pos_ids = Tensor::<B, 1, Int>::arange(0..t as i64, &device).reshape([1, t]);
        x = x + self.pos.forward(pos_ids); // broadcast [1,t,d] over batch
        for blk in &self.blocks {
            x = x.clone() + blk.attn(blk.norm1.forward(x.clone()));
            let n2 = blk.norm2.forward(x.clone());
            x = x + blk.moe(n2);
        }
        x = self.norm_f.forward(x);
        // tied unembedding: logits = x @ wq^T
        x.matmul(wq.transpose().unsqueeze_dim(0)) // [1,d,vocab] broadcast -> [b,t,vocab]
    }
}

// ---------------------------------------------------------------------------
// Stage A: verify the coder round-trips a skewed multi-symbol stream.
// ---------------------------------------------------------------------------
fn test_coder() {
    // deterministic pseudo-random symbol stream over a 2000-symbol alphabet
    let vocab = 2000usize;
    let n = 20_000usize;
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    // a fixed skewed prob vector (zipf-ish)
    let probs: Vec<f32> = (0..vocab).map(|i| 1.0 / (i as f32 + 1.0)).collect();
    let cdf = probs_to_cdf(&probs);
    // sample symbols from the cdf
    let syms: Vec<usize> = (0..n)
        .map(|_| {
            let f = (next() % CDF_TOTAL as u64) as u32;
            cdf.partition_point(|&c| c <= f) - 1
        })
        .collect();

    let mut enc = RangeEncoder::new();
    for &s in &syms {
        ac_encode_symbol(&mut enc, &cdf, s);
    }
    let bytes = enc.finish();

    let mut dec = RangeDecoder::new(&bytes);
    let mut ok = true;
    for &s in &syms {
        let d = ac_decode_symbol(&mut dec, &cdf);
        if d != s {
            ok = false;
            break;
        }
    }
    let bpb = 8.0 * bytes.len() as f64 / n as f64;
    println!(
        "coder round-trip: {} ({} syms, {} bytes, {:.3} bits/sym)",
        if ok { "OK" } else { "FAIL" },
        n,
        bytes.len(),
        bpb
    );
    assert!(ok, "range coder round-trip failed");
}

// xorshift for deterministic batch sampling
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn train<B: AutodiffBackend>(
    tokens: &[i32],
    cfg: &GptConfig,
    device: &B::Device,
    steps: usize,
) -> Gpt<B> {
    let mut model = cfg.init::<B>(device);
    let mut optim = AdamConfig::new().init();
    let lr = 3e-4;
    let (batch, ctx) = (32usize, cfg.ctx);
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    for step in 0..steps {
        let mut inp = Vec::with_capacity(batch * ctx);
        let mut tgt = Vec::with_capacity(batch * ctx);
        for _ in 0..batch {
            let s = (rng.next() as usize) % (tokens.len() - ctx - 1);
            for k in 0..ctx {
                inp.push(tokens[s + k]);
                tgt.push(tokens[s + k + 1]);
            }
        }
        let inp_t = Tensor::<B, 2, Int>::from_data(TensorData::new(inp, [batch, ctx]), device);
        let tgt_t = Tensor::<B, 2, Int>::from_data(TensorData::new(tgt, [batch, ctx]), device);
        let logits = model.forward(inp_t);
        let [bb, tt, vv] = logits.dims();
        let loss = CrossEntropyLossConfig::new()
            .init(device)
            .forward(logits.reshape([bb * tt, vv]), tgt_t.reshape([bb * tt]));
        let lval = loss.clone().into_scalar().elem::<f32>();
        let grads = loss.backward();
        let gp = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, gp);
        if step % 50 == 0 || step + 1 == steps {
            println!(
                "  step {step:4} loss {lval:.4}  ({:.3} bpb)",
                f64::from(lval) / std::f64::consts::LN_2
            );
        }
    }
    model
}

/// One next-token distribution given a context of token ids (last position).
fn predict<B: Backend>(model: &Gpt<B>, device: &B::Device, ctx_ids: &[i32]) -> Vec<f32> {
    let len = ctx_ids.len();
    let vocab = model.tok_w.val().dims()[0];
    let t = Tensor::<B, 2, Int>::from_data(TensorData::new(ctx_ids.to_vec(), [1, len]), device);
    let logits = model.forward(t); // [1, len, vocab]
    let last = logits
        .slice([0..1, len - 1..len, 0..vocab])
        .reshape([vocab]);
    softmax(last, 0).into_data().to_vec::<f32>().unwrap()
}

/// Block-reset context feed matching `src/v8.rs::step`: keep ≤ `ctx` tokens;
/// when the window fills, clear it and let the incoming token become position 0.
/// Returns the next-token distribution after processing `tok`.
fn feed<B: Backend>(
    model: &Gpt<B>,
    device: &B::Device,
    block: &mut Vec<i32>,
    ctx: usize,
    tok: i32,
) -> Vec<f32> {
    if block.len() == ctx {
        block.clear();
    }
    block.push(tok);
    predict(model, device, block)
}

/// Encode token ids autoregressively (BOS-seeded, block-reset context — same
/// behavior as the submission codec, so the bpb matches and it round-trips).
fn nn_encode<B: Backend>(model: &Gpt<B>, device: &B::Device, toks: &[i32], bos: i32) -> Vec<u8> {
    let ctx = model.pos.weight.val().dims()[0];
    let mut enc = RangeEncoder::new();
    let mut block = Vec::new();
    let mut probs = feed(model, device, &mut block, ctx, bos);
    for &id in toks {
        let cdf = probs_to_cdf(&probs);
        ac_encode_symbol(&mut enc, &cdf, id as usize);
        probs = feed(model, device, &mut block, ctx, id);
    }
    enc.finish()
}

fn nn_decode<B: Backend>(
    model: &Gpt<B>,
    device: &B::Device,
    archive: &[u8],
    n: usize,
    bos: i32,
) -> Vec<i32> {
    let ctx = model.pos.weight.val().dims()[0];
    let mut dec = RangeDecoder::new(archive);
    let mut out = Vec::with_capacity(n);
    let mut block = Vec::new();
    let mut probs = feed(model, device, &mut block, ctx, bos);
    for _ in 0..n {
        let cdf = probs_to_cdf(&probs);
        let sym = ac_decode_symbol(&mut dec, &cdf) as i32;
        out.push(sym);
        probs = feed(model, device, &mut block, ctx, sym);
    }
    out
}

// ---------------------------------------------------------------------------
// Weight packing — produce the submission blob the pure-Rust codec consumes.
// Must replicate BitLinear's quantization exactly: scale = mean(|w|)+1e-5,
// ternary = round(w/scale).clamp(-1,1). Format mirrors `src/v8.rs::Model`.
// ---------------------------------------------------------------------------
const BLOB_MAGIC: u32 = 0x3852_5A4C; // "LZR8" little-endian
const BLOB_VERSION: u32 = 3;

fn push_f32s(out: &mut Vec<u8>, vals: &[f32]) {
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn tensor_vec<B: Backend, const R: usize>(t: Tensor<B, R>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

/// Pack a row-major weight as scale f32 + 2-bit ternary, replicating
/// `ternary_ste`'s quantization exactly (scale = mean(|w|)+1e-5).
fn pack_ternary(out: &mut Vec<u8>, w: &[f32]) {
    let scale = (w.iter().map(|v| v.abs()).sum::<f32>() / w.len() as f32) + 1e-5;
    out.extend_from_slice(&scale.to_le_bytes());
    let mut byte = 0u8;
    let mut k = 0u32;
    for &v in w {
        let code: u8 = match (v / scale).round().clamp(-1.0, 1.0) as i32 {
            -1 => 0,
            0 => 1,
            _ => 2,
        };
        byte |= code << (2 * k);
        k += 1;
        if k == 4 {
            out.push(byte);
            byte = 0;
            k = 0;
        }
    }
    if k > 0 {
        out.push(byte);
    }
}

fn pack_bitlinear<B: Backend>(out: &mut Vec<u8>, lin: &BitLinear<B>) {
    pack_ternary(out, &tensor_vec(lin.weight.val()));
}

fn push_layernorm<B: Backend>(out: &mut Vec<u8>, ln: &LayerNorm<B>) {
    push_f32s(out, &tensor_vec(ln.gamma.val()));
    push_f32s(out, &tensor_vec(ln.beta.clone().unwrap().val()));
}

fn pack_model<B: Backend>(model: &Gpt<B>, cfg: &GptConfig, bpe: &bpe::Bpe) -> Vec<u8> {
    let mut out = Vec::new();
    for v in [
        BLOB_MAGIC,
        BLOB_VERSION,
        cfg.vocab as u32,
        cfg.d_model as u32,
        cfg.n_layers as u32,
        cfg.n_heads as u32,
        cfg.ffn as u32,
        cfg.ctx as u32,
        cfg.n_experts as u32,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&bpe.to_bytes());
    pack_ternary(&mut out, &tensor_vec(model.tok_w.val())); // ternary tied embedding
    push_f32s(&mut out, &tensor_vec(model.pos.weight.val())); // f32 pos
    push_layernorm(&mut out, &model.norm_f);
    for blk in &model.blocks {
        push_layernorm(&mut out, &blk.norm1);
        pack_bitlinear(&mut out, &blk.wq);
        pack_bitlinear(&mut out, &blk.wk);
        pack_bitlinear(&mut out, &blk.wv);
        pack_bitlinear(&mut out, &blk.wo);
        push_layernorm(&mut out, &blk.norm2);
        push_f32s(&mut out, &tensor_vec(blk.router.val())); // router (f32)
        for exp in &blk.experts {
            pack_bitlinear(&mut out, &exp.ff1);
            pack_bitlinear(&mut out, &exp.ff2);
        }
    }
    out
}

/// Train on GPU and evaluate the round-trip + bpb there (GPU is the faster
/// inference path; the pure-Rust CPU codec is exercised separately via
/// `lzr nn-test`). `test_toks` is `bpe.encode(test_bytes)`; bpb is bits per
/// original *byte* so it stays comparable across tokenizations.
fn train_and_eval(
    train_toks: &[i32],
    test_bytes: &[u8],
    test_toks: &[i32],
    cfg: &GptConfig,
    bpe: &bpe::Bpe,
    label: &str,
) {
    type GpuAd = Autodiff<Wgpu<f32, i32>>;
    let gdev = WgpuDevice::default();
    let nparams = cfg.init::<Wgpu<f32, i32>>(&gdev).num_params();
    println!(
        "[{label}] training on GPU ({nparams} params, vocab {})...",
        cfg.vocab
    );
    let model = train::<GpuAd>(train_toks, cfg, &gdev, 1000);

    let gpu_model = model.valid();
    let arc = nn_encode::<Wgpu<f32, i32>>(&gpu_model, &gdev, test_toks, 0);
    let dec = nn_decode::<Wgpu<f32, i32>>(&gpu_model, &gdev, &arc, test_toks.len(), 0);
    println!(
        "[{label}] GPU: roundtrip {}  bpb {:.3}  ({} bytes / {} tokens -> {} bytes)",
        dec == test_toks,
        8.0 * arc.len() as f64 / test_bytes.len() as f64,
        test_bytes.len(),
        test_toks.len(),
        arc.len()
    );

    // Pack the trained model into the submission blob (bitnet only — the
    // ternary path is what ships). Packed directly from the GPU model.
    if cfg.bit {
        let blob = pack_model(&gpu_model, cfg, bpe);
        std::fs::write("assets/v8weights.bin", &blob).expect("write blob");
        let ld_floor = 16.0 * blob.len() as f64 / 1e9;
        println!(
            "[{label}] packed blob: {} bytes ({:.2} KiB)  →  weights-only L(D) ≥ {:.4} bpb on enwik9",
            blob.len(),
            blob.len() as f64 / 1024.0,
            ld_floor,
        );
    }
}

fn main() {
    test_coder();
    let data = std::fs::read("assets/enwik8").expect("read enwik8");
    let mb = 1024 * 1024;

    // online-BPE (1K merges / MB), 2K vocab to keep this ~1M MoE test in budget.
    let bpe = bpe::Bpe::learn(&data, 2048);
    let vocab = bpe.vocab_size();
    // Train on enwik8 and evaluate on it — for Hutter the model is trained on the
    // exact data it compresses (weights ship as L(D)), so memorization is the
    // regime, not overfitting. eval slice is a prefix of the training span.
    let train_bytes = &data[..4 * mb];
    let train_toks: Vec<i32> = bpe.encode(train_bytes).iter().map(|&t| t as i32).collect();
    let eval_bytes = &data[..4096];
    let eval_toks: Vec<i32> = bpe.encode(eval_bytes).iter().map(|&t| t as i32).collect();
    println!(
        "BPE: vocab {vocab}; train {} MB -> {} tokens ({:.2} bytes/token)",
        train_bytes.len() / mb,
        train_toks.len(),
        train_bytes.len() as f64 / train_toks.len() as f64,
    );

    // ~1M-param MoE test: d=128, 2 layers, 4 heads, expert-ffn 48, 24 experts (top-2).
    let fp = GptConfig::new(vocab, 128, 2, 4, 48, 256, 24).with_bit(false);
    let bit = GptConfig::new(vocab, 128, 2, 4, 48, 256, 24).with_bit(true);
    train_and_eval(&train_toks, eval_bytes, &eval_toks, &fp, &bpe, "fp");
    train_and_eval(&train_toks, eval_bytes, &eval_toks, &bit, &bpe, "bitnet");
}
