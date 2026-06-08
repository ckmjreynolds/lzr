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

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, NdArray, Wgpu};
use burn::module::{AutodiffModule, Module};
use burn::nn::attention::{
    MhaInput, MultiHeadAttention, MultiHeadAttentionConfig, generate_autoregressive_mask,
};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::Distribution;
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::backend::AutodiffBackend;

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

#[derive(Module, Debug)]
struct Block<B: Backend> {
    norm1: LayerNorm<B>,
    attn: MultiHeadAttention<B>,
    norm2: LayerNorm<B>,
    ff1: BitLinear<B>,
    ff2: BitLinear<B>,
}

#[derive(Module, Debug)]
struct Gpt<B: Backend> {
    tok: Embedding<B>,
    pos: Embedding<B>,
    blocks: Vec<Block<B>>,
    norm_f: LayerNorm<B>,
}

#[derive(Config, Debug)]
struct GptConfig {
    vocab: usize,
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    ffn: usize,
    ctx: usize,
    #[config(default = false)]
    bit: bool,
}

impl GptConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Gpt<B> {
        let blocks = (0..self.n_layers)
            .map(|_| Block {
                norm1: LayerNormConfig::new(self.d_model).init(device),
                attn: MultiHeadAttentionConfig::new(self.d_model, self.n_heads)
                    .with_dropout(0.0)
                    .init(device),
                norm2: LayerNormConfig::new(self.d_model).init(device),
                ff1: BitLinear::new(self.d_model, self.ffn, self.bit, device),
                ff2: BitLinear::new(self.ffn, self.d_model, self.bit, device),
            })
            .collect();
        Gpt {
            tok: EmbeddingConfig::new(self.vocab, self.d_model).init(device),
            pos: EmbeddingConfig::new(self.ctx, self.d_model).init(device),
            blocks,
            norm_f: LayerNormConfig::new(self.d_model).init(device),
        }
    }
}

impl<B: Backend> Gpt<B> {
    /// tokens [batch, seq] Int -> logits [batch, seq, vocab].
    fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, t] = tokens.dims();
        let device = tokens.device();
        let mut x = self.tok.forward(tokens); // [b, t, d]
        let pos_ids = Tensor::<B, 1, Int>::arange(0..t as i64, &device).reshape([1, t]);
        x = x + self.pos.forward(pos_ids); // broadcast [1,t,d] over batch
        let mask = generate_autoregressive_mask::<B>(b, t, &device); // [b,t,t] bool, true=block
        for blk in &self.blocks {
            let normed = blk.norm1.forward(x.clone());
            let attn = blk
                .attn
                .forward(MhaInput::self_attn(normed).mask_attn(mask.clone()))
                .context;
            x = x + attn;
            let n2 = blk.norm2.forward(x.clone());
            x = x + blk.ff2.forward(gelu(blk.ff1.forward(n2)));
        }
        x = self.norm_f.forward(x);
        // tied unembedding: logits = x @ tok_weight^T
        let wt = self.tok.weight.val().transpose(); // [d, vocab]
        x.matmul(wt.unsqueeze_dim(0)) // [1,d,vocab] broadcast -> [b,t,vocab]
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

const VOCAB: usize = 256; // byte-level v1

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
    data: &[u8],
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
            let s = (rng.next() as usize) % (data.len() - ctx - 1);
            for k in 0..ctx {
                inp.push(i32::from(data[s + k]));
                tgt.push(i32::from(data[s + k + 1]));
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
    let t = Tensor::<B, 2, Int>::from_data(TensorData::new(ctx_ids.to_vec(), [1, len]), device);
    let logits = model.forward(t); // [1, len, vocab]
    let last = logits
        .slice([0..1, len - 1..len, 0..VOCAB])
        .reshape([VOCAB]);
    softmax(last, 0).into_data().to_vec::<f32>().unwrap()
}

/// Encode `bytes` autoregressively (BOS-prefixed context). Same forward path as decode.
fn nn_encode<B: Backend>(model: &Gpt<B>, device: &B::Device, bytes: &[u8], bos: i32) -> Vec<u8> {
    let mut enc = RangeEncoder::new();
    let mut ctx_ids = vec![bos];
    for &byte in bytes {
        let probs = predict(model, device, &ctx_ids);
        let cdf = probs_to_cdf(&probs);
        ac_encode_symbol(&mut enc, &cdf, byte as usize);
        ctx_ids.push(i32::from(byte));
    }
    enc.finish()
}

fn nn_decode<B: Backend>(
    model: &Gpt<B>,
    device: &B::Device,
    archive: &[u8],
    n: usize,
    bos: i32,
) -> Vec<u8> {
    let mut dec = RangeDecoder::new(archive);
    let mut out = Vec::with_capacity(n);
    let mut ctx_ids = vec![bos];
    for _ in 0..n {
        let probs = predict(model, device, &ctx_ids);
        let cdf = probs_to_cdf(&probs);
        let sym = ac_decode_symbol(&mut dec, &cdf);
        out.push(sym as u8);
        ctx_ids.push(sym as i32);
    }
    out
}

fn train_and_eval(data: &[u8], test: &[u8], cfg: &GptConfig, label: &str) {
    type GpuAd = Autodiff<Wgpu<f32, i32>>;
    let gdev = WgpuDevice::default();
    let nparams = cfg.init::<Wgpu<f32, i32>>(&gdev).num_params();
    println!("[{label}] training on GPU ({nparams} params)...");
    let model = train::<GpuAd>(&data[..256 * 1024], cfg, &gdev, 300);

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let path = format!("/tmp/v8model_{label}");
    model.clone().save_file(&path, &recorder).expect("save");

    let gpu_model = model.valid();
    let arc = nn_encode::<Wgpu<f32, i32>>(&gpu_model, &gdev, test, 0);
    let dec = nn_decode::<Wgpu<f32, i32>>(&gpu_model, &gdev, &arc, test.len(), 0);
    println!(
        "[{label}] GPU: roundtrip {}  bpb {:.3}  ({} -> {} bytes)",
        dec == test,
        8.0 * arc.len() as f64 / test.len() as f64,
        test.len(),
        arc.len()
    );

    let cdev = NdArrayDevice::Cpu;
    let cpu_model = cfg
        .init::<NdArray<f32, i32>>(&cdev)
        .load_file(&path, &recorder, &cdev)
        .expect("load");
    let arc_c = nn_encode::<NdArray<f32, i32>>(&cpu_model, &cdev, test, 0);
    let dec_c = nn_decode::<NdArray<f32, i32>>(&cpu_model, &cdev, &arc_c, test.len(), 0);
    println!(
        "[{label}] CPU: roundtrip {}  bpb {:.3}",
        dec_c == test,
        8.0 * arc_c.len() as f64 / test.len() as f64
    );
}

fn main() {
    test_coder();
    let data = std::fs::read("assets/enwik8").expect("read enwik8");
    let test = &data[256 * 1024..256 * 1024 + 240]; // held-out slice
    let fp = GptConfig::new(VOCAB, 128, 4, 4, 512, 256).with_bit(false);
    let bit = GptConfig::new(VOCAB, 128, 4, 4, 512, 256).with_bit(true);
    train_and_eval(&data, test, &fp, "fp");
    train_and_eval(&data, test, &bit, "bitnet");
}
