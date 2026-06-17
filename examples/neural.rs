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

use std::time::Instant;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::module::{AutodiffModule, Module};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::Distribution;
use burn::tensor::IndexingUpdateOp;
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

/// Experts per token (the max/mask-the-max/max gate selection hardcodes 2).
const TOP_K: usize = 2;

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

    /// Switch-style load-balance auxiliary loss from the full softmax `gate`
    /// and the top-k selection `mask` (both `[N, ne]`): `ne · Σ_e f_e · P_e`,
    /// where `f_e` is the fraction of routing slots expert `e` won and `P_e`
    /// its mean gate probability. Equals 1.0 at perfect balance, `ne/k` at
    /// full collapse. Without it the router collapses — measured 2026-06-10 on
    /// the L6/32-expert run: layers 4–5 routed every token to the same 2
    /// experts (usage entropy 1.0 bit) and the loss stalled at the unigram
    /// floor, while the L4/24-expert run (no aux loss, lucky) stayed alive.
    fn balance_aux(gate: &Tensor<B, 2>, mask: &Tensor<B, 2>) -> Tensor<B, 1> {
        let [nn, ne] = gate.dims();
        let f = mask.clone().sum_dim(0) / (nn * TOP_K) as f32; // [1, ne]
        let p = gate.clone().mean_dim(0); // [1, ne]
        (f * p).sum() * ne as f32
    }

    /// Top-2 mixture-of-experts FFN. Training computes all experts densely and
    /// masks to the top-2 (simple + correct; the codec does the sparse gather).
    /// Gate = softmax over router logits, renormalized over the chosen experts.
    /// Returns `(output, balance_aux)`.
    fn moe(&self, n2: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 1>) {
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
        let ne = self.experts.len();
        let aux = Self::balance_aux(
            &gate.clone().reshape([b * t, ne]),
            &mask.clone().reshape([b * t, ne]),
        );
        let masked = gate * mask;
        let denom = masked.clone().sum_dim(2); // [b,t,1]
        let gnorm = masked / denom;
        let mut out = Tensor::<B, 3>::zeros([b, t, d], &n2.device());
        for (e, exp) in self.experts.iter().enumerate() {
            let oe = exp.ff2.forward(gelu(exp.ff1.forward(n2.clone()))); // [b,t,d]
            let ge = gnorm.clone().slice([0..b, 0..t, e..e + 1]); // [b,t,1]
            out = out + oe * ge;
        }
        (out, aux)
    }

    /// Sparse top-2 MoE: identical math to `moe`, but each expert runs only on
    /// the tokens routed to it (gathered via `argwhere`/`select`) and its output
    /// is scattered back — ~`n_experts/TOP_K`× less compute *and* retained
    /// activation than computing all experts densely. Returns `(output, aux)`.
    fn moe_sparse(&self, n2: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let [b, t, d] = n2.dims();
        let nn = b * t;
        let x = n2.reshape([nn, d]); // [N, d]
        let rlogits = x.clone().matmul(self.router.val().transpose()); // [N, ne]
        let gate = softmax(rlogits, 1);
        let max1 = gate.clone().max_dim(1);
        let is_max = gate.clone().equal(max1.clone());
        let without = gate.clone().mask_fill(is_max, f32::NEG_INFINITY);
        let kth = without.max_dim(1);
        let mask = gate.clone().greater_equal(kth).float();
        let aux = Self::balance_aux(&gate, &mask);
        let masked = gate.clone() * mask;
        let gnorm = masked.clone() / masked.sum_dim(1); // [N, ne]; 0 except top-2
        let mut out = Tensor::<B, 2>::zeros([nn, d], &x.device());
        for (e, exp) in self.experts.iter().enumerate() {
            let col = gnorm.clone().slice([0..nn, e..e + 1]).reshape([nn]); // [N] gate (0 if unused)
            let aw = col.clone().greater_elem(0.0).argwhere(); // [n_e, 1]
            let ne = aw.dims()[0];
            if ne == 0 {
                continue;
            }
            let idx = aw.reshape([ne]); // [n_e] token indices
            let x_e = x.clone().select(0, idx.clone()).reshape([1, ne, d]); // [1,n_e,d]
            let oe = exp.ff2.forward(gelu(exp.ff1.forward(x_e))).reshape([ne, d]); // [n_e,d]
            let g_e = col.select(0, idx.clone()).reshape([ne, 1]); // [n_e,1]
            out = out.select_assign(0, idx, oe * g_e, IndexingUpdateOp::Add);
        }
        (out.reshape([b, t, d]), aux)
    }

    /// Dense **batched** MoE: every expert's FFN runs as a single batched matmul
    /// over a stacked `[e, …]` weight tensor, so the dispatch count is O(1) in
    /// the expert count instead of `moe_sparse`'s per-expert gather/FFN/scatter
    /// (whose `argwhere` forces a GPU→CPU sync per expert — the suspected
    /// dispatch wall). Numerically identical to `moe` (all experts, gated by the
    /// top-2 mask); the trade is E× activation memory for the dense `[e,N,ffn]`
    /// hidden. Returns `(output, aux)`.
    fn moe_batched(&self, n2: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let [b, t, d] = n2.dims();
        let nn = b * t;
        let e = self.experts.len();
        let x = n2.reshape([nn, d]); // [N, d]
        let rlogits = x.clone().matmul(self.router.val().transpose()); // [N, e]
        let gate = softmax(rlogits, 1);
        let max1 = gate.clone().max_dim(1);
        let is_max = gate.clone().equal(max1);
        let without = gate.clone().mask_fill(is_max, f32::NEG_INFINITY);
        let kth = without.max_dim(1);
        let mask = gate.clone().greater_equal(kth).float();
        let aux = Self::balance_aux(&gate, &mask);
        let masked = gate.clone() * mask;
        let gnorm = masked.clone() / masked.sum_dim(1); // [N, e], 0 except top-2

        // stack expert weights -> [e, out, in], per-expert ternary STE
        let w1 = Tensor::stack::<3>(self.experts.iter().map(|x| x.ff1.weight.val()).collect(), 0);
        let w2 = Tensor::stack::<3>(self.experts.iter().map(|x| x.ff2.weight.val()).collect(), 0);
        let tern3 = |w: Tensor<B, 3>| {
            let [e, _, _] = w.dims();
            let g = w
                .clone()
                .abs()
                .mean_dim(2)
                .mean_dim(1)
                .reshape([e, 1, 1])
                .add_scalar(1e-5);
            let wq = w.clone().div(g.clone()).round().clamp(-1.0, 1.0).mul(g);
            w.clone() + (wq - w).detach()
        };
        let w1q = tern3(w1); // [e, ffn, d]
        let w2q = tern3(w2); // [e, d, ffn]

        // int8 activation STE on x (per token), broadcast to all experts
        let act = |z: Tensor<B, 3>| {
            let s = z.clone().abs().max_dim(2).div_scalar(127.0).clamp_min(1e-5);
            let zq = z.clone().div(s.clone()).round().clamp(-127.0, 127.0).mul(s);
            z.clone() + (zq - z).detach()
        };
        let xe = act(x.reshape([1, nn, d])).repeat_dim(0, e); // [e, N, d]
        let h = gelu(xe.matmul(w1q.swap_dims(1, 2))); // [e,N,d]@[e,d,ffn]=[e,N,ffn]
        let o = act(h).matmul(w2q.swap_dims(1, 2)); // [e,N,ffn]@[e,ffn,d]=[e,N,d]

        let w = gnorm.swap_dims(0, 1).reshape([e, nn, 1]); // [e, N, 1]
        let out = (o * w).sum_dim(0).reshape([b, t, d]);
        (out, aux)
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
    /// tokens [batch, seq] Int -> (logits [batch, seq, vocab], summed MoE
    /// balance aux across blocks — add `λ·aux` to the training loss, ignore at
    /// eval).
    fn forward(&self, tokens: Tensor<B, 2, Int>) -> (Tensor<B, 3>, Tensor<B, 1>) {
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
        let mut aux = Tensor::<B, 1>::zeros([1], &device);
        for blk in &self.blocks {
            x = x.clone() + blk.attn(blk.norm1.forward(x.clone()));
            let n2 = blk.norm2.forward(x.clone());
            let (mo, a) = blk.moe_sparse(n2);
            x = x + mo;
            aux = aux + a;
        }
        x = self.norm_f.forward(x);
        // tied unembedding: logits = x @ wq^T
        (x.matmul(wq.transpose().unsqueeze_dim(0)), aux) // [1,d,vocab] broadcast -> [b,t,vocab]
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

// Checkpoint trio written every 1K steps: the shippable ternary blob, an f32
// record (resumable, unlike the quantized blob), and a step sidecar so the LR
// schedule continues where it left off. Adam moments are not saved; they
// rebuild within ~100 steps of a resume. Smoke mode gets its own paths so it
// can never clobber a real run's artifacts.
struct CkptPaths {
    blob: &'static str,
    f32_rec: &'static str, // BinFileRecorder appends ".bin"
    step: &'static str,
}
const RUN_CKPT: CkptPaths = CkptPaths {
    blob: "assets/v8weights.bin",
    f32_rec: "checkpoints/v8_train_f32",
    step: "checkpoints/v8_train_step.txt",
};
const SMOKE_CKPT: CkptPaths = CkptPaths {
    blob: "checkpoints/smoke_weights.bin",
    f32_rec: "checkpoints/smoke_f32",
    step: "checkpoints/smoke_step.txt",
};

/// Warmup-stable-decay: 1K-step linear warmup to the peak, hold flat, then a
/// cosine decay to peak/10 over the final `DECAY` steps of the horizon. The
/// flat region's trajectory is horizon-independent, so the decay can be
/// triggered on demand when the loss curve plateaus: stop the run, set `steps`
/// to current+`DECAY`, and `-- resume` — the schedule stays continuous. (A
/// plain cosine locks the horizon at launch; changing it mid-run reshapes the
/// whole curve. The first enwik9 run held LR flat throughout; annealing
/// mattered in the prior v4/v5 MoE runs.)
/// Peak LR, overridable via `PEAK_LR` env for stall diagnosis (default 3e-4).
fn peak_lr() -> f64 {
    static PEAK: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *PEAK.get_or_init(|| {
        std::env::var("PEAK_LR")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3e-4)
    })
}

/// Warmup length, overridable via `WARMUP_STEPS` env (default 1000; 0 means
/// flat peak from step 0 — the regime the successful 24M enwik9 run used).
fn warmup_steps() -> usize {
    static W: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *W.get_or_init(|| {
        std::env::var("WARMUP_STEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1000)
    })
}

fn lr_at(step: usize, steps: usize) -> f64 {
    let peak = peak_lr();
    let warmup = warmup_steps();
    const DECAY: usize = 10_000;
    let min = peak / 10.0;
    if step < warmup {
        return peak * (step + 1) as f64 / warmup as f64;
    }
    let decay_start = steps.saturating_sub(DECAY).max(warmup);
    if step < decay_start {
        return peak;
    }
    let t = (step - decay_start) as f64 / (steps - decay_start).max(1) as f64;
    min + 0.5 * (peak - min) * (1.0 + (std::f64::consts::PI * t.min(1.0)).cos())
}

fn train<B: AutodiffBackend>(
    tokens: &[i32],
    cfg: &GptConfig,
    device: &B::Device,
    steps: usize,
    bpe: &bpe::Bpe,
    resume: bool,
    ckpt: &CkptPaths,
) -> Gpt<B> {
    let mut model = cfg.init::<B>(device);
    let mut start_step = 0usize;
    if resume {
        let rec = BinFileRecorder::<FullPrecisionSettings>::new();
        model = model
            .load_file(ckpt.f32_rec, &rec, device)
            .expect("resume: load f32 checkpoint");
        start_step = std::fs::read_to_string(ckpt.step)
            .expect("resume: read step sidecar")
            .trim()
            .parse()
            .expect("resume: parse step");
        println!("  resumed from step {start_step} ({}.bin)", ckpt.f32_rec);
    }
    let mut optim = AdamConfig::new().init();
    let (batch, ctx) = (64usize, cfg.ctx);
    // re-seed per resume point so a resumed run samples fresh batches
    let mut rng = Rng(0x2545_F491_4F6C_DD1D ^ (start_step as u64).wrapping_mul(0x9E37_79B9));
    for step in start_step..steps {
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
        let (logits, aux) = model.forward(inp_t);
        let [bb, tt, vv] = logits.dims();
        let ce = CrossEntropyLossConfig::new()
            .init(device)
            .forward(logits.reshape([bb * tt, vv]), tgt_t.reshape([bb * tt]));
        // λ·aux keeps the routers load-balanced (each block contributes 1.0 at
        // perfect balance; the step log prints aux/n_layers so balanced ≈ 1.00)
        const LAMBDA: f32 = 0.01;
        let lval = ce.clone().into_scalar().elem::<f32>();
        let aval = aux.clone().into_scalar().elem::<f32>();
        let loss = ce + aux.mul_scalar(LAMBDA);
        let grads = loss.backward();
        let gp = GradientsParams::from_grads(grads, &model);
        let lr = lr_at(step, steps);
        model = optim.step(lr, model, gp);
        if step % 50 == 0 || step + 1 == steps {
            println!(
                "  step {step:5} loss {lval:.4}  ({:.3} bits/token, lr {lr:.2e}, aux {:.2})",
                f64::from(lval) / std::f64::consts::LN_2,
                aval / cfg.n_layers as f32,
            );
        }
        if step > 0 && step % 1000 == 0 {
            let blob = pack_model(&model, cfg, bpe);
            std::fs::write(ckpt.blob, &blob).expect("checkpoint write");
            let rec = BinFileRecorder::<FullPrecisionSettings>::new();
            std::fs::create_dir_all("checkpoints").expect("checkpoints dir");
            model
                .clone()
                .save_file(ckpt.f32_rec, &rec)
                .expect("f32 checkpoint save");
            // sidecar holds *completed* steps, so a resume starts at the next one
            std::fs::write(ckpt.step, format!("{}\n", step + 1)).expect("step sidecar write");
            println!(
                "  [checkpoint] step {step}: {} KiB blob + f32 record written",
                blob.len() / 1024
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
    let (logits, _aux) = model.forward(t); // [1, len, vocab]
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
const BLOB_VERSION: u32 = 4; // trit-packed: 5 trits/byte ≈ 1.6 bits/weight

fn push_f32s(out: &mut Vec<u8>, vals: &[f32]) {
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn tensor_vec<B: Backend, const R: usize>(t: Tensor<B, R>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

/// Pack a row-major weight as scale f32 + trit-packed ternary (v4: 5 trits per
/// byte as little-endian base-3 digits, digit = trit + 1 — ~1.6 bits/weight),
/// replicating `ternary_ste`'s quantization exactly (scale = mean(|w|)+1e-5).
fn pack_ternary(out: &mut Vec<u8>, w: &[f32]) {
    let scale = (w.iter().map(|v| v.abs()).sum::<f32>() / w.len() as f32) + 1e-5;
    out.extend_from_slice(&scale.to_le_bytes());
    let mut byte = 0u8;
    let mut pow = 1u8;
    let mut k = 0u32;
    for &v in w {
        let digit = ((v / scale).round().clamp(-1.0, 1.0) as i8 + 1) as u8;
        byte += digit * pow;
        k += 1;
        if k == 5 {
            out.push(byte);
            byte = 0;
            pow = 1;
            k = 0;
        } else {
            pow *= 3;
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
    steps: usize,
    label: &str,
    resume: bool,
) {
    type GpuAd = Autodiff<Wgpu<f32, i32>>;
    let gdev = WgpuDevice::default();
    let nparams = cfg.init::<Wgpu<f32, i32>>(&gdev).num_params();
    println!(
        "[{label}] training {steps} steps on GPU ({nparams} params, vocab {})...",
        cfg.vocab
    );
    let model = train::<GpuAd>(train_toks, cfg, &gdev, steps, bpe, resume, &RUN_CKPT);

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

/// Verify the sparse MoE equals the dense MoE (same top-2 math) AND that its
/// autodiff backward runs (the `select_assign` gradient is the real unknown).
/// Aborts before any expensive training if either fails.
fn validate_moe() {
    type B = Autodiff<Wgpu<f32, i32>>;
    let dev = WgpuDevice::default();
    let cfg = GptConfig::new(2048, 64, 1, 4, 48, 16, 24).with_bit(true);
    let model = cfg.init::<B>(&dev);
    let blk = &model.blocks[0];
    let x = Tensor::<B, 3>::random([2, 8, 64], Distribution::Normal(0.0, 1.0), &dev);
    let (dense, daux) = blk.moe(x.clone());
    let (sparse, saux) = blk.moe_sparse(x.clone());
    let diff = (dense - sparse.clone())
        .abs()
        .max()
        .into_scalar()
        .elem::<f32>();
    let adiff = (daux.into_scalar().elem::<f32>() - saux.clone().into_scalar().elem::<f32>()).abs();
    println!("MoE validation: max|dense - sparse| = {diff:.3e}, |aux Δ| = {adiff:.3e}");
    assert!(diff < 1e-3, "sparse MoE diverges from dense ({diff})");
    assert!(
        adiff < 1e-4,
        "aux diverges between dense and sparse ({adiff})"
    );
    let _ = (sparse.sum() + saux).backward(); // panics if select_assign/aux have no backward
    println!("MoE validation: sparse forward matches dense, backward OK.");
}

/// Benchmark the MoE FFN implementations on wgpu: the per-expert sparse path
/// (`moe_sparse`, one `argwhere`+gather+FFN+scatter per expert) vs the dense
/// batched path (`moe_batched`, all experts in O(1)-in-E bmm dispatches). Times
/// forward and forward+backward at realistic token counts across expert counts,
/// so we can see whether the dispatch wall is real on this backend and whether
/// batching breaks it (and at what activation-memory cost). Verifies the two
/// agree numerically first.
fn moebench() {
    type B = Autodiff<Wgpu<f32, i32>>;
    let dev = WgpuDevice::default();
    let (d, ffn, batch, ctx) = (256usize, 384usize, 64usize, 256usize);
    println!(
        "MoE bench (wgpu): d={d} ffn={ffn} tokens={} (batch {batch} × ctx {ctx}), fwd+bwd, 20 iters",
        batch * ctx
    );
    let flush = |z: Tensor<B, 3>| {
        let _ = z.sum().into_scalar();
    };
    for e in [16usize, 32, 64] {
        let cfg = GptConfig::new(16384, d, 1, 8, ffn, ctx, e).with_bit(true);
        let model = cfg.init::<B>(&dev);
        let blk = &model.blocks[0];
        let x = Tensor::<B, 3>::random([batch, ctx, d], Distribution::Normal(0.0, 1.0), &dev);

        let (od, _) = blk.moe(x.clone());
        let (ob, _) = blk.moe_batched(x.clone());
        let diff = (od - ob).abs().max().into_scalar().elem::<f32>();

        let bench = |sparse: bool| -> f64 {
            for _ in 0..3 {
                let (o, a) = if sparse {
                    blk.moe_sparse(x.clone())
                } else {
                    blk.moe_batched(x.clone())
                };
                let _ = (o.sum() + a.sum()).backward();
            }
            let t = Instant::now();
            let mut last = None;
            for _ in 0..20 {
                let (o, a) = if sparse {
                    blk.moe_sparse(x.clone())
                } else {
                    blk.moe_batched(x.clone())
                };
                let g = (o.sum() + a.sum()).backward();
                last = Some(blk.router.grad(&g).unwrap());
            }
            let _ = last.unwrap().into_data(); // force queue completion
            t.elapsed().as_secs_f64() * 1000.0 / 20.0
        };

        let sp = bench(true);
        let ba = bench(false);
        flush(blk.moe_batched(x).0);
        println!(
            "  E={e:3}  |dense−batched|={diff:.1e}  sparse {sp:7.1} ms  batched {ba:7.1} ms  speedup {:.2}×",
            sp / ba
        );
    }
}

/// Exercise train → checkpoint → resume end-to-end on a tiny model (~2 min)
/// before trusting it for a multi-day run. In-process resume is the same disk
/// round-trip a fresh process would do.
fn smoke() {
    type GpuAd = Autodiff<Wgpu<f32, i32>>;
    let gdev = WgpuDevice::default();
    let data = std::fs::read("assets/enwik8").expect("read enwik8");
    let slice = &data[..2 * 1024 * 1024];
    let bpe = bpe::Bpe::learn(slice, 512);
    let toks: Vec<i32> = bpe.encode(slice).iter().map(|&t| t as i32).collect();
    // Diagnostic knobs (env): depth, width, and a single-phase step count —
    // used to reproduce the L6 unigram-shelf stall at minutes-scale.
    let env_us = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let (layers, d_model, steps) = (
        env_us("SMOKE_LAYERS", 2),
        env_us("SMOKE_D", 64),
        env_us("SMOKE_STEPS", 0),
    );
    let heads = (d_model / 16).max(2);
    let cfg = GptConfig::new(bpe.vocab_size(), d_model, layers, heads, 48, 64, 4).with_bit(true);
    if steps > 0 {
        let mut hist = vec![0u64; bpe.vocab_size()];
        for &t in &toks {
            hist[t as usize] += 1;
        }
        let n = toks.len() as f64;
        let unigram: f64 = hist
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / n;
                -p * p.log2()
            })
            .sum();
        println!(
            "[smoke] diagnostic: L{layers} d{d_model}, {steps} steps, peak {:.1e}, warmup {}; token unigram floor {unigram:.3} bits",
            peak_lr(),
            warmup_steps(),
        );
        let _ = train::<GpuAd>(&toks, &cfg, &gdev, steps, &bpe, false, &SMOKE_CKPT);
        return;
    }
    validate_moe();
    println!("[smoke] fresh run, 1100 steps (f32 checkpoint lands at 1000)...");
    let _ = train::<GpuAd>(&toks, &cfg, &gdev, 1100, &bpe, false, &SMOKE_CKPT);
    println!("[smoke] resuming from the step-1000 checkpoint, to 1200...");
    let _ = train::<GpuAd>(&toks, &cfg, &gdev, 1200, &bpe, true, &SMOKE_CKPT);
    println!("[smoke] resume path OK (loss should continue near its pre-resume level)");
}

fn main() {
    // `-- resume` continues from checkpoints/v8_train_f32.bin at the saved step
    // (same config + data required; shape mismatches fail loudly on load).
    // `-- smoke` runs the tiny train→checkpoint→resume exercise instead.
    let resume = std::env::args().any(|a| a == "resume");
    if std::env::args().any(|a| a == "smoke") {
        smoke();
        return;
    }
    if std::env::args().any(|a| a == "moebench") {
        moebench();
        return;
    }
    test_coder();
    // validate_moe() already confirmed sparse == dense + backward; skip its
    // ~6 min one-time shader compile on relaunches (training compiles anyway).
    let data = std::fs::read("assets/enwik9").expect("read enwik9");
    let n = data.len();
    let mb = 1024 * 1024;

    // First real run on the FULL enwik9. online-BPE (1K merges / MB) to 16K vocab,
    // then tokenize all of enwik9 in 16 MB chunks (a whole-file encode would need
    // >10 GB for the linked-list structures; chunked is ~hundreds of MB and the
    // boundary tokens are negligible). Memorization regime; eval on a prefix.
    let bpe = bpe::Bpe::learn(&data, 16384);
    let vocab = bpe.vocab_size();
    let mut train_toks: Vec<i32> = Vec::with_capacity(n / 3);
    let mut off = 0;
    while off < n {
        let end = (off + 16 * mb).min(n);
        train_toks.extend(bpe.encode(&data[off..end]).iter().map(|&t| t as i32));
        off = end;
    }
    let eval_bytes: Vec<u8> = data[..4096].to_vec();
    let eval_toks: Vec<i32> = bpe.encode(&eval_bytes).iter().map(|&t| t as i32).collect();
    drop(data); // free the 1 GB; only the token array is needed from here
    println!(
        "BPE: vocab {vocab}; enwik9 -> {} tokens ({:.2} bytes/token)",
        train_toks.len(),
        n as f64 / train_toks.len() as f64,
    );

    // The net-best architecture: d=256, 6 layers, 32 experts, expert-ffn 384
    // ≈ 43.6M total / 8.25M active → full-enwik9 net 1.4160 (curve point 2).
    // bpb-vs-params curve (full-enwik9 net): 24.2M → 1.873, 43.6M → 1.4160.
    // Curve point 3 (d256/L6/64e, ~81.4M) was tested 06-16/17 and came out
    // net-WORSE: its L(C) lead over E32 stayed flat at ~0.024 (1 MB slice, across
    // 21/30/40K) — below the +0.06 trit-packed L(D) the extra 38M params cost. So
    // more experts give diminishing L(C) that does not cover their L(D); 43.6M is
    // at/near the net optimum for this stack. Keep 32 unless a new lever changes
    // the L(C) slope (RoPE, hybrid arm).
    let bit = GptConfig::new(vocab, 256, 6, 8, 384, 256, 32).with_bit(true);
    // `steps` is the WSD horizon — a *maximum* (~8.8 epochs, ~3.3 days at
    // 2.4 s/step), not a commitment: the LR holds flat after warmup, so when
    // the loss curve plateaus we stop, set this to current+10K, and `-- resume`
    // to run just the decay tail. Chosen over 50K because the 24M run was still
    // improving at 37.8K and this model is 1.8× bigger with slower-converging
    // ternary-QAT + MoE dynamics.
    train_and_eval(
        &train_toks,
        &eval_bytes,
        &eval_toks,
        &bit,
        &bpe,
        120_000,
        "bitnet",
        resume,
    );
}
