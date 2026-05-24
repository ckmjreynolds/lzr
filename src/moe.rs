// Phase 29 WIP: the module is fully tested but no consumer in the codec
// path wires it up yet (that lands as the v4 codec scaffold). The
// module-level `dead_code` allow keeps the unused warnings out of
// `-Dwarnings` until then.
#![allow(dead_code)]

//! Sparse top-1 `MoE` byte-level decoder transformer for v4.
//!
//! Inference-only port of the `MoEByteTransformer` in
//! `lzr-neural/src/lzr_neural/moe.py`. Backbone identical to the
//! dense [`crate::transformer::ByteTransformer`] except each block's
//! FFN is replaced by a sparse top-1 routed `MoE` FFN: a per-block
//! router projects `d_model → n_experts`, top-1 `argmax` selects
//! exactly one expert per token, the chosen expert's FFN runs and
//! its output is scaled by the router's softmax probability. Active
//! per-token compute matches the dense baseline at the same backbone
//! shape — the cost is total params (and `L(D)`), not active FLOPs.
//!
//! Weights come from the `.lzrm` binary written by
//! `lzr-neural/scripts/export_moe_weights.py`. The format mirrors
//! `.lzrn` exactly except for the header (one extra `n_experts`
//! field) and the per-layer block layout (router weights followed
//! by per-expert `(fc1, fc2)` pairs instead of the single dense
//! `(fc1, fc2)`).
//!
//! See also `JOURNAL.md` 2026-05-15 Phase 29 for the standalone-bpb
//! result that motivates this port.

use anyhow::{Result, bail};

use crate::transformer::{
    gelu_inplace, matmul_x_w_t, read_tensor, rmsnorm_inplace, softmax_inplace,
};

/// Magic header bytes: `'LZRM'` little-endian. Distinct from `.lzrn`
/// (`'LZRN'`) so a wrong-format load fails fast at the magic check
/// instead of partway through tensor reads.
const MAGIC: u32 = 0x4C5A_524D;
const VERSION: u32 = 1;

/// Bit-width choice for weight storage in the shipped binary. The
/// `.lzrm` file on disk is always `f32`; the bit width affects what
/// gets committed to a future shipped binary (and the simulated
/// precision applied at load time via [`MoeByteTransformer::apply_quantization`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Quantization {
    F32,
    /// Per-tensor symmetric int8. One scale per whole tensor; biased
    /// by per-tensor max-abs outliers.
    Int8,
    /// Per-tensor symmetric int4. As Int8 but 4-bit; near-broken on
    /// our model at this scale.
    Int4,
    /// Per-row (per-output-channel) symmetric int8. Each 2D weight
    /// tensor gets one scale per output row, computed from that row's
    /// max-abs. 1D tensors (norms) fall back to per-tensor since
    /// per-channel on 1D is a no-op. Adds a tiny scales-overhead to
    /// `L(D)` (~93 KB total for the current architecture, <0.001 bpb).
    Int8Ch,
    /// Per-row symmetric int4. Same shape handling as `Int8Ch`.
    Int4Ch,
    /// **Mixed precision**: FFN expert weights at per-channel int4,
    /// attention / router / embedding weights at per-channel int8,
    /// norms left at f32. Targets the `L(D)` sweet spot where the
    /// bulk of params (FFN) get aggressive quant but precision-
    /// sensitive small tensors stay safe. Per-row scale overhead is
    /// real for int4 (~6% on small rows) and tracked exactly by
    /// [`MoeByteTransformer::shipped_bytes`].
    Mixed4,
    /// Mixed3: same scheme as `Mixed4` but FFN at per-channel int3.
    /// Maximum aggressive L(D) saving (saves ~0.017 bpb vs Mixed4 on
    /// the wider model); risks larger L(C) hit since 3 bits is at
    /// the edge of what naive symmetric quant can represent.
    Mixed3,
    /// Mixed5: same scheme as `Mixed4` but FFN at per-channel int5.
    /// Less aggressive L(D) saving than Mixed4; safer L(C). Use as a
    /// fallback point if Mixed3 / Mixed4 both lose L(C) more than
    /// they save L(D).
    Mixed5,
    /// Same scheme as `Mixed5` but with **asymmetric** per-channel
    /// int5 on the FFN — each row gets both a f32 scale and a u8
    /// zero-point. Recovers precision when a row's weight
    /// distribution isn't centered at zero (post-training FFN
    /// weights often have a small bias). Per-row overhead +1 byte vs
    /// symmetric (the u8 zero-point), <1% of the row's quant cost
    /// on our architectures.
    Mixed5Asym,
}

impl Quantization {
    /// Parse from a `LZR_MOE_QUANT` env-var value. Unset / unrecognized
    /// returns `F32` so default behavior is unchanged.
    pub(crate) fn from_env(var: &str) -> Self {
        match std::env::var(var).ok().as_deref() {
            Some("int8") => Self::Int8,
            Some("int4") => Self::Int4,
            Some("int8ch") => Self::Int8Ch,
            Some("int4ch") => Self::Int4Ch,
            Some("mixed4") => Self::Mixed4,
            Some("mixed3") => Self::Mixed3,
            Some("mixed5") => Self::Mixed5,
            Some("mixed5asym") => Self::Mixed5Asym,
            _ => Self::F32,
        }
    }

    /// Bytes per parameter at this bit width — used by the legacy
    /// `L(D)` estimator. Per-channel variants understate slightly
    /// (ignore per-row scale overhead, ~3% on our architecture).
    /// Returns `None` for mixed-precision schemes whose bytes-per-
    /// param isn't uniform — callers should use
    /// [`MoeByteTransformer::shipped_bytes`] for those.
    pub(crate) const fn bytes_per_param(self) -> Option<f64> {
        match self {
            Self::F32 => Some(4.0),
            Self::Int8 | Self::Int8Ch => Some(1.0),
            Self::Int4 | Self::Int4Ch => Some(0.5),
            Self::Mixed4 | Self::Mixed3 | Self::Mixed5 | Self::Mixed5Asym => None,
        }
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
            Self::Int8Ch => "int8ch",
            Self::Int4Ch => "int4ch",
            Self::Mixed4 => "mixed4",
            Self::Mixed3 => "mixed3",
            Self::Mixed5 => "mixed5",
            Self::Mixed5Asym => "mixed5asym",
        }
    }

    /// FFN bit-width for the mixed schemes (panics for non-mixed).
    const fn ffn_bits(self) -> u32 {
        match self {
            Self::Mixed3 => 3,
            Self::Mixed4 => 4,
            Self::Mixed5 | Self::Mixed5Asym => 5,
            _ => panic!("ffn_bits called on non-mixed variant"),
        }
    }

    /// Whether the FFN uses asymmetric (range-based scale + zero-point)
    /// quantization. Currently only `Mixed5Asym`.
    const fn ffn_asymmetric(self) -> bool {
        matches!(self, Self::Mixed5Asym)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MoeConfig {
    pub(crate) n_layer: usize,
    pub(crate) n_head: usize,
    pub(crate) d_model: usize,
    pub(crate) d_ff: usize,
    pub(crate) context: usize,
    pub(crate) vocab_size: usize,
    pub(crate) n_experts: usize,
}

impl MoeConfig {
    pub(crate) const fn head_dim(&self) -> usize {
        self.d_model / self.n_head
    }
}

#[derive(Debug)]
pub(crate) struct ExpertWeights {
    fc1_w: Vec<f32>, // [d_ff, d_model]
    fc2_w: Vec<f32>, // [d_model, d_ff]
}

#[allow(clippy::struct_field_names)]
#[derive(Debug)]
pub(crate) struct MoeBlock {
    norm1_w: Vec<f32>,           // [d_model]
    qkv_w: Vec<f32>,             // [3 * d_model, d_model]
    proj_w: Vec<f32>,            // [d_model, d_model]
    norm2_w: Vec<f32>,           // [d_model]
    router_w: Vec<f32>,          // [n_experts, d_model]
    experts: Vec<ExpertWeights>, // len == n_experts
}

/// Phase 50C Day-4 follow-up: per-token routing trace produced by
/// [`MoeByteTransformer::dump_routing`]. All vectors are
/// `[n_layers, n_tokens]` row-major (layer-major).
///
/// Used by `lzr routing-analyze` to characterize what each expert
/// is doing — load distribution, routing confidence, byte-class
/// specialization. Intentionally minimal: only the per-token chosen
/// expert, the softmax probability of that choice, and the full
/// router-distribution entropy. The full 32-way prob distribution
/// is not stored (would balloon the dump 16× for marginal extra
/// analytic value).
#[derive(Debug)]
pub(crate) struct RoutingTrace {
    pub(crate) n_layers: usize,
    pub(crate) n_tokens: usize,
    pub(crate) n_experts: usize,
    /// `[n_layers, n_tokens]`. Expert index chosen by argmax at each (layer, position).
    pub(crate) expert_assignment: Vec<u32>,
    /// `[n_layers, n_tokens]`. Softmax probability of the chosen expert (gate value).
    pub(crate) gate_val: Vec<f32>,
    /// `[n_layers, n_tokens]`. Shannon entropy of the router softmax in nats.
    pub(crate) router_entropy_nats: Vec<f32>,
}

impl RoutingTrace {
    pub(crate) fn new(n_layers: usize, n_tokens: usize, n_experts: usize) -> Self {
        let total = n_layers * n_tokens;
        Self {
            n_layers,
            n_tokens,
            n_experts,
            expert_assignment: vec![0; total],
            gate_val: vec![0.0; total],
            router_entropy_nats: vec![0.0; total],
        }
    }
}

#[derive(Debug)]
pub(crate) struct MoeByteTransformer {
    pub(crate) cfg: MoeConfig,
    tok_emb: Vec<f32>, // [vocab_size, d_model]
    pos_emb: Vec<f32>, // [context, d_model]
    blocks: Vec<MoeBlock>,
    norm_f: Vec<f32>, // [d_model]
    /// Optional int8 packed cache of all matmul-side weights (Phase
    /// 50A). When `Some(...)`, [`MoeByteTransformer::forward_step`]
    /// dispatches to the integer path. Built by
    /// [`MoeByteTransformer::prepare_int_cache`] after q-dq.
    int_cache: Option<IntCache>,
}

#[derive(Debug)]
struct IntExpertCache {
    fc1: crate::int_inference::IntTensor,
    fc2: crate::int_inference::IntTensor,
}

#[derive(Debug)]
struct IntBlockCache {
    qkv: crate::int_inference::IntTensor,
    proj: crate::int_inference::IntTensor,
    router: crate::int_inference::IntTensor,
    experts: Vec<IntExpertCache>,
}

#[derive(Debug)]
struct IntCache {
    /// `tok_emb` packed as a single tensor (rows = vocab, cols = `d_model`).
    /// Used both for the final vocab projection (matmul) and the
    /// embedding lookup (dequant a row on demand).
    tok_emb: crate::int_inference::IntTensor,
    blocks: Vec<IntBlockCache>,
}

impl MoeByteTransformer {
    /// Parse a `.lzrm` byte buffer. Layout, in declaration order:
    ///
    /// ```text
    ///   header (36 bytes): magic, version, n_layer, n_head, d_model,
    ///                      d_ff, context, vocab_size, n_experts (all u32 LE)
    ///   tok_emb            [vocab_size, d_model]
    ///   pos_emb            [context, d_model]
    ///   for each layer:
    ///     norm1            [d_model]
    ///     qkv              [3*d_model, d_model]
    ///     proj             [d_model, d_model]
    ///     norm2            [d_model]
    ///     router           [n_experts, d_model]
    ///     for each expert:
    ///       fc1            [d_ff, d_model]
    ///       fc2            [d_model, d_ff]
    ///   norm_f             [d_model]
    /// ```
    pub(crate) fn load_lzrm(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 36 {
            bail!("lzrm buffer too short for header: {}", bytes.len());
        }
        let cfg = read_header(&bytes[..36])?;
        let mut cursor = 36usize;

        let tok_emb = read_tensor(bytes, &mut cursor, cfg.vocab_size * cfg.d_model)?;
        let pos_emb = read_tensor(bytes, &mut cursor, cfg.context * cfg.d_model)?;

        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            let norm1_w = read_tensor(bytes, &mut cursor, cfg.d_model)?;
            let qkv_w = read_tensor(bytes, &mut cursor, 3 * cfg.d_model * cfg.d_model)?;
            let proj_w = read_tensor(bytes, &mut cursor, cfg.d_model * cfg.d_model)?;
            let norm2_w = read_tensor(bytes, &mut cursor, cfg.d_model)?;
            let router_w = read_tensor(bytes, &mut cursor, cfg.n_experts * cfg.d_model)?;
            let mut experts = Vec::with_capacity(cfg.n_experts);
            for _ in 0..cfg.n_experts {
                let fc1_w = read_tensor(bytes, &mut cursor, cfg.d_ff * cfg.d_model)?;
                let fc2_w = read_tensor(bytes, &mut cursor, cfg.d_model * cfg.d_ff)?;
                experts.push(ExpertWeights { fc1_w, fc2_w });
            }
            blocks.push(MoeBlock {
                norm1_w,
                qkv_w,
                proj_w,
                norm2_w,
                router_w,
                experts,
            });
        }

        let norm_f = read_tensor(bytes, &mut cursor, cfg.d_model)?;
        if cursor != bytes.len() {
            bail!(
                "lzrm buffer has {} trailing bytes after parse",
                bytes.len() - cursor
            );
        }
        Ok(Self {
            cfg,
            tok_emb,
            pos_emb,
            blocks,
            norm_f,
            int_cache: None,
        })
    }

    /// Apply per-tensor symmetric quantize-then-dequantize in place to
    /// every weight tensor. This is a dev-time stand-in for actually
    /// shipping the binary at this bit width — the inference kernels
    /// still see `f32`, but the values are restricted to the grid that
    /// `bits`-bit quantization would produce. Lets us measure the
    /// `L(C)` cost of any `L(D)`-saving quantization choice without
    /// touching the `.lzrm` format or the matmul kernels.
    ///
    /// `Quantization::F32` is a no-op. `Int8` uses 8-bit symmetric,
    /// `Int4` uses 4-bit symmetric. Each tensor (embeddings, attention
    /// projections, router, every expert FFN, norms) gets its own
    /// per-tensor scale from its max-abs.
    pub(crate) fn apply_quantization(&mut self, q: Quantization) {
        if matches!(q, Quantization::F32) {
            return;
        }
        let cfg = self.cfg;
        let d = cfg.d_model;

        if matches!(
            q,
            Quantization::Mixed4
                | Quantization::Mixed3
                | Quantization::Mixed5
                | Quantization::Mixed5Asym
        ) {
            // FFN expert weights → per-channel at `ffn_bits` bits
            // (symmetric or asymmetric depending on variant);
            // everything else (attn / router / embeddings) →
            // per-channel int8 symmetric; norms stay at f32 since
            // they're tiny and high-precision matters.
            let ffn_bits = q.ffn_bits();
            let ffn_asym = q.ffn_asymmetric();
            let q_ffn = |w: &mut Vec<f32>, rows: usize, cols: usize| {
                if ffn_asym {
                    quantize_dequantize_per_channel_asym(w, rows, cols, ffn_bits);
                } else {
                    quantize_dequantize_per_channel(w, rows, cols, ffn_bits);
                }
            };
            quantize_dequantize_per_channel(&mut self.tok_emb, cfg.vocab_size, d, 8);
            quantize_dequantize_per_channel(&mut self.pos_emb, cfg.context, d, 8);
            for block in &mut self.blocks {
                quantize_dequantize_per_channel(&mut block.qkv_w, 3 * d, d, 8);
                quantize_dequantize_per_channel(&mut block.proj_w, d, d, 8);
                quantize_dequantize_per_channel(&mut block.router_w, cfg.n_experts, d, 8);
                for expert in &mut block.experts {
                    q_ffn(&mut expert.fc1_w, cfg.d_ff, d);
                    q_ffn(&mut expert.fc2_w, d, cfg.d_ff);
                }
            }
            return;
        }

        let (bits, per_channel) = match q {
            Quantization::F32
            | Quantization::Mixed4
            | Quantization::Mixed3
            | Quantization::Mixed5
            | Quantization::Mixed5Asym => {
                unreachable!("handled above")
            }
            Quantization::Int8 => (8, false),
            Quantization::Int4 => (4, false),
            Quantization::Int8Ch => (8, true),
            Quantization::Int4Ch => (4, true),
        };
        // 2D tensor: [rows, cols], per-channel quant is per-row.
        let q2d = |w: &mut Vec<f32>, rows: usize, cols: usize| {
            if per_channel {
                quantize_dequantize_per_channel(w, rows, cols, bits);
            } else {
                quantize_dequantize_inplace(w, bits);
            }
        };
        // 1D tensor: always per-tensor (per-channel on 1D is a no-op
        // since each element would be its own "channel").
        let q1d = |w: &mut Vec<f32>| {
            quantize_dequantize_inplace(w, bits);
        };

        q2d(&mut self.tok_emb, cfg.vocab_size, d);
        q2d(&mut self.pos_emb, cfg.context, d);
        for block in &mut self.blocks {
            q1d(&mut block.norm1_w);
            q2d(&mut block.qkv_w, 3 * d, d);
            q2d(&mut block.proj_w, d, d);
            q1d(&mut block.norm2_w);
            q2d(&mut block.router_w, cfg.n_experts, d);
            for expert in &mut block.experts {
                q2d(&mut expert.fc1_w, cfg.d_ff, d);
                q2d(&mut expert.fc2_w, d, cfg.d_ff);
            }
        }
        q1d(&mut self.norm_f);
    }

    /// Compute the exact shipped-weight byte count this model would
    /// pay under the given `Quantization`, including per-row scale
    /// overhead for per-channel and mixed schemes. Authoritative
    /// source for `L(D)` projection — callers should use this rather
    /// than `n_params × Quantization::bytes_per_param`.
    ///
    /// 1D tensors (norms): always per-tensor — one f32 scale per
    /// tensor (or f32 storage in `Mixed4`).
    /// 2D tensors at per-channel: `rows × (cols × bits/8 + 4)` bytes.
    /// 2D tensors at per-tensor: `rows × cols × bits/8 + 4` bytes.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub(crate) fn shipped_bytes(&self, q: Quantization) -> u64 {
        // Helper: bytes for a 2D tensor of shape [rows, cols] at bits-bit
        // per-channel (one f32 scale per row).
        let per_channel_2d = |rows: usize, cols: usize, bits: u32| -> u64 {
            let weight_bytes = (rows as u64) * ((cols as u64) * u64::from(bits)).div_ceil(8);
            let scale_bytes = (rows as u64) * 4;
            weight_bytes + scale_bytes
        };
        // Helper: bytes for a flat tensor of `n` elements at bits-bit
        // per-tensor (single f32 scale shared).
        let per_tensor = |n: usize, bits: u32| -> u64 {
            let weight_bytes = ((n as u64) * u64::from(bits)).div_ceil(8);
            weight_bytes + 4
        };
        // Helper: bytes for a flat tensor stored as f32 (no quant).
        let f32_bytes = |n: usize| -> u64 { (n as u64) * 4 };

        let cfg = self.cfg;
        let d = cfg.d_model;
        match q {
            Quantization::F32 => f32_bytes(self.total_params()),
            Quantization::Int8 => per_tensor(self.total_params(), 8),
            Quantization::Int4 => per_tensor(self.total_params(), 4),
            Quantization::Int8Ch | Quantization::Int4Ch => {
                let bits = if matches!(q, Quantization::Int8Ch) {
                    8
                } else {
                    4
                };
                let mut b = 0_u64;
                b += per_channel_2d(cfg.vocab_size, d, bits);
                b += per_channel_2d(cfg.context, d, bits);
                for block in &self.blocks {
                    b += per_tensor(block.norm1_w.len(), bits);
                    b += per_channel_2d(3 * d, d, bits);
                    b += per_channel_2d(d, d, bits);
                    b += per_tensor(block.norm2_w.len(), bits);
                    b += per_channel_2d(cfg.n_experts, d, bits);
                    for _ in &block.experts {
                        b += per_channel_2d(cfg.d_ff, d, bits);
                        b += per_channel_2d(d, cfg.d_ff, bits);
                    }
                }
                b += per_tensor(self.norm_f.len(), bits);
                b
            }
            Quantization::Mixed4
            | Quantization::Mixed3
            | Quantization::Mixed5
            | Quantization::Mixed5Asym => {
                // FFN per-channel int{3,4,5} (symmetric or
                // asymmetric); attn/router/emb per-channel int8;
                // norms f32.
                // Asymmetric adds 1 u8 zero-point per row on top of
                // the 4-byte f32 scale, so per-channel rows cost
                // `cols × bits / 8 + 5` instead of `+ 4`.
                let ffn_bits = q.ffn_bits();
                let ffn_per_row_overhead = if q.ffn_asymmetric() { 5_u64 } else { 4_u64 };
                let ffn_per_channel_2d = |rows: usize, cols: usize| -> u64 {
                    let weight_bytes =
                        (rows as u64) * ((cols as u64) * u64::from(ffn_bits)).div_ceil(8);
                    let scale_bytes = (rows as u64) * ffn_per_row_overhead;
                    weight_bytes + scale_bytes
                };
                let mut b = 0_u64;
                b += per_channel_2d(cfg.vocab_size, d, 8);
                b += per_channel_2d(cfg.context, d, 8);
                for block in &self.blocks {
                    b += f32_bytes(block.norm1_w.len());
                    b += per_channel_2d(3 * d, d, 8);
                    b += per_channel_2d(d, d, 8);
                    b += f32_bytes(block.norm2_w.len());
                    b += per_channel_2d(cfg.n_experts, d, 8);
                    for _ in &block.experts {
                        b += ffn_per_channel_2d(cfg.d_ff, d);
                        b += ffn_per_channel_2d(d, cfg.d_ff);
                    }
                }
                b += f32_bytes(self.norm_f.len());
                b
            }
        }
    }

    /// Total trainable parameter count — sum of every weight tensor's
    /// length. Used by callers to project shipped weight size at any
    /// bit width.
    pub(crate) fn total_params(&self) -> usize {
        let mut n = self.tok_emb.len() + self.pos_emb.len() + self.norm_f.len();
        for block in &self.blocks {
            n += block.norm1_w.len()
                + block.qkv_w.len()
                + block.proj_w.len()
                + block.norm2_w.len()
                + block.router_w.len();
            for expert in &block.experts {
                n += expert.fc1_w.len() + expert.fc2_w.len();
            }
        }
        n
    }

    /// Streaming forward step. One token in, `[vocab_size]` next-byte
    /// logits out. Per-token cost is identical to the dense
    /// transformer at the same backbone — only one expert FFN runs
    /// (chosen by router `argmax`), at the dense FFN's exact shape.
    /// The router itself is a single `[d_model → n_experts]` matvec
    /// per layer, negligible vs the FFN cost (~0.3% at `d_ff=256`,
    /// `n_experts=8`, `d_model=96`).
    #[allow(
        clippy::many_single_char_names,
        clippy::needless_range_loop,
        clippy::cast_precision_loss,
        clippy::suboptimal_flops,
        clippy::similar_names
    )]
    pub(crate) fn forward_step(&self, cache: &mut MoeKvCache, token: u32) -> Vec<f32> {
        let cfg = &self.cfg;
        assert_eq!(cache.layers.len(), cfg.n_layer);
        assert!(
            cache.pos < cfg.context,
            "MoE KV cache full at pos={} (context={}); caller must reset between chunks",
            cache.pos,
            cfg.context
        );
        assert!(
            (token as usize) < cfg.vocab_size,
            "token {token} out of vocab (size {})",
            cfg.vocab_size
        );
        let d = cfg.d_model;
        let h = cfg.n_head;
        let hd = cfg.head_dim();
        let p = cache.pos;

        let mut x = vec![0f32; d];
        let tok_row = &self.tok_emb[token as usize * d..(token as usize + 1) * d];
        let pos_row = &self.pos_emb[p * d..(p + 1) * d];
        for i in 0..d {
            x[i] = tok_row[i] + pos_row[i];
        }

        for (l, block) in self.blocks.iter().enumerate() {
            // Attention residual branch — identical to dense
            // transformer's forward_step; the only difference is the
            // FFN below uses MoE routing.
            let mut x_norm = vec![0f32; d];
            x_norm.copy_from_slice(&x);
            rmsnorm_inplace(&mut x_norm, &block.norm1_w);

            let mut qkv = vec![0f32; 3 * d];
            matmul_x_w_t(&x_norm, &block.qkv_w, &mut qkv, 1, d, 3 * d);

            let layer_cache = &mut cache.layers[l];
            layer_cache.k[p * d..(p + 1) * d].copy_from_slice(&qkv[d..2 * d]);
            layer_cache.v[p * d..(p + 1) * d].copy_from_slice(&qkv[2 * d..3 * d]);

            let mut attn_out = vec![0f32; d];
            for head in 0..h {
                let q_h = &qkv[head * hd..head * hd + hd];
                let scale = 1.0 / (hd as f32).sqrt();
                let mut scores = vec![0f32; p + 1];
                for j in 0..=p {
                    let kj = &layer_cache.k[j * d + head * hd..j * d + head * hd + hd];
                    let mut s = 0f32;
                    for i in 0..hd {
                        s += q_h[i] * kj[i];
                    }
                    scores[j] = s * scale;
                }
                softmax_inplace(&mut scores);

                let out_h = &mut attn_out[head * hd..head * hd + hd];
                for j in 0..=p {
                    let vj = &layer_cache.v[j * d + head * hd..j * d + head * hd + hd];
                    let s_j = scores[j];
                    for i in 0..hd {
                        out_h[i] += s_j * vj[i];
                    }
                }
            }

            let mut proj_out = vec![0f32; d];
            matmul_x_w_t(&attn_out, &block.proj_w, &mut proj_out, 1, d, d);
            for i in 0..d {
                x[i] += proj_out[i];
            }

            // MoE FFN residual branch.
            x_norm.copy_from_slice(&x);
            rmsnorm_inplace(&mut x_norm, &block.norm2_w);

            // Router: linear d_model → n_experts, then softmax → top-1.
            let mut router_logits = vec![0f32; cfg.n_experts];
            matmul_x_w_t(
                &x_norm,
                &block.router_w,
                &mut router_logits,
                1,
                d,
                cfg.n_experts,
            );
            let mut router_probs = router_logits.clone();
            softmax_inplace(&mut router_probs);
            let (expert_idx, gate_val) = argmax_with_value(&router_probs);

            // One expert's FFN, weighted by its router probability.
            let expert = &block.experts[expert_idx];
            let mut ff_hidden = vec![0f32; cfg.d_ff];
            matmul_x_w_t(&x_norm, &expert.fc1_w, &mut ff_hidden, 1, d, cfg.d_ff);
            gelu_inplace(&mut ff_hidden);

            let mut ff_out = vec![0f32; d];
            matmul_x_w_t(&ff_hidden, &expert.fc2_w, &mut ff_out, 1, cfg.d_ff, d);
            for i in 0..d {
                x[i] += ff_out[i] * gate_val;
            }
        }

        rmsnorm_inplace(&mut x, &self.norm_f);
        let mut logits = vec![0f32; cfg.vocab_size];
        matmul_x_w_t(&x, &self.tok_emb, &mut logits, 1, d, cfg.vocab_size);

        cache.pos += 1;
        logits
    }

    /// Pack every matmul-side weight tensor into per-channel-int8.
    /// Idempotent. After this call `int_cache.is_some()` and
    /// [`MoeByteTransformer::forward_step`] will dispatch to the
    /// integer path.
    /// True once `prepare_int_cache` has been called and the integer
    /// inference path is ready. Used by `MoeArm::precompute_for_encode`
    /// to gate batched-encode mode entry.
    pub(crate) const fn int_cache_ready(&self) -> bool {
        self.int_cache.is_some()
    }

    pub(crate) fn prepare_int_cache(&mut self) {
        use crate::int_inference::IntTensor;
        if self.int_cache.is_some() {
            return;
        }
        let cfg = self.cfg;
        let d = cfg.d_model;
        let tok_emb = IntTensor::pack_per_channel_sym(&self.tok_emb, cfg.vocab_size, d, 8);
        let blocks = self
            .blocks
            .iter()
            .map(|b| IntBlockCache {
                qkv: IntTensor::pack_per_channel_sym(&b.qkv_w, 3 * d, d, 8),
                proj: IntTensor::pack_per_channel_sym(&b.proj_w, d, d, 8),
                router: IntTensor::pack_per_channel_sym(&b.router_w, cfg.n_experts, d, 8),
                experts: b
                    .experts
                    .iter()
                    .map(|e| IntExpertCache {
                        fc1: IntTensor::pack_per_channel_sym(&e.fc1_w, cfg.d_ff, d, 8),
                        fc2: IntTensor::pack_per_channel_sym(&e.fc2_w, d, cfg.d_ff, 8),
                    })
                    .collect(),
            })
            .collect();
        self.int_cache = Some(IntCache { tok_emb, blocks });
    }

    /// Integer-path forward step (Phase 50A). Same API as
    /// [`MoeByteTransformer::forward_step`] but every matmul runs
    /// through int8 × int8 → int32 → f32 with dynamic per-token
    /// activation quantization. Element-wise ops (rmsnorm, softmax,
    /// gelu) and the attention dot-products stay in f32 because they
    /// are precision-sensitive or too small to benefit. Requires
    /// [`MoeByteTransformer::prepare_int_cache`] to have been called.
    #[allow(
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::needless_range_loop,
        clippy::too_many_lines,
        clippy::suboptimal_flops,
        clippy::cast_precision_loss
    )]
    pub(crate) fn forward_step_int(&self, cache: &mut MoeKvCache, token: u32) -> Vec<f32> {
        let ic = self
            .int_cache
            .as_ref()
            .expect("prepare_int_cache() must be called before forward_step_int");
        let cfg = &self.cfg;
        assert!(cache.pos < cfg.context);
        assert!((token as usize) < cfg.vocab_size);
        let d = cfg.d_model;
        let h = cfg.n_head;
        let hd = cfg.head_dim();
        let p = cache.pos;

        // Embedding: dequant a row of tok_emb on the fly + add pos_emb.
        let mut x = vec![0f32; d];
        let row = token as usize;
        let scale = ic.tok_emb.scales[row];
        let row_i8 = &ic.tok_emb.data[row * d..(row + 1) * d];
        let pos_row = &self.pos_emb[p * d..(p + 1) * d];
        for i in 0..d {
            x[i] = f32::from(row_i8[i]) * scale + pos_row[i];
        }

        let mut act_i8 = Vec::with_capacity(d.max(cfg.d_ff));
        for (l, block) in self.blocks.iter().enumerate() {
            let int_block = &ic.blocks[l];

            // Attention path.
            let mut x_norm = x.clone();
            rmsnorm_inplace(&mut x_norm, &block.norm1_w);

            // qkv: int matmul.
            let x_scale = crate::int_inference::quantize_act_i8(&x_norm, &mut act_i8);
            let mut qkv = vec![0f32; 3 * d];
            crate::int_inference::matmul_dispatch(&act_i8, x_scale, &int_block.qkv, &mut qkv);

            let layer_cache = &mut cache.layers[l];
            layer_cache.k[p * d..(p + 1) * d].copy_from_slice(&qkv[d..2 * d]);
            layer_cache.v[p * d..(p + 1) * d].copy_from_slice(&qkv[2 * d..3 * d]);

            // Attention dot products: stay f32 (tiny, precision-sensitive).
            let mut attn_out = vec![0f32; d];
            for head in 0..h {
                let q_h = &qkv[head * hd..head * hd + hd];
                let scale_attn = 1.0 / (hd as f32).sqrt();
                let mut scores = vec![0f32; p + 1];
                for j in 0..=p {
                    let kj = &layer_cache.k[j * d + head * hd..j * d + head * hd + hd];
                    let mut s = 0f32;
                    for i in 0..hd {
                        s += q_h[i] * kj[i];
                    }
                    scores[j] = s * scale_attn;
                }
                softmax_inplace(&mut scores);
                let out_h = &mut attn_out[head * hd..head * hd + hd];
                for j in 0..=p {
                    let vj = &layer_cache.v[j * d + head * hd..j * d + head * hd + hd];
                    let s_j = scores[j];
                    for i in 0..hd {
                        out_h[i] += s_j * vj[i];
                    }
                }
            }

            // proj: int matmul.
            let proj_scale = crate::int_inference::quantize_act_i8(&attn_out, &mut act_i8);
            let mut proj_out = vec![0f32; d];
            crate::int_inference::matmul_dispatch(
                &act_i8,
                proj_scale,
                &int_block.proj,
                &mut proj_out,
            );
            for i in 0..d {
                x[i] += proj_out[i];
            }

            // MoE FFN.
            x_norm.copy_from_slice(&x);
            rmsnorm_inplace(&mut x_norm, &block.norm2_w);

            // Router: int matmul.
            let r_scale = crate::int_inference::quantize_act_i8(&x_norm, &mut act_i8);
            let mut router_logits = vec![0f32; cfg.n_experts];
            crate::int_inference::matmul_dispatch(
                &act_i8,
                r_scale,
                &int_block.router,
                &mut router_logits,
            );
            let mut router_probs = router_logits.clone();
            softmax_inplace(&mut router_probs);
            let (expert_idx, gate_val) = argmax_with_value(&router_probs);

            // Expert FFN: two int matmuls, GELU between.
            let int_expert = &int_block.experts[expert_idx];
            let fc1_scale = crate::int_inference::quantize_act_i8(&x_norm, &mut act_i8);
            let mut ff_hidden = vec![0f32; cfg.d_ff];
            crate::int_inference::matmul_dispatch(
                &act_i8,
                fc1_scale,
                &int_expert.fc1,
                &mut ff_hidden,
            );
            gelu_inplace(&mut ff_hidden);

            let fc2_scale = crate::int_inference::quantize_act_i8(&ff_hidden, &mut act_i8);
            let mut ff_out = vec![0f32; d];
            crate::int_inference::matmul_dispatch(&act_i8, fc2_scale, &int_expert.fc2, &mut ff_out);
            for i in 0..d {
                x[i] += ff_out[i] * gate_val;
            }
        }

        rmsnorm_inplace(&mut x, &self.norm_f);
        let final_scale = crate::int_inference::quantize_act_i8(&x, &mut act_i8);
        let mut logits = vec![0f32; cfg.vocab_size];
        crate::int_inference::matmul_dispatch(&act_i8, final_scale, &ic.tok_emb, &mut logits);

        cache.pos += 1;
        logits
    }

    /// Phase 50C Day-4 follow-up: routing diagnostic. Runs the full
    /// batched forward and captures per-token routing decisions into
    /// a [`RoutingTrace`]. Used by `lzr routing-analyze` to
    /// characterize what each of the `n_experts` experts actually
    /// does — load distribution, routing entropy, and (with the
    /// codec's BPE table) byte-class × expert crosstabs.
    pub(crate) fn dump_routing(&self, tokens: &[u32]) -> RoutingTrace {
        let mut trace = RoutingTrace::new(self.cfg.n_layer, tokens.len(), self.cfg.n_experts);
        let _ = self.batched_forward_int_traced(tokens, Some(&mut trace));
        trace
    }

    /// Phase 50C Day-4 Phase-1 — CPU batched forward over a full
    /// token sequence, integer-inference path. Produces the same
    /// `[N * vocab_size]` logits a loop of
    /// [`MoeByteTransformer::forward_step_int`] would, but with the
    /// per-position work laid out so the heavy inner loops can be
    /// progressively replaced by batched GPU kernels in Day-4 Phase 2+.
    ///
    /// Bit-equivalence to the per-step path is the acceptance contract
    /// — every f32 reduction is computed with the same scalar
    /// summation order as `forward_step_int`, every per-token int8
    /// activation quant is identical, and the int8 GEMMs reuse
    /// `matmul_i8_i8_per_channel` (one call per position for now —
    /// the future GPU batched-GEMM kernel will produce the same i32
    /// accumulators so bit-exactness is preserved).
    ///
    /// Constraints: `tokens.len() <= cfg.context` (mirrors the
    /// per-step path's `cache.pos < cfg.context` assert). Callers
    /// chunk longer sequences themselves.
    /// Public entry point. Thin wrapper around
    /// [`MoeByteTransformer::batched_forward_int_traced`] with no
    /// routing capture — same signature and behavior as before
    /// Phase 50C Day-4 follow-up.
    pub(crate) fn batched_forward_int(&self, tokens: &[u32]) -> Vec<f32> {
        self.batched_forward_int_traced(tokens, None)
    }

    /// Shared implementation. When `trace` is `Some`, captures
    /// per-token routing decisions into it (used by `dump_routing`).
    /// Otherwise behaves identically to the plain
    /// `batched_forward_int`.
    #[allow(
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::needless_range_loop,
        clippy::too_many_lines,
        clippy::suboptimal_flops,
        clippy::cast_precision_loss
    )]
    pub(crate) fn batched_forward_int_traced(
        &self,
        tokens: &[u32],
        mut trace: Option<&mut RoutingTrace>,
    ) -> Vec<f32> {
        let ic = self
            .int_cache
            .as_ref()
            .expect("prepare_int_cache() must be called before batched_forward_int");
        let cfg = &self.cfg;
        let n = tokens.len();
        let d = cfg.d_model;
        let h = cfg.n_head;
        let hd = cfg.head_dim();
        let vocab = cfg.vocab_size;
        let n_experts = cfg.n_experts;
        let d_ff = cfg.d_ff;
        assert!(
            n <= cfg.context,
            "batched_forward_int: tokens.len()={n} exceeds context {} — caller must chunk",
            cfg.context
        );
        for &t in tokens {
            assert!((t as usize) < vocab, "token id {t} >= vocab_size {vocab}");
        }

        // Embedding: dequant tok_emb rows + add pos_emb, batched [N, d].
        let mut x = vec![0f32; n * d];
        for (i, &tok) in tokens.iter().enumerate() {
            let row = tok as usize;
            let scale = ic.tok_emb.scales[row];
            let row_i8 = &ic.tok_emb.data[row * d..(row + 1) * d];
            let pos_row = &self.pos_emb[i * d..(i + 1) * d];
            for c in 0..d {
                x[i * d + c] = f32::from(row_i8[c]) * scale + pos_row[c];
            }
        }

        // Per-layer K/V scratch, sized for one layer at a time (we
        // overwrite each layer's K/V into the same buffers).
        let mut k_buf = vec![0f32; n * d];
        let mut v_buf = vec![0f32; n * d];
        let mut act_i8 = Vec::with_capacity(d.max(d_ff));

        for (l, block) in self.blocks.iter().enumerate() {
            let int_block = &ic.blocks[l];

            // Pre-attention rmsnorm: per-position, same scalar reduction.
            let mut x_norm = x.clone();
            for i in 0..n {
                rmsnorm_inplace(&mut x_norm[i * d..(i + 1) * d], &block.norm1_w);
            }

            // qkv matmul: per-token act-quant + batched int8 GEMM (CPU or GPU).
            let mut qkv_x_i8 = vec![0i8; n * d];
            let mut qkv_x_scales = vec![0f32; n];
            for i in 0..n {
                qkv_x_scales[i] =
                    crate::int_inference::quantize_act_i8(&x_norm[i * d..(i + 1) * d], &mut act_i8);
                qkv_x_i8[i * d..(i + 1) * d].copy_from_slice(&act_i8);
            }
            let mut qkv = vec![0f32; n * 3 * d];
            crate::int_inference::batched_matmul_dispatch(
                &qkv_x_i8,
                &qkv_x_scales,
                &int_block.qkv,
                n,
                &mut qkv,
            );

            // Write K/V for this layer (used by attention below).
            for i in 0..n {
                k_buf[i * d..(i + 1) * d].copy_from_slice(&qkv[i * 3 * d + d..i * 3 * d + 2 * d]);
                v_buf[i * d..(i + 1) * d]
                    .copy_from_slice(&qkv[i * 3 * d + 2 * d..i * 3 * d + 3 * d]);
            }

            // Causal multi-head attention, per-position, same scalar
            // reductions as forward_step_int.
            let mut attn_out = vec![0f32; n * d];
            let scale_attn = 1.0 / (hd as f32).sqrt();
            for i in 0..n {
                for head in 0..h {
                    let q_h = &qkv[i * 3 * d + head * hd..i * 3 * d + head * hd + hd];
                    let mut scores = vec![0f32; i + 1];
                    for j in 0..=i {
                        let kj = &k_buf[j * d + head * hd..j * d + head * hd + hd];
                        let mut s = 0f32;
                        for c in 0..hd {
                            s += q_h[c] * kj[c];
                        }
                        scores[j] = s * scale_attn;
                    }
                    softmax_inplace(&mut scores);
                    let out_h = &mut attn_out[i * d + head * hd..i * d + head * hd + hd];
                    for j in 0..=i {
                        let vj = &v_buf[j * d + head * hd..j * d + head * hd + hd];
                        let s_j = scores[j];
                        for c in 0..hd {
                            out_h[c] += s_j * vj[c];
                        }
                    }
                }
            }

            // proj matmul: batched int8 GEMM, then residual add.
            let mut proj_x_i8 = vec![0i8; n * d];
            let mut proj_x_scales = vec![0f32; n];
            for i in 0..n {
                proj_x_scales[i] = crate::int_inference::quantize_act_i8(
                    &attn_out[i * d..(i + 1) * d],
                    &mut act_i8,
                );
                proj_x_i8[i * d..(i + 1) * d].copy_from_slice(&act_i8);
            }
            let mut proj_out = vec![0f32; n * d];
            crate::int_inference::batched_matmul_dispatch(
                &proj_x_i8,
                &proj_x_scales,
                &int_block.proj,
                n,
                &mut proj_out,
            );
            for i in 0..n * d {
                x[i] += proj_out[i];
            }

            // Pre-FFN rmsnorm.
            let mut x_norm2 = x.clone();
            for i in 0..n {
                rmsnorm_inplace(&mut x_norm2[i * d..(i + 1) * d], &block.norm2_w);
            }

            // Router: batched int8 GEMM, then per-token softmax+argmax.
            let mut router_x_i8 = vec![0i8; n * d];
            let mut router_x_scales = vec![0f32; n];
            for i in 0..n {
                router_x_scales[i] = crate::int_inference::quantize_act_i8(
                    &x_norm2[i * d..(i + 1) * d],
                    &mut act_i8,
                );
                router_x_i8[i * d..(i + 1) * d].copy_from_slice(&act_i8);
            }
            let mut router_logits = vec![0f32; n * n_experts];
            crate::int_inference::batched_matmul_dispatch(
                &router_x_i8,
                &router_x_scales,
                &int_block.router,
                n,
                &mut router_logits,
            );

            // Per-token expert selection (argmax of softmax router).
            let mut expert_assignment = vec![0usize; n];
            let mut gate_vals = vec![0f32; n];
            for i in 0..n {
                let mut probs = router_logits[i * n_experts..(i + 1) * n_experts].to_vec();
                softmax_inplace(&mut probs);
                let (expert_idx, gate_val) = argmax_with_value(&probs);
                expert_assignment[i] = expert_idx;
                gate_vals[i] = gate_val;
                // Routing capture (diagnostic — dump_routing path).
                if let Some(t) = trace.as_deref_mut() {
                    let idx = l * n + i;
                    t.expert_assignment[idx] = u32::try_from(expert_idx).expect("expert fits u32");
                    t.gate_val[idx] = gate_val;
                    let entropy: f32 = probs
                        .iter()
                        .filter(|&&p| p > 0.0)
                        .map(|&p| -p * p.ln())
                        .sum();
                    t.router_entropy_nats[idx] = entropy;
                }
            }

            // Per-expert gather → batched fc1 → gelu → batched fc2 → scatter.
            // Same per-token int sums as the per-token loop: each row of
            // the batched per-expert matmul is the same input × the same
            // weight rows as the original per-token call.
            let mut tokens_for_expert: Vec<Vec<usize>> = vec![Vec::new(); n_experts];
            for (i, &e) in expert_assignment.iter().enumerate() {
                tokens_for_expert[e].push(i);
            }
            for e in 0..n_experts {
                let toks = &tokens_for_expert[e];
                if toks.is_empty() {
                    continue;
                }
                let m_e = toks.len();
                let int_expert = &int_block.experts[e];

                // fc1: per-row quantize x_norm2 for the assigned tokens, then batched matmul.
                let mut fc1_x_i8 = vec![0i8; m_e * d];
                let mut fc1_x_scales = vec![0f32; m_e];
                for (k, &i) in toks.iter().enumerate() {
                    fc1_x_scales[k] = crate::int_inference::quantize_act_i8(
                        &x_norm2[i * d..(i + 1) * d],
                        &mut act_i8,
                    );
                    fc1_x_i8[k * d..(k + 1) * d].copy_from_slice(&act_i8);
                }
                let mut ff_hidden = vec![0f32; m_e * d_ff];
                crate::int_inference::batched_matmul_dispatch(
                    &fc1_x_i8,
                    &fc1_x_scales,
                    &int_expert.fc1,
                    m_e,
                    &mut ff_hidden,
                );

                // gelu per row (same as per-token path).
                for k in 0..m_e {
                    gelu_inplace(&mut ff_hidden[k * d_ff..(k + 1) * d_ff]);
                }

                // fc2: per-row quantize ff_hidden, batched matmul, scatter back to x.
                let mut fc2_x_i8 = vec![0i8; m_e * d_ff];
                let mut fc2_x_scales = vec![0f32; m_e];
                for k in 0..m_e {
                    fc2_x_scales[k] = crate::int_inference::quantize_act_i8(
                        &ff_hidden[k * d_ff..(k + 1) * d_ff],
                        &mut act_i8,
                    );
                    fc2_x_i8[k * d_ff..(k + 1) * d_ff].copy_from_slice(&act_i8);
                }
                let mut ff_out = vec![0f32; m_e * d];
                crate::int_inference::batched_matmul_dispatch(
                    &fc2_x_i8,
                    &fc2_x_scales,
                    &int_expert.fc2,
                    m_e,
                    &mut ff_out,
                );
                for (k, &i) in toks.iter().enumerate() {
                    let gate = gate_vals[i];
                    for c in 0..d {
                        x[i * d + c] += ff_out[k * d + c] * gate;
                    }
                }
            }
        }

        // Final rmsnorm + batched vocab projection.
        let mut final_x_i8 = vec![0i8; n * d];
        let mut final_x_scales = vec![0f32; n];
        for i in 0..n {
            rmsnorm_inplace(&mut x[i * d..(i + 1) * d], &self.norm_f);
            final_x_scales[i] =
                crate::int_inference::quantize_act_i8(&x[i * d..(i + 1) * d], &mut act_i8);
            final_x_i8[i * d..(i + 1) * d].copy_from_slice(&act_i8);
        }
        let mut logits = vec![0f32; n * vocab];
        crate::int_inference::batched_matmul_dispatch(
            &final_x_i8,
            &final_x_scales,
            &ic.tok_emb,
            n,
            &mut logits,
        );
        logits
    }

    /// Batched full-sequence forward — same return shape as
    /// [`crate::transformer::ByteTransformer::forward`]. Used only by
    /// the parity test (which compares against `PyTorch` logits
    /// dumped in batched mode); the codec path uses `forward_step`
    /// for KV caching.
    #[allow(clippy::many_single_char_names, clippy::needless_range_loop)]
    pub(crate) fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let t = tokens.len();
        let vocab = self.cfg.vocab_size;
        let mut all_logits = vec![0f32; t * vocab];
        let mut cache = self.new_kv_cache();
        for (i, &tok) in tokens.iter().enumerate() {
            let step = self.forward_step(&mut cache, tok);
            all_logits[i * vocab..(i + 1) * vocab].copy_from_slice(&step);
        }
        all_logits
    }

    pub(crate) fn new_kv_cache(&self) -> MoeKvCache {
        MoeKvCache::new(&self.cfg)
    }
}

#[derive(Debug)]
pub(crate) struct MoeKvCache {
    pub(crate) layers: Vec<MoeKvCacheLayer>,
    pub(crate) pos: usize,
}

#[derive(Debug)]
pub(crate) struct MoeKvCacheLayer {
    pub(crate) k: Vec<f32>, // [context, d_model]
    pub(crate) v: Vec<f32>, // [context, d_model]
}

impl MoeKvCache {
    pub(crate) fn new(cfg: &MoeConfig) -> Self {
        let layers = (0..cfg.n_layer)
            .map(|_| MoeKvCacheLayer {
                k: vec![0f32; cfg.context * cfg.d_model],
                v: vec![0f32; cfg.context * cfg.d_model],
            })
            .collect();
        Self { layers, pos: 0 }
    }

    pub(crate) const fn reset(&mut self) {
        self.pos = 0;
    }
}

fn read_header(buf: &[u8]) -> Result<MoeConfig> {
    let read_u32 = |off: usize| -> u32 {
        u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
    };
    let magic = read_u32(0);
    if magic != MAGIC {
        bail!("lzrm magic mismatch: got 0x{magic:08x}, expected 0x{MAGIC:08x}");
    }
    let version = read_u32(4);
    if version != VERSION {
        bail!("lzrm version mismatch: got {version}, expected {VERSION}");
    }
    Ok(MoeConfig {
        n_layer: read_u32(8) as usize,
        n_head: read_u32(12) as usize,
        d_model: read_u32(16) as usize,
        d_ff: read_u32(20) as usize,
        context: read_u32(24) as usize,
        vocab_size: read_u32(28) as usize,
        n_experts: read_u32(32) as usize,
    })
}

/// `argmax` over a probability slice, returning the index and the
/// corresponding value. Equivalent to `PyTorch`'s `probs.max(dim=-1)`.
fn argmax_with_value(probs: &[f32]) -> (usize, f32) {
    let mut best_i = 0usize;
    let mut best_v = probs[0];
    for (i, &v) in probs.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    (best_i, best_v)
}

/// Per-tensor symmetric quantize-then-dequantize in place. Picks a
/// scale from the tensor's max-abs so the highest-magnitude weight
/// maps to `+/- max_q`; everything in between rounds to the nearest
/// quantization grid point. Zeros stay zero. A tensor of all-zero
/// (or near-zero) values is left untouched.
///
/// `bits` is the signed bit width (e.g., 8 for int8, 4 for int4).
/// The grid runs from `-(2^(bits-1) - 1)` to `+(2^(bits-1) - 1)`
/// — symmetric, so the negative reserved slot (`-(2^(bits-1))`) is
/// unused. This costs one grid point per tensor but keeps the scale
/// computation trivially correct for both signs.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn quantize_dequantize_inplace(values: &mut [f32], bits: u32) {
    assert!((1..32).contains(&bits), "bits must be in 1..32");
    let max_abs = values.iter().fold(0_f32, |a, &v| a.max(v.abs()));
    if max_abs == 0.0 {
        return;
    }
    let max_q = ((1_u32 << (bits - 1)) - 1) as f32;
    let scale = max_abs / max_q;
    let inv_scale = scale.recip();
    for v in values.iter_mut() {
        let q = (*v * inv_scale).round().clamp(-max_q, max_q);
        *v = q * scale;
    }
}

/// Per-row symmetric quantize-then-dequantize for a row-major 2D
/// tensor of shape `[rows, cols]` stored as a flat `Vec<f32>`. Each
/// row gets its own scale from its own max-abs, so a row with mostly
/// small weights doesn't have its precision wasted by an outlier in
/// a different row.
///
/// Shipped `L(D)` per such tensor is `rows * cols * (bits/8) +
/// rows * 4` (the per-row f32 scales) — the scale overhead is
/// ~3% on our architecture, accounted for in `main.rs` reporting.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn quantize_dequantize_per_channel(values: &mut [f32], rows: usize, cols: usize, bits: u32) {
    assert!((1..32).contains(&bits), "bits must be in 1..32");
    assert_eq!(values.len(), rows * cols, "shape mismatch");
    let max_q = ((1_u32 << (bits - 1)) - 1) as f32;
    for row in values.chunks_exact_mut(cols) {
        let max_abs = row.iter().fold(0_f32, |a, &v| a.max(v.abs()));
        if max_abs == 0.0 {
            continue;
        }
        let scale = max_abs / max_q;
        let inv_scale = scale.recip();
        for v in row.iter_mut() {
            let q = (*v * inv_scale).round().clamp(-max_q, max_q);
            *v = q * scale;
        }
    }
}

/// Asymmetric per-channel quantize-then-dequantize: each row of the
/// `[rows, cols]` 2D tensor gets a `(scale, zero_point)` pair derived
/// from its `(min, max)` so the full quant grid covers the row's
/// actual range — useful when a row's weights aren't centered at
/// zero (post-training FFN weights often acquire a small bias).
///
/// Quant: `q = round((w - w_min) / scale)` clamped to `[0, n_levels-1]`.
/// Dequant: `w' = q * scale + w_min`.
///
/// Shipped storage per row is `cols * bits / 8` weight bytes + 4
/// (f32 scale) + 1 (u8 zero-point) — the per-row overhead vs
/// symmetric is +1 byte, accounted for in [`MoeByteTransformer::shipped_bytes`].
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn quantize_dequantize_per_channel_asym(values: &mut [f32], rows: usize, cols: usize, bits: u32) {
    assert!((2..32).contains(&bits), "asym needs at least 2 bits");
    assert_eq!(values.len(), rows * cols, "shape mismatch");
    let n_levels = (1_u32 << bits) - 1; // e.g., 31 for int5 (0..=31)
    let n_levels_f = n_levels as f32;
    for row in values.chunks_exact_mut(cols) {
        let (mut w_min, mut w_max) = (row[0], row[0]);
        for &v in row.iter() {
            if v < w_min {
                w_min = v;
            }
            if v > w_max {
                w_max = v;
            }
        }
        let range = w_max - w_min;
        if range == 0.0 {
            continue;
        }
        let scale = range / n_levels_f;
        let inv_scale = scale.recip();
        for v in row.iter_mut() {
            let q = ((*v - w_min) * inv_scale).round().clamp(0.0, n_levels_f);
            *v = q * scale + w_min;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parity check against the `PyTorch` `MoEByteTransformer`'s
    /// batched forward. Reads `ckpt.lzrm` (weights) and
    /// `ckpt.logits.bin` (reference logits for input `[0..seq_len)`)
    /// from `lzr-neural/ckpts/`. Skips cleanly if artifacts aren't
    /// present.
    ///
    /// Reference artifacts generated by:
    /// ```text
    ///   uv run scripts/export_moe_weights.py --ckpt ckpts/<name>.pt
    ///   uv run scripts/dump_moe_logits_for_parity.py --ckpt ckpts/<name>.pt --seq-len 32
    /// ```
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn run_moe_parity_check(name: &str) {
        let lzrm_path = format!("/Users/creynolds/Programming/lzr-neural/ckpts/{name}.lzrm");
        let logits_path =
            format!("/Users/creynolds/Programming/lzr-neural/ckpts/{name}.logits.bin");
        if !std::path::Path::new(&lzrm_path).exists()
            || !std::path::Path::new(&logits_path).exists()
        {
            eprintln!("MoE parity artifacts for {name} missing — skipping");
            return;
        }
        let lzrm = std::fs::read(&lzrm_path).unwrap();
        let model = MoeByteTransformer::load_lzrm(&lzrm).unwrap();
        let logits_bytes = std::fs::read(&logits_path).unwrap();
        let t = u32::from_le_bytes([
            logits_bytes[0],
            logits_bytes[1],
            logits_bytes[2],
            logits_bytes[3],
        ]) as usize;
        let vocab = u32::from_le_bytes([
            logits_bytes[4],
            logits_bytes[5],
            logits_bytes[6],
            logits_bytes[7],
        ]) as usize;
        assert_eq!(vocab, model.cfg.vocab_size);
        let mut reference = Vec::with_capacity(t * vocab);
        for chunk in logits_bytes[8..].chunks_exact(4) {
            reference.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        assert_eq!(reference.len(), t * vocab);

        let input: Vec<u32> = (0..t).map(|i| i as u32).collect();
        let actual = model.forward(&input);
        assert_eq!(actual.len(), reference.len());

        let mut max_abs = 0f32;
        let mut sum_abs = 0f32;
        for (a, e) in actual.iter().zip(reference.iter()) {
            let d = (a - e).abs();
            if d > max_abs {
                max_abs = d;
            }
            sum_abs += d;
        }
        let mean_abs = sum_abs / actual.len() as f32;
        eprintln!(
            "MoE logit parity ({name}): t={t} vocab={vocab}  max_abs={max_abs:.6}  mean_abs={mean_abs:.6}"
        );
        assert!(
            max_abs < 1e-3,
            "MoE logit divergence too large for {name}: max_abs={max_abs}"
        );
    }

    #[test]
    fn moe_pytorch_logit_parity_nano_plus_equivalent() {
        run_moe_parity_check("moe_nano_plus_equivalent_42_step20000");
    }

    /// Phase 50C Day-4 Phase-1 diagnostic: wall comparison of the
    /// new batched CPU path vs the per-step loop on a small sequence
    /// from the Phase-45 checkpoint. Both paths run identical work;
    /// the difference, if any, comes from fewer per-call `Vec` allocs
    /// in the batched path. Run on demand:
    /// ```text
    /// cargo test --release moe::tests::batched_vs_per_step_cpu_wall \
    ///   -- --nocapture --ignored
    /// ```
    #[test]
    #[ignore = "diagnostic — run on demand"]
    fn batched_vs_per_step_cpu_wall() {
        let lzrm_path = "/Users/creynolds/Programming/lzr-neural/ckpts/\
                         moe_tok_wider_16k_xlong_lowlr_42_step240000.lzrm";
        if !std::path::Path::new(lzrm_path).exists() {
            eprintln!("Phase-45 .lzrm missing — skipping bench");
            return;
        }
        let lzrm = std::fs::read(lzrm_path).expect("read .lzrm");
        let mut model = MoeByteTransformer::load_lzrm(&lzrm).expect("load");
        model.apply_quantization(Quantization::Mixed5Asym);
        model.prepare_int_cache();
        let vocab_u32 = u32::try_from(model.cfg.vocab_size).expect("vocab fits u32");
        let tokens: Vec<u32> = (0..256u32).map(|i| (i * 37 + 11) % vocab_u32).collect();

        // per-step
        let t0 = std::time::Instant::now();
        let iters = 3u32;
        for _ in 0..iters {
            let mut cache = model.new_kv_cache();
            for &t in &tokens {
                std::hint::black_box(model.forward_step_int(&mut cache, t));
            }
        }
        let per_step_ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(iters);

        // batched
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            std::hint::black_box(model.batched_forward_int(&tokens));
        }
        let batched_ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(iters);

        eprintln!(
            "wall on {tokens_len} tokens × {iters} iters:\n  per-step CPU: {per_step_ms:>8.1} ms / iter\n  batched  CPU: {batched_ms:>8.1} ms / iter  ({ratio:.2}× per-step)",
            tokens_len = tokens.len(),
            ratio = batched_ms / per_step_ms,
        );
    }

    /// Phase 50C Day-4 Phase-1 acceptance gate. The new batched
    /// CPU forward must produce **bit-identical** logits to a loop
    /// of the per-step int forward over the same token sequence.
    /// Bit-identity, not "approximately equal" — every f32 in the
    /// logit slab must `to_bits()`-match the per-step reference.
    ///
    /// Uses the production Phase-45 checkpoint. Skips cleanly if
    /// the artifact isn't present (e.g., on CI machines without
    /// access to `lzr-neural/ckpts/`).
    #[test]
    fn batched_forward_int_bit_matches_per_step_phase45() {
        let lzrm_path = "/Users/creynolds/Programming/lzr-neural/ckpts/\
                         moe_tok_wider_16k_xlong_lowlr_42_step240000.lzrm";
        if !std::path::Path::new(lzrm_path).exists() {
            eprintln!("Phase-45 .lzrm missing — skipping batched parity test");
            return;
        }
        let lzrm = std::fs::read(lzrm_path).expect("read .lzrm");
        let mut model = MoeByteTransformer::load_lzrm(&lzrm).expect("load");
        model.apply_quantization(Quantization::Mixed5Asym);
        model.prepare_int_cache();
        let vocab = model.cfg.vocab_size;

        let vocab_u32 = u32::try_from(vocab).expect("vocab fits u32");
        let tokens: Vec<u32> = (0..64u32).map(|i| (i * 37 + 11) % vocab_u32).collect();

        let mut per_step_logits = vec![0f32; tokens.len() * vocab];
        let mut cache = model.new_kv_cache();
        for (i, &t) in tokens.iter().enumerate() {
            let step = model.forward_step_int(&mut cache, t);
            per_step_logits[i * vocab..(i + 1) * vocab].copy_from_slice(&step);
        }

        let batched = model.batched_forward_int(&tokens);
        assert_eq!(batched.len(), per_step_logits.len());

        let mut exact = 0usize;
        let mut max_abs = 0f32;
        for (a, b) in batched.iter().zip(per_step_logits.iter()) {
            if a.to_bits() == b.to_bits() {
                exact += 1;
            }
            let d = (a - b).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        eprintln!(
            "batched vs per-step parity (Phase-45): {exact}/{} bit-exact, max_abs={max_abs:e}",
            batched.len()
        );
        assert_eq!(
            exact,
            batched.len(),
            "batched_forward_int must bit-match per-step; max_abs={max_abs:e}"
        );
    }

    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn load_lzrm_round_trip_minimal() {
        // Hand-crafted .lzrm for a minimal config (n_layer=1, d_model=2,
        // n_head=1, d_ff=2, context=2, vocab=3, n_experts=2). Confirms
        // header parsing and tensor stream consumption.
        let cfg_bytes: Vec<u8> = [
            MAGIC.to_le_bytes(),
            VERSION.to_le_bytes(),
            1u32.to_le_bytes(), // n_layer
            1u32.to_le_bytes(), // n_head
            2u32.to_le_bytes(), // d_model
            2u32.to_le_bytes(), // d_ff
            2u32.to_le_bytes(), // context
            3u32.to_le_bytes(), // vocab_size
            2u32.to_le_bytes(), // n_experts
        ]
        .concat();
        let n_floats = 3 * 2 // tok_emb
            + 2 * 2          // pos_emb
            + 2              // norm1
            + 6 * 2          // qkv (3*d_model x d_model = 6 x 2)
            + 2 * 2          // proj
            + 2              // norm2
            + 2 * 2          // router (n_experts x d_model)
            + 2 * (2 * 2     // expert fc1 (d_ff x d_model)
                 + 2 * 2)    // expert fc2 (d_model x d_ff)
            + 2; // norm_f
        let mut bytes = cfg_bytes;
        for i in 0..n_floats {
            bytes.extend_from_slice(&f32::from(i as u16).to_le_bytes());
        }
        let model = MoeByteTransformer::load_lzrm(&bytes).expect("load");
        assert_eq!(model.cfg.n_layer, 1);
        assert_eq!(model.cfg.d_model, 2);
        assert_eq!(model.cfg.vocab_size, 3);
        assert_eq!(model.cfg.n_experts, 2);
        assert_eq!(model.tok_emb.len(), 6);
        assert_eq!(model.pos_emb.len(), 4);
        assert_eq!(model.blocks.len(), 1);
        assert_eq!(model.blocks[0].experts.len(), 2);
    }
}
