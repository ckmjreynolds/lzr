//! Architecture constants for the byte-level `BitNet` transformer.
//!
//! These constants define the submission model's shape and pin the exact
//! packed weights layout. They are `const` so the weight-file length check
//! at the `include_bytes!` site can be a compile-time assertion.

/// Byte vocabulary: one symbol per possible byte value.
pub(crate) const VOCAB: usize = 256;

/// Model width. Must be divisible by 256 for k-quant alignment (CLAUDE.md).
pub(crate) const D_MODEL: usize = 256;

/// Number of transformer blocks.
pub(crate) const N_LAYERS: usize = 2;

/// Attention heads per layer. `D_MODEL` must be divisible by `N_HEADS`.
pub(crate) const N_HEADS: usize = 4;

/// Per-head dimension. Derived constraint: `N_HEADS * HEAD_DIM == D_MODEL`.
pub(crate) const HEAD_DIM: usize = D_MODEL / N_HEADS;

/// MLP hidden-size multiplier relative to `D_MODEL`.
pub(crate) const MLP_MULT: usize = 1;

/// `SwiGLU` MLP hidden width.
pub(crate) const D_FF: usize = D_MODEL * MLP_MULT;

/// Training sequence length and inference context window.
pub(crate) const CONTEXT_LEN: usize = 256;

/// `RoPE` base frequency.
pub(crate) const ROPE_THETA: f32 = 10_000.0;

/// `RMSNorm` numerical stability epsilon.
pub(crate) const RMS_EPS: f32 = 1.0e-5;

const _: () = {
    assert!(
        D_MODEL % 256 == 0,
        "D_MODEL must be divisible by 256 (CLAUDE.md)"
    );
    assert!(
        N_HEADS * HEAD_DIM == D_MODEL,
        "N_HEADS * HEAD_DIM must equal D_MODEL"
    );
    assert!(D_MODEL % 32 == 0, "matmul inner loop unrolls by 32");
};

/// Packed 2-bit ternary weight byte count for an `out_dim × in_dim` matrix.
/// Four ternary weights fit in one byte (2 bits each).
pub(crate) const fn packed_bytes(out_dim: usize, in_dim: usize) -> usize {
    (out_dim * in_dim).div_ceil(4)
}

/// Per-layer ternary matrix sizes.
///
/// Q / K / V / O are `D_MODEL × D_MODEL`; `SwiGLU` `w1 / w3` are `D_FF × D_MODEL`,
/// and `w2` is `D_MODEL × D_FF`.
pub(crate) const PACKED_QKVO_BYTES: usize = packed_bytes(D_MODEL, D_MODEL);
pub(crate) const SCALE_QKVO_F32S: usize = D_MODEL;
pub(crate) const PACKED_W1_BYTES: usize = packed_bytes(D_FF, D_MODEL);
pub(crate) const PACKED_W3_BYTES: usize = packed_bytes(D_FF, D_MODEL);
pub(crate) const PACKED_W2_BYTES: usize = packed_bytes(D_MODEL, D_FF);
pub(crate) const SCALE_W1_F32S: usize = D_FF;
pub(crate) const SCALE_W3_F32S: usize = D_FF;
pub(crate) const SCALE_W2_F32S: usize = D_MODEL;

/// Compute the cos / sin tables used by `RoPE` for every `(position, j)` pair
/// with `position ∈ [0, CONTEXT_LEN)` and `j ∈ [0, HEAD_DIM / 2)`.
///
/// Both tables are `CONTEXT_LEN * HEAD_DIM/2` long, in row-major
/// `(position, j)` order. Shared between the candle-nn training forward pass
/// and the pure-Rust inference kernel so numerics stay aligned.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub(crate) fn rope_cos_sin_tables() -> (Vec<f32>, Vec<f32>) {
    let half = HEAD_DIM / 2;
    let mut cos = vec![0.0_f32; CONTEXT_LEN * half];
    let mut sin = vec![0.0_f32; CONTEXT_LEN * half];
    for pos in 0..CONTEXT_LEN {
        for j in 0..half {
            let freq = 1.0_f64 / f64::from(ROPE_THETA).powf(2.0 * j as f64 / HEAD_DIM as f64);
            let angle = pos as f64 * freq;
            cos[pos * half + j] = angle.cos() as f32;
            sin[pos * half + j] = angle.sin() as f32;
        }
    }
    (cos, sin)
}

/// Total on-disk bytes for the embedded packed weights blob.
///
/// Layout: `tok_emb` (f32) + per layer: `attn_norm` (f32), Q/K/V/O (each packed + f32 scale),
/// `mlp_norm` (f32), w1/w2/w3 (each packed + f32 scale).
pub(crate) const PACKED_WEIGHTS_LEN: usize = {
    let tok_emb = VOCAB * D_MODEL * 4;
    let attn_norm = D_MODEL * 4;
    let mlp_norm = D_MODEL * 4;
    let qkvo_one = PACKED_QKVO_BYTES + SCALE_QKVO_F32S * 4;
    let w1 = PACKED_W1_BYTES + SCALE_W1_F32S * 4;
    let w2 = PACKED_W2_BYTES + SCALE_W2_F32S * 4;
    let w3 = PACKED_W3_BYTES + SCALE_W3_F32S * 4;
    let per_layer = attn_norm + 4 * qkvo_one + mlp_norm + w1 + w2 + w3;
    tok_emb + N_LAYERS * per_layer
};
