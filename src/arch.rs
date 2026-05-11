//! Architecture constants for the byte-level RWKV v4 model.
//!
//! RWKV v4 replaces transformer attention with linear-time-mix over a
//! learned time-decay state. There is no KV cache, no `RoPE`; inference
//! progresses in O(1) per token regardless of stream length. Model state
//! at each layer is five `D_MODEL`-sized vectors (`shifted_x_tm`,
//! `shifted_x_cm`, `aa`, `bb`, `pp`) — the pp vector lives in log-space
//! for the WKV numerical-stability trick.
//!
//! Constants are `const` so that the weight-file length check at the
//! `include_bytes!` site is a compile-time assertion.

/// Symbol vocabulary. Reverted to byte-level on 2026-05-09 with the
/// pivot to a deterministic-first ensemble: type-routed mixing,
/// order-N PPM, LZ pre-pass, etc. (see 2026-05-09 journal entry, when
/// written). The neural component is shelved while the deterministic
/// stack is iterated — when we re-introduce a neural arm it'll plug
/// in as one component of a mixed ensemble, and the tokenizer
/// question gets re-decided then.
pub(crate) const VOCAB: usize = 256;

/// Length of the cumulative distribution function passed to the
/// arithmetic coder: one entry per symbol plus the trailing `TOTAL`
/// sentinel.
pub(crate) const CDF_LEN: usize = VOCAB + 1;

/// Model width. Must be divisible by 256 for k-quant alignment (CLAUDE.md)
/// and `≤ 512` so every ternary matrix's `in_dim` stays within the TL1 NEON
/// kernel's i16-accumulator bound. Bumping beyond 512 requires widening
/// the LUT-kernel accumulators to i32 or adding periodic flush.
///
/// `D_MODEL = 256`, `N_LAYERS = 8`, `CM_MULT = 1` — the **4M scale-up**
/// for the second neural arm in the phase-2 ensemble. The 1M run
/// (2026-05-09) plateaued at info-bpb 1.86, mixed into the LZ-routed
/// codec at 1.991 bpb / 249 MB on 1 GB. Standalone-neural's gap to the
/// deterministic floor was the dominant remaining lever; bumping to
/// 4M (4× ternary parameter count, ~2× per-step training time) targets
/// the 0.10–0.20 bpb improvement the 1M-vs-2026-04-24-baseline gap
/// suggests is on the table.
pub(crate) const D_MODEL: usize = 256;

/// Number of RWKV blocks. Each block is one time-mix + one channel-mix.
pub(crate) const N_LAYERS: usize = 8;

/// Channel-mix hidden-size multiplier relative to `D_MODEL`. RWKV v4
/// traditionally uses `4×`, but `D_FF = 4·D_MODEL = 2048` would exceed
/// both TL1 (`≤ 512`) and TL2 (`≤ 768`) accumulator bounds on the
/// channel-mix V projection (`in_dim = D_FF`). Hold at `1×` for the
/// initial model; widening requires kernel changes.
pub(crate) const CM_MULT: usize = 1;

/// Channel-mix hidden width.
pub(crate) const D_FF: usize = D_MODEL * CM_MULT;

/// `RMSNorm` numerical stability epsilon. RWKV v4 originally uses
/// `LayerNorm`; we use `RMSNorm` to match the kernel path already built
/// out and to save the per-norm bias parameter.
pub(crate) const RMS_EPS: f32 = 1.0e-5;

const _: () = {
    assert!(
        D_MODEL % 256 == 0,
        "D_MODEL must be divisible by 256 (CLAUDE.md)"
    );
    assert!(D_MODEL % 32 == 0, "matmul inner loop unrolls by 32");
};

/// Packed 2-bit ternary weight byte count for an `out_dim × in_dim` matrix.
/// Four ternary weights fit in one byte (2 bits each).
pub(crate) const fn packed_bytes(out_dim: usize, in_dim: usize) -> usize {
    (out_dim * in_dim).div_ceil(4)
}

// Token-embedding ternary matrix size. `tok_emb` is `VOCAB × D_MODEL`,
// stored as `I2_S` packed weights + per-row f32 absmax scale. Same
// kernel used by the per-layer matrices, with the LM head reusing the
// embedding via weight-tied output projection.
pub(crate) const PACKED_TOK_EMB_BYTES: usize = packed_bytes(VOCAB, D_MODEL);
pub(crate) const SCALE_TOK_EMB_F32S: usize = VOCAB;

// Per-layer ternary matrix sizes.
//
// Time-mix has four projections: receptance `R`, key `K`, value `V`,
// output `O`. All are `D_MODEL × D_MODEL`.
pub(crate) const PACKED_TM_BYTES: usize = packed_bytes(D_MODEL, D_MODEL);
pub(crate) const SCALE_TM_F32S: usize = D_MODEL;

// Channel-mix has three projections: key `K` (expansion to `D_FF`),
// value `V` (contraction back to `D_MODEL`), receptance `R`
// (`D_MODEL × D_MODEL`).
pub(crate) const PACKED_CM_K_BYTES: usize = packed_bytes(D_FF, D_MODEL);
pub(crate) const SCALE_CM_K_F32S: usize = D_FF;
pub(crate) const PACKED_CM_V_BYTES: usize = packed_bytes(D_MODEL, D_FF);
pub(crate) const SCALE_CM_V_F32S: usize = D_MODEL;
pub(crate) const PACKED_CM_R_BYTES: usize = packed_bytes(D_MODEL, D_MODEL);
pub(crate) const SCALE_CM_R_F32S: usize = D_MODEL;

/// Total on-disk bytes for the embedded packed weights blob.
///
/// Layout per layer:
///   `tm_norm` (f32), `tm_mix_r` (f32), `tm_mix_k` (f32), `tm_mix_v` (f32),
///   `time_decay` (f32), `time_first` (f32), `tm_R` + scale, `tm_K` + scale,
///   `tm_V` + scale, `tm_O` + scale,
///   `cm_norm` (f32), `cm_mix_k` (f32), `cm_mix_r` (f32),
///   `cm_K` + scale, `cm_V` + scale, `cm_R` + scale.
///
/// Global: `tok_emb` (ternary `I2_S` packed + per-row f32 scale),
/// initial `ln0` (f32), final `ln_f` (f32).
pub(crate) const PACKED_WEIGHTS_LEN: usize = {
    let tok_emb = PACKED_TOK_EMB_BYTES + SCALE_TOK_EMB_F32S * 4;
    let ln0 = D_MODEL * 4;
    let ln_f = D_MODEL * 4;

    let tm_norm = D_MODEL * 4;
    let tm_mix = 3 * D_MODEL * 4; // time_mix_r, time_mix_k, time_mix_v
    let wkv_params = 2 * D_MODEL * 4; // time_decay, time_first
    let tm_proj = 4 * (PACKED_TM_BYTES + SCALE_TM_F32S * 4);

    let cm_norm = D_MODEL * 4;
    let cm_mix = 2 * D_MODEL * 4; // channel_mix_k, channel_mix_r
    let cm_k = PACKED_CM_K_BYTES + SCALE_CM_K_F32S * 4;
    let cm_v = PACKED_CM_V_BYTES + SCALE_CM_V_F32S * 4;
    let cm_r = PACKED_CM_R_BYTES + SCALE_CM_R_F32S * 4;

    let per_layer = tm_norm + tm_mix + wkv_params + tm_proj + cm_norm + cm_mix + cm_k + cm_v + cm_r;
    tok_emb + ln0 + ln_f + N_LAYERS * per_layer
};
