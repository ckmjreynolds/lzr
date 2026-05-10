//! `BitNet` ternary matvec kernels (ported from `bitnet.cpp`'s `I2_S` path).
//!
//! This module is a Rust rewrite of Microsoft's `bitnet.cpp` inference
//! kernels, adapted to our single-threaded, byte-level compressor use case.

// Dense math notation (`I2_S`, `idx ∈ {0..=8}`, `K = 2`, etc.) in the
// section-header doc comments triggers many false positives for the
// `doc_markdown` lint. Allow it file-wide — the rest of the crate doesn't
// have this density of math terms.
#![allow(clippy::doc_markdown)]
//!
//! # Storage format: block-interleaved, `QK_I2_S = 64`
//!
//! Each tensor is cut into **blocks** of 64 consecutive weights (along the
//! `in_dim` axis). A block is stored as **16 bytes** of packed ternary
//! codes, laid out the same way `bitnet.cpp` does on ARM:
//!
//! Byte `b` (`0 ≤ b < 16`) carries four 2-bit codes for the weights at
//! positions `{b, b + 16, b + 32, b + 48}` within the block:
//!
//! ```text
//!     bits[7:6] = code at position b       (group 0)
//!     bits[5:4] = code at position b + 16  (group 1)
//!     bits[3:2] = code at position b + 32  (group 2)
//!     bits[1:0] = code at position b + 48  (group 3)
//! ```
//!
//! **Code → value** (unchanged from the previous encoding):
//!
//! | code | value |
//! |-----:|------:|
//! | `0b00` | `-1` |
//! | `0b01` | ` 0` |
//! | `0b10` | `+1` |
//! | `0b11` | unused |
//!
//! The unpack is a subtract: `w_i8 = (code as i8) - 1`.
//!
//! # Per-row scales
//!
//! We retain **per-output-row `f32` scales** (`w_scale[out_dim]`) rather
//! than `bitnet.cpp`'s single global scale. Per-row absmean is what the
//! `BitNet` b1.58 paper specifies, and it's what our QAT training produces.
//! `bitnet.cpp`'s simplification works for its flag-from-HF-pretrained
//! pipeline where the original training may not have used per-row scales.
//!
//! # Activation quantization
//!
//! 8-bit per-vector absmax (`quantize_activations`). Matches the paper's
//! per-token spec; `bitnet.cpp`'s per-tensor is a throughput trade that
//! doesn't apply to our batch-1 inference.
//!
//! # Kernels
//!
//! - [`matvec_ternary_scalar`]: portable fallback.
//! - `matvec_ternary_neon` (aarch64 + `dotprod`): matches `bitnet.cpp`'s
//!   `ggml_vec_dot_i2_i8_s_1x1` structure. 16-byte vector load → 4 × 2-bit
//!   unpack (shr + and) → 4 × `vdotq_s32` against the 4 corresponding i8×16
//!   activation chunks (from positions `[0..16)`, `[16..32)`, `[32..48)`,
//!   `[48..64)` within the block's activation slice).
//! - `matvec_ternary_avx2` (`x86_64` + `avx2`): `pmaddubsw`-based `I2_S`,
//!   matching `bitnet.cpp`'s x86 path.

/// Block size in weights. `QK_I2_S` in `bitnet.cpp` on ARM.
pub(crate) const QK_I2_S: usize = 64;

/// Bytes of packed ternary per block (2 bits × 64 weights / 8 bits per byte).
pub(crate) const PACKED_BYTES_PER_BLOCK: usize = QK_I2_S / 4;

/// Quantize `x` into 8-bit signed integers with a shared absmax scale.
///
/// Returns the scale `s` such that `x[i] ≈ s * (x_q[i] as f32)`. Zero-input
/// vectors return `s = 0.0`; the matvec short-circuits on zero scale.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn quantize_activations(x: &[f32], x_q: &mut [i8]) -> f32 {
    debug_assert_eq!(x.len(), x_q.len());
    let mut absmax = 0.0_f32;
    for &v in x {
        let a = v.abs();
        if a > absmax {
            absmax = a;
        }
    }
    if absmax == 0.0 {
        for q in x_q.iter_mut() {
            *q = 0;
        }
        return 0.0;
    }
    let scale = absmax / 127.0;
    let inv = 1.0 / scale;
    for (src, dst) in x.iter().zip(x_q.iter_mut()) {
        let q = (src * inv).round().clamp(-127.0, 127.0);
        *dst = q as i8;
    }
    scale
}

/// Pack `weights` (ternary `{-1, 0, 1}` stored as `i8`) into the
/// block-interleaved 2-bit format.
///
/// Layout: `weights.len()` must be a multiple of `QK_I2_S`; `packed.len()`
/// must be `weights.len() / 4`. Blocks are packed sequentially.
#[cfg(any(feature = "training", test))]
pub(crate) fn pack_ternary(weights: &[i8], packed: &mut [u8]) {
    assert_eq!(weights.len() % QK_I2_S, 0);
    assert_eq!(packed.len(), weights.len() / 4);
    let n_blocks = weights.len() / QK_I2_S;
    let group_size = QK_I2_S / 4; // 16 on ARM sizing

    for block_idx in 0..n_blocks {
        let w_base = block_idx * QK_I2_S;
        let p_base = block_idx * PACKED_BYTES_PER_BLOCK;
        for b in 0..group_size {
            // Codes at positions {b, b+16, b+32, b+48} in the block.
            let c0 = weight_to_code(weights[w_base + b]);
            let c1 = weight_to_code(weights[w_base + b + group_size]);
            let c2 = weight_to_code(weights[w_base + b + 2 * group_size]);
            let c3 = weight_to_code(weights[w_base + b + 3 * group_size]);
            // Layout: bits[7:6]=c0, [5:4]=c1, [3:2]=c2, [1:0]=c3.
            // Matches bitnet.cpp's
            //     temp = q8[j] << (6 - 2 * group_idx)
            //     packed[group_pos] |= temp
            packed[p_base + b] = (c0 << 6) | (c1 << 4) | (c2 << 2) | c3;
        }
    }
}

#[cfg(any(feature = "training", test))]
#[inline]
fn weight_to_code(w: i8) -> u8 {
    match w {
        -1 => 0b00,
        0 => 0b01,
        1 => 0b10,
        other => panic!("pack_ternary: value {other} is not in {{-1, 0, 1}}"),
    }
}

/// Dispatch entry point. Selects the fastest kernel available at runtime.
pub(crate) fn matvec_ternary(
    packed: &[u8],
    w_scale: &[f32],
    x_q: &[i8],
    x_scale: f32,
    out: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            // SAFETY: dotprod availability verified above.
            #[allow(unsafe_code)]
            unsafe {
                matvec_ternary_neon(packed, w_scale, x_q, x_scale, out);
            }
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 availability verified above.
            #[allow(unsafe_code)]
            unsafe {
                matvec_ternary_avx2(packed, w_scale, x_q, x_scale, out);
            }
            return;
        }
    }

    matvec_ternary_scalar(packed, w_scale, x_q, x_scale, out);
}

/// Scalar reference / fallback kernel.
///
/// Operates on the same block-interleaved format as the SIMD kernels so
/// the diff-tests can drive all three from the same inputs. `i32 → f32`
/// precision is bounded by the quantization noise.
#[inline(never)]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_wrap)]
pub(crate) fn matvec_ternary_scalar(
    packed: &[u8],
    w_scale: &[f32],
    x_q: &[i8],
    x_scale: f32,
    out: &mut [f32],
) {
    let out_dim = w_scale.len();
    let in_dim = x_q.len();
    debug_assert_eq!(packed.len(), out_dim * in_dim / 4);
    debug_assert_eq!(out.len(), out_dim);
    debug_assert_eq!(
        in_dim % QK_I2_S,
        0,
        "in_dim must be divisible by QK_I2_S = {QK_I2_S}"
    );

    if x_scale == 0.0 {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        return;
    }

    let packed_per_row = in_dim / 4;
    let blocks_per_row = in_dim / QK_I2_S;
    let group_size = QK_I2_S / 4;

    for row in 0..out_dim {
        let row_packed = &packed[row * packed_per_row..(row + 1) * packed_per_row];
        let mut acc: i32 = 0;

        for block_idx in 0..blocks_per_row {
            let p_base = block_idx * PACKED_BYTES_PER_BLOCK;
            let x_base = block_idx * QK_I2_S;
            for b in 0..group_size {
                let byte = row_packed[p_base + b];
                // Same layout as the packer: c0 @ [7:6], c1 @ [5:4], c2 @
                // [3:2], c3 @ [1:0]. Subtracting 1 maps {0,1,2} → {-1,0,+1}.
                let c0 = ((byte >> 6) & 0b11) as i8;
                let c1 = ((byte >> 4) & 0b11) as i8;
                let c2 = ((byte >> 2) & 0b11) as i8;
                let c3 = (byte & 0b11) as i8;
                acc += i32::from(x_q[x_base + b]) * i32::from(c0 - 1);
                acc += i32::from(x_q[x_base + b + group_size]) * i32::from(c1 - 1);
                acc += i32::from(x_q[x_base + b + 2 * group_size]) * i32::from(c2 - 1);
                acc += i32::from(x_q[x_base + b + 3 * group_size]) * i32::from(c3 - 1);
            }
        }

        out[row] = (acc as f32) * w_scale[row] * x_scale;
    }
}

/// NEON + dotprod kernel, ported from `bitnet.cpp`'s
/// `ggml_vec_dot_i2_i8_s_1x1`.
///
/// Per block of 64 weights:
/// 1. `vld1q_u8` → 16 bytes of packed codes.
/// 2. `vshrq_n_u8` + `vandq_u8` × 4 → 4 × `uint8x16_t` holding the codes
///    for groups 0..3 (16 lanes each).
/// 3. Reinterpret as i8, subtract 1 to map `{0,1,2} → {-1,0,+1}`.
/// 4. `vld1q_s8` × 4 → 4 × `int8x16_t` of activations from positions
///    `[0..16), [16..32), [32..48), [48..64)` within the block.
/// 5. `vdotq_s32` × 4, accumulating into one `int32x4_t`.
/// 6. `vaddvq_s32` → scalar sum once per row.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[target_feature(enable = "neon,dotprod")]
#[allow(unsafe_code, clippy::cast_precision_loss)]
unsafe fn matvec_ternary_neon(
    packed: &[u8],
    w_scale: &[f32],
    x_q: &[i8],
    x_scale: f32,
    out: &mut [f32],
) {
    use std::arch::aarch64::{
        vaddvq_s32, vandq_u8, vdotq_s32, vdupq_n_s8, vdupq_n_s32, vdupq_n_u8, vld1q_s8, vld1q_u8,
        vreinterpretq_s8_u8, vshrq_n_u8, vsubq_s8,
    };

    let out_dim = w_scale.len();
    let in_dim = x_q.len();
    debug_assert_eq!(packed.len(), out_dim * in_dim / 4);
    debug_assert_eq!(out.len(), out_dim);
    debug_assert_eq!(in_dim % QK_I2_S, 0);

    if x_scale == 0.0 {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        return;
    }

    let packed_per_row = in_dim / 4;
    let blocks_per_row = in_dim / QK_I2_S;

    // SAFETY: NEON + dotprod available per dispatcher check. All pointer
    // dereferences stay within asserted lengths.
    unsafe {
        let mask = vdupq_n_u8(0x03);
        let one_s8 = vdupq_n_s8(1);

        for row in 0..out_dim {
            let row_packed = packed.as_ptr().add(row * packed_per_row);
            let mut acc = vdupq_n_s32(0);

            for block_idx in 0..blocks_per_row {
                let p_ptr = row_packed.add(block_idx * PACKED_BYTES_PER_BLOCK);
                let x_ptr = x_q.as_ptr().add(block_idx * QK_I2_S);

                // One 16-byte load → the whole block.
                let xq8_3 = vld1q_u8(p_ptr);
                // Per-group shifts. Matches bitnet.cpp:
                //     xq8_2 = vshrq_n_u8(xq8_3, 2);
                //     xq8_1 = vshrq_n_u8(xq8_3, 4);
                //     xq8_0 = vshrq_n_u8(xq8_3, 6);
                let xq8_2 = vshrq_n_u8::<2>(xq8_3);
                let xq8_1 = vshrq_n_u8::<4>(xq8_3);
                let xq8_0 = vshrq_n_u8::<6>(xq8_3);

                // Mask to 2 bits; reinterpret as i8; subtract 1 → {-1,0,+1}.
                let q0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(xq8_0, mask)), one_s8);
                let q1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(xq8_1, mask)), one_s8);
                let q2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(xq8_2, mask)), one_s8);
                let q3 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(xq8_3, mask)), one_s8);

                // Load the 4 × 16-byte activation chunks from the current
                // block's activation slice.
                let y0 = vld1q_s8(x_ptr);
                let y1 = vld1q_s8(x_ptr.add(16));
                let y2 = vld1q_s8(x_ptr.add(32));
                let y3 = vld1q_s8(x_ptr.add(48));

                // 4 × SDOT — each multiplies 16 i8 weight / activation
                // pairs and accumulates groups of 4 into the 4 i32 lanes
                // of `acc`.
                acc = vdotq_s32(acc, q0, y0);
                acc = vdotq_s32(acc, q1, y1);
                acc = vdotq_s32(acc, q2, y2);
                acc = vdotq_s32(acc, q3, y3);
            }

            let sum = vaddvq_s32(acc);
            out[row] = (sum as f32) * w_scale[row] * x_scale;
        }
    }
}

/// AVX2 kernel, ported from `bitnet.cpp`'s `_mm256_maddubs_epi16` pattern.
///
/// Packed weights are u8 in `{0, 1, 2}`; activations are i8. `pmaddubsw`
/// issues u8 × i8 → i16 pair-wise adds; we then widen i16 → i32 via
/// `_mm256_madd_epi16` with a 1-vector. The `+1` bias is corrected at row
/// end via `−Σ x`.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code, clippy::cast_precision_loss, clippy::cast_possible_wrap)]
unsafe fn matvec_ternary_avx2(
    packed: &[u8],
    w_scale: &[f32],
    x_q: &[i8],
    x_scale: f32,
    out: &mut [f32],
) {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm_unpackhi_epi64,
        _mm256_add_epi16, _mm256_add_epi32, _mm256_and_si256, _mm256_castsi256_si128,
        _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
        _mm256_set1_epi8, _mm256_set1_epi16, _mm256_setzero_si256, _mm256_srli_epi16,
    };

    let out_dim = w_scale.len();
    let in_dim = x_q.len();
    debug_assert_eq!(packed.len(), out_dim * in_dim / 4);
    debug_assert_eq!(out.len(), out_dim);
    // AVX2 variant works on 128-weight blocks like bitnet.cpp's x86 path,
    // but we keep our on-disk layout at QK_I2_S=64 for NEON alignment and
    // iterate two blocks per inner step. `in_dim` must therefore be a
    // multiple of 128.
    debug_assert_eq!(in_dim % 128, 0);

    if x_scale == 0.0 {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        return;
    }

    let packed_per_row = in_dim / 4;

    // SAFETY: AVX2 enabled per dispatcher; all offsets stay in bounds.
    unsafe {
        let mask = _mm256_set1_epi8(0x03);
        let one16 = _mm256_set1_epi16(1);

        // Σ x used for the +1 bias correction once per row.
        let mut x_sum: i32 = 0;
        for &v in x_q.iter() {
            x_sum += i32::from(v);
        }

        for row in 0..out_dim {
            let row_packed = packed.as_ptr().add(row * packed_per_row);
            let mut accu = _mm256_setzero_si256();

            // 32 packed bytes per iter × 4 codes per byte = 128 weights.
            let mut byte_off = 0usize;
            let mut x_off = 0usize;
            while byte_off + 32 <= packed_per_row {
                let xq8_3 = _mm256_loadu_si256(row_packed.add(byte_off).cast::<__m256i>());
                let xq8_2 = _mm256_srli_epi16::<2>(xq8_3);
                let xq8_1 = _mm256_srli_epi16::<4>(xq8_3);
                let xq8_0 = _mm256_srli_epi16::<6>(xq8_3);

                let q0 = _mm256_and_si256(xq8_0, mask);
                let q1 = _mm256_and_si256(xq8_1, mask);
                let q2 = _mm256_and_si256(xq8_2, mask);
                let q3 = _mm256_and_si256(xq8_3, mask);

                let y0 = _mm256_loadu_si256(x_q.as_ptr().add(x_off).cast::<__m256i>());
                let y1 = _mm256_loadu_si256(x_q.as_ptr().add(x_off + 32).cast::<__m256i>());
                let y2 = _mm256_loadu_si256(x_q.as_ptr().add(x_off + 64).cast::<__m256i>());
                let y3 = _mm256_loadu_si256(x_q.as_ptr().add(x_off + 96).cast::<__m256i>());

                let p0 = _mm256_maddubs_epi16(q0, y0);
                let p1 = _mm256_maddubs_epi16(q1, y1);
                let p2 = _mm256_maddubs_epi16(q2, y2);
                let p3 = _mm256_maddubs_epi16(q3, y3);

                let s01 = _mm256_add_epi16(p0, p1);
                let s23 = _mm256_add_epi16(p2, p3);
                let accu16 = _mm256_add_epi16(s01, s23);
                accu = _mm256_add_epi32(accu, _mm256_madd_epi16(accu16, one16));

                byte_off += 32;
                x_off += 128;
            }

            // Horizontal sum of the 8 i32 lanes.
            let hi = _mm256_extracti128_si256::<1>(accu);
            let sum128: __m128i = _mm_add_epi32(_mm256_castsi256_si128(accu), hi);
            let hi64 = _mm_unpackhi_epi64(sum128, sum128);
            let sum64 = _mm_add_epi32(hi64, sum128);
            let hi32 = _mm_shuffle_epi32::<{ (2 << 6) | (3 << 4) | (0 << 2) | 1 }>(sum64);
            let biased = _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));

            // Remove the +1 bias: Σ (code − 1) · x = biased − Σ x.
            let true_sum = biased - x_sum;
            out[row] = (true_sum as f32) * w_scale[row] * x_scale;
        }
    }
}

// ---------------------------------------------------------------------------
// TL1 (K = 2 lookup table) — aarch64 NEON path.
//
// TL1 is bitnet.cpp's ARM lookup kernel. Two ternary weights are packed into
// one 4-bit index:
//
//     idx = (w0 + 1) + 3 · (w1 + 1)        w ∈ {−1, 0, +1}  →  idx ∈ 0..=8
//
// Before the row loop, for every pair of consecutive activations (x0, x1),
// we precompute a 16-entry LUT:
//
//     lut[(w0+1) + 3·(w1+1)] = w0 · x0 + w1 · x1
//
// entries 9..15 stay zero. At 7-bit activation precision (|x| ≤ 63 after
// a `>> 1` arith shift of the 8-bit input), each LUT entry is in [-128, 126]
// and fits in `i8` exactly.
//
// # Packing layout — 16-row output tiles
//
// To let `vqtbl1q_s8` amortize a single 16-entry LUT across 16 shuffle
// lanes, we tile the output dim by `M_TILE_TL1 = 16` rows. For a given
// tile, at every *pair-pair* `pp ∈ [0, in_dim/4)` (two consecutive pairs),
// we store 16 bytes — one per row in the tile. Within each byte:
//
//     high nibble (bits 7..4) = idx at pair  2·pp
//     low  nibble (bits 3..0) = idx at pair  2·pp + 1
//
// Total: `(out_dim/16) · (in_dim/4) · 16 = out_dim · in_dim / 4` bytes,
// identical to `I2_S` (both pack 2 bits per weight on average).
//
// # Kernel structure
//
// For each output-row tile:
//   acc = int16x8 pair (lanes 0..15 = rows 0..15 of tile)
//   for each pair-pair `pp`:
//     bytes = vld1q_u8(tile, pp)                   # 16 bytes
//     lut_hi = vld1q_s8(lut + (2·pp)   · 16)
//     lut_lo = vld1q_s8(lut + (2·pp+1) · 16)
//     p_hi = vqtbl1q_s8(lut_hi, bytes >> 4)        # 16 i8 partial sums
//     p_lo = vqtbl1q_s8(lut_lo, bytes & 0xF)       # 16 i8 partial sums
//     acc += widen(p_hi) + widen(p_lo)
//   out[tile] = (acc as f32) * 2.0 * x_scale * w_scale[row]
//
// The `× 2.0` corrects for the `>> 1` applied when building the LUT.

/// Output rows processed per tile in TL1. Matches `vqtbl1q_s8`'s 16-lane shape.
pub(crate) const M_TILE_TL1: usize = 16;

/// Bytes per TL1 LUT slice (one per activation pair).
pub(crate) const TL1_LUT_STRIDE: usize = 16;

/// Max `in_dim` supported by the i16-accumulator TL1 kernel without overflow.
///
/// Worst-case per-pair partial sum magnitude is `2 · 63 = 126`, and there
/// are `in_dim / 2` pairs, so `|acc| ≤ 63 · in_dim`. `i16` max is 32767 →
/// safe as long as `in_dim ≤ 520`. We use 512 for a tidy power-of-two cap;
/// larger `in_dim` falls back to the I2_S dispatch path.
pub(crate) const TL1_MAX_IN_DIM: usize = 512;

/// Packed TL1 size for an `out_dim × in_dim` ternary matrix.
///
/// Same size as I2_S (one 2-bit code per weight on average).
pub(crate) const fn tl1_packed_bytes(out_dim: usize, in_dim: usize) -> usize {
    out_dim * in_dim / 4
}

/// `true` when `out_dim`/`in_dim` are in the shape range the TL1 kernel
/// supports. Callers use this to decide whether to populate a TL1 buffer.
pub(crate) const fn tl1_supports(out_dim: usize, in_dim: usize) -> bool {
    out_dim % M_TILE_TL1 == 0 && in_dim % 4 == 0 && in_dim <= TL1_MAX_IN_DIM
}

#[inline]
fn encode_tl1_pair(w0: i8, w1: i8) -> u8 {
    debug_assert!((-1..=1).contains(&w0) && (-1..=1).contains(&w1));
    // (w0 + 1), (w1 + 1) ∈ {0, 1, 2}; idx ∈ {0..=8}.
    #[allow(clippy::cast_sign_loss)]
    let c0 = (w0 + 1) as u8;
    #[allow(clippy::cast_sign_loss)]
    let c1 = (w1 + 1) as u8;
    c0 + 3 * c1
}

/// Pack row-major ternary weights into the TL1 tiled layout.
///
/// `weights` is `out_dim × in_dim` row-major i8 in `{-1, 0, 1}`. `tl1_packed`
/// has length `tl1_packed_bytes(out_dim, in_dim)`.
pub(crate) fn pack_tl1(weights: &[i8], tl1_packed: &mut [u8], out_dim: usize, in_dim: usize) {
    assert_eq!(weights.len(), out_dim * in_dim);
    assert_eq!(tl1_packed.len(), tl1_packed_bytes(out_dim, in_dim));
    assert!(tl1_supports(out_dim, in_dim));

    let n_tiles = out_dim / M_TILE_TL1;
    let pair_pairs = in_dim / 4;
    let tile_stride = pair_pairs * 16;

    for t in 0..n_tiles {
        for pp in 0..pair_pairs {
            let p_hi = 2 * pp;
            let p_lo = 2 * pp + 1;
            for r in 0..M_TILE_TL1 {
                let row = t * M_TILE_TL1 + r;
                let row_base = row * in_dim;
                let idx_hi = encode_tl1_pair(
                    weights[row_base + 2 * p_hi],
                    weights[row_base + 2 * p_hi + 1],
                );
                let idx_lo = encode_tl1_pair(
                    weights[row_base + 2 * p_lo],
                    weights[row_base + 2 * p_lo + 1],
                );
                tl1_packed[t * tile_stride + pp * 16 + r] = (idx_hi << 4) | idx_lo;
            }
        }
    }
}

/// Convert an existing `I2_S`-packed buffer into TL1 layout.
///
/// This is the runtime bridge used at weights-load time so the on-disk
/// format can stay I2_S while the hot inference path reads TL1.
pub(crate) fn repack_i2s_to_tl1(
    i2s_packed: &[u8],
    tl1_packed: &mut [u8],
    out_dim: usize,
    in_dim: usize,
) {
    assert_eq!(i2s_packed.len(), out_dim * in_dim / 4);
    assert_eq!(tl1_packed.len(), tl1_packed_bytes(out_dim, in_dim));
    assert!(tl1_supports(out_dim, in_dim));
    assert_eq!(in_dim % QK_I2_S, 0);

    // Unpack I2_S into row-major ternary i8, then `pack_tl1`. Cost:
    // one-shot at load; fusing the two conversions would complicate the
    // already-intricate I2_S block-interleaved addressing.
    let mut flat = vec![0i8; out_dim * in_dim];
    unpack_i2s_to_rowmajor(i2s_packed, &mut flat, out_dim, in_dim);
    pack_tl1(&flat, tl1_packed, out_dim, in_dim);
}

/// Unpack a block-interleaved I2_S buffer into a flat row-major `{-1, 0, 1}`
/// `i8` array. Used as an intermediate when converting to TL1 / TL2 / etc.
#[allow(clippy::cast_possible_wrap)]
pub(crate) fn unpack_i2s_to_rowmajor(
    packed: &[u8],
    weights: &mut [i8],
    out_dim: usize,
    in_dim: usize,
) {
    assert_eq!(packed.len(), out_dim * in_dim / 4);
    assert_eq!(weights.len(), out_dim * in_dim);
    assert_eq!(in_dim % QK_I2_S, 0);

    let packed_per_row = in_dim / 4;
    let n_blocks_per_row = in_dim / QK_I2_S;
    let group_size = QK_I2_S / 4;

    for row in 0..out_dim {
        let p_row = &packed[row * packed_per_row..(row + 1) * packed_per_row];
        let w_row = &mut weights[row * in_dim..(row + 1) * in_dim];
        for block in 0..n_blocks_per_row {
            let p_base = block * PACKED_BYTES_PER_BLOCK;
            let w_base = block * QK_I2_S;
            for b in 0..group_size {
                let byte = p_row[p_base + b];
                w_row[w_base + b] = (((byte >> 6) & 0b11) as i8) - 1;
                w_row[w_base + b + group_size] = (((byte >> 4) & 0b11) as i8) - 1;
                w_row[w_base + b + 2 * group_size] = (((byte >> 2) & 0b11) as i8) - 1;
                w_row[w_base + b + 3 * group_size] = ((byte & 0b11) as i8) - 1;
            }
        }
    }
}

/// Dequantize a single I2_S-packed row into f32 via per-row scale.
///
/// Caller passes the row's packed bytes (length `in_dim / 4`) and the
/// row's f32 scale; the function decodes ternary values from the
/// block-interleaved layout (matching [`unpack_i2s_to_rowmajor`]) and
/// writes `scale * (-1 | 0 | +1)` into `out[..in_dim]`.
///
/// Used by the embedding lookup: `tok_emb` is stored as I2_S, but model
/// inference needs an f32 row vector for `RMSNorm` and the residual
/// path. Per-row dequant is `O(in_dim)` and runs once per token.
#[allow(clippy::cast_possible_wrap)]
pub(crate) fn dequantize_i2s_row(packed_row: &[u8], scale: f32, in_dim: usize, out: &mut [f32]) {
    assert_eq!(packed_row.len(), in_dim / 4);
    assert_eq!(out.len(), in_dim);
    assert_eq!(in_dim % QK_I2_S, 0);

    let n_blocks = in_dim / QK_I2_S;
    let group_size = QK_I2_S / 4;
    for block in 0..n_blocks {
        let p_base = block * PACKED_BYTES_PER_BLOCK;
        let w_base = block * QK_I2_S;
        for b in 0..group_size {
            let byte = packed_row[p_base + b];
            let v0 = f32::from((((byte >> 6) & 0b11) as i8) - 1);
            let v1 = f32::from((((byte >> 4) & 0b11) as i8) - 1);
            let v2 = f32::from((((byte >> 2) & 0b11) as i8) - 1);
            let v3 = f32::from(((byte & 0b11) as i8) - 1);
            out[w_base + b] = scale * v0;
            out[w_base + b + group_size] = scale * v1;
            out[w_base + b + 2 * group_size] = scale * v2;
            out[w_base + b + 3 * group_size] = scale * v3;
        }
    }
}

/// Build the TL1 activation LUT.
///
/// `x_q` is the 8-bit absmax-quantized activation vector. We right-shift
/// each entry by 1 to get a 7-bit value in `[-64, 63]`, so the worst-case
/// two-term partial sum `w0·x0 + w1·x1` with `|w| ≤ 1` stays in `i8` range
/// `[-128, 127]`. The kernel multiplies the final sum by `2.0 · x_scale` at
/// row-write time to undo the shift.
///
/// `lut` must have length `(x_q.len() / 2) · 16` and **must be
/// zero-initialized by the caller**. We only write the 9 valid entries per
/// pair; indices 9..15 are never read (the packer can only emit 0..8, and
/// NEON's `vqtbl1q_s8` returns zero for out-of-range indices).
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn build_tl1_lut(x_q: &[i8], lut: &mut [i8]) {
    let n_pairs = x_q.len() / 2;
    debug_assert_eq!(x_q.len() % 2, 0);
    debug_assert_eq!(lut.len(), n_pairs * TL1_LUT_STRIDE);
    for p in 0..n_pairs {
        let x0 = x_q[2 * p] >> 1;
        let x1 = x_q[2 * p + 1] >> 1;
        let slot = &mut lut[p * TL1_LUT_STRIDE..p * TL1_LUT_STRIDE + TL1_LUT_STRIDE];
        for w0 in -1i8..=1 {
            for w1 in -1i8..=1 {
                // (w + 1) ∈ {0, 1, 2}; non-negative so the `as usize` cast
                // is sign-safe, not a loss.
                #[allow(clippy::cast_sign_loss)]
                let idx = ((w0 + 1) as usize) + 3 * ((w1 + 1) as usize);
                let val = i32::from(w0) * i32::from(x0) + i32::from(w1) * i32::from(x1);
                // val ∈ [-128, 126] — fits i8.
                slot[idx] = val as i8;
            }
        }
    }
}

/// Scalar TL1 matvec — correctness oracle for the NEON kernel.
///
/// Only the NEON kernel ships on aarch64 (NEON is baseline), and TL1 is
/// unused on other arches (x86_64 prefers TL2, scalar fallback paths use
/// `I2_S`). The scalar kernel exists solely to diff-test the NEON path.
#[cfg(test)]
#[inline(never)]
#[allow(clippy::cast_precision_loss)]
pub(crate) fn matvec_ternary_tl1_scalar(
    tl1_packed: &[u8],
    w_scale: &[f32],
    lut: &[i8],
    x_scale: f32,
    in_dim: usize,
    out: &mut [f32],
) {
    let out_dim = w_scale.len();
    debug_assert_eq!(tl1_packed.len(), tl1_packed_bytes(out_dim, in_dim));
    debug_assert_eq!(lut.len(), (in_dim / 2) * TL1_LUT_STRIDE);
    debug_assert_eq!(out.len(), out_dim);
    debug_assert!(tl1_supports(out_dim, in_dim));

    if x_scale == 0.0 {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        return;
    }

    let pair_pairs = in_dim / 4;
    let tile_stride = pair_pairs * 16;
    let n_tiles = out_dim / M_TILE_TL1;
    let combined_scale = 2.0 * x_scale;

    for t in 0..n_tiles {
        for r in 0..M_TILE_TL1 {
            let mut acc: i32 = 0;
            for pp in 0..pair_pairs {
                let byte = tl1_packed[t * tile_stride + pp * 16 + r];
                let high = (byte >> 4) & 0x0F;
                let low = byte & 0x0F;
                let p_hi = 2 * pp;
                let p_lo = 2 * pp + 1;
                acc += i32::from(lut[p_hi * TL1_LUT_STRIDE + usize::from(high)]);
                acc += i32::from(lut[p_lo * TL1_LUT_STRIDE + usize::from(low)]);
            }
            let row = t * M_TILE_TL1 + r;
            out[row] = (acc as f32) * combined_scale * w_scale[row];
        }
    }
}

/// NEON TL1 kernel.
///
/// 16 output rows per tile, `vqtbl1q_s8` broadcasts a single 16-entry LUT
/// across all 16 lanes so one shuffle computes 16 row-partial sums. The
/// `lut` is pre-built by the caller (`build_tl1_lut`) so that consumers
/// sharing an activation vector (e.g. Q/K/V, w1/w3) rebuild it once
/// instead of once per matvec.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, clippy::cast_precision_loss, clippy::too_many_lines)]
unsafe fn matvec_ternary_tl1_neon(
    tl1_packed: &[u8],
    w_scale: &[f32],
    lut: &[i8],
    x_scale: f32,
    in_dim: usize,
    out: &mut [f32],
) {
    use std::arch::aarch64::{
        vaddq_s16, vandq_u8, vcvtq_f32_s32, vdupq_n_f32, vdupq_n_s16, vdupq_n_u8, vget_low_s8,
        vget_low_s16, vld1q_f32, vld1q_s8, vld1q_u8, vmovl_high_s8, vmovl_high_s16, vmovl_s8,
        vmovl_s16, vmulq_f32, vqtbl1q_s8, vshrq_n_u8, vst1q_f32,
    };

    let out_dim = w_scale.len();
    debug_assert_eq!(tl1_packed.len(), tl1_packed_bytes(out_dim, in_dim));
    debug_assert_eq!(lut.len(), (in_dim / 2) * TL1_LUT_STRIDE);
    debug_assert_eq!(out.len(), out_dim);
    debug_assert!(tl1_supports(out_dim, in_dim));

    if x_scale == 0.0 {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        return;
    }

    let pair_pairs = in_dim / 4;
    let tile_stride = pair_pairs * 16;
    let n_tiles = out_dim / M_TILE_TL1;
    let combined_scale = 2.0 * x_scale;

    // SAFETY: NEON is baseline on aarch64. All pointer dereferences stay
    // within slice bounds (checked via debug_assert above).
    unsafe {
        let nibble_mask = vdupq_n_u8(0x0F);
        let scale_broadcast = vdupq_n_f32(combined_scale);

        for t in 0..n_tiles {
            let mut acc_lo = vdupq_n_s16(0); // rows 0..7
            let mut acc_hi = vdupq_n_s16(0); // rows 8..15

            for pp in 0..pair_pairs {
                let bytes = vld1q_u8(tl1_packed.as_ptr().add(t * tile_stride + pp * 16));
                let high = vshrq_n_u8::<4>(bytes);
                let low = vandq_u8(bytes, nibble_mask);

                let lut_hi = vld1q_s8(lut.as_ptr().add(2 * pp * TL1_LUT_STRIDE));
                let lut_lo = vld1q_s8(lut.as_ptr().add((2 * pp + 1) * TL1_LUT_STRIDE));

                let p_hi = vqtbl1q_s8(lut_hi, high);
                let p_lo = vqtbl1q_s8(lut_lo, low);

                acc_lo = vaddq_s16(acc_lo, vmovl_s8(vget_low_s8(p_hi)));
                acc_hi = vaddq_s16(acc_hi, vmovl_high_s8(p_hi));
                acc_lo = vaddq_s16(acc_lo, vmovl_s8(vget_low_s8(p_lo)));
                acc_hi = vaddq_s16(acc_hi, vmovl_high_s8(p_lo));
            }

            // Widen i16 → i32 across the four row-quartets.
            let acc_0 = vmovl_s16(vget_low_s16(acc_lo));
            let acc_1 = vmovl_high_s16(acc_lo);
            let acc_2 = vmovl_s16(vget_low_s16(acc_hi));
            let acc_3 = vmovl_high_s16(acc_hi);

            let f_0 = vcvtq_f32_s32(acc_0);
            let f_1 = vcvtq_f32_s32(acc_1);
            let f_2 = vcvtq_f32_s32(acc_2);
            let f_3 = vcvtq_f32_s32(acc_3);

            let row_base = t * M_TILE_TL1;
            let ws_0 = vld1q_f32(w_scale.as_ptr().add(row_base));
            let ws_1 = vld1q_f32(w_scale.as_ptr().add(row_base + 4));
            let ws_2 = vld1q_f32(w_scale.as_ptr().add(row_base + 8));
            let ws_3 = vld1q_f32(w_scale.as_ptr().add(row_base + 12));

            // Multiply order matches the scalar kernel exactly:
            // (f * combined_scale) * w_scale → bit-identical f32.
            let r_0 = vmulq_f32(vmulq_f32(f_0, scale_broadcast), ws_0);
            let r_1 = vmulq_f32(vmulq_f32(f_1, scale_broadcast), ws_1);
            let r_2 = vmulq_f32(vmulq_f32(f_2, scale_broadcast), ws_2);
            let r_3 = vmulq_f32(vmulq_f32(f_3, scale_broadcast), ws_3);

            vst1q_f32(out.as_mut_ptr().add(row_base), r_0);
            vst1q_f32(out.as_mut_ptr().add(row_base + 4), r_1);
            vst1q_f32(out.as_mut_ptr().add(row_base + 8), r_2);
            vst1q_f32(out.as_mut_ptr().add(row_base + 12), r_3);
        }
    }
}

// ---------------------------------------------------------------------------
// Arch-neutral LUT glue.
//
// `Weights` populates a per-arch "LUT-packed" buffer at load time and the
// `matvec_prequant` call site reads it without knowing TL1 vs. TL2.
//
// - aarch64 → TL1 bytes; same size as I2_S.
// - x86_64  → TL2 bytes; ~4/3 × I2_S (one 6-bit index per byte).
// - other   → empty; the caller must use `matvec_ternary` (I2_S) directly.

/// Bytes required for the arch-preferred LUT packing of an `out_dim × in_dim`
/// matrix. Returns 0 on architectures without a LUT kernel.
#[cfg(target_arch = "aarch64")]
pub(crate) const fn lut_packed_bytes(out_dim: usize, in_dim: usize) -> usize {
    tl1_packed_bytes(out_dim, in_dim)
}

#[cfg(target_arch = "x86_64")]
pub(crate) const fn lut_packed_bytes(out_dim: usize, in_dim: usize) -> usize {
    tl2_packed_bytes(out_dim, in_dim)
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(crate) const fn lut_packed_bytes(_: usize, _: usize) -> usize {
    0
}

/// `true` if the arch-preferred LUT kernel supports the given shape.
/// Returns `false` on architectures without a LUT kernel.
#[cfg(target_arch = "aarch64")]
pub(crate) const fn lut_supports(out_dim: usize, in_dim: usize) -> bool {
    tl1_supports(out_dim, in_dim)
}

#[cfg(target_arch = "x86_64")]
pub(crate) const fn lut_supports(out_dim: usize, in_dim: usize) -> bool {
    tl2_supports(out_dim, in_dim)
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(crate) const fn lut_supports(_: usize, _: usize) -> bool {
    false
}

/// Convert an I2_S-packed tensor into the arch-preferred LUT layout.
///
/// No-op (empty buffers) on architectures without a LUT kernel.
#[cfg(target_arch = "aarch64")]
pub(crate) fn repack_i2s_to_lut(i2s: &[u8], lut: &mut [u8], out_dim: usize, in_dim: usize) {
    repack_i2s_to_tl1(i2s, lut, out_dim, in_dim);
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn repack_i2s_to_lut(i2s: &[u8], lut: &mut [u8], out_dim: usize, in_dim: usize) {
    repack_i2s_to_tl2(i2s, lut, out_dim, in_dim);
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(crate) fn repack_i2s_to_lut(_i2s: &[u8], _lut: &mut [u8], _out_dim: usize, _in_dim: usize) {
    // No LUT kernel on this arch; caller should not allocate a LUT buffer.
    debug_assert!(_lut.is_empty());
}

/// Number of `i8` LUT scratch bytes needed for a given `in_dim`, in the
/// arch-preferred LUT format. Zero on architectures without a LUT kernel.
#[cfg(target_arch = "aarch64")]
pub(crate) const fn lut_scratch_bytes(in_dim: usize) -> usize {
    (in_dim / 2) * TL1_LUT_STRIDE
}

#[cfg(target_arch = "x86_64")]
pub(crate) const fn lut_scratch_bytes(in_dim: usize) -> usize {
    in_dim.div_ceil(K_TL2) * TL2_LUT_STRIDE
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(crate) const fn lut_scratch_bytes(_: usize) -> usize {
    0
}

/// Build the arch-preferred LUT from an already-quantized activation.
///
/// Caller must ensure `lut.len() == lut_scratch_bytes(x_q.len())` and
/// that `lut` is zero-initialized (we only write the valid entries per
/// group; see `build_tl1_lut` / `build_tl2_lut`).
#[cfg(target_arch = "aarch64")]
pub(crate) fn build_lut(x_q: &[i8], lut: &mut [i8]) {
    build_tl1_lut(x_q, lut);
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn build_lut(x_q: &[i8], lut: &mut [i8]) {
    build_tl2_lut(x_q, lut);
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(crate) fn build_lut(_: &[i8], lut: &mut [i8]) {
    debug_assert!(lut.is_empty());
}

/// Dispatch a matvec against the arch-preferred LUT layout.
///
/// `lut` must already be built via `build_lut(x_q, lut)`. Hoisting the LUT
/// build lets callers that share an activation (Q/K/V or w1/w3) build
/// once and reuse across matvecs.
#[cfg(target_arch = "aarch64")]
pub(crate) fn matvec_ternary_lut(
    lut_packed: &[u8],
    w_scale: &[f32],
    lut: &[i8],
    x_scale: f32,
    in_dim: usize,
    out: &mut [f32],
) {
    // SAFETY: NEON is baseline on aarch64.
    #[allow(unsafe_code)]
    unsafe {
        matvec_ternary_tl1_neon(lut_packed, w_scale, lut, x_scale, in_dim, out);
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn matvec_ternary_lut(
    lut_packed: &[u8],
    w_scale: &[f32],
    lut: &[i8],
    x_scale: f32,
    in_dim: usize,
    out: &mut [f32],
) {
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 availability verified above.
        #[allow(unsafe_code)]
        unsafe {
            matvec_ternary_tl2_avx2(lut_packed, w_scale, lut, x_scale, in_dim, out);
        }
    } else {
        matvec_ternary_tl2_scalar(lut_packed, w_scale, lut, x_scale, in_dim, out);
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(crate) fn matvec_ternary_lut(_: &[u8], _: &[f32], _: &[i8], _: f32, _: usize, _: &mut [f32]) {
    unreachable!("matvec_ternary_lut should not be called on this arch");
}

// ---------------------------------------------------------------------------
// TL2 (K = 3 lookup table) — x86_64 AVX2 path.
//
// Three ternary weights are packed into one 6-bit index:
//
//     idx = (w0 + 1) + 3·(w1 + 1) + 9·(w2 + 1)   w ∈ {-1, 0, +1}
//                                                → idx ∈ {0..=26}
//
// Per activation-triple `(x0, x1, x2)` we precompute a 32-byte LUT split
// into two 16-byte sub-LUTs so a pair of `vpshufb` lookups covers the full
// 32-entry index range per AVX2 lane:
//
//     sub_lut_0[idx]        = w0·x0 + w1·x1 + w2·x2   for idx ∈ 0..=15
//     sub_lut_1[idx - 16]   = …                       for idx ∈ 16..=26
//     sub_lut_1[idx - 16]   = 0                       for idx ∈ 27..=31
//
// 7-bit-minus-1 activation precision: `x_q >> 2` gives values in `[-32, 31]`,
// so `|w0·x0 + w1·x1 + w2·x2| ≤ 3·32 = 96` fits i8. The row-end scale is
// `4 · x_scale · w_scale[row]` (the ×4 undoes the `>> 2`).
//
// # Packing layout — 32-row output tiles
//
// To amortize the `vpshufb` across an AVX2 lane width, we tile the output
// dim by `M_TILE_TL2 = 32` rows. Per `(tile, triple)` we store 32 bytes,
// one 6-bit index per row. Tails: `in_dim` need not be divisible by 3.
// We pad the logical activation vector up to the next multiple of 3 with
// zeros, which means the trailing triple's weights come from a virtual
// position whose stored weight must also be 0 (packed as code for `w=0`).
//
// Total size: `(out_dim / 32) · n_triples · 32 = out_dim · n_triples` bytes,
// where `n_triples = ceil(in_dim / 3)` — about 4/3× I2_S storage.
//
// # Inner kernel (AVX2)
//
// For each 32-row tile and each triple:
//     indices  = load 32 bytes of packed indices
//     lut_0    = broadcast sub_lut_0 across both 128-bit lanes
//     lut_1    = broadcast sub_lut_1 across both 128-bit lanes
//     low_idx  = indices | (is_high ? 0x80 : 0)        # force MSB when ≥ 16
//     high_idx = indices − 16                          # negative when < 16
//     p0       = vpshufb(lut_0, low_idx)               # sub_lut_0[idx]  if idx < 16 else 0
//     p1       = vpshufb(lut_1, high_idx)              # sub_lut_1[idx−16] if idx ≥ 16 else 0
//     partial  = p0 | p1                               # exactly one is nonzero per lane
//     acc     += widen(partial)                        # i8 → i16, add to row accumulators
//
// Per-iter cost: 1 load + 2 pshufb + 1 blend + 2 widen + 2 add ≈ 8 AVX2 ops
// for 32 lanes × 3 weights per lane = 96 MAC-equivalents. About 12 MACs per
// AVX2 op, comparable to TL1 NEON's 6.4 MACs per NEON op (AVX2 is 2× wider).

// TL2 helpers are only called on x86_64 runtime (and from tests on any
// arch). On aarch64 non-test builds they're dead; the `allow(dead_code)`
// on each item silences the resulting warnings without hiding real misuse
// on x86_64 (where the LUT dispatch path pulls them in).

/// Weights per TL2 index.
#[cfg(any(test, target_arch = "x86_64"))]
pub(crate) const K_TL2: usize = 3;

/// Output rows processed per tile in TL2. Matches AVX2's 32-lane shape.
#[cfg(any(test, target_arch = "x86_64"))]
pub(crate) const M_TILE_TL2: usize = 32;

/// Bytes per TL2 LUT slice (one per triple): two 16-byte sub-LUTs.
#[cfg(any(test, target_arch = "x86_64"))]
pub(crate) const TL2_LUT_STRIDE: usize = 32;

/// Max `in_dim` supported by the i16-accumulator TL2 kernel without overflow.
///
/// Worst-case per-triple partial is `3 · 32 = 96`, and there are `ceil(in_dim/3)`
/// triples, so `|acc| ≤ 32 · in_dim` in the worst case. `i16` max is 32767 →
/// safe for `in_dim ≤ 1020`. We use 768 to match the `CLAUDE.md` d_model set.
#[cfg(any(test, target_arch = "x86_64"))]
pub(crate) const TL2_MAX_IN_DIM: usize = 768;

/// Number of TL2 triples for a given `in_dim` (rounded up — trailing
/// positions pad with zero).
#[cfg(any(test, target_arch = "x86_64"))]
pub(crate) const fn tl2_n_triples(in_dim: usize) -> usize {
    in_dim.div_ceil(K_TL2)
}

/// Packed TL2 size for an `out_dim × in_dim` ternary matrix.
#[cfg(any(test, target_arch = "x86_64"))]
pub(crate) const fn tl2_packed_bytes(out_dim: usize, in_dim: usize) -> usize {
    out_dim * tl2_n_triples(in_dim)
}

/// `true` when `out_dim`/`in_dim` are in the shape range the TL2 kernel
/// supports. `in_dim % 3` need not be zero (the kernel pads internally),
/// but `out_dim` must be divisible by `M_TILE_TL2`.
#[cfg(any(test, target_arch = "x86_64"))]
pub(crate) const fn tl2_supports(out_dim: usize, in_dim: usize) -> bool {
    out_dim % M_TILE_TL2 == 0 && in_dim <= TL2_MAX_IN_DIM && in_dim > 0
}

#[cfg(any(test, target_arch = "x86_64"))]
#[inline]
fn encode_tl2_triple(w0: i8, w1: i8, w2: i8) -> u8 {
    debug_assert!((-1..=1).contains(&w0) && (-1..=1).contains(&w1) && (-1..=1).contains(&w2));
    #[allow(clippy::cast_sign_loss)]
    let c0 = (w0 + 1) as u8;
    #[allow(clippy::cast_sign_loss)]
    let c1 = (w1 + 1) as u8;
    #[allow(clippy::cast_sign_loss)]
    let c2 = (w2 + 1) as u8;
    c0 + 3 * c1 + 9 * c2
}

/// Pack row-major ternary weights into the TL2 tiled layout.
///
/// `weights` is `out_dim × in_dim` row-major i8 in `{-1, 0, 1}`. Trailing
/// in-dim positions past `in_dim` are treated as zero (padding to the next
/// multiple of 3).
#[cfg(any(test, target_arch = "x86_64"))]
pub(crate) fn pack_tl2(weights: &[i8], tl2_packed: &mut [u8], out_dim: usize, in_dim: usize) {
    assert_eq!(weights.len(), out_dim * in_dim);
    assert_eq!(tl2_packed.len(), tl2_packed_bytes(out_dim, in_dim));
    assert!(tl2_supports(out_dim, in_dim));

    let n_triples = tl2_n_triples(in_dim);
    let n_tiles = out_dim / M_TILE_TL2;
    let tile_stride = n_triples * M_TILE_TL2;

    let get_w = |row: usize, j: usize| -> i8 {
        if j < in_dim {
            weights[row * in_dim + j]
        } else {
            0
        }
    };

    for t in 0..n_tiles {
        for tr in 0..n_triples {
            let j0 = 3 * tr;
            let j1 = 3 * tr + 1;
            let j2 = 3 * tr + 2;
            for r in 0..M_TILE_TL2 {
                let row = t * M_TILE_TL2 + r;
                let idx = encode_tl2_triple(get_w(row, j0), get_w(row, j1), get_w(row, j2));
                tl2_packed[t * tile_stride + tr * M_TILE_TL2 + r] = idx;
            }
        }
    }
}

/// Convert an existing `I2_S`-packed buffer into TL2 layout.
#[cfg(any(test, target_arch = "x86_64"))]
pub(crate) fn repack_i2s_to_tl2(
    i2s_packed: &[u8],
    tl2_packed: &mut [u8],
    out_dim: usize,
    in_dim: usize,
) {
    assert_eq!(i2s_packed.len(), out_dim * in_dim / 4);
    assert_eq!(tl2_packed.len(), tl2_packed_bytes(out_dim, in_dim));
    assert!(tl2_supports(out_dim, in_dim));
    assert_eq!(in_dim % QK_I2_S, 0);

    let mut flat = vec![0i8; out_dim * in_dim];
    unpack_i2s_to_rowmajor(i2s_packed, &mut flat, out_dim, in_dim);
    pack_tl2(&flat, tl2_packed, out_dim, in_dim);
}

/// Build the TL2 activation LUT.
///
/// `x_q` is the 8-bit absmax-quantized activation vector (length `in_dim`).
/// Internally we right-shift by 2 to get values in `[-32, 31]`, keeping the
/// three-term partial sum in `i8` range. The kernel multiplies the final
/// sum by `4 · x_scale` at row-write time to undo the shift.
///
/// `lut` must have length `n_triples · TL2_LUT_STRIDE` where
/// `n_triples = ceil(x_q.len() / 3)`, and **must be zero-initialized by the
/// caller**. We only write the 27 valid entries per triple; AVX2 `vpshufb`
/// returns zero for out-of-range indices, and the packer can only emit 0..26.
#[cfg(any(test, target_arch = "x86_64"))]
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn build_tl2_lut(x_q: &[i8], lut: &mut [i8]) {
    let n_triples = tl2_n_triples(x_q.len());
    debug_assert_eq!(lut.len(), n_triples * TL2_LUT_STRIDE);
    for tr in 0..n_triples {
        let j0 = 3 * tr;
        let j1 = 3 * tr + 1;
        let j2 = 3 * tr + 2;
        let x0 = if j0 < x_q.len() { x_q[j0] >> 2 } else { 0 };
        let x1 = if j1 < x_q.len() { x_q[j1] >> 2 } else { 0 };
        let x2 = if j2 < x_q.len() { x_q[j2] >> 2 } else { 0 };
        let slot = &mut lut[tr * TL2_LUT_STRIDE..tr * TL2_LUT_STRIDE + TL2_LUT_STRIDE];
        for w0 in -1i8..=1 {
            for w1 in -1i8..=1 {
                for w2 in -1i8..=1 {
                    // (w + 1) ∈ {0, 1, 2}; non-negative so the `as usize`
                    // casts are sign-safe, not a loss.
                    #[allow(clippy::cast_sign_loss)]
                    let idx =
                        ((w0 + 1) as usize) + 3 * ((w1 + 1) as usize) + 9 * ((w2 + 1) as usize);
                    let val = i32::from(w0) * i32::from(x0)
                        + i32::from(w1) * i32::from(x1)
                        + i32::from(w2) * i32::from(x2);
                    // val ∈ [-96, 96] — fits i8.
                    slot[idx] = val as i8;
                }
            }
        }
    }
}

/// Scalar TL2 matvec — correctness oracle for the AVX2 kernel. Takes a
/// pre-built LUT so that consumers sharing an activation vector rebuild
/// the LUT once instead of once per matvec.
#[cfg(any(test, target_arch = "x86_64"))]
#[inline(never)]
#[allow(clippy::cast_precision_loss)]
pub(crate) fn matvec_ternary_tl2_scalar(
    tl2_packed: &[u8],
    w_scale: &[f32],
    lut: &[i8],
    x_scale: f32,
    in_dim: usize,
    out: &mut [f32],
) {
    let out_dim = w_scale.len();
    debug_assert_eq!(tl2_packed.len(), tl2_packed_bytes(out_dim, in_dim));
    debug_assert_eq!(lut.len(), tl2_n_triples(in_dim) * TL2_LUT_STRIDE);
    debug_assert_eq!(out.len(), out_dim);
    debug_assert!(tl2_supports(out_dim, in_dim));

    if x_scale == 0.0 {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        return;
    }

    let n_triples = tl2_n_triples(in_dim);

    let n_tiles = out_dim / M_TILE_TL2;
    let tile_stride = n_triples * M_TILE_TL2;
    let combined_scale = 4.0 * x_scale;

    for t in 0..n_tiles {
        for r in 0..M_TILE_TL2 {
            let mut acc: i32 = 0;
            for tr in 0..n_triples {
                let idx = tl2_packed[t * tile_stride + tr * M_TILE_TL2 + r];
                acc += i32::from(lut[tr * TL2_LUT_STRIDE + usize::from(idx)]);
            }
            let row = t * M_TILE_TL2 + r;
            out[row] = (acc as f32) * combined_scale * w_scale[row];
        }
    }
}

/// AVX2 TL2 kernel.
///
/// **UNVERIFIED ON X86** — this codepath has only been built against the
/// spec and the scalar oracle. First real execution happens on an AVX2 host
/// (CI or the Hutter judging machine). The scalar kernel is the correctness
/// ground truth; any AVX2 vs. scalar disagreement must be treated as a bug
/// in the AVX2 implementation.
///
/// 32 output rows per tile. Per-triple, a pair of `vpshufb` lookups covers
/// the 27 valid index values (indices 0..15 via `sub_lut_0`, 16..26 via
/// `sub_lut_1`). The unused 27..31 range in `sub_lut_1` is zero so stray
/// indices read as zero (they should not occur from a correct packer).
#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code, clippy::cast_precision_loss, clippy::too_many_lines)]
unsafe fn matvec_ternary_tl2_avx2(
    tl2_packed: &[u8],
    w_scale: &[f32],
    lut: &[i8],
    x_scale: f32,
    in_dim: usize,
    out: &mut [f32],
) {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm_loadu_si128, _mm256_add_epi16, _mm256_and_si256,
        _mm256_broadcastsi128_si256, _mm256_castsi256_si128, _mm256_cmpgt_epi8,
        _mm256_cvtepi8_epi16, _mm256_cvtepi16_epi32, _mm256_cvtepi32_ps, _mm256_extracti128_si256,
        _mm256_loadu_ps, _mm256_loadu_si256, _mm256_mul_ps, _mm256_or_si256, _mm256_set1_epi8,
        _mm256_set1_ps, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_storeu_ps,
        _mm256_sub_epi8,
    };

    let out_dim = w_scale.len();
    debug_assert_eq!(tl2_packed.len(), tl2_packed_bytes(out_dim, in_dim));
    debug_assert_eq!(lut.len(), tl2_n_triples(in_dim) * TL2_LUT_STRIDE);
    debug_assert_eq!(out.len(), out_dim);
    debug_assert!(tl2_supports(out_dim, in_dim));

    if x_scale == 0.0 {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        return;
    }

    let n_triples = tl2_n_triples(in_dim);
    let n_tiles = out_dim / M_TILE_TL2;
    let tile_stride = n_triples * M_TILE_TL2;
    let combined_scale = 4.0 * x_scale;

    // SAFETY: AVX2 verified by dispatcher. All pointer derefs stay in range.
    unsafe {
        let sixteen_v = _mm256_set1_epi8(16);
        let fifteen_v = _mm256_set1_epi8(15);
        // `0x80 as i8 == -128` — the MSB-only bit pattern for pshufb's
        // "return zero" sentinel.
        let highbit_v = _mm256_set1_epi8(-128);
        let scale_v = _mm256_set1_ps(combined_scale);

        for t in 0..n_tiles {
            let mut acc_lo = _mm256_setzero_si256(); // 16 i16 (rows 0..15)
            let mut acc_hi = _mm256_setzero_si256(); // 16 i16 (rows 16..31)

            for tr in 0..n_triples {
                let indices = _mm256_loadu_si256(
                    tl2_packed
                        .as_ptr()
                        .add(t * tile_stride + tr * M_TILE_TL2)
                        .cast::<__m256i>(),
                );

                // Broadcast each per-triple 16-byte sub-LUT across both 128-bit lanes.
                let sub0_128: __m128i =
                    _mm_loadu_si128(lut.as_ptr().add(tr * TL2_LUT_STRIDE).cast::<__m128i>());
                let sub1_128: __m128i =
                    _mm_loadu_si128(lut.as_ptr().add(tr * TL2_LUT_STRIDE + 16).cast::<__m128i>());
                let lut_0 = _mm256_broadcastsi128_si256(sub0_128);
                let lut_1 = _mm256_broadcastsi128_si256(sub1_128);

                // Route indices: lanes with idx ≥ 16 set the MSB of their
                // low-path index (→ pshufb returns 0); lanes with idx < 16
                // produce a negative high-path index (same outcome for the
                // high-path pshufb).
                let is_high = _mm256_cmpgt_epi8(indices, fifteen_v);
                let force_msb = _mm256_and_si256(is_high, highbit_v);
                let low_idx = _mm256_or_si256(indices, force_msb);
                let high_idx = _mm256_sub_epi8(indices, sixteen_v);

                let p0 = _mm256_shuffle_epi8(lut_0, low_idx);
                let p1 = _mm256_shuffle_epi8(lut_1, high_idx);
                let partial = _mm256_or_si256(p0, p1); // one is zero per lane

                // Widen i8 → i16 in two halves, accumulate.
                let lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(partial));
                let hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(partial));
                acc_lo = _mm256_add_epi16(acc_lo, lo16);
                acc_hi = _mm256_add_epi16(acc_hi, hi16);
            }

            // Widen i16 → i32 → f32, scale, store. 4 chunks of 8 rows.
            let row_base = t * M_TILE_TL2;
            let i32_0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc_lo));
            let i32_1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(acc_lo));
            let i32_2 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc_hi));
            let i32_3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(acc_hi));

            let f_0 = _mm256_cvtepi32_ps(i32_0);
            let f_1 = _mm256_cvtepi32_ps(i32_1);
            let f_2 = _mm256_cvtepi32_ps(i32_2);
            let f_3 = _mm256_cvtepi32_ps(i32_3);

            let ws_0 = _mm256_loadu_ps(w_scale.as_ptr().add(row_base));
            let ws_1 = _mm256_loadu_ps(w_scale.as_ptr().add(row_base + 8));
            let ws_2 = _mm256_loadu_ps(w_scale.as_ptr().add(row_base + 16));
            let ws_3 = _mm256_loadu_ps(w_scale.as_ptr().add(row_base + 24));

            // Match scalar order: (f * combined_scale) * w_scale.
            let r_0 = _mm256_mul_ps(_mm256_mul_ps(f_0, scale_v), ws_0);
            let r_1 = _mm256_mul_ps(_mm256_mul_ps(f_1, scale_v), ws_1);
            let r_2 = _mm256_mul_ps(_mm256_mul_ps(f_2, scale_v), ws_2);
            let r_3 = _mm256_mul_ps(_mm256_mul_ps(f_3, scale_v), ws_3);

            _mm256_storeu_ps(out.as_mut_ptr().add(row_base), r_0);
            _mm256_storeu_ps(out.as_mut_ptr().add(row_base + 8), r_1);
            _mm256_storeu_ps(out.as_mut_ptr().add(row_base + 16), r_2);
            _mm256_storeu_ps(out.as_mut_ptr().add(row_base + 24), r_3);
        }
    }
}

#[cfg(test)]
// Correctness tests assert exact f32 equality — integer accumulator + one
// final f32 multiply means bit-identical results across kernels.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn rand_case(rng: &mut rand::rngs::StdRng) -> (Vec<u8>, Vec<f32>, Vec<i8>, f32, usize) {
        use rand::Rng;
        let out_dim = 1 + rng.random_range(0..4);
        // Use multiple of 128 so AVX2 runs too; NEON only needs multiple of QK_I2_S.
        let blocks = 2 * (1 + rng.random_range(0..3));
        let in_dim = QK_I2_S * blocks;
        let weights: Vec<i8> = (0..out_dim * in_dim)
            .map(|_| rng.random_range(-1..=1))
            .collect();
        let mut packed = vec![0u8; out_dim * in_dim / 4];
        pack_ternary(&weights, &mut packed);
        let w_scale: Vec<f32> = (0..out_dim).map(|_| rng.random_range(0.01..2.0)).collect();
        let x_q: Vec<i8> = (0..in_dim).map(|_| rng.random_range(-127..=127)).collect();
        let x_scale = rng.random_range(0.0001..0.5);
        (packed, w_scale, x_q, x_scale, out_dim)
    }

    #[allow(clippy::cast_precision_loss)]
    fn naive_matvec(
        weights: &[i8],
        w_scale: &[f32],
        x_q: &[i8],
        x_scale: f32,
        out_dim: usize,
        in_dim: usize,
        out: &mut [f32],
    ) {
        for row in 0..out_dim {
            let mut acc: i32 = 0;
            for j in 0..in_dim {
                acc += i32::from(weights[row * in_dim + j]) * i32::from(x_q[j]);
            }
            out[row] = (acc as f32) * w_scale[row] * x_scale;
        }
    }

    #[test]
    fn pack_round_trip_scalar() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x1234_5678);
        for _ in 0..10 {
            let (packed, w_scale, x_q, x_scale, out_dim) = rand_case(&mut rng);
            let in_dim = x_q.len();

            // Recover original weights from the I2_S-packed bytes.
            let mut weights = vec![0i8; out_dim * in_dim];
            unpack_i2s_to_rowmajor(&packed, &mut weights, out_dim, in_dim);

            let mut out_naive = vec![0.0_f32; out_dim];
            naive_matvec(
                &weights,
                &w_scale,
                &x_q,
                x_scale,
                out_dim,
                in_dim,
                &mut out_naive,
            );

            let mut out_scalar = vec![0.0_f32; out_dim];
            matvec_ternary_scalar(&packed, &w_scale, &x_q, x_scale, &mut out_scalar);

            assert_eq!(out_naive, out_scalar);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar() {
        use rand::SeedableRng;
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xFEED_C0DE);
        for _ in 0..20 {
            let (packed, w_scale, x_q, x_scale, out_dim) = rand_case(&mut rng);
            let mut out_scalar = vec![0.0_f32; out_dim];
            let mut out_neon = vec![0.0_f32; out_dim];
            matvec_ternary_scalar(&packed, &w_scale, &x_q, x_scale, &mut out_scalar);
            #[allow(unsafe_code)]
            // SAFETY: dotprod check above.
            unsafe {
                matvec_ternary_neon(&packed, &w_scale, &x_q, x_scale, &mut out_neon);
            }
            assert_eq!(out_scalar, out_neon);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar() {
        use rand::SeedableRng;
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xBEEF_FACE);
        for _ in 0..20 {
            let (packed, w_scale, x_q, x_scale, out_dim) = rand_case(&mut rng);
            let mut out_scalar = vec![0.0_f32; out_dim];
            let mut out_avx = vec![0.0_f32; out_dim];
            matvec_ternary_scalar(&packed, &w_scale, &x_q, x_scale, &mut out_scalar);
            #[allow(unsafe_code)]
            // SAFETY: avx2 check above.
            unsafe {
                matvec_ternary_avx2(&packed, &w_scale, &x_q, x_scale, &mut out_avx);
            }
            assert_eq!(out_scalar, out_avx);
        }
    }

    #[test]
    fn activation_quant_zero_vector() {
        let x = [0.0_f32; QK_I2_S];
        let mut xq = [0i8; QK_I2_S];
        let s = quantize_activations(&x, &mut xq);
        assert_eq!(s, 0.0);
        assert_eq!(xq, [0i8; QK_I2_S]);
    }

    /// Test-only naive matvec that mimics the TL1 kernel's activation
    /// pre-shift, so scalar / NEON TL1 kernels can be compared against
    /// ground truth to within quantization noise.
    #[allow(clippy::cast_precision_loss)]
    fn naive_matvec_tl1(
        weights: &[i8],
        w_scale: &[f32],
        x_q: &[i8],
        x_scale: f32,
        out_dim: usize,
        in_dim: usize,
        out: &mut [f32],
    ) {
        for row in 0..out_dim {
            let mut acc: i32 = 0;
            for j in 0..in_dim {
                let x_shifted = i32::from(x_q[j] >> 1);
                acc += i32::from(weights[row * in_dim + j]) * x_shifted;
            }
            out[row] = (acc as f32) * 2.0 * x_scale * w_scale[row];
        }
    }

    /// `(weights, tl1_packed, w_scale, x_q, x_scale, out_dim, in_dim)`.
    type Tl1Case = (Vec<i8>, Vec<u8>, Vec<f32>, Vec<i8>, f32, usize, usize);

    fn rand_tl1_case(rng: &mut rand::rngs::StdRng) -> Tl1Case {
        use rand::Rng;
        // out_dim divisible by 16, in_dim divisible by QK_I2_S (so I2_S
        // repack stays valid) and ≤ TL1_MAX_IN_DIM.
        let n_tiles = 1 + rng.random_range(0..4);
        let out_dim = n_tiles * M_TILE_TL1;
        let blocks = 1 + rng.random_range(0..3);
        let in_dim = QK_I2_S * blocks; // 64 or 128 or 192
        let weights: Vec<i8> = (0..out_dim * in_dim)
            .map(|_| rng.random_range(-1..=1))
            .collect();
        let mut tl1 = vec![0u8; tl1_packed_bytes(out_dim, in_dim)];
        pack_tl1(&weights, &mut tl1, out_dim, in_dim);
        let w_scale: Vec<f32> = (0..out_dim).map(|_| rng.random_range(0.01..2.0)).collect();
        let x_q: Vec<i8> = (0..in_dim).map(|_| rng.random_range(-127..=127)).collect();
        let x_scale = rng.random_range(0.0001..0.5);
        (weights, tl1, w_scale, x_q, x_scale, out_dim, in_dim)
    }

    #[test]
    fn tl1_lut_valid_entries_are_partial_sums() {
        // Verify LUT is built correctly for a simple, hand-checkable pair.
        let x_q = [60i8, -40, 0, 0];
        let mut lut = vec![0i8; 2 * TL1_LUT_STRIDE];
        build_tl1_lut(&x_q, &mut lut);
        // pair 0: x0 = 60 >> 1 = 30, x1 = -40 >> 1 = -20 (arith right shift
        // on negative rounds toward -inf, so -40 >> 1 = -20).
        let x0 = 60i8 >> 1;
        let x1 = -40i8 >> 1;
        for w0 in -1i8..=1 {
            for w1 in -1i8..=1 {
                #[allow(clippy::cast_sign_loss)]
                let idx = ((w0 + 1) as usize) + 3 * ((w1 + 1) as usize);
                let expected = i32::from(w0) * i32::from(x0) + i32::from(w1) * i32::from(x1);
                assert_eq!(i32::from(lut[idx]), expected, "idx {idx}, w0 {w0}, w1 {w1}");
            }
        }
    }

    /// Test helper: build TL1 LUT from `x_q` and run the scalar kernel.
    fn run_tl1_scalar(tl1: &[u8], w_scale: &[f32], x_q: &[i8], x_scale: f32, out: &mut [f32]) {
        let mut lut = vec![0i8; (x_q.len() / 2) * TL1_LUT_STRIDE];
        build_tl1_lut(x_q, &mut lut);
        matvec_ternary_tl1_scalar(tl1, w_scale, &lut, x_scale, x_q.len(), out);
    }

    #[test]
    fn pack_tl1_roundtrip_via_scalar_matches_naive() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xA1_B2_C3_D4);
        for _ in 0..10 {
            let (weights, tl1, w_scale, x_q, x_scale, out_dim, in_dim) = rand_tl1_case(&mut rng);
            let mut out_scalar = vec![0.0_f32; out_dim];
            run_tl1_scalar(&tl1, &w_scale, &x_q, x_scale, &mut out_scalar);
            let mut out_naive = vec![0.0_f32; out_dim];
            naive_matvec_tl1(
                &weights,
                &w_scale,
                &x_q,
                x_scale,
                out_dim,
                in_dim,
                &mut out_naive,
            );
            assert_eq!(out_scalar, out_naive);
        }
    }

    #[test]
    fn repack_i2s_to_tl1_matches_direct_pack() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x11_22_33_44);
        for _ in 0..5 {
            let (weights, tl1_direct, _, _, _, out_dim, in_dim) = rand_tl1_case(&mut rng);
            let mut i2s = vec![0u8; out_dim * in_dim / 4];
            pack_ternary(&weights, &mut i2s);
            let mut tl1_via_i2s = vec![0u8; tl1_packed_bytes(out_dim, in_dim)];
            repack_i2s_to_tl1(&i2s, &mut tl1_via_i2s, out_dim, in_dim);
            assert_eq!(tl1_direct, tl1_via_i2s);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn tl1_neon_matches_tl1_scalar() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xDE_AD_BE_EF);
        for _ in 0..20 {
            let (_, tl1, w_scale, x_q, x_scale, out_dim, in_dim) = rand_tl1_case(&mut rng);
            let mut lut = vec![0i8; (in_dim / 2) * TL1_LUT_STRIDE];
            build_tl1_lut(&x_q, &mut lut);
            let mut out_scalar = vec![0.0_f32; out_dim];
            matvec_ternary_tl1_scalar(&tl1, &w_scale, &lut, x_scale, in_dim, &mut out_scalar);
            let mut out_neon = vec![0.0_f32; out_dim];
            #[allow(unsafe_code)]
            // SAFETY: NEON is baseline on aarch64.
            unsafe {
                matvec_ternary_tl1_neon(&tl1, &w_scale, &lut, x_scale, in_dim, &mut out_neon);
            }
            assert_eq!(out_scalar, out_neon);
        }
    }

    /// TL1 vs I2_S: not bit-exact (TL1 arith-shifts activations by 1) but
    /// the per-row absolute error is provably `≤ in_dim · x_scale · |w_scale|`.
    ///
    /// Derivation: the residual `x_q[j] - 2·(x_q[j] >> 1)` equals the low
    /// bit of `x_q[j]` (both signs; arith shift rounds toward -∞). Each
    /// weight is in `{-1, 0, +1}`, so the per-row accumulator error in
    /// integer space is `≤ in_dim`. Multiplying by `x_scale · w_scale[row]`
    /// converts to f32 output space.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn tl1_scalar_within_worstcase_bound_of_i2s_scalar() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xCA_FE_BA_BE);
        for _ in 0..10 {
            let (weights, tl1, w_scale, x_q, x_scale, out_dim, in_dim) = rand_tl1_case(&mut rng);
            let mut i2s_packed = vec![0u8; out_dim * in_dim / 4];
            pack_ternary(&weights, &mut i2s_packed);
            let mut out_i2s = vec![0.0_f32; out_dim];
            matvec_ternary_scalar(&i2s_packed, &w_scale, &x_q, x_scale, &mut out_i2s);
            let mut out_tl1 = vec![0.0_f32; out_dim];
            run_tl1_scalar(&tl1, &w_scale, &x_q, x_scale, &mut out_tl1);
            for row in 0..out_dim {
                let bound = (in_dim as f32) * x_scale * w_scale[row].abs();
                let diff = (out_i2s[row] - out_tl1[row]).abs();
                assert!(
                    diff <= bound * 1.001, // tiny f32-rounding headroom
                    "row {row}: i2s = {}, tl1 = {}, diff = {}, bound = {}",
                    out_i2s[row],
                    out_tl1[row],
                    diff,
                    bound
                );
            }
        }
    }

    // ---- TL2 tests ---------------------------------------------------------

    /// Naive matvec mimicking the TL2 activation pre-shift (`x_q >> 2`) so
    /// the scalar TL2 kernel can be compared against ground truth bit-exactly
    /// (modulo f32 multiply order).
    #[allow(clippy::cast_precision_loss)]
    fn naive_matvec_tl2(
        weights: &[i8],
        w_scale: &[f32],
        x_q: &[i8],
        x_scale: f32,
        out_dim: usize,
        in_dim: usize,
        out: &mut [f32],
    ) {
        for row in 0..out_dim {
            let mut acc: i32 = 0;
            for j in 0..in_dim {
                acc += i32::from(weights[row * in_dim + j]) * i32::from(x_q[j] >> 2);
            }
            out[row] = (acc as f32) * 4.0 * x_scale * w_scale[row];
        }
    }

    /// `(weights, tl2_packed, w_scale, x_q, x_scale, out_dim, in_dim)`.
    type Tl2Case = (Vec<i8>, Vec<u8>, Vec<f32>, Vec<i8>, f32, usize, usize);

    fn rand_tl2_case(rng: &mut rand::rngs::StdRng) -> Tl2Case {
        use rand::Rng;
        // out_dim divisible by 32 (TL2 tile); in_dim a multiple of 64 so we
        // can also pack as I2_S for cross-checks. in_dim=64 covers the
        // triple-padding edge case (64 % 3 != 0).
        let n_tiles = 1 + rng.random_range(0..3);
        let out_dim = n_tiles * M_TILE_TL2;
        let blocks = 1 + rng.random_range(0..3);
        let in_dim = QK_I2_S * blocks;
        let weights: Vec<i8> = (0..out_dim * in_dim)
            .map(|_| rng.random_range(-1..=1))
            .collect();
        let mut tl2 = vec![0u8; tl2_packed_bytes(out_dim, in_dim)];
        pack_tl2(&weights, &mut tl2, out_dim, in_dim);
        let w_scale: Vec<f32> = (0..out_dim).map(|_| rng.random_range(0.01..2.0)).collect();
        let x_q: Vec<i8> = (0..in_dim).map(|_| rng.random_range(-127..=127)).collect();
        let x_scale = rng.random_range(0.0001..0.5);
        (weights, tl2, w_scale, x_q, x_scale, out_dim, in_dim)
    }

    #[test]
    fn tl2_lut_valid_entries_are_partial_sums() {
        // One triple, hand-checkable.
        let x_q = [60i8, -40, 80];
        let mut lut = vec![0i8; tl2_n_triples(x_q.len()) * TL2_LUT_STRIDE];
        build_tl2_lut(&x_q, &mut lut);
        let x0 = 60i8 >> 2;
        let x1 = -40i8 >> 2;
        let x2 = 80i8 >> 2;
        for w0 in -1i8..=1 {
            for w1 in -1i8..=1 {
                for w2 in -1i8..=1 {
                    #[allow(clippy::cast_sign_loss)]
                    let idx =
                        ((w0 + 1) as usize) + 3 * ((w1 + 1) as usize) + 9 * ((w2 + 1) as usize);
                    let expected = i32::from(w0) * i32::from(x0)
                        + i32::from(w1) * i32::from(x1)
                        + i32::from(w2) * i32::from(x2);
                    assert_eq!(
                        i32::from(lut[idx]),
                        expected,
                        "idx {idx}, w0 {w0}, w1 {w1}, w2 {w2}"
                    );
                }
            }
        }
        // idx 27..31 must be zero.
        for (idx, b) in lut.iter().enumerate().take(32).skip(27) {
            assert_eq!(*b, 0, "invalid idx {idx} should be zero");
        }
    }

    /// Test helper: build TL2 LUT from `x_q` and run the scalar kernel.
    fn run_tl2_scalar(tl2: &[u8], w_scale: &[f32], x_q: &[i8], x_scale: f32, out: &mut [f32]) {
        let mut lut = vec![0i8; tl2_n_triples(x_q.len()) * TL2_LUT_STRIDE];
        build_tl2_lut(x_q, &mut lut);
        matvec_ternary_tl2_scalar(tl2, w_scale, &lut, x_scale, x_q.len(), out);
    }

    #[test]
    fn pack_tl2_roundtrip_via_scalar_matches_naive() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x55_66_77_88);
        for _ in 0..10 {
            let (weights, tl2, w_scale, x_q, x_scale, out_dim, in_dim) = rand_tl2_case(&mut rng);
            let mut out_scalar = vec![0.0_f32; out_dim];
            run_tl2_scalar(&tl2, &w_scale, &x_q, x_scale, &mut out_scalar);
            let mut out_naive = vec![0.0_f32; out_dim];
            naive_matvec_tl2(
                &weights,
                &w_scale,
                &x_q,
                x_scale,
                out_dim,
                in_dim,
                &mut out_naive,
            );
            assert_eq!(out_scalar, out_naive);
        }
    }

    #[test]
    fn repack_i2s_to_tl2_matches_direct_pack() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x99_AA_BB_CC);
        for _ in 0..5 {
            let (weights, tl2_direct, _, _, _, out_dim, in_dim) = rand_tl2_case(&mut rng);
            let mut i2s = vec![0u8; out_dim * in_dim / 4];
            pack_ternary(&weights, &mut i2s);
            let mut tl2_via_i2s = vec![0u8; tl2_packed_bytes(out_dim, in_dim)];
            repack_i2s_to_tl2(&i2s, &mut tl2_via_i2s, out_dim, in_dim);
            assert_eq!(tl2_direct, tl2_via_i2s);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn tl2_avx2_matches_tl2_scalar() {
        use rand::SeedableRng;
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xD0_D0_D0_D0);
        for _ in 0..20 {
            let (_, tl2, w_scale, x_q, x_scale, out_dim, in_dim) = rand_tl2_case(&mut rng);
            let mut lut = vec![0i8; tl2_n_triples(in_dim) * TL2_LUT_STRIDE];
            build_tl2_lut(&x_q, &mut lut);
            let mut out_scalar = vec![0.0_f32; out_dim];
            matvec_ternary_tl2_scalar(&tl2, &w_scale, &lut, x_scale, in_dim, &mut out_scalar);
            let mut out_avx2 = vec![0.0_f32; out_dim];
            #[allow(unsafe_code)]
            // SAFETY: AVX2 checked above.
            unsafe {
                matvec_ternary_tl2_avx2(&tl2, &w_scale, &lut, x_scale, in_dim, &mut out_avx2);
            }
            assert_eq!(out_scalar, out_avx2);
        }
    }

    /// TL2 vs I2_S: per-row error `≤ 3 · in_dim · x_scale · |w_scale|` since
    /// `x_q >> 2` discards 2 bits per activation and each weight contributes
    /// up to `|w| ≤ 1`. See `tl1_scalar_within_worstcase_bound_of_i2s_scalar`
    /// for the TL1 derivation.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn tl2_scalar_within_worstcase_bound_of_i2s_scalar() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xE0_E0_E0_E0);
        for _ in 0..10 {
            let (weights, tl2, w_scale, x_q, x_scale, out_dim, in_dim) = rand_tl2_case(&mut rng);
            let mut i2s_packed = vec![0u8; out_dim * in_dim / 4];
            pack_ternary(&weights, &mut i2s_packed);
            let mut out_i2s = vec![0.0_f32; out_dim];
            matvec_ternary_scalar(&i2s_packed, &w_scale, &x_q, x_scale, &mut out_i2s);
            let mut out_tl2 = vec![0.0_f32; out_dim];
            run_tl2_scalar(&tl2, &w_scale, &x_q, x_scale, &mut out_tl2);
            for row in 0..out_dim {
                let bound = 3.0 * (in_dim as f32) * x_scale * w_scale[row].abs();
                let diff = (out_i2s[row] - out_tl2[row]).abs();
                assert!(
                    diff <= bound * 1.001,
                    "row {row}: i2s = {}, tl2 = {}, diff = {}, bound = {}",
                    out_i2s[row],
                    out_tl2[row],
                    diff,
                    bound
                );
            }
        }
    }
}
