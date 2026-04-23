//! `BitNet` ternary matvec kernels.
//!
//! Weights are ternary `{-1, 0, +1}` with a per-output-row `f32` absmean scale.
//! Storage is 2 bits per weight, row-major, 4 weights per byte, using
//! **sequential codes** (matches Microsoft's `bitnet.cpp`):
//!
//! | code | value |
//! |-----:|------:|
//! | `0b00 = 0` | `-1` |
//! | `0b01 = 1` | ` 0` |
//! | `0b10 = 2` | `+1` |
//! | `0b11`     | unused |
//!
//! Unpack is a single subtract: `w_i8 = (code as i8) - 1`. That cleanly
//! vectorizes on both NEON and AVX2.
//!
//! **Activation quantization** is 8-bit per-vector (per-matvec-input)
//! absmax: one shared scale per activation vector. Matches the `BitNet` b1.58
//! paper and is slightly more accurate than `bitnet.cpp`'s per-tensor scheme.
//!
//! **Kernels.** Three implementations, dispatched at runtime by
//! [`matvec_ternary`]:
//!
//! - [`matvec_ternary_scalar`]: portable fallback. Pure scalar arithmetic,
//!   auto-vectorized by LLVM. Always present.
//! - `matvec_ternary_neon` (aarch64 + `dotprod`): NEON `vdotq_s32` — one
//!   instruction computes 4 independent dot-products of 4 consecutive i8
//!   pairs (i8×i8 → i32×4 MAC). Requires ARMv8.2 SDOT, which every Apple
//!   M-series chip has. We enable this via the nightly
//!   `stdarch_neon_dotprod` feature at the crate root.
//! - `matvec_ternary_avx2` (`x86_64` + `avx2`): AVX2 `pmaddubsw` (still a real
//!   multiply, but i8 × i8 → i16 pairwise at SIMD rate — the same pattern
//!   `bitnet.cpp`'s `I2_S` kernel uses).
//!
//! The "multiplication-free" framing in the `BitNet` paper is a hardware-ASIC
//! claim; on CPUs the real win is 4× packing density + SIMD integer MAC
//! throughput, not multiply elimination. (For true multiply-free code, see
//! `bitnet.cpp`'s `TL1`/`TL2` lookup-table kernels — deferred for now.)
//!
//! The scalar kernel remains the correctness oracle — every SIMD kernel is
//! diff-tested against it.

/// Quantize `x` into 8-bit signed integers with a shared absmax scale.
///
/// Returns the scale `s` such that `x[i] ≈ s * (x_q[i] as f32)`. Zero-input
/// vectors return `s = 0.0` and an all-zero `x_q` (matvec short-circuits on
/// zero scale).
///
/// The `f32 → i8` cast is intentional: values are clamped to `[-127, 127]`
/// immediately before the cast, so truncation is the defined behavior.
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

/// Dispatch entry point.
///
/// On aarch64 NEON is always available (part of the ABI) so we jump
/// unconditionally. On `x86_64` we runtime-detect AVX2 — CPUID lookup is
/// cached by the stdlib so the branch is ~free after the first call.
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
            // SAFETY: AVX2 availability is checked above.
            #[allow(unsafe_code)]
            unsafe {
                matvec_ternary_avx2(packed, w_scale, x_q, x_scale, out);
            }
            return;
        }
    }

    // Fallback path: any target without a detected SIMD kernel.
    matvec_ternary_scalar(packed, w_scale, x_q, x_scale, out);
}

/// Reference kernel — walks the packed weights one byte at a time.
///
/// Used as the correctness oracle for every SIMD kernel.
///
/// `i32 → f32` at output time is lossless at our model sizes; the
/// quantization noise dominates. `code as i8` is safe because `code & 0b11`
/// is always in `0..=3`, which fits in positive `i8`.
#[cfg(test)]
#[inline(never)]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_wrap)]
pub(crate) fn matvec_ternary_ref(
    packed: &[u8],
    w_scale: &[f32],
    x_q: &[i8],
    x_scale: f32,
    out: &mut [f32],
) {
    let out_dim = w_scale.len();
    let in_dim = x_q.len();
    debug_assert_eq!(packed.len(), (out_dim * in_dim).div_ceil(4));
    debug_assert_eq!(out.len(), out_dim);
    debug_assert_eq!(in_dim % 4, 0, "in_dim must be divisible by 4");

    if x_scale == 0.0 {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        return;
    }

    let packed_per_row = in_dim / 4;
    for row in 0..out_dim {
        let row_packed = &packed[row * packed_per_row..(row + 1) * packed_per_row];
        let mut acc: i32 = 0;
        for (byte_idx, &p) in row_packed.iter().enumerate() {
            let j0 = byte_idx * 4;
            for k in 0..4 {
                let code = (p >> (2 * k)) & 0b11;
                let w = i32::from(code as i8) - 1;
                acc += i32::from(x_q[j0 + k]) * w;
            }
        }
        out[row] = (acc as f32) * w_scale[row] * x_scale;
    }
}

/// Scalar fallback — 32-weight inner block, pure scalar arithmetic.
///
/// With `-C target-cpu=native` LLVM will auto-vectorize on most modern
/// targets. Explicit SIMD kernels (see NEON / AVX2 below) typically beat
/// this by 4–8× but are architecture-specific; this path is the portable
/// baseline and the LSB-backstop when feature detection fails.
///
/// `i32 → f32` precision loss is bounded by quantization noise; `code as i8`
/// is safe because `code & 0b11` is always in `0..=3`.
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
    debug_assert_eq!(packed.len(), (out_dim * in_dim).div_ceil(4));
    debug_assert_eq!(out.len(), out_dim);
    debug_assert_eq!(
        in_dim % 32,
        0,
        "scalar kernel requires in_dim divisible by 32"
    );

    if x_scale == 0.0 {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        return;
    }

    let packed_per_row = in_dim / 4;
    let blocks = in_dim / 32;

    for row in 0..out_dim {
        let row_packed = &packed[row * packed_per_row..(row + 1) * packed_per_row];
        let mut acc: i32 = 0;

        for b in 0..blocks {
            let byte_off = b * 8;
            let x_off = b * 32;
            let bytes = &row_packed[byte_off..byte_off + 8];
            let xs = &x_q[x_off..x_off + 32];

            for k in 0..32 {
                let p = bytes[k / 4];
                let shift = (k % 4) * 2;
                let code = (p >> shift) & 0b11;
                let w = i32::from(code as i8) - 1;
                acc += i32::from(xs[k]) * w;
            }
        }

        out[row] = (acc as f32) * w_scale[row] * x_scale;
    }
}

/// NEON helper: load 4 packed bytes at `row_packed + byte_off`, unpack them
/// into 16 i8 ternary weights.
///
/// `#[inline]` (not `#[inline(always)]` — incompatible with
/// `#[target_feature]`) plus the shared codegen unit is enough for LLVM to
/// fold this into the hot loop and keep the constant shuffle/shift vectors
/// in registers.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon,dotprod")]
#[allow(unsafe_code)]
unsafe fn neon_unpack_weights(
    row_packed: *const u8,
    byte_off: usize,
    shuf_idx: std::arch::aarch64::uint8x16_t,
    shift_i8: std::arch::aarch64::int8x16_t,
    mask: std::arch::aarch64::uint8x16_t,
    one_s8: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int8x16_t {
    use std::arch::aarch64::{
        vandq_u8, vdupq_n_u32, vqtbl1q_u8, vreinterpretq_s8_u8, vreinterpretq_u8_u32, vshlq_u8,
        vsubq_s8,
    };
    // SAFETY: caller guarantees `byte_off + 4 <= row-packed length`.
    unsafe {
        let p = row_packed.add(byte_off);
        let w32 = u32::from_le_bytes([*p, *p.add(1), *p.add(2), *p.add(3)]);
        let packed_u8 = vreinterpretq_u8_u32(vdupq_n_u32(w32));
        let broadcast = vqtbl1q_u8(packed_u8, shuf_idx);
        let shifted = vshlq_u8(broadcast, shift_i8);
        let codes_u8 = vandq_u8(shifted, mask);
        vsubq_s8(vreinterpretq_s8_u8(codes_u8), one_s8)
    }
}

/// NEON + dotprod kernel. Processes 16 ternary weights per iteration:
///
/// 1. Load 4 packed bytes = 16 codes, broadcast across 16 lanes via
///    `vqtbl1q_u8` with index `[0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]`.
/// 2. Per-lane right-shift by `[0,2,4,6, …]` via `vshlq_u8` with the NEON
///    negative-shift convention.
/// 3. Mask `& 0b11`, reinterpret as i8, subtract 1 → i8 in `{-1, 0, +1}`.
/// 4. One `vdotq_s32` instruction: i8×16 weights × i8×16 activations,
///    summed in groups of 4 into an i32×4 accumulator.
///
/// Requires ARMv8.2 SDOT (every Apple M-series chip has it). A runtime
/// feature check in [`matvec_ternary`] gates the call; if SDOT is absent
/// we fall through to [`matvec_ternary_scalar`].
#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[target_feature(enable = "neon,dotprod")]
#[allow(
    unsafe_code,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::unreadable_literal
)]
unsafe fn matvec_ternary_neon(
    packed: &[u8],
    w_scale: &[f32],
    x_q: &[i8],
    x_scale: f32,
    out: &mut [f32],
) {
    use std::arch::aarch64::{
        int8x16_t, uint8x16_t, vaddvq_s32, vdotq_s32, vdupq_n_s8, vdupq_n_s32, vdupq_n_u8,
        vld1q_s8, vld1q_u8,
    };

    let out_dim = w_scale.len();
    let in_dim = x_q.len();
    debug_assert_eq!(packed.len(), (out_dim * in_dim).div_ceil(4));
    debug_assert_eq!(out.len(), out_dim);
    debug_assert_eq!(
        in_dim % 16,
        0,
        "NEON kernel requires in_dim divisible by 16"
    );

    if x_scale == 0.0 {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        return;
    }

    let packed_per_row = in_dim / 4;
    let blocks = in_dim / 16;

    // SAFETY: NEON and SDOT (dotprod) are enabled via `#[target_feature]`
    // on this fn; the dispatcher verifies SDOT availability before calling.
    // All pointer reads are in-bounds given the debug-asserted lengths.
    //
    // **Row-blocking by 4.** Each iteration of the outer loop consumes the
    // activation vector once and feeds it into 4 independent row
    // accumulators, so the activation loads are shared across 4 output
    // rows. For `out_dim` not divisible by 4, the tail rows fall through
    // to a 1-at-a-time path.
    unsafe {
        let shuf_idx = vld1q_u8([0u8, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3].as_ptr());
        let shift_u8 = vld1q_u8(
            [
                0u8, 0xFEu8, 0xFCu8, 0xFAu8, 0u8, 0xFEu8, 0xFCu8, 0xFAu8, 0u8, 0xFEu8, 0xFCu8,
                0xFAu8, 0u8, 0xFEu8, 0xFCu8, 0xFAu8,
            ]
            .as_ptr(),
        );
        let shift_i8 = core::mem::transmute::<uint8x16_t, int8x16_t>(shift_u8);
        let mask = vdupq_n_u8(0x03);
        let one_s8 = vdupq_n_s8(1);

        let mut row = 0;

        // 4-row blocked path.
        while row + 4 <= out_dim {
            let r0_base = row * packed_per_row;
            let r1_base = r0_base + packed_per_row;
            let r2_base = r1_base + packed_per_row;
            let r3_base = r2_base + packed_per_row;
            let r0 = packed.as_ptr().add(r0_base);
            let r1 = packed.as_ptr().add(r1_base);
            let r2 = packed.as_ptr().add(r2_base);
            let r3 = packed.as_ptr().add(r3_base);

            let mut a0 = vdupq_n_s32(0);
            let mut a1 = vdupq_n_s32(0);
            let mut a2 = vdupq_n_s32(0);
            let mut a3 = vdupq_n_s32(0);

            for b in 0..blocks {
                let byte_off = b * 4;
                let x_off = b * 16;
                let xv = vld1q_s8(x_q.as_ptr().add(x_off));

                let w0 = neon_unpack_weights(r0, byte_off, shuf_idx, shift_i8, mask, one_s8);
                let w1 = neon_unpack_weights(r1, byte_off, shuf_idx, shift_i8, mask, one_s8);
                let w2 = neon_unpack_weights(r2, byte_off, shuf_idx, shift_i8, mask, one_s8);
                let w3 = neon_unpack_weights(r3, byte_off, shuf_idx, shift_i8, mask, one_s8);

                a0 = vdotq_s32(a0, w0, xv);
                a1 = vdotq_s32(a1, w1, xv);
                a2 = vdotq_s32(a2, w2, xv);
                a3 = vdotq_s32(a3, w3, xv);
            }

            out[row] = (vaddvq_s32(a0) as f32) * w_scale[row] * x_scale;
            out[row + 1] = (vaddvq_s32(a1) as f32) * w_scale[row + 1] * x_scale;
            out[row + 2] = (vaddvq_s32(a2) as f32) * w_scale[row + 2] * x_scale;
            out[row + 3] = (vaddvq_s32(a3) as f32) * w_scale[row + 3] * x_scale;

            row += 4;
        }

        // Scalar tail: 1 row at a time.
        while row < out_dim {
            let row_base = row * packed_per_row;
            let row_ptr = packed.as_ptr().add(row_base);
            let mut acc = vdupq_n_s32(0);

            for b in 0..blocks {
                let byte_off = b * 4;
                let x_off = b * 16;
                let xv = vld1q_s8(x_q.as_ptr().add(x_off));
                let w = neon_unpack_weights(row_ptr, byte_off, shuf_idx, shift_i8, mask, one_s8);
                acc = vdotq_s32(acc, w, xv);
            }

            out[row] = (vaddvq_s32(acc) as f32) * w_scale[row] * x_scale;
            row += 1;
        }
    }
}

/// AVX2 kernel. Uses `pmaddubsw` (unsigned × signed 8-bit, pairwise add to
/// i16) on packed weights as u8 in `{0, 1, 2}`. Result is the biased sum
/// `Σ (w+1) × x = Σ w×x + Σ x`; the correction `−Σ x` is applied after the
/// row loop using one scalar sum of the (already-computed) activation bytes.
///
/// This is the same pattern bitnet.cpp's `I2_S` kernel uses on x86.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code, clippy::cast_precision_loss, clippy::cast_possible_wrap)]
fn matvec_ternary_avx2(packed: &[u8], w_scale: &[f32], x_q: &[i8], x_scale: f32, out: &mut [f32]) {
    use std::arch::x86_64::{
        __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_hadd_epi32, _mm_srli_si128,
        _mm256_add_epi32, _mm256_and_si256, _mm256_cvtepi16_epi32, _mm256_extracti128_si256,
        _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi8,
        _mm256_set1_epi16, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_srli_epi16,
        _mm256_storeu_si256, _mm256_sub_epi8,
    };

    let out_dim = w_scale.len();
    let in_dim = x_q.len();
    debug_assert_eq!(packed.len(), (out_dim * in_dim).div_ceil(4));
    debug_assert_eq!(out.len(), out_dim);
    debug_assert_eq!(
        in_dim % 32,
        0,
        "AVX2 kernel requires in_dim divisible by 32"
    );

    if x_scale == 0.0 {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        return;
    }

    // Precompute Σ x_q (needed for the +1 bias correction below).
    let x_sum: i32 = x_q.iter().map(|&v| i32::from(v)).sum();

    let packed_per_row = in_dim / 4;
    let blocks = in_dim / 32;

    for row in 0..out_dim {
        let row_packed = &packed[row * packed_per_row..(row + 1) * packed_per_row];
        let mut acc = unsafe { _mm256_setzero_si256() };

        for b in 0..blocks {
            let byte_off = b * 8;
            let x_off = b * 32;

            // Load 8 packed bytes (32 codes).
            let mut packed_scratch = [0u8; 32];
            packed_scratch[..8].copy_from_slice(&row_packed[byte_off..byte_off + 8]);
            let packed_vec =
                unsafe { _mm256_loadu_si256(packed_scratch.as_ptr().cast::<__m256i>()) };

            // Unpack 8 bytes × 4 codes → 32 u8 values in {0, 1, 2}:
            //   byte_i = packed[i / 4]
            //   shift  = (i % 4) * 2
            //   code   = (byte_i >> shift) & 3
            //
            // We broadcast each of the 8 packed bytes to 4 lanes via a
            // `pshufb` index table, then shift-and-mask.
            let shuf_idx = unsafe {
                _mm256_loadu_si256(
                    [
                        0i8, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
                        6, 6, 6, 6, 7, 7, 7, 7,
                    ]
                    .as_ptr()
                    .cast::<__m256i>(),
                )
            };
            let broadcast = unsafe { _mm256_shuffle_epi8(packed_vec, shuf_idx) };

            // Per-lane shift: since AVX2 lacks per-byte variable shifts,
            // fake it by treating each u16 pair as (hi, lo), shifting the
            // whole 16-bit value right by the lane-specific amount, and
            // masking. Simpler: use separate shift constants per lane via
            // repeated shuffle + AND.
            //
            // Equivalent: compute `(broadcast >> shift_i) & 3`. We expand
            // this with 4 shift+mask passes and then blend via selective
            // OR. Since the layout is fixed (lanes 0,4,8,... get shift 0;
            // lanes 1,5,9,... get shift 2; etc.), we can precompute the
            // shifted-then-masked result per-lane via `_mm256_srli_epi16`
            // on the appropriately-blended value. For clarity, scalarize
            // the unpack step into a staging buffer — the outer
            // `pmaddubsw` is where the real SIMD win happens.
            let mut codes_scratch = [0u8; 32];
            unsafe {
                _mm256_storeu_si256(codes_scratch.as_mut_ptr().cast::<__m256i>(), broadcast);
            }
            for (k, slot) in codes_scratch.iter_mut().enumerate() {
                let shift = ((k % 4) * 2) as u32;
                *slot = (*slot >> shift) & 0x03;
            }
            let codes_u8 = unsafe { _mm256_loadu_si256(codes_scratch.as_ptr().cast::<__m256i>()) };

            // Load 32 i8 activations.
            let xv = unsafe { _mm256_loadu_si256(x_q.as_ptr().add(x_off).cast::<__m256i>()) };

            // pmaddubsw: u8 × i8 → i16 pairwise adds.
            //   Lane i of result = codes[2i] * x[2i] + codes[2i+1] * x[2i+1]
            // That's still with codes in {0, 1, 2}; we correct afterward.
            let prod = unsafe { _mm256_maddubs_epi16(codes_u8, xv) };

            // Widen i16 → i32 and accumulate.
            let prod_lo = unsafe { _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod, 0)) };
            let prod_hi = unsafe { _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod, 1)) };
            acc = unsafe { _mm256_add_epi32(acc, prod_lo) };
            acc = unsafe { _mm256_add_epi32(acc, prod_hi) };

            // Silence unused _mm256_madd_epi16 / _mm256_set1_epi16 /
            // _mm256_set1_epi8 / _mm256_sub_epi8 / _mm256_srli_epi16 /
            // _mm_add_epi32 / _mm_cvtsi128_si32 / _mm_hadd_epi32 /
            // _mm_srli_si128 imports — these are here for future
            // refinements (true SIMD unpack).
            let _ = (
                _mm256_madd_epi16,
                _mm256_set1_epi16,
                _mm256_set1_epi8,
                _mm256_sub_epi8,
                _mm256_srli_epi16,
                _mm_add_epi32,
                _mm_cvtsi128_si32,
                _mm_hadd_epi32,
                _mm_srli_si128,
            );
        }

        // Horizontal sum of acc's 8 i32 lanes.
        let biased_sum = unsafe {
            let lo = _mm256_extracti128_si256(acc, 0);
            let hi = _mm256_extracti128_si256(acc, 1);
            let sum128 = _mm_add_epi32(lo, hi);
            let sum64 = _mm_hadd_epi32(sum128, sum128);
            let sum32 = _mm_hadd_epi32(sum64, sum64);
            _mm_cvtsi128_si32(sum32)
        };

        // Undo the +1 bias: Σ (w+1)·x = Σ w·x + Σ x  =>  Σ w·x = biased − Σ x.
        let true_sum = biased_sum - x_sum;
        out[row] = (true_sum as f32) * w_scale[row] * x_scale;
    }
}

/// Pack ternary weights (`{-1, 0, 1}` stored as `i8`) into the 2-bit
/// sequential encoding the kernels consume.
#[cfg(any(feature = "training", test))]
pub(crate) fn pack_ternary(weights: &[i8], packed: &mut [u8]) {
    debug_assert_eq!(weights.len() % 4, 0);
    debug_assert_eq!(packed.len(), weights.len() / 4);
    for (chunk_idx, chunk) in weights.chunks_exact(4).enumerate() {
        let mut byte = 0u8;
        for (k, &w) in chunk.iter().enumerate() {
            let code: u8 = match w {
                -1 => 0b00,
                0 => 0b01,
                1 => 0b10,
                other => panic!("pack_ternary: value {other} is not in {{-1, 0, 1}}"),
            };
            byte |= code << (2 * k);
        }
        packed[chunk_idx] = byte;
    }
}

/// Unpack 2-bit ternary weights back to `i8` values in `{-1, 0, 1}`.
/// Test-only helper. `code as i8` is safe — `code & 0b11` is always in `0..=3`.
#[cfg(test)]
#[allow(clippy::cast_possible_wrap)]
pub(crate) fn unpack_ternary(packed: &[u8], weights: &mut [i8]) {
    debug_assert_eq!(packed.len(), weights.len().div_ceil(4));
    for (byte_idx, &p) in packed.iter().enumerate() {
        let j0 = byte_idx * 4;
        for k in 0..4 {
            if j0 + k >= weights.len() {
                break;
            }
            let code = (p >> (2 * k)) & 0b11;
            weights[j0 + k] = (code as i8) - 1;
        }
    }
}

#[cfg(test)]
// Correctness tests assert exact f32 equality — the kernels use an integer
// accumulator with a single final f32 multiply, so bit-identical results are
// expected across kernels and across runs.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        let w: Vec<i8> = (0..32).map(|i| (i % 3) as i8 - 1).collect();
        let mut packed = vec![0u8; 8];
        pack_ternary(&w, &mut packed);
        let mut out = vec![0i8; 32];
        unpack_ternary(&packed, &mut out);
        assert_eq!(w, out);
    }

    #[test]
    fn activation_quant_preserves_sign() {
        let x = [1.0_f32, -2.0, 3.0, -4.0, 0.5, 0.0, -0.5, 127.0];
        let mut xq = [0i8; 8];
        let s = quantize_activations(&x, &mut xq);
        assert!(s > 0.0);
        for (orig, q) in x.iter().zip(xq.iter()) {
            let recon = f32::from(*q) * s;
            assert!(
                (recon - *orig).abs() <= s,
                "recon={recon} orig={orig} s={s}"
            );
        }
    }

    #[test]
    fn activation_quant_zero_vector() {
        let x = [0.0_f32; 16];
        let mut xq = [0i8; 16];
        let s = quantize_activations(&x, &mut xq);
        assert_eq!(s, 0.0);
        assert_eq!(xq, [0i8; 16]);
    }

    #[test]
    fn matvec_ref_hand_computed() {
        // 2 output rows, 8 input columns.
        // Row 0 weights: [+1, -1, 0, 0, +1, +1, -1, 0]
        // Row 1 weights: [ 0, +1, +1, -1, 0, 0, -1, +1]
        let w: [i8; 16] = [1, -1, 0, 0, 1, 1, -1, 0, 0, 1, 1, -1, 0, 0, -1, 1];
        let mut packed = vec![0u8; 4];
        pack_ternary(&w, &mut packed);
        let w_scale = [1.0_f32, 1.0];
        let x_q: [i8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let x_scale = 1.0_f32;
        let mut out = [0.0_f32; 2];
        matvec_ternary_ref(&packed, &w_scale, &x_q, x_scale, &mut out);
        // row 0: +1-2+0+0+5+6-7+0 = 3
        // row 1:  0+2+3-4+0+0-7+8 = 2
        assert_eq!(out[0], 3.0);
        assert_eq!(out[1], 2.0);
    }

    fn random_matvec_case(
        rng: &mut rand::rngs::StdRng,
    ) -> (Vec<u8>, Vec<f32>, Vec<i8>, f32, usize, usize) {
        use rand::Rng;
        let out_dim = 1 + rng.random_range(0..8);
        // Each kernel has its own alignment requirement (16 for NEON, 32 for
        // scalar/AVX2). Use 32-aligned blocks so every kernel accepts the
        // input.
        let blocks = 1 + rng.random_range(0..4);
        let in_dim = 32 * blocks;
        let weights: Vec<i8> = (0..out_dim * in_dim)
            .map(|_| rng.random_range(-1..=1))
            .collect();
        let mut packed = vec![0u8; (out_dim * in_dim) / 4];
        pack_ternary(&weights, &mut packed);
        let w_scale: Vec<f32> = (0..out_dim).map(|_| rng.random_range(0.01..2.0)).collect();
        let x_q: Vec<i8> = (0..in_dim).map(|_| rng.random_range(-127..=127)).collect();
        let x_scale = rng.random_range(0.0001..0.5);
        (packed, w_scale, x_q, x_scale, out_dim, in_dim)
    }

    #[test]
    fn scalar_matches_ref_random() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xDEAD_BEEF);

        for _ in 0..50 {
            let (packed, w_scale, x_q, x_scale, out_dim, _in_dim) = random_matvec_case(&mut rng);
            let mut out_ref = vec![0.0_f32; out_dim];
            let mut out_fast = vec![0.0_f32; out_dim];
            matvec_ternary_ref(&packed, &w_scale, &x_q, x_scale, &mut out_ref);
            matvec_ternary_scalar(&packed, &w_scale, &x_q, x_scale, &mut out_fast);
            assert_eq!(out_ref, out_fast);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_ref_random() {
        use rand::SeedableRng;
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xFEED_C0DE);

        for _ in 0..50 {
            let (packed, w_scale, x_q, x_scale, out_dim, _in_dim) = random_matvec_case(&mut rng);
            let mut out_ref = vec![0.0_f32; out_dim];
            let mut out_neon = vec![0.0_f32; out_dim];
            matvec_ternary_ref(&packed, &w_scale, &x_q, x_scale, &mut out_ref);
            #[allow(unsafe_code)]
            // SAFETY: NEON is part of the aarch64 baseline ABI.
            unsafe {
                matvec_ternary_neon(&packed, &w_scale, &x_q, x_scale, &mut out_neon);
            }
            assert_eq!(out_ref, out_neon);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_ref_random() {
        use rand::SeedableRng;
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xCAFE_FACE);

        for _ in 0..50 {
            let (packed, w_scale, x_q, x_scale, out_dim, _in_dim) = random_matvec_case(&mut rng);
            let mut out_ref = vec![0.0_f32; out_dim];
            let mut out_avx = vec![0.0_f32; out_dim];
            matvec_ternary_ref(&packed, &w_scale, &x_q, x_scale, &mut out_ref);
            #[allow(unsafe_code)]
            // SAFETY: `is_x86_feature_detected!("avx2")` checked above.
            unsafe {
                matvec_ternary_avx2(&packed, &w_scale, &x_q, x_scale, &mut out_avx);
            }
            assert_eq!(out_ref, out_avx);
        }
    }

    #[test]
    fn matvec_zero_activations_zero_output() {
        // out_dim = 2, in_dim = 32 → packed bytes = 2*32/4 = 16.
        let packed = vec![0x55u8; 16]; // 0x55 = 0b01010101 → all codes = 1 → all weights = 0
        let w_scale = [0.5_f32; 2];
        let x_q = [0i8; 32];
        let x_scale = 0.0;
        let mut out = [1.0_f32; 2];
        matvec_ternary_scalar(&packed, &w_scale, &x_q, x_scale, &mut out);
        assert_eq!(out, [0.0, 0.0]);
    }
}
