//! `BitNet` ternary matvec kernels.
//!
//! Weights are ternary `{-1, 0, +1}` with a per-output-row `f32` absmean scale.
//! Storage is 2 bits per weight, row-major, 4 weights per byte:
//!   `0b00 = 0`, `0b01 = +1`, `0b11 = -1`. `0b10` is unused so sign-extension
//!   from 2 bits yields the correct i8 value for every valid code.
//!
//! Inputs are quantized to 8 bits (per-vector absmax) immediately before each
//! matvec. The kernel accumulates `i32` and converts to `f32` once per row,
//! multiplying by the per-row weight scale and the activation scale.
//!
//! Per the 2026-04-21 JOURNAL.md finding, clean scalar loops with
//! `RUSTFLAGS="-C target-cpu=native"` auto-vectorize to ~75 GOPS on Apple
//! NEON, while manual `wide::i16x8` SIMD was measured 25× slower. This file
//! keeps the inner loops plain integer arithmetic so LLVM can lift them into
//! `SMLAL` / `PMADDUBSW`-family instructions on whatever target hardware.

/// Sign-extend a 2-bit code in `{0b00, 0b01, 0b11}` to `i8` in `{0, +1, -1}`.
///
/// `#[inline(always)]` is deliberate: this runs inside every matvec inner
/// iteration, and the function body collapses to a shift + sign-extend pair.
/// The `u8 as i8` reinterprets bit patterns intentionally (the byte is at
/// most `0xC0`, which maps to `-64`, so the subsequent arithmetic shift
/// restores `-1`).
#[inline(always)]
#[allow(clippy::inline_always, clippy::cast_possible_wrap)]
const fn code_to_i8(code: u8) -> i8 {
    ((code << 6) as i8) >> 6
}

/// Quantize `x` into 8-bit signed integers with a shared absmax scale.
///
/// Returns the scale `s` such that `x[i] ≈ s * (x_q[i] as f32)`. Zero-input
/// vectors return `s = 0.0` and an all-zero `x_q` (the matvec handles the
/// zero-scale case by short-circuiting).
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

/// Reference kernel — walks the packed weights one byte at a time.
///
/// This is the correctness oracle: straightforward, no block unrolling, no
/// word loads. Fast kernels are diff-tested against this one.
///
/// The `i32 → f32` cast at output time loses precision when the dot product
/// exceeds 2^24, which is possible for wide inputs. For this kernel's
/// `in_dim ≤ ~4096` and ternary weights the accumulator fits in f32's
/// representable range; the precision loss is in the same regime as the
/// activation and weight quantization, i.e. dominated by them.
#[cfg(test)]
#[inline(never)]
#[allow(clippy::cast_precision_loss)]
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
                let w = code_to_i8(code);
                acc += i32::from(x_q[j0 + k]) * i32::from(w);
            }
        }
        out[row] = (acc as f32) * w_scale[row] * x_scale;
    }
}

/// Fast kernel — 32-weight inner block, pure scalar arithmetic.
///
/// LLVM auto-vectorizes the 32-wide inner loop to NEON / AVX2 SMLAL-family
/// instructions with `-C target-cpu=native`. Deliberately avoids `std::arch`
/// intrinsics and portable-SIMD crates (JOURNAL.md 2026-04-21).
///
/// Precision-loss allow: see [`matvec_ternary_ref`].
#[inline(never)]
#[allow(clippy::cast_precision_loss)]
pub(crate) fn matvec_ternary(
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
        "fast kernel requires in_dim divisible by 32; call matvec_ternary_ref otherwise"
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

            // Fixed-size 32 inner loop helps LLVM unroll and schedule.
            for k in 0..32 {
                let p = bytes[k / 4];
                let shift = (k % 4) * 2;
                let code = (p >> shift) & 0b11;
                let w = code_to_i8(code);
                acc += i32::from(xs[k]) * i32::from(w);
            }
        }

        out[row] = (acc as f32) * w_scale[row] * x_scale;
    }
}

/// Pack a slice of ternary weights (values in `{-1, 0, 1}` stored as `i8`) into
/// the 2-bit packed format the kernels consume.
#[cfg(any(feature = "training", test))]
pub(crate) fn pack_ternary(weights: &[i8], packed: &mut [u8]) {
    debug_assert_eq!(weights.len() % 4, 0);
    debug_assert_eq!(packed.len(), weights.len() / 4);
    for (chunk_idx, chunk) in weights.chunks_exact(4).enumerate() {
        let mut byte = 0u8;
        for (k, &w) in chunk.iter().enumerate() {
            let code: u8 = match w {
                0 => 0b00,
                1 => 0b01,
                -1 => 0b11,
                other => panic!("pack_ternary: value {other} is not in {{-1, 0, 1}}"),
            };
            byte |= code << (2 * k);
        }
        packed[chunk_idx] = byte;
    }
}

/// Unpack 2-bit ternary weights back to `i8` values in `{-1, 0, 1}`.
/// Primarily for test harnesses and debugging.
#[cfg(test)]
pub(crate) fn unpack_ternary(packed: &[u8], weights: &mut [i8]) {
    debug_assert_eq!(packed.len(), weights.len().div_ceil(4));
    for (byte_idx, &p) in packed.iter().enumerate() {
        let j0 = byte_idx * 4;
        for k in 0..4 {
            if j0 + k >= weights.len() {
                break;
            }
            let code = (p >> (2 * k)) & 0b11;
            weights[j0 + k] = code_to_i8(code);
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
    fn code_to_i8_maps_correctly() {
        assert_eq!(code_to_i8(0b00), 0);
        assert_eq!(code_to_i8(0b01), 1);
        assert_eq!(code_to_i8(0b11), -1);
    }

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

    #[test]
    fn matvec_fast_matches_ref_random() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xDEAD_BEEF);

        for _ in 0..50 {
            let out_dim = 1 + rng.random_range(0..8);
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

            let mut out_ref = vec![0.0_f32; out_dim];
            let mut out_fast = vec![0.0_f32; out_dim];
            matvec_ternary_ref(&packed, &w_scale, &x_q, x_scale, &mut out_ref);
            matvec_ternary(&packed, &w_scale, &x_q, x_scale, &mut out_fast);
            assert_eq!(out_ref, out_fast);
        }
    }

    #[test]
    fn matvec_zero_activations_zero_output() {
        // out_dim = 2, in_dim = 32 → packed bytes = 2*32/4 = 16.
        let packed = vec![0u8; 16];
        let w_scale = [0.5_f32; 2];
        let x_q = [0i8; 32];
        let x_scale = 0.0;
        let mut out = [1.0_f32; 2];
        matvec_ternary(&packed, &w_scale, &x_q, x_scale, &mut out);
        assert_eq!(out, [0.0, 0.0]);
    }
}
