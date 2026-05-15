// Phase 25a WIP: the module is fully tested but no consumer in the codec
// path wires it up yet (that lands in Phase 25c). The module-level
// `dead_code` allow keeps the unused warnings out of `-Dwarnings` until then.
#![allow(dead_code)]

//! Byte-level decoder transformer for the Step C neural arm.
//!
//! Inference-only port of the architecture in
//! `/Users/creynolds/Programming/lzr-neural/src/lzr_neural/model.py`.
//! Architecture: [`RMSNorm`] pre-norm, causal multi-head attention with
//! no bias, [`GELU`] (exact erf-based, matching `PyTorch`'s default)
//! feed-forward, tied input/output embeddings, learned absolute
//! position embeddings.
//!
//! Forward pass is **batched over the full sequence** (no KV cache) for
//! the first phase. The KV-cache variant lands once the parity test
//! against `PyTorch` is green — same kernels, just incremental.
//!
//! Memory layout: every weight is a row-major `Vec<f32>` in
//! `[out_dim, in_dim]` order (`PyTorch`'s `nn.Linear.weight` convention).
//! That makes `matmul(x, W)` compute `x @ W^T` — same as `PyTorch`'s
//! `F.linear`.
//!
//! Weights come from the `.lzrn` binary written by
//! `lzr-neural/scripts/export_weights.py`; the header format is
//! mirrored exactly here in [`read_header`].
//!
//! [`RMSNorm`]: https://arxiv.org/abs/1910.07467
//! [`GELU`]: https://arxiv.org/abs/1606.08415

use anyhow::{Result, anyhow, bail};

/// Magic header bytes: `'LZRN'` little-endian. Catches accidental loads
/// of the wrong file type before we read garbage as floats.
const MAGIC: u32 = 0x4C5A_524E;
const VERSION: u32 = 1;

#[derive(Debug, Clone, Copy)]
pub(crate) struct TransformerConfig {
    pub(crate) n_layer: usize,
    pub(crate) n_head: usize,
    pub(crate) d_model: usize,
    pub(crate) d_ff: usize,
    pub(crate) context: usize,
    pub(crate) vocab_size: usize,
}

impl TransformerConfig {
    pub(crate) const fn head_dim(&self) -> usize {
        self.d_model / self.n_head
    }
}

/// Per-layer weights. All fields end in `_w` because all of them are
/// weight tensors — clippy's struct-field-name lint flags this as a
/// "common postfix" but the suffix carries information: it
/// distinguishes the storage tensors from any future bias / scale
/// fields that may appear (none are added for this architecture, but
/// the convention matches the export-script layout).
#[allow(clippy::struct_field_names)]
#[derive(Debug)]
pub(crate) struct Block {
    norm1_w: Vec<f32>, // [d_model]
    qkv_w: Vec<f32>,   // [3 * d_model, d_model]
    proj_w: Vec<f32>,  // [d_model, d_model]
    norm2_w: Vec<f32>, // [d_model]
    fc1_w: Vec<f32>,   // [d_ff, d_model]
    fc2_w: Vec<f32>,   // [d_model, d_ff]
}

#[derive(Debug)]
pub(crate) struct ByteTransformer {
    pub(crate) cfg: TransformerConfig,
    tok_emb: Vec<f32>, // [vocab_size, d_model]
    pos_emb: Vec<f32>, // [context, d_model]
    blocks: Vec<Block>,
    norm_f: Vec<f32>, // [d_model]
}

impl ByteTransformer {
    /// Parse a `.lzrn` byte buffer into a populated model.
    /// See `lzr-neural/scripts/export_weights.py` for the canonical
    /// layout documentation; this function is the only place that
    /// has to stay in lock-step with that script.
    pub(crate) fn load_lzrn(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 32 {
            bail!("lzrn buffer too short for header: {}", bytes.len());
        }
        let cfg = read_header(&bytes[..32])?;
        let mut cursor = 32usize;

        let tok_emb = read_tensor(bytes, &mut cursor, cfg.vocab_size * cfg.d_model)?;
        let pos_emb = read_tensor(bytes, &mut cursor, cfg.context * cfg.d_model)?;

        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            let norm1_w = read_tensor(bytes, &mut cursor, cfg.d_model)?;
            let qkv_w = read_tensor(bytes, &mut cursor, 3 * cfg.d_model * cfg.d_model)?;
            let proj_w = read_tensor(bytes, &mut cursor, cfg.d_model * cfg.d_model)?;
            let norm2_w = read_tensor(bytes, &mut cursor, cfg.d_model)?;
            let fc1_w = read_tensor(bytes, &mut cursor, cfg.d_ff * cfg.d_model)?;
            let fc2_w = read_tensor(bytes, &mut cursor, cfg.d_model * cfg.d_ff)?;
            blocks.push(Block {
                norm1_w,
                qkv_w,
                proj_w,
                norm2_w,
                fc1_w,
                fc2_w,
            });
        }

        let norm_f = read_tensor(bytes, &mut cursor, cfg.d_model)?;
        if cursor != bytes.len() {
            bail!(
                "lzrn buffer has {} trailing bytes after parse",
                bytes.len() - cursor
            );
        }
        Ok(Self {
            cfg,
            tok_emb,
            pos_emb,
            blocks,
            norm_f,
        })
    }

    /// Batched full-sequence forward. Returns logits as a flat
    /// `Vec<f32>` of shape `[seq_len, vocab_size]`, row-major.
    ///
    /// The kernel-style index loops (`for i in 0..t`) are intentional —
    /// they map cleanly to auto-vec'd inner loops in scalar Rust and
    /// keep the math readable as offsets into row-major Vec storage.
    /// Single-character names (`t`, `d`, `i`, `k`) are conventional in
    /// transformer kernels; clippy's many-single-char-names lint is
    /// suppressed for this and the related kernel functions.
    #[allow(clippy::many_single_char_names, clippy::needless_range_loop)]
    pub(crate) fn forward(&self, tokens: &[u8]) -> Vec<f32> {
        let t = tokens.len();
        let d = self.cfg.d_model;
        assert!(
            t <= self.cfg.context,
            "seq_len {t} > context {}",
            self.cfg.context
        );

        // x = tok_emb[tokens] + pos_emb[0..t]   [t, d_model]
        let mut x = vec![0f32; t * d];
        for (i, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            let dst = &mut x[i * d..(i + 1) * d];
            let tok_row = &self.tok_emb[tok * d..(tok + 1) * d];
            let pos_row = &self.pos_emb[i * d..(i + 1) * d];
            for k in 0..d {
                dst[k] = tok_row[k] + pos_row[k];
            }
        }

        let mut scratch = TransformerScratch::new(&self.cfg, t);
        for block in &self.blocks {
            self.run_block(block, &mut x, &mut scratch, t);
        }

        // Final RMSNorm.
        for i in 0..t {
            let row = &mut x[i * d..(i + 1) * d];
            rmsnorm_inplace(row, &self.norm_f);
        }

        // Tied output projection: logits = x @ tok_emb^T  → [t, vocab].
        let mut logits = vec![0f32; t * self.cfg.vocab_size];
        matmul_x_w_t(&x, &self.tok_emb, &mut logits, t, d, self.cfg.vocab_size);
        logits
    }

    #[allow(
        clippy::many_single_char_names,
        clippy::needless_range_loop,
        clippy::cast_precision_loss,
        clippy::suboptimal_flops,
        clippy::similar_names
    )]
    fn run_block(&self, block: &Block, x: &mut [f32], s: &mut TransformerScratch, t: usize) {
        let d = self.cfg.d_model;
        let h = self.cfg.n_head;
        let hd = self.cfg.head_dim();

        // attention residual branch
        for i in 0..t {
            let dst = &mut s.h[i * d..(i + 1) * d];
            let row = &x[i * d..(i + 1) * d];
            dst.copy_from_slice(row);
            rmsnorm_inplace(dst, &block.norm1_w);
        }

        // qkv = h_norm @ qkv_w^T  → [t, 3*d_model]
        matmul_x_w_t(&s.h, &block.qkv_w, &mut s.qkv, t, d, 3 * d);

        // Per-head attention.
        for head in 0..h {
            // Build q, k, v for this head as contiguous [t, hd] slices.
            for i in 0..t {
                let row = &s.qkv[i * 3 * d..(i + 1) * 3 * d];
                let q_src = &row[head * hd..head * hd + hd];
                let k_src = &row[d + head * hd..d + head * hd + hd];
                let v_src = &row[2 * d + head * hd..2 * d + head * hd + hd];
                s.q[i * hd..(i + 1) * hd].copy_from_slice(q_src);
                s.k[i * hd..(i + 1) * hd].copy_from_slice(k_src);
                s.v[i * hd..(i + 1) * hd].copy_from_slice(v_src);
            }

            // scores [t, t] = q @ k^T / sqrt(hd)  with causal mask.
            let scale = 1.0 / (hd as f32).sqrt();
            for i in 0..t {
                for j in 0..t {
                    if j > i {
                        s.scores[i * t + j] = f32::NEG_INFINITY;
                    } else {
                        let qi = &s.q[i * hd..(i + 1) * hd];
                        let kj = &s.k[j * hd..(j + 1) * hd];
                        let mut acc = 0.0f32;
                        for k in 0..hd {
                            acc += qi[k] * kj[k];
                        }
                        s.scores[i * t + j] = acc * scale;
                    }
                }
            }
            // softmax per row.
            for i in 0..t {
                let row = &mut s.scores[i * t..(i + 1) * t];
                softmax_inplace(row);
            }
            // out_head [t, hd] = scores @ v.
            for i in 0..t {
                let out_row = &mut s.attn_out[i * d + head * hd..i * d + head * hd + hd];
                for k in 0..hd {
                    out_row[k] = 0.0;
                }
                for j in 0..=i {
                    let s_ij = s.scores[i * t + j];
                    let vj = &s.v[j * hd..(j + 1) * hd];
                    for k in 0..hd {
                        out_row[k] += s_ij * vj[k];
                    }
                }
            }
        }

        // attn_out @ proj^T → s.h2 [t, d_model]
        matmul_x_w_t(&s.attn_out, &block.proj_w, &mut s.h2, t, d, d);

        // residual: x += s.h2
        for i in 0..t * d {
            x[i] += s.h2[i];
        }

        // FFN residual branch
        for i in 0..t {
            let dst = &mut s.h[i * d..(i + 1) * d];
            let row = &x[i * d..(i + 1) * d];
            dst.copy_from_slice(row);
            rmsnorm_inplace(dst, &block.norm2_w);
        }

        // ff_hidden = h_norm @ fc1^T  → [t, d_ff]
        matmul_x_w_t(&s.h, &block.fc1_w, &mut s.ff_hidden, t, d, self.cfg.d_ff);
        gelu_inplace(&mut s.ff_hidden);
        // ff_out = ff_hidden @ fc2^T → [t, d_model]
        matmul_x_w_t(&s.ff_hidden, &block.fc2_w, &mut s.h2, t, self.cfg.d_ff, d);

        for i in 0..t * d {
            x[i] += s.h2[i];
        }
    }
}

/// Per-forward scratch buffers (reused across blocks).
#[allow(clippy::struct_field_names)]
struct TransformerScratch {
    h: Vec<f32>,   // [t, d_model] — normalized input to attn or ff
    qkv: Vec<f32>, // [t, 3*d_model]
    q: Vec<f32>,   // [t, head_dim]
    k: Vec<f32>,
    v: Vec<f32>,
    scores: Vec<f32>,    // [t, t]
    attn_out: Vec<f32>,  // [t, d_model]
    h2: Vec<f32>,        // [t, d_model] — residual branch output
    ff_hidden: Vec<f32>, // [t, d_ff]
}

impl TransformerScratch {
    fn new(cfg: &TransformerConfig, t: usize) -> Self {
        Self {
            h: vec![0.0; t * cfg.d_model],
            qkv: vec![0.0; t * 3 * cfg.d_model],
            q: vec![0.0; t * cfg.head_dim()],
            k: vec![0.0; t * cfg.head_dim()],
            v: vec![0.0; t * cfg.head_dim()],
            scores: vec![0.0; t * t],
            attn_out: vec![0.0; t * cfg.d_model],
            h2: vec![0.0; t * cfg.d_model],
            ff_hidden: vec![0.0; t * cfg.d_ff],
        }
    }
}

fn read_header(buf: &[u8]) -> Result<TransformerConfig> {
    let read_u32 = |off: usize| -> u32 {
        u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
    };
    let magic = read_u32(0);
    if magic != MAGIC {
        bail!("lzrn magic mismatch: got 0x{magic:08x}, expected 0x{MAGIC:08x}");
    }
    let version = read_u32(4);
    if version != VERSION {
        bail!("lzrn version mismatch: got {version}, expected {VERSION}");
    }
    Ok(TransformerConfig {
        n_layer: read_u32(8) as usize,
        n_head: read_u32(12) as usize,
        d_model: read_u32(16) as usize,
        d_ff: read_u32(20) as usize,
        context: read_u32(24) as usize,
        vocab_size: read_u32(28) as usize,
    })
}

fn read_tensor(bytes: &[u8], cursor: &mut usize, n_floats: usize) -> Result<Vec<f32>> {
    let n_bytes = n_floats * 4;
    if *cursor + n_bytes > bytes.len() {
        return Err(anyhow!(
            "lzrn truncated: need {} bytes at offset {}, have {}",
            n_bytes,
            *cursor,
            bytes.len() - *cursor,
        ));
    }
    let mut out = Vec::with_capacity(n_floats);
    let slice = &bytes[*cursor..*cursor + n_bytes];
    for chunk in slice.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    *cursor += n_bytes;
    Ok(out)
}

/// Row-major matmul: `out[m, n] = x[m, k] @ w[n, k]^T`. `PyTorch`'s
/// `nn.Linear(in=k, out=n).weight` has shape `[n, k]`, so this matches
/// `F.linear(x, w) = x @ w^T`.
#[allow(
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::suboptimal_flops
)]
fn matmul_x_w_t(x: &[f32], w: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(x.len(), m * k);
    assert_eq!(w.len(), n * k);
    assert_eq!(out.len(), m * n);
    for i in 0..m {
        let x_row = &x[i * k..(i + 1) * k];
        let out_row = &mut out[i * n..(i + 1) * n];
        for j in 0..n {
            let w_row = &w[j * k..(j + 1) * k];
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += x_row[p] * w_row[p];
            }
            out_row[j] = acc;
        }
    }
}

/// In-place `RMSNorm`: `x = x / sqrt(mean(x^2) + eps) * weight`.
/// Eps matches `PyTorch` model: `1e-5`.
#[allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::suboptimal_flops,
    clippy::items_after_statements
)]
fn rmsnorm_inplace(x: &mut [f32], weight: &[f32]) {
    assert_eq!(x.len(), weight.len());
    const EPS: f32 = 1e-5;
    let mut sum_sq = 0.0f32;
    for &v in x.iter() {
        sum_sq += v * v;
    }
    let mean_sq = sum_sq / x.len() as f32;
    let scale = (mean_sq + EPS).sqrt().recip();
    for i in 0..x.len() {
        x[i] = x[i] * scale * weight[i];
    }
}

/// `PyTorch`'s default `F.gelu` uses the exact erf-based formulation:
/// `gelu(x) = x * 0.5 * (1.0 + erf(x / sqrt(2)))`. Using libm-style
/// stable erf approximation (Abramowitz & Stegun 7.1.26, max abs error
/// ~1.5e-7) — within `f32` precision.
#[allow(clippy::suboptimal_flops)]
fn gelu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        let z = *v * std::f32::consts::FRAC_1_SQRT_2;
        *v = *v * 0.5 * (1.0 + erf_approx(z));
    }
}

/// Abramowitz & Stegun erf approximation 7.1.26. Coefficients trimmed
/// to `f32` precision (`PyTorch`'s `torch.erf` itself uses higher-precision
/// constants internally, but the residual difference is well below the
/// parity threshold).
#[allow(clippy::suboptimal_flops)]
fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152) * t) + 1.421_413_7) * t - 0.284_496_74) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    sign * y
}

/// In-place numerically-stable softmax over the slice. Treats
/// `f32::NEG_INFINITY` entries as masked (their exp is 0, so they
/// contribute nothing to the sum and end up with output 0).
fn softmax_inplace(x: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = sum.recip();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_matches_pytorch_unit_weight() {
        // x = [1, 2, 3, 4], weight = 1 → rms = sqrt(7.5) = 2.7386...
        // x / rms = [0.3651, 0.7302, 1.0954, 1.4605]
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        rmsnorm_inplace(&mut x, &w);
        let expected = [0.3651, 0.7302, 1.0954, 1.4605];
        for (a, e) in x.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 5e-3, "rmsnorm: {a} vs {e}");
        }
    }

    #[test]
    fn gelu_matches_pytorch_at_known_points() {
        // GELU(0) = 0, GELU(1) ≈ 0.8412, GELU(-1) ≈ -0.1587, GELU(2) ≈ 1.9546.
        let mut x = vec![0.0f32, 1.0, -1.0, 2.0];
        gelu_inplace(&mut x);
        let expected = [0.0, 0.8412, -0.1587, 1.9546];
        for (a, e) in x.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 5e-3, "gelu: {a} vs {e}");
        }
    }

    #[test]
    fn softmax_matches_known() {
        // softmax([1, 2, 3]) ≈ [0.0900, 0.2447, 0.6652]
        let mut x = vec![1.0f32, 2.0, 3.0];
        softmax_inplace(&mut x);
        let expected = [0.0900, 0.2447, 0.6652];
        for (a, e) in x.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 5e-3, "softmax: {a} vs {e}");
        }
        let s: f32 = x.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn matmul_simple_2x2() {
        // x = [[1,2],[3,4]]  (m=2, k=2)
        // w = [[5,6],[7,8]]  (n=2, k=2) — represents linear weight [out=2, in=2]
        // out = x @ w^T = [[1*5+2*6, 1*7+2*8],[3*5+4*6, 3*7+4*8]]
        //               = [[17, 23],[39, 53]]
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut out = vec![0.0f32; 4];
        matmul_x_w_t(&x, &w, &mut out, 2, 2, 2);
        assert_eq!(out, vec![17.0, 23.0, 39.0, 53.0]);
    }

    /// Parity check against the `PyTorch` forward pass. Reads
    /// `ckpt.lzrn` (weights) and `ckpt.logits.bin` (reference logits
    /// for input `[0..seq_len)`) from the `lzr-neural` sibling repo.
    /// Skips cleanly if the artifacts aren't present so this test is
    /// safe to leave in the suite on machines that don't have them.
    ///
    /// The reference artifacts are generated by:
    ///   `uv run scripts/export_weights.py --ckpt ckpts/small_42_step2000.pt`
    ///   `uv run scripts/dump_logits_for_parity.py --ckpt ckpts/small_42_step2000.pt --seq-len 32`
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn run_parity_check(name: &str) {
        let lzrn_path = format!("/Users/creynolds/Programming/lzr-neural/ckpts/{name}.lzrn");
        let logits_path = format!("/Users/creynolds/Programming/lzr-neural/ckpts/{name}.logits.bin");
        if !std::path::Path::new(&lzrn_path).exists()
            || !std::path::Path::new(&logits_path).exists()
        {
            eprintln!("parity artifacts for {name} missing — skipping");
            return;
        }
        let lzrn = std::fs::read(&lzrn_path).unwrap();
        let model = ByteTransformer::load_lzrn(&lzrn).unwrap();
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

        let input: Vec<u8> = (0..t as u8).collect();
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
            "logit parity ({name}): t={t} vocab={vocab}  max_abs={max_abs:.6}  mean_abs={mean_abs:.6}"
        );
        assert!(
            max_abs < 1e-3,
            "logit divergence too large for {name}: max_abs={max_abs}"
        );
    }

    #[test]
    fn pytorch_logit_parity_small_42_step2000() {
        run_parity_check("small_42_step2000");
    }

    #[test]
    fn pytorch_logit_parity_medium_42_step20000() {
        run_parity_check("medium_42_step20000");
    }

    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn load_lzrn_round_trip() {
        // Hand-crafted .lzrn for a minimal config (n_layer=1, d_model=2,
        // n_head=1, d_ff=2, context=2, vocab=3). Confirms header parsing
        // and tensor stream consumption.
        let cfg_bytes: Vec<u8> = [
            MAGIC.to_le_bytes(),
            VERSION.to_le_bytes(),
            1u32.to_le_bytes(), // n_layer
            1u32.to_le_bytes(), // n_head
            2u32.to_le_bytes(), // d_model
            2u32.to_le_bytes(), // d_ff
            2u32.to_le_bytes(), // context
            3u32.to_le_bytes(), // vocab_size
        ]
        .concat();
        let n_floats = 3 * 2 // tok_emb
            + 2 * 2          // pos_emb
            + 2              // norm1
            + 6 * 2          // qkv (3*d_model x d_model = 6 x 2)
            + 2 * 2          // proj
            + 2              // norm2
            + 2 * 2          // fc1
            + 2 * 2          // fc2
            + 2; // norm_f
        let mut bytes = cfg_bytes;
        for i in 0..n_floats {
            bytes.extend_from_slice(&f32::from(i as u16).to_le_bytes());
        }
        let model = ByteTransformer::load_lzrn(&bytes).expect("load");
        assert_eq!(model.cfg.n_layer, 1);
        assert_eq!(model.cfg.d_model, 2);
        assert_eq!(model.cfg.vocab_size, 3);
        assert_eq!(model.tok_emb.len(), 6);
        assert_eq!(model.pos_emb.len(), 4);
        assert_eq!(model.blocks.len(), 1);
    }
}
