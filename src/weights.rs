//! Weight blob layout and accessors.
//!
//! The packed weights file is a single contiguous byte buffer with a fixed,
//! compile-time-known layout (driven by [`crate::arch`]). No per-tensor
//! metadata, no safetensors; the raw-weights file's total length must equal
//! [`crate::arch::PACKED_WEIGHTS_LEN`]. Checkpoint files use the same body
//! prefixed by a 16-byte header (`LZRCKPT1` magic + u64 step count).
//!
//! Body layout (all LE for the f32 fields):
//! ```text
//! tok_emb: [f32; VOCAB * D_MODEL]
//! per layer (N_LAYERS times):
//!   attn_norm: [f32; D_MODEL]
//!   q_packed:  [u8; D_MODEL*D_MODEL/4]   q_scale: [f32; D_MODEL]
//!   k_packed, k_scale
//!   v_packed, v_scale
//!   o_packed, o_scale
//!   mlp_norm:  [f32; D_MODEL]
//!   w1_packed: [u8; D_FF*D_MODEL/4]      w1_scale: [f32; D_FF]
//!   w2_packed: [u8; D_MODEL*D_FF/4]      w2_scale: [f32; D_MODEL]
//!   w3_packed: [u8; D_FF*D_MODEL/4]      w3_scale: [f32; D_FF]
//! ```

#[cfg(feature = "training")]
use std::io::Write;
use std::path::Path;

use crate::arch::{
    D_MODEL, N_LAYERS, PACKED_QKVO_BYTES, PACKED_W1_BYTES, PACKED_W2_BYTES, PACKED_W3_BYTES,
    PACKED_WEIGHTS_LEN, SCALE_QKVO_F32S, SCALE_W1_F32S, SCALE_W2_F32S, SCALE_W3_F32S, VOCAB,
};

/// 8-byte ASCII magic for checkpoint files.
pub(crate) const CKPT_MAGIC: &[u8; 8] = b"LZRCKPT1";

const TOK_EMB_BYTES: usize = VOCAB * D_MODEL * 4;
const ATTN_NORM_BYTES: usize = D_MODEL * 4;
const MLP_NORM_BYTES: usize = D_MODEL * 4;
const SCALE_QKVO_BYTES: usize = SCALE_QKVO_F32S * 4;
const SCALE_W1_BYTES: usize = SCALE_W1_F32S * 4;
const SCALE_W2_BYTES: usize = SCALE_W2_F32S * 4;
const SCALE_W3_BYTES: usize = SCALE_W3_F32S * 4;

#[derive(Clone, Copy, Debug)]
struct LayerLayout {
    attn_norm: usize,
    q_packed: usize,
    q_scale: usize,
    k_packed: usize,
    k_scale: usize,
    v_packed: usize,
    v_scale: usize,
    o_packed: usize,
    o_scale: usize,
    mlp_norm: usize,
    w1_packed: usize,
    w1_scale: usize,
    w2_packed: usize,
    w2_scale: usize,
    w3_packed: usize,
    w3_scale: usize,
    end: usize,
}

const fn layer_layout_starting_at(start: usize) -> LayerLayout {
    let attn_norm = start;
    let q_packed = attn_norm + ATTN_NORM_BYTES;
    let q_scale = q_packed + PACKED_QKVO_BYTES;
    let k_packed = q_scale + SCALE_QKVO_BYTES;
    let k_scale = k_packed + PACKED_QKVO_BYTES;
    let v_packed = k_scale + SCALE_QKVO_BYTES;
    let v_scale = v_packed + PACKED_QKVO_BYTES;
    let o_packed = v_scale + SCALE_QKVO_BYTES;
    let o_scale = o_packed + PACKED_QKVO_BYTES;
    let mlp_norm = o_scale + SCALE_QKVO_BYTES;
    let w1_packed = mlp_norm + MLP_NORM_BYTES;
    let w1_scale = w1_packed + PACKED_W1_BYTES;
    let w2_packed = w1_scale + SCALE_W1_BYTES;
    let w2_scale = w2_packed + PACKED_W2_BYTES;
    let w3_packed = w2_scale + SCALE_W2_BYTES;
    let w3_scale = w3_packed + PACKED_W3_BYTES;
    let end = w3_scale + SCALE_W3_BYTES;
    LayerLayout {
        attn_norm,
        q_packed,
        q_scale,
        k_packed,
        k_scale,
        v_packed,
        v_scale,
        o_packed,
        o_scale,
        mlp_norm,
        w1_packed,
        w1_scale,
        w2_packed,
        w2_scale,
        w3_packed,
        w3_scale,
        end,
    }
}

const fn all_layer_layouts() -> [LayerLayout; N_LAYERS] {
    let zero = layer_layout_starting_at(0);
    let mut out = [zero; N_LAYERS];
    let mut cursor = TOK_EMB_BYTES;
    let mut i = 0;
    while i < N_LAYERS {
        out[i] = layer_layout_starting_at(cursor);
        cursor = out[i].end;
        i += 1;
    }
    out
}

const LAYERS: [LayerLayout; N_LAYERS] = all_layer_layouts();

const _: () = {
    assert!(
        LAYERS[N_LAYERS - 1].end == PACKED_WEIGHTS_LEN,
        "Layer offset table disagrees with PACKED_WEIGHTS_LEN — update arch.rs"
    );
};

/// One ternary matrix: packed weights + per-row scale.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TernaryMatrix<'a> {
    pub packed: &'a [u8],
    pub scale: &'a [f32],
}

/// Borrowed view of one transformer block's tensors.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerView<'a> {
    pub attn_norm: &'a [f32],
    pub q: TernaryMatrix<'a>,
    pub k: TernaryMatrix<'a>,
    pub v: TernaryMatrix<'a>,
    pub o: TernaryMatrix<'a>,
    pub mlp_norm: &'a [f32],
    pub w1: TernaryMatrix<'a>,
    pub w2: TernaryMatrix<'a>,
    pub w3: TernaryMatrix<'a>,
}

/// Owned, parsed weights.
///
/// All f32 tensors are copied into a single aligned `Vec<f32>` at construction
/// time (so inference can get properly-aligned slices without `unsafe`); all
/// packed tensors stay as bytes (the kernel reads them one u8 at a time so no
/// alignment is required).
#[derive(Debug)]
pub(crate) struct Weights {
    f32s: Vec<f32>,
    packed: Vec<u8>,
    f32_ix: F32Index,
    pk_ix: [PackedIndex; N_LAYERS],
}

#[derive(Debug)]
struct F32Index {
    tok_emb: usize,
    layers: [LayerF32Index; N_LAYERS],
}

#[derive(Clone, Copy, Debug)]
struct LayerF32Index {
    attn_norm: usize,
    q_scale: usize,
    k_scale: usize,
    v_scale: usize,
    o_scale: usize,
    mlp_norm: usize,
    w1_scale: usize,
    w2_scale: usize,
    w3_scale: usize,
}

#[derive(Clone, Copy, Debug)]
struct PackedIndex {
    q: usize,
    k: usize,
    v: usize,
    o: usize,
    w1: usize,
    w2: usize,
    w3: usize,
}

impl Weights {
    /// Parse a raw weights blob (no checkpoint header). Length must equal
    /// [`PACKED_WEIGHTS_LEN`].
    ///
    /// Linear top-to-bottom parse reads cleanly in the canonical layer order
    /// the file format uses — factoring into sub-functions would hide that.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn from_bytes(buf: &[u8]) -> anyhow::Result<Self> {
        if buf.len() != PACKED_WEIGHTS_LEN {
            anyhow::bail!(
                "weights blob length {} != expected {PACKED_WEIGHTS_LEN}",
                buf.len()
            );
        }
        let mut f32s = Vec::with_capacity(PACKED_WEIGHTS_LEN / 4);
        let mut packed = Vec::with_capacity(PACKED_WEIGHTS_LEN);

        let tok_emb_off = f32s.len();
        read_f32s(buf, 0, VOCAB * D_MODEL, &mut f32s);

        let mut layers_f32: [LayerF32Index; N_LAYERS] = [LayerF32Index {
            attn_norm: 0,
            q_scale: 0,
            k_scale: 0,
            v_scale: 0,
            o_scale: 0,
            mlp_norm: 0,
            w1_scale: 0,
            w2_scale: 0,
            w3_scale: 0,
        }; N_LAYERS];
        let mut layers_pk: [PackedIndex; N_LAYERS] = [PackedIndex {
            q: 0,
            k: 0,
            v: 0,
            o: 0,
            w1: 0,
            w2: 0,
            w3: 0,
        }; N_LAYERS];

        for (i, layout) in LAYERS.iter().enumerate() {
            layers_f32[i].attn_norm = f32s.len();
            read_f32s(buf, layout.attn_norm, D_MODEL, &mut f32s);

            (layers_pk[i].q, layers_f32[i].q_scale) = read_pk_scale(
                buf,
                &mut packed,
                &mut f32s,
                layout.q_packed,
                PACKED_QKVO_BYTES,
                layout.q_scale,
                SCALE_QKVO_F32S,
            );
            (layers_pk[i].k, layers_f32[i].k_scale) = read_pk_scale(
                buf,
                &mut packed,
                &mut f32s,
                layout.k_packed,
                PACKED_QKVO_BYTES,
                layout.k_scale,
                SCALE_QKVO_F32S,
            );
            (layers_pk[i].v, layers_f32[i].v_scale) = read_pk_scale(
                buf,
                &mut packed,
                &mut f32s,
                layout.v_packed,
                PACKED_QKVO_BYTES,
                layout.v_scale,
                SCALE_QKVO_F32S,
            );
            (layers_pk[i].o, layers_f32[i].o_scale) = read_pk_scale(
                buf,
                &mut packed,
                &mut f32s,
                layout.o_packed,
                PACKED_QKVO_BYTES,
                layout.o_scale,
                SCALE_QKVO_F32S,
            );

            layers_f32[i].mlp_norm = f32s.len();
            read_f32s(buf, layout.mlp_norm, D_MODEL, &mut f32s);

            (layers_pk[i].w1, layers_f32[i].w1_scale) = read_pk_scale(
                buf,
                &mut packed,
                &mut f32s,
                layout.w1_packed,
                PACKED_W1_BYTES,
                layout.w1_scale,
                SCALE_W1_F32S,
            );
            (layers_pk[i].w2, layers_f32[i].w2_scale) = read_pk_scale(
                buf,
                &mut packed,
                &mut f32s,
                layout.w2_packed,
                PACKED_W2_BYTES,
                layout.w2_scale,
                SCALE_W2_F32S,
            );
            (layers_pk[i].w3, layers_f32[i].w3_scale) = read_pk_scale(
                buf,
                &mut packed,
                &mut f32s,
                layout.w3_packed,
                PACKED_W3_BYTES,
                layout.w3_scale,
                SCALE_W3_F32S,
            );
        }

        Ok(Self {
            f32s,
            packed,
            f32_ix: F32Index {
                tok_emb: tok_emb_off,
                layers: layers_f32,
            },
            pk_ix: layers_pk,
        })
    }

    /// Load weights from a raw-bytes file (no checkpoint header).
    pub(crate) fn load_raw<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path.as_ref())?;
        Self::from_bytes(&bytes)
    }

    /// Load weights from a checkpoint file (16-byte header + body).
    /// Returns `(weights, step)`.
    pub(crate) fn load_checkpoint<P: AsRef<Path>>(path: P) -> anyhow::Result<(Self, u64)> {
        let bytes = std::fs::read(path.as_ref())?;
        if bytes.len() != 16 + PACKED_WEIGHTS_LEN {
            anyhow::bail!(
                "checkpoint length {} != expected {} (16-byte header + {PACKED_WEIGHTS_LEN} body)",
                bytes.len(),
                16 + PACKED_WEIGHTS_LEN
            );
        }
        if &bytes[0..8] != CKPT_MAGIC {
            anyhow::bail!("checkpoint magic mismatch");
        }
        let step = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let weights = Self::from_bytes(&bytes[16..])?;
        Ok((weights, step))
    }

    /// Token embedding matrix, `VOCAB × D_MODEL` in row-major layout.
    pub(crate) fn tok_emb(&self) -> &[f32] {
        let o = self.f32_ix.tok_emb;
        &self.f32s[o..o + VOCAB * D_MODEL]
    }

    pub(crate) fn layer(&self, i: usize) -> LayerView<'_> {
        let f = &self.f32_ix.layers[i];
        let p = &self.pk_ix[i];
        LayerView {
            attn_norm: &self.f32s[f.attn_norm..f.attn_norm + D_MODEL],
            q: TernaryMatrix {
                packed: &self.packed[p.q..p.q + PACKED_QKVO_BYTES],
                scale: &self.f32s[f.q_scale..f.q_scale + SCALE_QKVO_F32S],
            },
            k: TernaryMatrix {
                packed: &self.packed[p.k..p.k + PACKED_QKVO_BYTES],
                scale: &self.f32s[f.k_scale..f.k_scale + SCALE_QKVO_F32S],
            },
            v: TernaryMatrix {
                packed: &self.packed[p.v..p.v + PACKED_QKVO_BYTES],
                scale: &self.f32s[f.v_scale..f.v_scale + SCALE_QKVO_F32S],
            },
            o: TernaryMatrix {
                packed: &self.packed[p.o..p.o + PACKED_QKVO_BYTES],
                scale: &self.f32s[f.o_scale..f.o_scale + SCALE_QKVO_F32S],
            },
            mlp_norm: &self.f32s[f.mlp_norm..f.mlp_norm + D_MODEL],
            w1: TernaryMatrix {
                packed: &self.packed[p.w1..p.w1 + PACKED_W1_BYTES],
                scale: &self.f32s[f.w1_scale..f.w1_scale + SCALE_W1_F32S],
            },
            w2: TernaryMatrix {
                packed: &self.packed[p.w2..p.w2 + PACKED_W2_BYTES],
                scale: &self.f32s[f.w2_scale..f.w2_scale + SCALE_W2_F32S],
            },
            w3: TernaryMatrix {
                packed: &self.packed[p.w3..p.w3 + PACKED_W3_BYTES],
                scale: &self.f32s[f.w3_scale..f.w3_scale + SCALE_W3_F32S],
            },
        }
    }
}

/// One layer's tensors for [`write_weights`].
#[cfg(feature = "training")]
pub(crate) struct LayerTensors<'a> {
    pub attn_norm: &'a [f32],
    pub q_packed: &'a [u8],
    pub q_scale: &'a [f32],
    pub k_packed: &'a [u8],
    pub k_scale: &'a [f32],
    pub v_packed: &'a [u8],
    pub v_scale: &'a [f32],
    pub o_packed: &'a [u8],
    pub o_scale: &'a [f32],
    pub mlp_norm: &'a [f32],
    pub w1_packed: &'a [u8],
    pub w1_scale: &'a [f32],
    pub w2_packed: &'a [u8],
    pub w2_scale: &'a [f32],
    pub w3_packed: &'a [u8],
    pub w3_scale: &'a [f32],
}

/// Serialize the model tensors into the canonical weight-blob byte format.
#[cfg(feature = "training")]
pub(crate) fn write_weights(
    tok_emb: &[f32],
    layers: &[LayerTensors<'_>; N_LAYERS],
    out: &mut Vec<u8>,
) {
    out.clear();
    out.reserve(PACKED_WEIGHTS_LEN);
    write_f32s(tok_emb, out);
    for l in layers {
        write_f32s(l.attn_norm, out);
        out.extend_from_slice(l.q_packed);
        write_f32s(l.q_scale, out);
        out.extend_from_slice(l.k_packed);
        write_f32s(l.k_scale, out);
        out.extend_from_slice(l.v_packed);
        write_f32s(l.v_scale, out);
        out.extend_from_slice(l.o_packed);
        write_f32s(l.o_scale, out);
        write_f32s(l.mlp_norm, out);
        out.extend_from_slice(l.w1_packed);
        write_f32s(l.w1_scale, out);
        out.extend_from_slice(l.w2_packed);
        write_f32s(l.w2_scale, out);
        out.extend_from_slice(l.w3_packed);
        write_f32s(l.w3_scale, out);
    }
    debug_assert_eq!(out.len(), PACKED_WEIGHTS_LEN);
}

/// Write a checkpoint file: 16-byte header (magic + step LE u64) + body.
#[cfg(feature = "training")]
pub(crate) fn write_checkpoint<W: Write>(
    out: &mut W,
    step: u64,
    tok_emb: &[f32],
    layers: &[LayerTensors<'_>; N_LAYERS],
) -> std::io::Result<()> {
    out.write_all(CKPT_MAGIC)?;
    out.write_all(&step.to_le_bytes())?;
    let mut body = Vec::new();
    write_weights(tok_emb, layers, &mut body);
    out.write_all(&body)?;
    Ok(())
}

fn read_f32s(buf: &[u8], offset: usize, count: usize, out: &mut Vec<f32>) {
    for i in 0..count {
        let b = &buf[offset + i * 4..offset + i * 4 + 4];
        out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    }
}

/// Copy one `packed + scale` pair from `buf` into the parsed buffers.
/// Returns `(packed_offset, scale_offset)` into `packed` / `f32s` so the
/// caller can record indices into its layer tables.
fn read_pk_scale(
    buf: &[u8],
    packed: &mut Vec<u8>,
    f32s: &mut Vec<f32>,
    packed_src: usize,
    packed_len: usize,
    scale_src: usize,
    scale_count: usize,
) -> (usize, usize) {
    let pk_off = packed.len();
    packed.extend_from_slice(&buf[packed_src..packed_src + packed_len]);
    let scale_off = f32s.len();
    read_f32s(buf, scale_src, scale_count, f32s);
    (pk_off, scale_off)
}

#[cfg(feature = "training")]
fn write_f32s(src: &[f32], out: &mut Vec<u8>) {
    for &x in src {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_parses_and_views_have_right_sizes() {
        let buf = vec![0u8; PACKED_WEIGHTS_LEN];
        let w = Weights::from_bytes(&buf).unwrap();
        assert_eq!(w.tok_emb().len(), VOCAB * D_MODEL);
        for i in 0..N_LAYERS {
            let lv = w.layer(i);
            assert_eq!(lv.attn_norm.len(), D_MODEL);
            assert_eq!(lv.mlp_norm.len(), D_MODEL);
            assert_eq!(lv.q.packed.len(), PACKED_QKVO_BYTES);
            assert_eq!(lv.q.scale.len(), SCALE_QKVO_F32S);
            assert_eq!(lv.w1.packed.len(), PACKED_W1_BYTES);
            assert_eq!(lv.w1.scale.len(), SCALE_W1_F32S);
            assert_eq!(lv.w2.packed.len(), PACKED_W2_BYTES);
            assert_eq!(lv.w2.scale.len(), SCALE_W2_F32S);
            assert_eq!(lv.w3.packed.len(), PACKED_W3_BYTES);
            assert_eq!(lv.w3.scale.len(), SCALE_W3_F32S);
        }
    }

    #[test]
    fn rejects_wrong_size_blob() {
        let buf = vec![0u8; PACKED_WEIGHTS_LEN - 1];
        assert!(Weights::from_bytes(&buf).is_err());
    }

    #[test]
    fn checkpoint_header_roundtrip() {
        let buf = vec![0u8; PACKED_WEIGHTS_LEN];
        // Manually build a checkpoint with header.
        let mut ckpt = Vec::with_capacity(16 + PACKED_WEIGHTS_LEN);
        ckpt.extend_from_slice(CKPT_MAGIC);
        ckpt.extend_from_slice(&1234_u64.to_le_bytes());
        ckpt.extend_from_slice(&buf);

        let dir = std::env::temp_dir().join("lzr_ckpt_roundtrip_test.bin");
        std::fs::write(&dir, &ckpt).unwrap();
        let (_w, step) = Weights::load_checkpoint(&dir).unwrap();
        assert_eq!(step, 1234);
        std::fs::remove_file(&dir).ok();
    }
}
