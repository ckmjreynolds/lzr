//! Weight blob layout and accessors for the RWKV v4 byte-level model.
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
//! ln0:     [f32; D_MODEL]
//! ln_f:    [f32; D_MODEL]
//! per layer (N_LAYERS times):
//!   tm_norm:      [f32; D_MODEL]
//!   time_mix_r:   [f32; D_MODEL]
//!   time_mix_k:   [f32; D_MODEL]
//!   time_mix_v:   [f32; D_MODEL]
//!   time_decay:   [f32; D_MODEL]
//!   time_first:   [f32; D_MODEL]
//!   tm_r_packed + tm_r_scale
//!   tm_k_packed + tm_k_scale
//!   tm_v_packed + tm_v_scale
//!   tm_o_packed + tm_o_scale
//!   cm_norm:      [f32; D_MODEL]
//!   channel_mix_k:[f32; D_MODEL]
//!   channel_mix_r:[f32; D_MODEL]
//!   cm_k_packed + cm_k_scale  (D_FF × D_MODEL)
//!   cm_v_packed + cm_v_scale  (D_MODEL × D_FF)
//!   cm_r_packed + cm_r_scale  (D_MODEL × D_MODEL)
//! ```

#[cfg(feature = "training")]
use std::io::Write;
use std::path::Path;

use crate::arch::{
    D_FF, D_MODEL, N_LAYERS, PACKED_CM_K_BYTES, PACKED_CM_R_BYTES, PACKED_CM_V_BYTES,
    PACKED_TM_BYTES, PACKED_WEIGHTS_LEN, SCALE_CM_K_F32S, SCALE_CM_R_F32S, SCALE_CM_V_F32S,
    SCALE_TM_F32S, VOCAB,
};
use crate::bitnet::{lut_packed_bytes, lut_supports, repack_i2s_to_lut};

/// 8-byte ASCII magic for checkpoint files.
pub(crate) const CKPT_MAGIC: &[u8; 8] = b"LZRCKPT1";

const TOK_EMB_BYTES: usize = VOCAB * D_MODEL * 4;
const LN0_BYTES: usize = D_MODEL * 4;
const LN_F_BYTES: usize = D_MODEL * 4;

const TM_NORM_BYTES: usize = D_MODEL * 4;
const TIME_MIX_BYTES: usize = D_MODEL * 4; // per time_mix_* vector
const TIME_DECAY_BYTES: usize = D_MODEL * 4;
const TIME_FIRST_BYTES: usize = D_MODEL * 4;
const SCALE_TM_BYTES: usize = SCALE_TM_F32S * 4;

const CM_NORM_BYTES: usize = D_MODEL * 4;
const CHANNEL_MIX_BYTES: usize = D_MODEL * 4;
const SCALE_CM_K_BYTES: usize = SCALE_CM_K_F32S * 4;
const SCALE_CM_V_BYTES: usize = SCALE_CM_V_F32S * 4;
const SCALE_CM_R_BYTES: usize = SCALE_CM_R_F32S * 4;

/// Per-matrix LUT buffer sizes (0 on architectures without a LUT kernel).
const LUT_TM_BYTES: usize = lut_packed_bytes(D_MODEL, D_MODEL);
const LUT_CM_K_BYTES: usize = lut_packed_bytes(D_FF, D_MODEL);
const LUT_CM_V_BYTES: usize = lut_packed_bytes(D_MODEL, D_FF);
const LUT_CM_R_BYTES: usize = lut_packed_bytes(D_MODEL, D_MODEL);
const LUT_LAYER_BYTES: usize = 4 * LUT_TM_BYTES + LUT_CM_K_BYTES + LUT_CM_V_BYTES + LUT_CM_R_BYTES;
const LUT_TOTAL_BYTES: usize = N_LAYERS * LUT_LAYER_BYTES;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const _: () = {
    assert!(
        lut_supports(D_MODEL, D_MODEL),
        "LUT kernel does not support D_MODEL x D_MODEL shapes — widen the supported range"
    );
    assert!(
        lut_supports(D_FF, D_MODEL),
        "LUT kernel does not support cm_k shape"
    );
    assert!(
        lut_supports(D_MODEL, D_FF),
        "LUT kernel does not support cm_v shape"
    );
};

#[derive(Clone, Copy, Debug)]
struct LayerLayout {
    tm_norm: usize,
    time_mix_r: usize,
    time_mix_k: usize,
    time_mix_v: usize,
    time_decay: usize,
    time_first: usize,
    tm_r_packed: usize,
    tm_r_scale: usize,
    tm_k_packed: usize,
    tm_k_scale: usize,
    tm_v_packed: usize,
    tm_v_scale: usize,
    tm_o_packed: usize,
    tm_o_scale: usize,
    cm_norm: usize,
    channel_mix_k: usize,
    channel_mix_r: usize,
    cm_k_packed: usize,
    cm_k_scale: usize,
    cm_v_packed: usize,
    cm_v_scale: usize,
    cm_r_packed: usize,
    cm_r_scale: usize,
    end: usize,
}

// The per-tensor offset names (`tm_r_packed` / `tm_r_scale`, etc.) mirror
// the literal weight-file layout; clippy's similar-names heuristic flags
// every packed/scale pair.
#[allow(clippy::similar_names)]
const fn layer_layout_starting_at(start: usize) -> LayerLayout {
    let tm_norm = start;
    let time_mix_r = tm_norm + TM_NORM_BYTES;
    let time_mix_k = time_mix_r + TIME_MIX_BYTES;
    let time_mix_v = time_mix_k + TIME_MIX_BYTES;
    let time_decay = time_mix_v + TIME_MIX_BYTES;
    let time_first = time_decay + TIME_DECAY_BYTES;
    let tm_r_packed = time_first + TIME_FIRST_BYTES;
    let tm_r_scale = tm_r_packed + PACKED_TM_BYTES;
    let tm_k_packed = tm_r_scale + SCALE_TM_BYTES;
    let tm_k_scale = tm_k_packed + PACKED_TM_BYTES;
    let tm_v_packed = tm_k_scale + SCALE_TM_BYTES;
    let tm_v_scale = tm_v_packed + PACKED_TM_BYTES;
    let tm_o_packed = tm_v_scale + SCALE_TM_BYTES;
    let tm_o_scale = tm_o_packed + PACKED_TM_BYTES;
    let cm_norm = tm_o_scale + SCALE_TM_BYTES;
    let channel_mix_k = cm_norm + CM_NORM_BYTES;
    let channel_mix_r = channel_mix_k + CHANNEL_MIX_BYTES;
    let cm_k_packed = channel_mix_r + CHANNEL_MIX_BYTES;
    let cm_k_scale = cm_k_packed + PACKED_CM_K_BYTES;
    let cm_v_packed = cm_k_scale + SCALE_CM_K_BYTES;
    let cm_v_scale = cm_v_packed + PACKED_CM_V_BYTES;
    let cm_r_packed = cm_v_scale + SCALE_CM_V_BYTES;
    let cm_r_scale = cm_r_packed + PACKED_CM_R_BYTES;
    let end = cm_r_scale + SCALE_CM_R_BYTES;
    LayerLayout {
        tm_norm,
        time_mix_r,
        time_mix_k,
        time_mix_v,
        time_decay,
        time_first,
        tm_r_packed,
        tm_r_scale,
        tm_k_packed,
        tm_k_scale,
        tm_v_packed,
        tm_v_scale,
        tm_o_packed,
        tm_o_scale,
        cm_norm,
        channel_mix_k,
        channel_mix_r,
        cm_k_packed,
        cm_k_scale,
        cm_v_packed,
        cm_v_scale,
        cm_r_packed,
        cm_r_scale,
        end,
    }
}

const fn all_layer_layouts() -> [LayerLayout; N_LAYERS] {
    let zero = layer_layout_starting_at(0);
    let mut out = [zero; N_LAYERS];
    let mut cursor = TOK_EMB_BYTES + LN0_BYTES + LN_F_BYTES;
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

/// One ternary matrix: I2_S-packed weights + arch-preferred LUT-packed
/// weights + per-row scale.
///
/// `lut_packed` is an empty slice on architectures without a LUT kernel; the
/// inference path must check and fall back to `packed` + scalar matvec.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TernaryMatrix<'a> {
    pub packed: &'a [u8],
    pub lut_packed: &'a [u8],
    pub scale: &'a [f32],
}

/// Borrowed view of one RWKV block's tensors.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerView<'a> {
    pub tm_norm: &'a [f32],
    pub time_mix_r: &'a [f32],
    pub time_mix_k: &'a [f32],
    pub time_mix_v: &'a [f32],
    pub time_decay: &'a [f32],
    pub time_first: &'a [f32],
    pub tm_r: TernaryMatrix<'a>,
    pub tm_k: TernaryMatrix<'a>,
    pub tm_v: TernaryMatrix<'a>,
    pub tm_o: TernaryMatrix<'a>,
    pub cm_norm: &'a [f32],
    pub channel_mix_k: &'a [f32],
    pub channel_mix_r: &'a [f32],
    pub cm_k: TernaryMatrix<'a>,
    pub cm_v: TernaryMatrix<'a>,
    pub cm_r: TernaryMatrix<'a>,
}

/// Owned, parsed weights.
#[derive(Debug)]
pub(crate) struct Weights {
    f32s: Vec<f32>,
    packed: Vec<u8>,
    /// Arch-preferred LUT packing (`TL1` on aarch64, `TL2` on `x86_64`,
    /// empty elsewhere). Built once at load time from `packed`.
    lut: Vec<u8>,
    f32_ix: F32Index,
    pk_ix: [TensorIndex; N_LAYERS],
    lut_ix: [TensorIndex; N_LAYERS],
}

#[derive(Debug)]
struct F32Index {
    tok_emb: usize,
    ln0: usize,
    ln_f: usize,
    layers: [LayerF32Index; N_LAYERS],
}

#[derive(Clone, Copy, Debug)]
struct LayerF32Index {
    tm_norm: usize,
    time_mix_r: usize,
    time_mix_k: usize,
    time_mix_v: usize,
    time_decay: usize,
    time_first: usize,
    tm_r_scale: usize,
    tm_k_scale: usize,
    tm_v_scale: usize,
    tm_o_scale: usize,
    cm_norm: usize,
    channel_mix_k: usize,
    channel_mix_r: usize,
    cm_k_scale: usize,
    cm_v_scale: usize,
    cm_r_scale: usize,
}

/// Per-layer byte offsets into either the `I2_S` `packed` buffer or the
/// arch-preferred `lut` buffer. Seven ternary tensors per block.
#[derive(Clone, Copy, Debug, Default)]
struct TensorIndex {
    tm_r: usize,
    tm_k: usize,
    tm_v: usize,
    tm_o: usize,
    cm_k: usize,
    cm_v: usize,
    cm_r: usize,
}

impl Weights {
    /// Parse a raw weights blob (no checkpoint header). Length must equal
    /// [`PACKED_WEIGHTS_LEN`].
    pub(crate) fn from_bytes(buf: &[u8]) -> anyhow::Result<Self> {
        Self::from_bytes_inner(buf, false)
    }

    #[allow(clippy::too_many_lines)]
    fn from_bytes_inner(buf: &[u8], force_i2s: bool) -> anyhow::Result<Self> {
        if buf.len() != PACKED_WEIGHTS_LEN {
            anyhow::bail!(
                "weights blob length {} != expected {PACKED_WEIGHTS_LEN}",
                buf.len()
            );
        }
        let mut f32s = Vec::with_capacity(PACKED_WEIGHTS_LEN / 4);
        let mut packed = Vec::with_capacity(PACKED_WEIGHTS_LEN);
        let mut lut = Vec::with_capacity(LUT_TOTAL_BYTES);

        let tok_emb_off = f32s.len();
        read_f32s(buf, 0, VOCAB * D_MODEL, &mut f32s);
        let ln0_off = f32s.len();
        read_f32s(buf, TOK_EMB_BYTES, D_MODEL, &mut f32s);
        let ln_f_off = f32s.len();
        read_f32s(buf, TOK_EMB_BYTES + LN0_BYTES, D_MODEL, &mut f32s);

        let zero_f32_idx = LayerF32Index {
            tm_norm: 0,
            time_mix_r: 0,
            time_mix_k: 0,
            time_mix_v: 0,
            time_decay: 0,
            time_first: 0,
            tm_r_scale: 0,
            tm_k_scale: 0,
            tm_v_scale: 0,
            tm_o_scale: 0,
            cm_norm: 0,
            channel_mix_k: 0,
            channel_mix_r: 0,
            cm_k_scale: 0,
            cm_v_scale: 0,
            cm_r_scale: 0,
        };
        let mut layers_f32: [LayerF32Index; N_LAYERS] = [zero_f32_idx; N_LAYERS];
        let mut layers_pk: [TensorIndex; N_LAYERS] = [TensorIndex::default(); N_LAYERS];
        let mut layers_lut: [TensorIndex; N_LAYERS] = [TensorIndex::default(); N_LAYERS];

        for (i, layout) in LAYERS.iter().enumerate() {
            layers_f32[i].tm_norm = f32s.len();
            read_f32s(buf, layout.tm_norm, D_MODEL, &mut f32s);
            layers_f32[i].time_mix_r = f32s.len();
            read_f32s(buf, layout.time_mix_r, D_MODEL, &mut f32s);
            layers_f32[i].time_mix_k = f32s.len();
            read_f32s(buf, layout.time_mix_k, D_MODEL, &mut f32s);
            layers_f32[i].time_mix_v = f32s.len();
            read_f32s(buf, layout.time_mix_v, D_MODEL, &mut f32s);
            layers_f32[i].time_decay = f32s.len();
            read_f32s(buf, layout.time_decay, D_MODEL, &mut f32s);
            layers_f32[i].time_first = f32s.len();
            read_f32s(buf, layout.time_first, D_MODEL, &mut f32s);

            (
                layers_pk[i].tm_r,
                layers_lut[i].tm_r,
                layers_f32[i].tm_r_scale,
            ) = read_pk_scale(
                buf,
                &mut packed,
                &mut lut,
                &mut f32s,
                layout.tm_r_packed,
                PACKED_TM_BYTES,
                layout.tm_r_scale,
                SCALE_TM_F32S,
                D_MODEL,
                D_MODEL,
                force_i2s,
            );
            (
                layers_pk[i].tm_k,
                layers_lut[i].tm_k,
                layers_f32[i].tm_k_scale,
            ) = read_pk_scale(
                buf,
                &mut packed,
                &mut lut,
                &mut f32s,
                layout.tm_k_packed,
                PACKED_TM_BYTES,
                layout.tm_k_scale,
                SCALE_TM_F32S,
                D_MODEL,
                D_MODEL,
                force_i2s,
            );
            (
                layers_pk[i].tm_v,
                layers_lut[i].tm_v,
                layers_f32[i].tm_v_scale,
            ) = read_pk_scale(
                buf,
                &mut packed,
                &mut lut,
                &mut f32s,
                layout.tm_v_packed,
                PACKED_TM_BYTES,
                layout.tm_v_scale,
                SCALE_TM_F32S,
                D_MODEL,
                D_MODEL,
                force_i2s,
            );
            (
                layers_pk[i].tm_o,
                layers_lut[i].tm_o,
                layers_f32[i].tm_o_scale,
            ) = read_pk_scale(
                buf,
                &mut packed,
                &mut lut,
                &mut f32s,
                layout.tm_o_packed,
                PACKED_TM_BYTES,
                layout.tm_o_scale,
                SCALE_TM_F32S,
                D_MODEL,
                D_MODEL,
                force_i2s,
            );

            layers_f32[i].cm_norm = f32s.len();
            read_f32s(buf, layout.cm_norm, D_MODEL, &mut f32s);
            layers_f32[i].channel_mix_k = f32s.len();
            read_f32s(buf, layout.channel_mix_k, D_MODEL, &mut f32s);
            layers_f32[i].channel_mix_r = f32s.len();
            read_f32s(buf, layout.channel_mix_r, D_MODEL, &mut f32s);

            (
                layers_pk[i].cm_k,
                layers_lut[i].cm_k,
                layers_f32[i].cm_k_scale,
            ) = read_pk_scale(
                buf,
                &mut packed,
                &mut lut,
                &mut f32s,
                layout.cm_k_packed,
                PACKED_CM_K_BYTES,
                layout.cm_k_scale,
                SCALE_CM_K_F32S,
                D_FF,
                D_MODEL,
                force_i2s,
            );
            (
                layers_pk[i].cm_v,
                layers_lut[i].cm_v,
                layers_f32[i].cm_v_scale,
            ) = read_pk_scale(
                buf,
                &mut packed,
                &mut lut,
                &mut f32s,
                layout.cm_v_packed,
                PACKED_CM_V_BYTES,
                layout.cm_v_scale,
                SCALE_CM_V_F32S,
                D_MODEL,
                D_FF,
                force_i2s,
            );
            (
                layers_pk[i].cm_r,
                layers_lut[i].cm_r,
                layers_f32[i].cm_r_scale,
            ) = read_pk_scale(
                buf,
                &mut packed,
                &mut lut,
                &mut f32s,
                layout.cm_r_packed,
                PACKED_CM_R_BYTES,
                layout.cm_r_scale,
                SCALE_CM_R_F32S,
                D_MODEL,
                D_MODEL,
                force_i2s,
            );
        }

        // Once the LUT buffer is built, the I2_S `packed` bytes are dead
        // RSS on LUT-supported arches (aarch64 / x86_64) — `matvec_prequant`
        // always routes to the LUT kernel there. Drop them. On unsupported
        // arches or under `force_i2s` diagnostic mode, `lut` stayed empty
        // and `packed` is the live buffer.
        if !lut.is_empty() {
            packed = Vec::new();
        }

        Ok(Self {
            f32s,
            packed,
            lut,
            f32_ix: F32Index {
                tok_emb: tok_emb_off,
                ln0: ln0_off,
                ln_f: ln_f_off,
                layers: layers_f32,
            },
            pk_ix: layers_pk,
            lut_ix: layers_lut,
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
        Self::load_checkpoint_inner(path)
    }

    fn load_checkpoint_inner<P: AsRef<Path>>(path: P) -> anyhow::Result<(Self, u64)> {
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
        let weights = Self::from_bytes_inner(&bytes[16..], false)?;
        Ok((weights, step))
    }

    /// Token embedding matrix, `VOCAB × D_MODEL` in row-major layout.
    pub(crate) fn tok_emb(&self) -> &[f32] {
        let o = self.f32_ix.tok_emb;
        &self.f32s[o..o + VOCAB * D_MODEL]
    }

    /// Initial `RMSNorm` weight applied to the token embedding before the
    /// first block.
    pub(crate) fn ln0(&self) -> &[f32] {
        let o = self.f32_ix.ln0;
        &self.f32s[o..o + D_MODEL]
    }

    /// Final `RMSNorm` weight applied to the last block's output before the
    /// LM head.
    pub(crate) fn ln_f(&self) -> &[f32] {
        let o = self.f32_ix.ln_f;
        &self.f32s[o..o + D_MODEL]
    }

    pub(crate) fn layer(&self, i: usize) -> LayerView<'_> {
        let f = &self.f32_ix.layers[i];
        let p = &self.pk_ix[i];
        let l = &self.lut_ix[i];
        // Each buffer returns `&[]` when its source `Vec` is empty:
        //   - `packed` empty when LUT was built and I2_S bytes were dropped.
        //   - `lut` empty when LUT was skipped (non-LUT arch, or the
        //     diagnostic `from_bytes_force_i2s` path).
        // `matvec_prequant` dispatches on `lut_packed.is_empty()`, so these
        // guards route transparently.
        let pk = |off: usize, len: usize| -> &[u8] {
            if self.packed.is_empty() {
                &[]
            } else {
                &self.packed[off..off + len]
            }
        };
        let lp = |off: usize, len: usize| -> &[u8] {
            if self.lut.is_empty() {
                &[]
            } else {
                &self.lut[off..off + len]
            }
        };
        LayerView {
            tm_norm: &self.f32s[f.tm_norm..f.tm_norm + D_MODEL],
            time_mix_r: &self.f32s[f.time_mix_r..f.time_mix_r + D_MODEL],
            time_mix_k: &self.f32s[f.time_mix_k..f.time_mix_k + D_MODEL],
            time_mix_v: &self.f32s[f.time_mix_v..f.time_mix_v + D_MODEL],
            time_decay: &self.f32s[f.time_decay..f.time_decay + D_MODEL],
            time_first: &self.f32s[f.time_first..f.time_first + D_MODEL],
            tm_r: TernaryMatrix {
                packed: pk(p.tm_r, PACKED_TM_BYTES),
                lut_packed: lp(l.tm_r, LUT_TM_BYTES),
                scale: &self.f32s[f.tm_r_scale..f.tm_r_scale + SCALE_TM_F32S],
            },
            tm_k: TernaryMatrix {
                packed: pk(p.tm_k, PACKED_TM_BYTES),
                lut_packed: lp(l.tm_k, LUT_TM_BYTES),
                scale: &self.f32s[f.tm_k_scale..f.tm_k_scale + SCALE_TM_F32S],
            },
            tm_v: TernaryMatrix {
                packed: pk(p.tm_v, PACKED_TM_BYTES),
                lut_packed: lp(l.tm_v, LUT_TM_BYTES),
                scale: &self.f32s[f.tm_v_scale..f.tm_v_scale + SCALE_TM_F32S],
            },
            tm_o: TernaryMatrix {
                packed: pk(p.tm_o, PACKED_TM_BYTES),
                lut_packed: lp(l.tm_o, LUT_TM_BYTES),
                scale: &self.f32s[f.tm_o_scale..f.tm_o_scale + SCALE_TM_F32S],
            },
            cm_norm: &self.f32s[f.cm_norm..f.cm_norm + D_MODEL],
            channel_mix_k: &self.f32s[f.channel_mix_k..f.channel_mix_k + D_MODEL],
            channel_mix_r: &self.f32s[f.channel_mix_r..f.channel_mix_r + D_MODEL],
            cm_k: TernaryMatrix {
                packed: pk(p.cm_k, PACKED_CM_K_BYTES),
                lut_packed: lp(l.cm_k, LUT_CM_K_BYTES),
                scale: &self.f32s[f.cm_k_scale..f.cm_k_scale + SCALE_CM_K_F32S],
            },
            cm_v: TernaryMatrix {
                packed: pk(p.cm_v, PACKED_CM_V_BYTES),
                lut_packed: lp(l.cm_v, LUT_CM_V_BYTES),
                scale: &self.f32s[f.cm_v_scale..f.cm_v_scale + SCALE_CM_V_F32S],
            },
            cm_r: TernaryMatrix {
                packed: pk(p.cm_r, PACKED_CM_R_BYTES),
                lut_packed: lp(l.cm_r, LUT_CM_R_BYTES),
                scale: &self.f32s[f.cm_r_scale..f.cm_r_scale + SCALE_CM_R_F32S],
            },
        }
    }
}

/// One layer's tensors for [`write_weights`].
#[cfg(feature = "training")]
pub(crate) struct LayerTensors<'a> {
    pub tm_norm: &'a [f32],
    pub time_mix_r: &'a [f32],
    pub time_mix_k: &'a [f32],
    pub time_mix_v: &'a [f32],
    pub time_decay: &'a [f32],
    pub time_first: &'a [f32],
    pub tm_r_packed: &'a [u8],
    pub tm_r_scale: &'a [f32],
    pub tm_k_packed: &'a [u8],
    pub tm_k_scale: &'a [f32],
    pub tm_v_packed: &'a [u8],
    pub tm_v_scale: &'a [f32],
    pub tm_o_packed: &'a [u8],
    pub tm_o_scale: &'a [f32],
    pub cm_norm: &'a [f32],
    pub channel_mix_k: &'a [f32],
    pub channel_mix_r: &'a [f32],
    pub cm_k_packed: &'a [u8],
    pub cm_k_scale: &'a [f32],
    pub cm_v_packed: &'a [u8],
    pub cm_v_scale: &'a [f32],
    pub cm_r_packed: &'a [u8],
    pub cm_r_scale: &'a [f32],
}

/// Serialize the model tensors into the canonical weight-blob byte format.
#[cfg(feature = "training")]
pub(crate) fn write_weights(
    tok_emb: &[f32],
    ln0: &[f32],
    ln_f: &[f32],
    layers: &[LayerTensors<'_>; N_LAYERS],
    out: &mut Vec<u8>,
) {
    out.clear();
    out.reserve(PACKED_WEIGHTS_LEN);
    write_f32s(tok_emb, out);
    write_f32s(ln0, out);
    write_f32s(ln_f, out);
    for l in layers {
        write_f32s(l.tm_norm, out);
        write_f32s(l.time_mix_r, out);
        write_f32s(l.time_mix_k, out);
        write_f32s(l.time_mix_v, out);
        write_f32s(l.time_decay, out);
        write_f32s(l.time_first, out);
        out.extend_from_slice(l.tm_r_packed);
        write_f32s(l.tm_r_scale, out);
        out.extend_from_slice(l.tm_k_packed);
        write_f32s(l.tm_k_scale, out);
        out.extend_from_slice(l.tm_v_packed);
        write_f32s(l.tm_v_scale, out);
        out.extend_from_slice(l.tm_o_packed);
        write_f32s(l.tm_o_scale, out);
        write_f32s(l.cm_norm, out);
        write_f32s(l.channel_mix_k, out);
        write_f32s(l.channel_mix_r, out);
        out.extend_from_slice(l.cm_k_packed);
        write_f32s(l.cm_k_scale, out);
        out.extend_from_slice(l.cm_v_packed);
        write_f32s(l.cm_v_scale, out);
        out.extend_from_slice(l.cm_r_packed);
        write_f32s(l.cm_r_scale, out);
    }
    debug_assert_eq!(out.len(), PACKED_WEIGHTS_LEN);
}

/// Write a checkpoint file: 16-byte header (magic + step LE u64) + body.
#[cfg(feature = "training")]
pub(crate) fn write_checkpoint<W: Write>(
    out: &mut W,
    step: u64,
    tok_emb: &[f32],
    ln0: &[f32],
    ln_f: &[f32],
    layers: &[LayerTensors<'_>; N_LAYERS],
) -> std::io::Result<()> {
    out.write_all(CKPT_MAGIC)?;
    out.write_all(&step.to_le_bytes())?;
    let mut body = Vec::with_capacity(PACKED_WEIGHTS_LEN);
    write_weights(tok_emb, ln0, ln_f, layers, &mut body);
    out.write_all(&body)?;
    Ok(())
}

fn read_f32s(buf: &[u8], byte_off: usize, count: usize, out: &mut Vec<f32>) {
    for i in 0..count {
        let o = byte_off + i * 4;
        let bytes: [u8; 4] = buf[o..o + 4].try_into().unwrap();
        out.push(f32::from_le_bytes(bytes));
    }
}

#[allow(clippy::too_many_arguments)]
fn read_pk_scale(
    buf: &[u8],
    packed: &mut Vec<u8>,
    lut: &mut Vec<u8>,
    f32s: &mut Vec<f32>,
    packed_src: usize,
    packed_len: usize,
    scale_src: usize,
    scale_count: usize,
    out_dim: usize,
    in_dim: usize,
    force_i2s: bool,
) -> (usize, usize, usize) {
    let pk_off = packed.len();
    packed.extend_from_slice(&buf[packed_src..packed_src + packed_len]);

    let lut_off = lut.len();
    let lut_len = lut_packed_bytes(out_dim, in_dim);
    if lut_len > 0 && !force_i2s {
        lut.resize(lut_off + lut_len, 0);
        let i2s_bytes = &packed[pk_off..pk_off + packed_len];
        repack_i2s_to_lut(
            i2s_bytes,
            &mut lut[lut_off..lut_off + lut_len],
            out_dim,
            in_dim,
        );
    }

    let scale_off = f32s.len();
    read_f32s(buf, scale_src, scale_count, f32s);
    (pk_off, lut_off, scale_off)
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
        assert_eq!(w.ln0().len(), D_MODEL);
        assert_eq!(w.ln_f().len(), D_MODEL);
        for i in 0..N_LAYERS {
            let lv = w.layer(i);
            assert_eq!(lv.tm_norm.len(), D_MODEL);
            assert_eq!(lv.time_mix_r.len(), D_MODEL);
            assert_eq!(lv.time_mix_k.len(), D_MODEL);
            assert_eq!(lv.time_mix_v.len(), D_MODEL);
            assert_eq!(lv.time_decay.len(), D_MODEL);
            assert_eq!(lv.time_first.len(), D_MODEL);
            assert_eq!(lv.cm_norm.len(), D_MODEL);
            assert_eq!(lv.channel_mix_k.len(), D_MODEL);
            assert_eq!(lv.channel_mix_r.len(), D_MODEL);
            for mat in [&lv.tm_r, &lv.tm_k, &lv.tm_v, &lv.tm_o, &lv.cm_r] {
                assert_eq!(mat.scale.len(), D_MODEL);
                // Exactly one of packed or lut_packed is populated.
                assert!(!mat.packed.is_empty() || !mat.lut_packed.is_empty());
            }
            assert_eq!(lv.cm_k.scale.len(), D_FF);
            assert_eq!(lv.cm_v.scale.len(), D_MODEL);
        }
    }
}
