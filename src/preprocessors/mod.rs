//! Byte preprocessors — reversible byte→byte pipeline stages applied ahead of the entropy coder.
//!
//! Each is a [`Transform`] (defined in [`crate::transform`]): it rewrites the byte
//! stream on the encode side and exactly inverts it on decode. This file holds the
//! text-folding stages ([`casefold`], [`entities`]) plus the text-detection helpers
//! they share, and the byte-BPE Re-Pair grammar tokenizer ([`repair`], which emits a
//! plain-byte grammar and does not use those helpers). The trait and the
//! [`crate::transform::Pipeline`] that chains all stages live in [`crate::transform`].

mod casefold;
mod entities;
mod repair;

/// Re-exported so the byte stages can refer to `super::Transform`.
pub(crate) use crate::transform::Transform;
pub(crate) use casefold::CaseFolding;
pub(crate) use entities::EntityFolding;
#[cfg_attr(feature = "bench-internals", visibility::make(pub))]
pub(crate) use repair::RepairTokenizer;

/// Minimum fraction of bytes that must be common-text bytes for the input to count
/// as text (below this, a byte stage skips its transform and passes through).
pub(super) const TEXT_FRACTION: f64 = 0.95;

/// Header mode byte: the input was passed through unchanged.
pub(super) const MODE_PASSTHROUGH: u8 = 0;
/// Header mode byte: the payload is folded (followed by the stage's chosen control bytes).
pub(super) const MODE_FOLDED: u8 = 1;

/// Decompression-bomb ceiling for the expanding `repair` stage: a small self-describing grammar can
/// otherwise expand without bound, so the stage caps its total output here.
pub(super) const MAX_EXPANSION_BYTES: u64 = 1 << 31;

/// Whether `byte` is a byte we expect to see in plain text: printable ASCII plus
/// tab, newline, and carriage return. Shared by the byte preprocessors' text gate.
pub(super) const fn is_text_byte(byte: u8) -> bool {
    matches!(byte, b'\t' | b'\n' | b'\r' | 0x20..=0x7E)
}

/// Whether an input of `len` bytes, `text_bytes` of which are [`is_text_byte`],
/// counts as text for the byte preprocessors' fold gate.
#[expect(clippy::cast_precision_loss, reason = "byte counts are far under 2^53")]
pub(super) fn looks_like_text(len: usize, text_bytes: usize) -> bool {
    len != 0 && text_bytes as f64 >= len as f64 * TEXT_FRACTION
}

/// The self-describing pass-through encoding: a [`MODE_PASSTHROUGH`] byte followed
/// by the input verbatim. Byte stages emit this when they have nothing to fold.
pub(super) fn passthrough(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() + 1);
    out.push(MODE_PASSTHROUGH);
    out.extend_from_slice(input);
    out
}

/// Dispatch a folded/pass-through stream on its leading mode byte, the shared
/// inverse of [`passthrough`] and a stage's folded encoding. Empty input (never
/// produced by `forward`) restores as empty; `decode_folded` receives the bytes
/// after the mode byte for the folded path; `stage` names the caller for errors.
pub(super) fn inverse_framed(
    input: &[u8],
    stage: &str,
    decode_folded: impl FnOnce(&[u8]) -> anyhow::Result<Vec<u8>>,
) -> anyhow::Result<Vec<u8>> {
    let Some((&mode, rest)) = input.split_first() else {
        // Empty input is not something `forward` ever produces; treat it as an
        // empty restoration rather than panicking on corrupt decode data.
        return Ok(Vec::new());
    };
    match mode {
        MODE_PASSTHROUGH => Ok(rest.to_vec()),
        MODE_FOLDED => decode_folded(rest),
        other => anyhow::bail!("unknown {stage} mode byte {other}"),
    }
}

/// The `N` lowest byte values absent from `present`, or `None` if fewer than `N`
/// exist. Byte preprocessors use this to claim unused byte values as their control
/// symbols (deterministically, so the decoder derives the same set from its own
/// scan of the reconstructed stream).
pub(super) fn spare_bytes<const N: usize>(present: &[bool; 256]) -> Option<[u8; N]> {
    let mut unused = (0u8..=255).filter(|&b| !present[usize::from(b)]);
    let mut out = [0u8; N];
    for slot in &mut out {
        *slot = unused.next()?;
    }
    Some(out)
}
