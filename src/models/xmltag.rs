//! XML-tag model: a hashed `StateMap` keyed on the **nearest enclosing tag name** plus the bit-tree
//! node.
//!
//! `MediaWiki` dumps (enwik) are very regular XML: `<id>` bodies are digits, `<timestamp>` bodies are
//! ISO dates, `<title>`/`<text>` bodies are prose, `<username>`/`<comment>` their own registers. This
//! model finds the most recent `<` within a short back-window, reads the element name that follows it
//! (post-casefold, so lowercased; a leading `/` of a closing tag is skipped), and conditions the
//! prediction on that name. Field contents are then predicted per-field rather than blended together.
//!
//! It abstains when no `<` sits within the window — i.e. deep inside a long `<text>` body, where the
//! order-N and word models already carry the context. It is a pure function of the finalized bytes
//! (`hist`), so encode and decode stay in lock-step; it only feeds the mixer, so a bug degrades ratio,
//! never correctness.

use super::statemap::StateMap;
use super::{Context, Model, byte_hash, hashed_bits};

/// How far back to scan for the opening `<` of the enclosing tag. Beyond this we treat the position as
/// free text (abstain). Covers the short structural fields (`<id>`, `<timestamp>`, `<username>`, …)
/// without paying for a long scan inside `<text>` bodies.
const MAX_BACK: usize = 64;

/// Cap on the tag-name length folded into the context (element names are short).
const MAX_NAME: usize = 16;

/// Predicts each bit from the nearest enclosing tag name plus the bit-tree node.
#[derive(Debug)]
pub(crate) struct XmlTagModel {
    /// Right-shift folding the multiplicative hash down to the table's index width.
    shift: u32,
    /// The adaptive probability map, one slot per (hashed) context.
    sm: StateMap,
    /// Slot chosen by the last [`XmlTagModel::predict`], reused by the paired [`XmlTagModel::update`];
    /// `None` when the model abstained (no nearby tag).
    idx: Option<usize>,
}

impl XmlTagModel {
    /// A fresh model. `capacity` (the framed byte count) sizes the hashed table via [`hashed_bits`] so
    /// encode and decode agree.
    pub(crate) fn new(capacity: usize) -> Self {
        let bits = hashed_bits(capacity);
        Self {
            shift: u64::BITS - bits,
            sm: StateMap::new(1 << bits),
            idx: None,
        }
    }

    /// The map slot for the current bit, or `None` to abstain: the nearest `<`-introduced tag name
    /// within [`MAX_BACK`] bytes, folded with the bit-tree node. Abstains when no `<` is in range or the
    /// tag has no name (e.g. `</` at the window edge).
    #[expect(
        clippy::cast_possible_truncation,
        reason = "The hashed value is folded down to `shift` bits, so it indexes the table exactly."
    )]
    fn slot(&self, ctx: &Context, hist: &[u8]) -> Option<usize> {
        let window = &hist[hist.len().saturating_sub(MAX_BACK)..];
        let lt = window.iter().rposition(|&b| b == b'<')?;
        // Skip the '<' and a possible '/' of a closing tag, then read the element-name bytes.
        let mut i = lt + 1;
        if window.get(i) == Some(&b'/') {
            i += 1;
        }
        let name = window[i..].iter().copied().take_while(u8::is_ascii_lowercase).take(MAX_NAME);
        let mut len = 0;
        let hash = byte_hash(ctx.node(), name.inspect(|_| len += 1));
        (len > 0).then(|| (hash >> self.shift) as usize)
    }
}

impl Model for XmlTagModel {
    fn predict(&mut self, ctx: &Context, hist: &[u8]) -> i32 {
        self.idx = self.slot(ctx, hist);
        self.idx.map_or(0, |idx| self.sm.predict(idx))
    }

    fn update(&mut self, _ctx: &Context, _hist: &[u8], bit: u8) {
        if let Some(idx) = self.idx {
            self.sm.update(idx, bit);
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    /// Abstains in free text (no nearby `<`) and keys on the enclosing element otherwise.
    #[test]
    fn keys_on_enclosing_tag() {
        let m = XmlTagModel::new(1 << 16);
        let ctx = Context::new();
        // No '<' in range => abstain.
        assert_eq!(m.slot(&ctx, b"plain running text with no markup at all here"), None);
        // Inside an <id> body: the '<id>' is a few bytes back.
        assert!(m.slot(&ctx, b"<id>12345").is_some());
        // Same element name => same slot, whether opening or closing tag introduced it.
        assert_eq!(m.slot(&ctx, b"<title>Foo"), m.slot(&ctx, b"</title>Foo"));
        // Different elements land on different slots.
        assert_ne!(m.slot(&ctx, b"<id>9"), m.slot(&ctx, b"<timestamp>9"));
    }

    /// Deterministic (encode/decode must agree).
    #[test]
    fn is_deterministic() {
        let m = XmlTagModel::new(1 << 16);
        let ctx = Context::new();
        assert_eq!(m.slot(&ctx, b"<comment>hi"), m.slot(&ctx, b"<comment>hi"));
    }
}
