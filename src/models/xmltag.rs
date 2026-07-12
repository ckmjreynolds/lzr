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

use super::statemap::SlotMap;
use super::{Context, Model, hashed_bits, hashed_slot};

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
    /// The adaptive probability map, keyed on the (hashed) enclosing tag name; abstains off any tag.
    map: SlotMap,
}

impl XmlTagModel {
    /// A fresh model. `capacity` (the framed byte count) sizes the hashed table via [`hashed_bits`] so
    /// encode and decode agree.
    pub(crate) fn new(capacity: usize) -> Self {
        let bits = hashed_bits(capacity);
        Self {
            shift: u64::BITS - bits,
            map: SlotMap::new(1 << bits),
        }
    }

    /// The map slot for the current bit, or `None` to abstain: the nearest `<`-introduced tag name
    /// within [`MAX_BACK`] bytes, folded with the bit-tree node. Abstains when no `<` is in range or the
    /// tag has no name (e.g. `</` at the window edge).
    fn slot(&self, ctx: &Context, hist: &[u8]) -> Option<usize> {
        let window = &hist[hist.len().saturating_sub(MAX_BACK)..];
        let lt = window.iter().rposition(|&b| b == b'<')?;
        // Skip the '<' and a possible '/' of a closing tag, then read the element-name bytes.
        let mut i = lt + 1;
        if window.get(i) == Some(&b'/') {
            i += 1;
        }
        let name = window[i..].iter().copied().take_while(u8::is_ascii_lowercase).take(MAX_NAME);
        hashed_slot(ctx.node(), name, self.shift)
    }
}

impl Model for XmlTagModel {
    fn predict(&mut self, ctx: &Context, hist: &[u8]) -> i32 {
        let slot = self.slot(ctx, hist);
        self.map.predict(slot)
    }

    fn update(&mut self, _ctx: &Context, _hist: &[u8], bit: u8) {
        self.map.update(bit);
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
