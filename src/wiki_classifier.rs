//! Wiki sub-mode classifier — runs inside Content alongside the XML
//! `Classifier`. Moore-style FSM updated after each committed Content
//! byte; both encoder and decoder maintain one and so agree on the
//! sub-mode of every byte without signaling, mirroring how the XML
//! classifier carries zero overhead.
//!
//! Three sub-modes (chosen by the recon in [`analyze_wiki`]; headings
//! were dropped because they're 0.3% of Content — not worth the
//! line-start tracking):
//! - `Plain` — Content outside any wiki structure (~19% of Content).
//! - `Link` — between `[[` and the matching `]]` (~76% of Content).
//!   Includes image links with thumb captions; the inside alphabet
//!   is constrained (mostly letters + a small set of separators).
//! - `Template` — between `{{` and matching `}}` (~4.5%). Templates
//!   nest heavily, so depth-tracked.
//!
//! ## No-lookahead FSM
//!
//! Decode can't see byte i+1 when it advances the FSM after byte i
//! — byte i+1 hasn't been decoded yet, and the FSM is the *input*
//! to that decode. So we keep a one-byte memory `prev` and detect
//! paired tokens (`[[`, `]]`, `{{`, `}}`) when the second byte
//! arrives. Depth changes apply AFTER the second byte is consumed,
//! which means the closing `]]` itself is still in Link mode and
//! the byte immediately after it switches to Plain. This is an
//! asymmetry the per-sub-mode literal models absorb naturally.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum WikiSub {
    Plain = 0,
    Link = 1,
    Template = 2,
}

impl WikiSub {
    pub(crate) const fn idx(self) -> usize {
        self as usize
    }
}

pub(crate) const N_WIKI_SUB: usize = 3;

/// Phase-22: 5-mode fine wiki sub-classifier. Splits `Link` into
/// `LinkTarget` (before the `|`) and `LinkDisplay` (after the `|`);
/// splits `Template` into `TemplateName` and `TemplateArg`. The
/// recon (2026-05-13) showed these four sub-modes have sharply
/// different distributions: `LinkTarget` is Zipf-skewed page names,
/// `LinkDisplay` is mostly prose, `TemplateName` is a small set of
/// canonical names, `TemplateArg` is mixed structured / prose data.
///
/// The `|` flag is global per type rather than per nesting level,
/// so deeply nested templates with `|` only in the inner one get a
/// few outer bytes misclassified as `TemplateArg`. Cheap to ignore.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum WikiFine {
    Plain = 0,
    LinkTarget = 1,
    LinkDisplay = 2,
    TemplateName = 3,
    TemplateArg = 4,
}

impl WikiFine {
    pub(crate) const fn idx(self) -> usize {
        self as usize
    }
}

#[allow(dead_code)]
pub(crate) const N_WIKI_FINE: usize = 5;

/// Mirror of [`WikiClassifier`] with `|`-flag tracking to split
/// Link into `LinkTarget` / `LinkDisplay` and Template into
/// `TemplateName` / `TemplateArg`. Same Moore-FSM construction:
/// both encoder and decoder run it deterministically so no
/// signaling bits are needed.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WikiFineClassifier {
    link_depth: u16,
    template_depth: u16,
    seen_pipe_in_link: bool,
    seen_pipe_in_template: bool,
    prev: Option<u8>,
}

impl WikiFineClassifier {
    pub(crate) const fn new() -> Self {
        Self {
            link_depth: 0,
            template_depth: 0,
            seen_pipe_in_link: false,
            seen_pipe_in_template: false,
            prev: None,
        }
    }

    pub(crate) const fn current_sub(self) -> WikiFine {
        if self.link_depth > 0 {
            if self.seen_pipe_in_link {
                WikiFine::LinkDisplay
            } else {
                WikiFine::LinkTarget
            }
        } else if self.template_depth > 0 {
            if self.seen_pipe_in_template {
                WikiFine::TemplateArg
            } else {
                WikiFine::TemplateName
            }
        } else {
            WikiFine::Plain
        }
    }

    /// Phase-23Q: total nesting depth (links + templates), saturating
    /// to `u16::MAX`. Exposed so downstream predictors can distinguish
    /// "depth 1 link inside Plain" from "depth 2 template inside a
    /// link" — the [`Self::current_sub`] enum only reveals the
    /// innermost mode, not how deep we are.
    pub(crate) const fn depth_total(self) -> u16 {
        self.link_depth.saturating_add(self.template_depth)
    }

    pub(crate) const fn advance(&mut self, byte: u8) {
        let pair_open_link = matches!(self.prev, Some(b'[')) && byte == b'[';
        let pair_close_link = matches!(self.prev, Some(b']')) && byte == b']';
        let pair_open_tpl = matches!(self.prev, Some(b'{')) && byte == b'{';
        let pair_close_tpl = matches!(self.prev, Some(b'}')) && byte == b'}';

        if pair_open_link {
            self.link_depth = self.link_depth.saturating_add(1);
            if self.link_depth == 1 {
                self.seen_pipe_in_link = false;
            }
            self.prev = None;
        } else if pair_close_link && self.link_depth > 0 {
            self.link_depth -= 1;
            if self.link_depth == 0 {
                self.seen_pipe_in_link = false;
            }
            self.prev = None;
        } else if pair_open_tpl {
            self.template_depth = self.template_depth.saturating_add(1);
            if self.template_depth == 1 {
                self.seen_pipe_in_template = false;
            }
            self.prev = None;
        } else if pair_close_tpl && self.template_depth > 0 {
            self.template_depth -= 1;
            if self.template_depth == 0 {
                self.seen_pipe_in_template = false;
            }
            self.prev = None;
        } else if byte == b'|' {
            if self.link_depth > 0 {
                self.seen_pipe_in_link = true;
            } else if self.template_depth > 0 {
                self.seen_pipe_in_template = true;
            }
            self.prev = None;
        } else if matches!(byte, b'[' | b']' | b'{' | b'}') {
            self.prev = Some(byte);
        } else {
            self.prev = None;
        }
    }
}

impl Default for WikiFineClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// One-byte memory for paired-token detection. Reset to `None` after
/// a pair fires so `[[[` doesn't double-count.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WikiClassifier {
    link_depth: u16,
    template_depth: u16,
    prev: Option<u8>,
}

impl WikiClassifier {
    pub(crate) const fn new() -> Self {
        Self {
            link_depth: 0,
            template_depth: 0,
            prev: None,
        }
    }

    /// Sub-mode that applies to the *next* byte given current depth.
    /// Link takes priority over Template (matches the recon
    /// categorization).
    pub(crate) const fn current_sub(self) -> WikiSub {
        if self.link_depth > 0 {
            WikiSub::Link
        } else if self.template_depth > 0 {
            WikiSub::Template
        } else {
            WikiSub::Plain
        }
    }

    /// Consume one committed Content byte. State transitions fire on
    /// the *second* byte of a pair (`[[`, `]]`, `{{`, `}}`) and reset
    /// `prev` so a triple like `[[[` opens exactly once.
    pub(crate) const fn advance(&mut self, byte: u8) {
        let pair_open_link = matches!(self.prev, Some(b'[')) && byte == b'[';
        let pair_close_link = matches!(self.prev, Some(b']')) && byte == b']';
        let pair_open_tpl = matches!(self.prev, Some(b'{')) && byte == b'{';
        let pair_close_tpl = matches!(self.prev, Some(b'}')) && byte == b'}';

        if pair_open_link {
            self.link_depth = self.link_depth.saturating_add(1);
            self.prev = None;
        } else if pair_close_link && self.link_depth > 0 {
            self.link_depth -= 1;
            self.prev = None;
        } else if pair_open_tpl {
            self.template_depth = self.template_depth.saturating_add(1);
            self.prev = None;
        } else if pair_close_tpl && self.template_depth > 0 {
            self.template_depth -= 1;
            self.prev = None;
        } else if matches!(byte, b'[' | b']' | b'{' | b'}') {
            self.prev = Some(byte);
        } else {
            self.prev = None;
        }
    }
}

impl Default for WikiClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run the classifier over `bytes`; return the sub-mode of each.
    fn classify(bytes: &[u8]) -> Vec<WikiSub> {
        let mut c = WikiClassifier::new();
        let mut out = Vec::with_capacity(bytes.len());
        for &b in bytes {
            out.push(c.current_sub());
            c.advance(b);
        }
        out
    }

    #[test]
    fn plain_text_stays_plain() {
        let v = classify(b"hello world");
        assert!(v.iter().all(|s| *s == WikiSub::Plain));
    }

    #[test]
    fn link_categorizes_inside_bytes() {
        // "[[Foo]]" — depth flips on the second '[' and second ']',
        // so:
        //   '['  Plain (depth was 0; prev becomes '[')
        //   '['  Plain (advance fires open: depth → 1)
        //   'F'  Link  (depth is 1)
        //   'o'  Link
        //   'o'  Link
        //   ']'  Link  (depth still 1; prev becomes ']')
        //   ']'  Link  (advance fires close: depth → 0)
        let v = classify(b"[[Foo]]");
        assert_eq!(
            v,
            vec![
                WikiSub::Plain,
                WikiSub::Plain,
                WikiSub::Link,
                WikiSub::Link,
                WikiSub::Link,
                WikiSub::Link,
                WikiSub::Link,
            ],
        );
    }

    #[test]
    fn nested_link_keeps_link_mode_throughout() {
        let v = classify(b"[[Outer|[[Inner]] alt]]");
        // Bytes 2..end are all inside at least one link.
        for (i, s) in v.iter().enumerate().skip(2) {
            assert_eq!(*s, WikiSub::Link, "byte {i} ({:?})", v[i]);
        }
    }

    #[test]
    fn template_inside_bytes() {
        let v = classify(b"{{infobox}}");
        // Inside-template bytes are at positions 2..=10 (skipping the
        // first `{{` opener).
        for (i, s) in v.iter().enumerate().take(11).skip(2) {
            assert_eq!(*s, WikiSub::Template, "byte {i}");
        }
    }

    #[test]
    fn link_inside_template_takes_link_priority() {
        let s = b"{{infobox|param=[[Foo]]}}";
        let v = classify(s);
        let f_idx = b"{{infobox|param=[[".len();
        assert_eq!(v[f_idx], WikiSub::Link);
        assert_eq!(v[f_idx + 1], WikiSub::Link);
        assert_eq!(v[f_idx + 2], WikiSub::Link);
    }

    #[test]
    fn triple_open_only_counts_once() {
        // "[[[" opens once: prev becomes '[' after byte 0, fires
        // open on byte 1 (depth=1, prev reset). Byte 2 is a lone '['
        // with prev=None — prev becomes '[' again but never fires.
        let mut c = WikiClassifier::new();
        for &b in b"[[[" {
            c.advance(b);
        }
        // Depth stayed at 1.
        c.advance(b'X');
        assert_eq!(c.current_sub(), WikiSub::Link);
    }

    #[test]
    fn unbalanced_close_doesnt_underflow() {
        let mut c = WikiClassifier::new();
        for &b in b"]]" {
            c.advance(b);
        }
        assert_eq!(c.current_sub(), WikiSub::Plain);
    }
}
