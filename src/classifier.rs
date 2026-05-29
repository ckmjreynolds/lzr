//! Streaming, deterministic mode classifier for enwik9-style `MediaWiki`
//! XML. At every byte position it reports the *current mode* before
//! the byte is consumed; encoder and decoder run the same FSM and so
//! agree on the mode of every byte without any signaling.
//!
//! ### Modes
//! - `Content` — `PCData` *outside* an article body: the page-metadata
//!   text (titles, ids, timestamps, usernames, comments), plus the
//!   `<` that opens the next tag.
//! - `TextContent` — `PCData` *inside* `<text ...>...</text>`: the
//!   article wikitext, which is the stream the v6 neural arm predicts.
//! - `TagStructure` — bytes inside `<...>` including the closing `>`,
//!   plus the opening `"` of an attribute value.
//! - `AttrValue` — bytes inside an attribute `"..."`, plus the
//!   closing `"`.
//!
//! Splitting `TextContent` from `Content` is the v6 structural lever:
//! the article body goes to the neural arm; the XML skeleton
//! (`TagStructure` + `AttrValue` + metadata `Content`) is near-free to
//! code deterministically (~0.05 bpb under LZMA on enwik9).
//!
//! ### Moore-FSM emission, not Mealy
//!
//! The decoder doesn't have the next byte yet — it's about to decode
//! it. So the mode it picks the codec by must depend on the state
//! *before* the byte, not on the byte itself. We use a Moore machine:
//! the state determines the mode, the byte determines the *next*
//! state. Trigger bytes (`<`, `>`, `"`) are therefore classified in
//! the previous mode rather than the one they open, an asymmetry the
//! per-mode codecs absorb naturally.
//!
//! ### Why `<text>` tracking is sound on valid XML
//!
//! The dump is well-formed XML, so any `<` inside article wikitext is
//! escaped (`&lt;`); the only literal `<` within `<text>` `PCData` is
//! the `</text>` that closes it. The closing content state is thus
//! determined entirely by the tag name: an opening, non-self-closing
//! `<text ...>` enters `TextContent`; every other tag (including
//! `</text>`) returns to `Content`. Confirmed empirically — enwik9's
//! top content "words" include `quot`/`lt`/`gt`/`amp`, i.e. `<`, `>`,
//! `"`, `&` are entity-encoded inside text.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub(crate) enum Mode {
    Content = 0,
    TextContent = 1,
    TagStructure = 2,
    AttrValue = 3,
}

impl Mode {
    pub(crate) const fn component_name(self) -> &'static str {
        match self {
            Self::Content => "content",
            Self::TextContent => "text",
            Self::TagStructure => "tag",
            Self::AttrValue => "attr",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Content,
    TextContent,
    Tag,
    AttrValue,
}

/// Number of leading tag-name bytes retained — enough to match `text`
/// plus one delimiter and reject names that merely start with `text`.
const NAME_LEN: usize = 5;

/// Streaming MediaWiki-XML mode classifier. Cheap to copy.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Classifier {
    state: State,
    /// Leading bytes of the tag currently being scanned (valid only in
    /// `Tag`/`AttrValue`); reset on each `<`.
    name: [u8; NAME_LEN],
    name_len: u8,
    /// Previous byte consumed — used to detect a self-closing `/>`.
    prev: u8,
}

impl Classifier {
    /// Construct a fresh classifier in the content state (the natural
    /// starting state for `enwik9`, which opens with `<mediawiki>`).
    pub(crate) const fn new() -> Self {
        Self {
            state: State::Content,
            name: [0; NAME_LEN],
            name_len: 0,
            prev: 0,
        }
    }

    /// Mode of the byte about to be consumed.
    pub(crate) const fn current_mode(self) -> Mode {
        match self.state {
            State::Content => Mode::Content,
            State::TextContent => Mode::TextContent,
            State::Tag => Mode::TagStructure,
            State::AttrValue => Mode::AttrValue,
        }
    }

    /// Consume `byte` and transition to the state for the next byte.
    pub(crate) fn advance(&mut self, byte: u8) {
        match self.state {
            State::Content | State::TextContent => {
                if byte == b'<' {
                    self.state = State::Tag;
                    self.name_len = 0;
                }
            }
            State::Tag => match byte {
                b'"' => self.state = State::AttrValue,
                b'>' => {
                    let opening = self.name_len > 0 && self.name[0] != b'/';
                    let is_text = self.name_len >= 4
                        && &self.name[0..4] == b"text"
                        && (self.name_len == 4
                            || matches!(self.name[4], b' ' | b'>' | b'/' | b'\t' | b'\n'));
                    let self_closing = self.prev == b'/';
                    self.state = if opening && is_text && !self_closing {
                        State::TextContent
                    } else {
                        State::Content
                    };
                }
                _ => {
                    if (self.name_len as usize) < NAME_LEN {
                        self.name[self.name_len as usize] = byte;
                        self.name_len += 1;
                    }
                }
            },
            State::AttrValue => {
                if byte == b'"' {
                    self.state = State::Tag;
                }
            }
        }
        self.prev = byte;
    }
}

impl Default for Classifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run the classifier over `input` and return the per-byte modes.
    fn classify(input: &[u8]) -> Vec<Mode> {
        let mut c = Classifier::new();
        let mut out = Vec::with_capacity(input.len());
        for &b in input {
            out.push(c.current_mode());
            c.advance(b);
        }
        out
    }

    #[test]
    fn simple_open_close_tag() {
        // PCData inside a non-text tag is plain `Content`, not text.
        let input = b"<page>meta</page>";
        let modes = classify(input);
        let expected = [
            Mode::Content,      // '<'
            Mode::TagStructure, // 'p'
            Mode::TagStructure, // 'a'
            Mode::TagStructure, // 'g'
            Mode::TagStructure, // 'e'
            Mode::TagStructure, // '>'
            Mode::Content,      // 'm'
            Mode::Content,      // 'e'
            Mode::Content,      // 't'
            Mode::Content,      // 'a'
            Mode::Content,      // '<'
            Mode::TagStructure, // '/'
            Mode::TagStructure, // 'p'
            Mode::TagStructure, // 'a'
            Mode::TagStructure, // 'g'
            Mode::TagStructure, // 'e'
            Mode::TagStructure, // '>'
        ];
        assert_eq!(modes, expected);
    }

    #[test]
    fn attribute_value_is_separated() {
        let input = b"<a foo=\"bar\">x</a>";
        let modes = classify(input);
        let attr: Vec<usize> = modes
            .iter()
            .enumerate()
            .filter_map(|(i, m)| (*m == Mode::AttrValue).then_some(i))
            .collect();
        // 'b'(8) 'a'(9) 'r'(10) and closing '"'(11) are AttrValue.
        assert_eq!(attr, vec![8, 9, 10, 11]);
        // 'x' after the tag is plain Content (not inside <text>).
        assert_eq!(modes[13], Mode::Content);
    }

    #[test]
    fn text_body_is_its_own_mode() {
        let input = b"<text>hi</text>";
        let modes = classify(input);
        let expected = [
            Mode::Content,      // '<'
            Mode::TagStructure, // 't'
            Mode::TagStructure, // 'e'
            Mode::TagStructure, // 'x'
            Mode::TagStructure, // 't'
            Mode::TagStructure, // '>'
            Mode::TextContent,  // 'h'
            Mode::TextContent,  // 'i'
            Mode::TextContent,  // '<'  (closing </text> begins)
            Mode::TagStructure, // '/'
            Mode::TagStructure, // 't'
            Mode::TagStructure, // 'e'
            Mode::TagStructure, // 'x'
            Mode::TagStructure, // 't'
            Mode::TagStructure, // '>'
        ];
        assert_eq!(modes, expected);
    }

    #[test]
    fn text_tag_with_attributes_enters_text() {
        let input = b"<text xml:space=\"preserve\">body</text>";
        let modes = classify(input);
        let body_start = input.iter().position(|&b| b == b'>').unwrap() + 1;
        for m in &modes[body_start..body_start + 4] {
            assert_eq!(
                *m,
                Mode::TextContent,
                "text-body byte should be TextContent"
            );
        }
    }

    #[test]
    fn self_closing_text_tag_does_not_enter_text() {
        // Empty article: <text .../> must NOT open TextContent.
        let input = b"<text foo=\"1\"/><x>m</x>";
        let modes = classify(input);
        assert!(
            !modes.contains(&Mode::TextContent),
            "self-closing <text/> must not produce TextContent"
        );
    }

    #[test]
    fn tag_named_textfoo_is_not_text() {
        // A tag whose name merely starts with "text" is not <text>.
        let input = b"<textfoo>z</textfoo>";
        let modes = classify(input);
        assert!(
            !modes.contains(&Mode::TextContent),
            "<textfoo> must not be treated as <text>"
        );
    }

    #[test]
    fn non_trigger_bytes_never_change_state() {
        let mut c = Classifier::new();
        for b in 0u8..=255u8 {
            if matches!(b, b'<' | b'>' | b'"') {
                continue;
            }
            let before = c.current_mode();
            c.advance(b);
            assert_eq!(c.current_mode(), before, "non-trigger byte {b:#x}");
        }
    }
}
