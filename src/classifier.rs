//! Streaming, deterministic mode classifier for enwik9-style XML.
//!
//! At every byte position the classifier reports the *current mode*
//! before the byte is consumed; encoder and decoder run the same FSM
//! and so agree on the mode of every byte without any signaling.
//!
//! Three modes for Phase 1:
//! - `Content` — `PCData` (text between tags), plus the `<` that opens
//!   the next tag (the `<` is decoded in `InContent` state because
//!   the state transition fires *after* the byte; see Moore-FSM note
//!   below).
//! - `TagStructure` — bytes inside `<...>` including the closing `>`,
//!   plus the opening `"` of an attribute value (same Moore-FSM
//!   reason).
//! - `AttrValue` — bytes inside an attribute `"..."`, plus the
//!   closing `"` (Moore-FSM, again).
//!
//! ### Moore-FSM emission, not Mealy
//!
//! The decoder doesn't have the next byte yet — it's about to decode
//! it. So the mode it picks the codec by must depend on the state
//! *before* the byte, not on the byte itself. We use a Moore machine:
//! the state determines the mode, the byte determines the *next* state.
//! Trigger bytes (`<`, `>`, `"`) are therefore classified in the
//! previous mode rather than the one they open, which is an asymmetry
//! the per-mode codecs absorb naturally.
//!
//! Sub-modes inside `Content` (prose vs wiki markup vs numeric vs
//! URL) are left to the prose codec in Phase 3.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub(crate) enum Mode {
    Content = 0,
    TagStructure = 1,
    AttrValue = 2,
}

impl Mode {
    pub(crate) const fn component_name(self) -> &'static str {
        match self {
            Self::Content => "content",
            Self::TagStructure => "tag",
            Self::AttrValue => "attr",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Content,
    Tag,
    AttrValue,
}

/// Streaming XML mode classifier. Cheap to copy (one byte).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Classifier {
    state: State,
}

impl Classifier {
    /// Construct a fresh classifier in the content state (the natural
    /// starting state for `enwik9`, which opens with `<mediawiki>`).
    pub(crate) const fn new() -> Self {
        Self {
            state: State::Content,
        }
    }

    /// Mode of the byte about to be consumed.
    pub(crate) const fn current_mode(self) -> Mode {
        match self.state {
            State::Content => Mode::Content,
            State::Tag => Mode::TagStructure,
            State::AttrValue => Mode::AttrValue,
        }
    }

    /// Consume `byte` and transition to the state for the next byte.
    pub(crate) const fn advance(&mut self, byte: u8) {
        self.state = match (self.state, byte) {
            (State::Content, b'<') | (State::AttrValue, b'"') => State::Tag,
            (State::Tag, b'>') => State::Content,
            (State::Tag, b'"') => State::AttrValue,
            (s, _) => s,
        };
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

    /// Run the classifier over `input` and return the sequence of
    /// per-byte modes.
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
        let input = b"<page>text</page>";
        let modes = classify(input);
        let expected = [
            Mode::Content,      // '<'   (state InContent → next InTag)
            Mode::TagStructure, // 'p'
            Mode::TagStructure, // 'a'
            Mode::TagStructure, // 'g'
            Mode::TagStructure, // 'e'
            Mode::TagStructure, // '>'   (state InTag → next InContent)
            Mode::Content,      // 't'
            Mode::Content,      // 'e'
            Mode::Content,      // 'x'
            Mode::Content,      // 't'
            Mode::Content,      // '<'   (state InContent → next InTag)
            Mode::TagStructure, // '/'
            Mode::TagStructure, // 'p'
            Mode::TagStructure, // 'a'
            Mode::TagStructure, // 'g'
            Mode::TagStructure, // 'e'
            Mode::TagStructure, // '>'   (state InTag → next InContent)
        ];
        assert_eq!(modes, expected);
    }

    #[test]
    fn attribute_value_is_separated() {
        // <a foo="bar">x</a>
        let input = b"<a foo=\"bar\">x</a>";
        let modes = classify(input);
        let expected = [
            Mode::Content,      // '<'
            Mode::TagStructure, // 'a'
            Mode::TagStructure, // ' '
            Mode::TagStructure, // 'f'
            Mode::TagStructure, // 'o'
            Mode::TagStructure, // 'o'
            Mode::TagStructure, // '='
            Mode::TagStructure, // '"'  (state InTag → next InAttrValue)
            Mode::AttrValue,    // 'b'
            Mode::AttrValue,    // 'a'
            Mode::AttrValue,    // 'r'
            Mode::AttrValue,    // '"'  (state InAttrValue → next InTag)
            Mode::TagStructure, // '>'  (state InTag → next InContent)
            Mode::Content,      // 'x'
            Mode::Content,      // '<'
            Mode::TagStructure, // '/'
            Mode::TagStructure, // 'a'
            Mode::TagStructure, // '>'
        ];
        assert_eq!(modes, expected);
    }

    #[test]
    fn classifier_handles_multiple_attributes() {
        let input = b"<x a=\"1\" b=\"22\">y</x>";
        //         idx 0123456789...
        let modes = classify(input);
        let attr_positions: Vec<usize> = modes
            .iter()
            .enumerate()
            .filter_map(|(i, m)| (*m == Mode::AttrValue).then_some(i))
            .collect();
        // First '"' at byte 5 is decoded in `Tag` state → TagStructure.
        // Value '1' (6) and closing '"' (7) decode in `AttrValue`.
        // Second opening '"' at byte 11 is `Tag` again. Then '2' (12),
        // '2' (13), closing '"' (14) are `AttrValue`.
        assert_eq!(attr_positions, vec![6, 7, 12, 13, 14]);
    }

    #[test]
    fn brackets_inside_attr_value_dont_escape_value_mode() {
        // The classifier doesn't track entities or escaping; any byte
        // other than '"' inside an attribute value stays in AttrValue
        // mode. This mirrors how a real attribute value can contain
        // `<` and `>` as escaped entities (`&lt;`, `&gt;`) — they
        // appear as plain characters in the byte stream.
        let input = b"<x a=\"<>\">y</x>";
        let modes = classify(input);
        let attr_count = modes.iter().filter(|m| **m == Mode::AttrValue).count();
        // 3 bytes of value content + closing quote = 4 bytes
        // attributed to AttrValue ('<', '>', and the closing '"').
        assert_eq!(attr_count, 3);
    }

    #[test]
    fn classifier_state_diverges_only_on_brackets_and_quotes() {
        // Bytes other than '<', '>', '"' don't change state. Verify
        // by walking a synthetic stream of non-trigger bytes.
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
