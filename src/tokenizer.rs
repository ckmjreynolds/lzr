//! Tokenization helpers for the word-level codec (Phase 13).
//!
//! Splits a Content byte slice into alternating word-class and
//! separator-class runs:
//! - **Word**: a maximal run of `[a-zA-Z]+`. Case is extracted to a
//!   small pattern code so the dictionary stores only the lowercase
//!   form (CDR's case-folding insight — without it the dict
//!   fragments badly: "The"/"the"/"THE" eat three slots).
//! - **Separator**: a maximal run of `[^a-zA-Z]+`. Includes spaces,
//!   punctuation, digits, wiki markup, newlines, multi-byte UTF-8
//!   payload bytes, etc. Stored as-is in the dictionary.
//!
//! The boundary rule (letter vs non-letter) is unambiguous, so the
//! decoder can determine token class from the first byte of each
//! reconstructed run.

/// Token class — determined by whether the first byte of the run is
/// an ASCII letter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TokenClass {
    Word,
    Separator,
}

impl TokenClass {
    pub(crate) const fn from_byte(b: u8) -> Self {
        if b.is_ascii_alphabetic() {
            Self::Word
        } else {
            Self::Separator
        }
    }
}

/// Find the end of the run starting at `pos`. Caller already knows
/// the class from `buf[pos]`. Returns the exclusive end index.
pub(crate) fn run_end(buf: &[u8], pos: usize) -> usize {
    debug_assert!(pos < buf.len());
    let class = TokenClass::from_byte(buf[pos]);
    let mut end = pos + 1;
    while end < buf.len() && TokenClass::from_byte(buf[end]) == class {
        end += 1;
    }
    end
}

/// Case pattern of a word run. Stored as a 2-bit code in the
/// archive; `Mixed` triggers an additional per-letter bitmask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum CasePattern {
    AllLower = 0,
    TitleCase = 1,
    AllUpper = 2,
    Mixed = 3,
}

impl CasePattern {
    pub(crate) const fn idx(self) -> usize {
        self as usize
    }

    pub(crate) const fn from_idx(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::AllLower),
            1 => Some(Self::TitleCase),
            2 => Some(Self::AllUpper),
            3 => Some(Self::Mixed),
            _ => None,
        }
    }
}

/// Classify the case pattern of a word run. The four categories
/// cover the overwhelming majority of English orthography; the
/// `Mixed` category is the escape for "iPhone"/"McDonald"-style
/// proper nouns and is paid for with a per-letter bitmask.
pub(crate) fn classify_case(word: &[u8]) -> CasePattern {
    let any_upper = word.iter().any(u8::is_ascii_uppercase);
    let any_lower = word.iter().any(u8::is_ascii_lowercase);
    if !any_upper {
        return CasePattern::AllLower;
    }
    if !any_lower {
        return CasePattern::AllUpper;
    }
    // Has both upper and lower; check title-case shape.
    if word[0].is_ascii_uppercase() && word[1..].iter().all(u8::is_ascii_lowercase) {
        return CasePattern::TitleCase;
    }
    CasePattern::Mixed
}

/// Lowercase a word run. The resulting bytes form the dictionary
/// key for this token.
pub(crate) fn lowercase(word: &[u8]) -> Vec<u8> {
    word.iter().map(u8::to_ascii_lowercase).collect()
}

/// Reconstruct the original word from its lowercase form and case
/// pattern. `mixed_mask` must be `Some` for `CasePattern::Mixed` and
/// is otherwise unused.
pub(crate) fn apply_case(
    lower: &[u8],
    pattern: CasePattern,
    mixed_mask: Option<&[bool]>,
) -> Vec<u8> {
    match pattern {
        CasePattern::AllLower => lower.to_vec(),
        CasePattern::AllUpper => lower.iter().map(u8::to_ascii_uppercase).collect(),
        CasePattern::TitleCase => {
            let mut v = lower.to_vec();
            if let Some(first) = v.first_mut() {
                *first = first.to_ascii_uppercase();
            }
            v
        }
        CasePattern::Mixed => {
            let mask = mixed_mask.expect("Mixed pattern needs a mask");
            assert_eq!(mask.len(), lower.len(), "mixed mask length mismatch");
            lower
                .iter()
                .zip(mask.iter())
                .map(|(b, &upper)| if upper { b.to_ascii_uppercase() } else { *b })
                .collect()
        }
    }
}

/// Compute the per-letter uppercase bitmask for a `Mixed` word.
/// `mask[i] = true` ⇔ `word[i]` is uppercase.
pub(crate) fn mixed_mask(word: &[u8]) -> Vec<bool> {
    word.iter().map(u8::is_ascii_uppercase).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_end_word() {
        assert_eq!(run_end(b"Hello world", 0), 5);
    }

    #[test]
    fn run_end_separator() {
        assert_eq!(run_end(b" .,!hello", 0), 4);
    }

    #[test]
    fn run_end_to_eof() {
        assert_eq!(run_end(b"abcdef", 0), 6);
    }

    #[test]
    fn case_classification() {
        assert_eq!(classify_case(b"hello"), CasePattern::AllLower);
        assert_eq!(classify_case(b"Hello"), CasePattern::TitleCase);
        assert_eq!(classify_case(b"HELLO"), CasePattern::AllUpper);
        assert_eq!(classify_case(b"iPhone"), CasePattern::Mixed);
        assert_eq!(classify_case(b"X"), CasePattern::AllUpper);
        assert_eq!(classify_case(b"x"), CasePattern::AllLower);
    }

    #[test]
    fn lowercase_idempotent_on_lowercase() {
        assert_eq!(lowercase(b"hello"), b"hello");
    }

    #[test]
    fn lowercase_strips_case() {
        assert_eq!(lowercase(b"Hello"), b"hello");
        assert_eq!(lowercase(b"HELLO"), b"hello");
        assert_eq!(lowercase(b"iPhone"), b"iphone");
    }

    #[test]
    fn case_roundtrip_all_lower() {
        let lower = b"hello";
        assert_eq!(apply_case(lower, CasePattern::AllLower, None), b"hello");
    }

    #[test]
    fn case_roundtrip_title() {
        assert_eq!(apply_case(b"hello", CasePattern::TitleCase, None), b"Hello");
    }

    #[test]
    fn case_roundtrip_upper() {
        assert_eq!(apply_case(b"hello", CasePattern::AllUpper, None), b"HELLO");
    }

    #[test]
    fn case_roundtrip_mixed() {
        let word = b"iPhone";
        let lower = lowercase(word);
        let pattern = classify_case(word);
        assert_eq!(pattern, CasePattern::Mixed);
        let mask = mixed_mask(word);
        assert_eq!(apply_case(&lower, pattern, Some(&mask)), word);
    }

    #[test]
    fn case_roundtrip_mcdonald() {
        let word = b"McDonald";
        let lower = lowercase(word);
        let pattern = classify_case(word);
        assert_eq!(pattern, CasePattern::Mixed);
        let mask = mixed_mask(word);
        assert_eq!(apply_case(&lower, pattern, Some(&mask)), word);
    }
}
