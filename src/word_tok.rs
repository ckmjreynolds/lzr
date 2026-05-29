//! v6 content tokenizer: word / digit / symbol segmentation with
//! capitalization as a side-channel.
//!
//! Operates on the `TextContent` stream (the article body the
//! [`crate::classifier`] isolates). Segmentation is a deterministic
//! function of the bytes — the encoder emits a canonical token
//! sequence and the decoder concatenates each token's bytes back, so
//! there is no Moore/zero-signaling hazard (unlike a byte-stream
//! router).
//!
//! ### Token classes
//! - **word** — a maximal run of ASCII letters whose lowercased form
//!   is in the capped vocabulary, carried as `(id, case)` where `case`
//!   is `Lower`/`Title`/`Upper`. Capitalization is thus a 2-bit
//!   side-channel rather than a vocabulary axis.
//! - **escape word** — a letter run that is out-of-vocab *or* has a
//!   mixed-case pattern the 2-bit channel can't express (e.g.
//!   `iPhone`): spelled literally (`Esc(bytes)`), counted as one
//!   `<esc>` token plus one letter token per byte. This is the
//!   bounded tail-handler, ~5–9% of words.
//! - **digit** — a single ASCII digit (per the v6 spec, digits stay
//!   individual).
//! - **symbol** — any other single byte (punctuation, whitespace,
//!   markup chars, UTF-8 continuation bytes). All 256 byte values are
//!   representable, so roundtrip is total.
//!
//! Non-ASCII letters are not part of the word class here (the escape
//! alphabet stays bounded at the 52 ASCII letters); their bytes fall
//! through to `symbol`, preserving them losslessly.

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Case {
    Lower,
    Title,
    Upper,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Tok {
    Word { id: u32, case: Case },
    Esc(Vec<u8>),
    Digit(u8),
    Sym(u8),
}

/// Per-class token accounting for a content stream.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TokStats {
    pub content_bytes: u64,
    pub words_in_vocab: u64,
    pub esc_words: u64,
    pub esc_letter_tokens: u64,
    pub digits: u64,
    pub symbols: u64,
    /// Single-space symbol tokens — candidates for a "space follows by
    /// default" side-channel that removes them from the token stream.
    pub single_space: u64,
}

impl TokStats {
    /// Total tokens the neural arm would predict: one per in-vocab
    /// word, plus `<esc>` + per-letter tokens for escapes, plus digits
    /// and symbols.
    pub(crate) const fn total_tokens(&self) -> u64 {
        self.words_in_vocab + self.esc_words + self.esc_letter_tokens + self.digits + self.symbols
    }
}

pub(crate) struct WordTok {
    word_to_id: HashMap<Vec<u8>, u32>,
}

impl WordTok {
    /// Build from an ordered list of lowercase words (index = id).
    pub(crate) fn from_words(words: &[Vec<u8>]) -> Self {
        let mut word_to_id = HashMap::with_capacity(words.len());
        for (i, w) in words.iter().enumerate() {
            word_to_id.insert(w.clone(), u32::try_from(i).expect("vocab id fits u32"));
        }
        Self { word_to_id }
    }

    /// Parse a newline-delimited lowercase word file (frequency order).
    pub(crate) fn from_word_file(text: &str) -> Self {
        let words: Vec<Vec<u8>> = text
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| l.as_bytes().to_vec())
            .collect();
        Self::from_words(&words)
    }

    pub(crate) fn vocab_len(&self) -> usize {
        self.word_to_id.len()
    }

    /// Classify a letter run into `(lowercased, case)`, or `None` if
    /// the case pattern isn't `Lower`/`Title`/`Upper` (mixed case →
    /// caller escapes it).
    fn classify_word(word: &[u8]) -> Option<(Vec<u8>, Case)> {
        let lower: Vec<u8> = word.iter().map(u8::to_ascii_lowercase).collect();
        if word.iter().all(u8::is_ascii_lowercase) {
            Some((lower, Case::Lower))
        } else if word.iter().all(u8::is_ascii_uppercase) {
            Some((lower, Case::Upper))
        } else if word[0].is_ascii_uppercase() && word[1..].iter().all(u8::is_ascii_lowercase) {
            Some((lower, Case::Title))
        } else {
            None
        }
    }

    /// Tokenize `content`, invoking `sink` for each token in order.
    pub(crate) fn tokenize_into(&self, content: &[u8], mut sink: impl FnMut(Tok)) {
        let n = content.len();
        let mut i = 0;
        while i < n {
            let b = content[i];
            if b.is_ascii_alphabetic() {
                let start = i;
                while i < n && content[i].is_ascii_alphabetic() {
                    i += 1;
                }
                let word = &content[start..i];
                match Self::classify_word(word) {
                    Some((lower, case)) if self.word_to_id.contains_key(&lower) => {
                        sink(Tok::Word {
                            id: self.word_to_id[&lower],
                            case,
                        });
                    }
                    _ => sink(Tok::Esc(word.to_vec())),
                }
            } else if b.is_ascii_digit() {
                sink(Tok::Digit(b));
                i += 1;
            } else {
                sink(Tok::Sym(b));
                i += 1;
            }
        }
    }

    pub(crate) fn tokenize(&self, content: &[u8]) -> Vec<Tok> {
        let mut out = Vec::new();
        self.tokenize_into(content, |t| out.push(t));
        out
    }

    /// Stream `content` and accumulate per-class token counts without
    /// materializing the token vector.
    pub(crate) fn stats(&self, content: &[u8]) -> TokStats {
        let mut s = TokStats {
            content_bytes: content.len() as u64,
            ..TokStats::default()
        };
        self.tokenize_into(content, |t| match t {
            Tok::Word { .. } => s.words_in_vocab += 1,
            Tok::Esc(bytes) => {
                s.esc_words += 1;
                s.esc_letter_tokens += bytes.len() as u64;
            }
            Tok::Digit(_) => s.digits += 1,
            Tok::Sym(b) => {
                s.symbols += 1;
                if b == b' ' {
                    s.single_space += 1;
                }
            }
        });
        s
    }

    /// Reconstruct the original bytes from a token sequence.
    pub(crate) fn detokenize(&self, toks: &[Tok], id_to_word: &[Vec<u8>]) -> Vec<u8> {
        let mut out = Vec::new();
        for t in toks {
            match t {
                Tok::Word { id, case } => {
                    let w = &id_to_word[*id as usize];
                    match case {
                        Case::Lower => out.extend_from_slice(w),
                        Case::Upper => out.extend(w.iter().map(u8::to_ascii_uppercase)),
                        Case::Title => {
                            out.push(w[0].to_ascii_uppercase());
                            out.extend_from_slice(&w[1..]);
                        }
                    }
                }
                Tok::Esc(bytes) => out.extend_from_slice(bytes),
                Tok::Digit(b) | Tok::Sym(b) => out.push(*b),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> (WordTok, Vec<Vec<u8>>) {
        let words: Vec<Vec<u8>> = ["the", "cat", "dog", "category"]
            .iter()
            .map(|s| s.as_bytes().to_vec())
            .collect();
        (WordTok::from_words(&words), words)
    }

    #[test]
    fn roundtrips_mixed_content() {
        let (tok, id_to_word) = vocab();
        // in-vocab words (Lower/Title/Upper), an OOV word (zzz), a
        // mixed-case word (iPhone-like), digits, markup symbols.
        let content = b"The cat ate 42 dogs!! [[Category|xQz]] THE";
        let toks = tok.tokenize(content);
        let back = tok.detokenize(&toks, &id_to_word);
        assert_eq!(back, content, "roundtrip must be byte-exact");
    }

    #[test]
    fn case_is_a_side_channel() {
        let (tok, _) = vocab();
        let toks = tok.tokenize(b"the The THE");
        let cases: Vec<Case> = toks
            .iter()
            .filter_map(|t| match t {
                Tok::Word { case, .. } => Some(*case),
                _ => None,
            })
            .collect();
        assert_eq!(cases, vec![Case::Lower, Case::Title, Case::Upper]);
    }

    #[test]
    fn mixed_case_and_oov_escape() {
        let (tok, id_to_word) = vocab();
        // "caT" is mixed-case (not Lower/Title/Upper) -> escape.
        // "xyz" is OOV -> escape. Both must roundtrip exactly.
        let content = b"caT xyz";
        let toks = tok.tokenize(content);
        assert!(matches!(toks[0], Tok::Esc(_)), "mixed-case -> escape");
        assert!(matches!(toks[2], Tok::Esc(_)), "OOV -> escape");
        assert_eq!(tok.detokenize(&toks, &id_to_word), content);
    }

    #[test]
    fn digits_are_individual() {
        let (tok, _) = vocab();
        let toks = tok.tokenize(b"1987");
        assert_eq!(toks.len(), 4);
        assert!(toks.iter().all(|t| matches!(t, Tok::Digit(_))));
    }

    #[test]
    fn stats_accounting_matches_token_stream() {
        let (tok, _) = vocab();
        let content = b"The cat 4 !! xyz";
        let s = tok.stats(content);
        let toks = tok.tokenize(content);
        // total_tokens counts <esc>+letters for escapes; the
        // materialized Vec has one entry per Esc, so recompute.
        let materialized: u64 = toks
            .iter()
            .map(|t| match t {
                Tok::Esc(b) => 1 + b.len() as u64,
                _ => 1,
            })
            .sum();
        assert_eq!(s.total_tokens(), materialized);
    }
}
