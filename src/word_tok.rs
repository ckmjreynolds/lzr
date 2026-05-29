//! v6 content tokenizer: word / digit / symbol-run segmentation with
//! capitalization and lone-space as side-channels.
//!
//! Operates on the `TextContent` stream (the article body the
//! [`crate::classifier`] isolates). Segmentation is a deterministic
//! function of the bytes — the encoder emits a canonical token
//! sequence and the decoder concatenates each token's bytes back, so
//! there is no Moore/zero-signaling hazard.
//!
//! ### Token classes
//! - **word** — a maximal run of ASCII letters whose lowercased form
//!   is in the capped vocabulary, carried as `(id, case)` where `case`
//!   is `Lower`/`Title`/`Upper`. Capitalization is a 2-bit
//!   side-channel, not a vocabulary axis.
//! - **escape word** — an out-of-vocab or mixed-case letter run,
//!   spelled literally (`Esc(bytes)`); the bounded tail-handler.
//! - **digit** — a single ASCII digit (v6 keeps digits individual).
//! - **symbol run** — a maximal run of non-alphanumeric bytes,
//!   segmented greedily (longest-match) against a symbol-run
//!   vocabulary so frequent markup like `[[`, `]]`, `]] ` collapse to
//!   one token; any single byte is representable, so roundtrip is
//!   total.
//!
//! ### Lone-space side-channel
//! Every word/digit/escape token carries a `space_after` bit: a single
//! `' '` immediately following the token is consumed by the bit rather
//! than emitted as a token. This removes the ~120M lone inter-word
//! spaces (≈26% of raw tokens) from the stream at the cost of one
//! highly-predictable bit per token. Symbol-run tokens never set it
//! (a maximal non-alnum run is always followed by an alphanumeric
//! byte), so spacing inside/after markup stays in the runs.

use std::collections::{HashMap, HashSet};

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
    Sym(Vec<u8>),
}

/// A token plus its lone-space side-channel bit.
pub(crate) type Unit = (Tok, bool);

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TokStats {
    pub words_in_vocab: u64,
    pub esc_words: u64,
    pub esc_letter_tokens: u64,
    pub digits: u64,
    pub sym_tokens: u64,
    /// Tokens whose `space_after` bit was set (lone spaces removed).
    pub space_after: u64,
}

impl TokStats {
    /// Tokens the neural arm predicts: in-vocab words, `<esc>` +
    /// per-letter for escapes, individual digits, and symbol-run
    /// tokens. The `space_after` bits are a side-channel, not tokens.
    pub(crate) const fn total_tokens(&self) -> u64 {
        self.words_in_vocab
            + self.esc_words
            + self.esc_letter_tokens
            + self.digits
            + self.sym_tokens
    }
}

pub(crate) struct WordTok {
    word_to_id: HashMap<Vec<u8>, u32>,
    symruns: HashSet<Vec<u8>>,
    symrun_maxlen: usize,
}

impl WordTok {
    /// Build from an ordered lowercase word list (index = id). No
    /// symbol-run merging (single-byte symbols only).
    pub(crate) fn from_words(words: &[Vec<u8>]) -> Self {
        let mut word_to_id = HashMap::with_capacity(words.len());
        for (i, w) in words.iter().enumerate() {
            word_to_id.insert(w.clone(), u32::try_from(i).expect("vocab id fits u32"));
        }
        Self {
            word_to_id,
            symruns: HashSet::new(),
            symrun_maxlen: 1,
        }
    }

    /// Attach a symbol-run vocabulary (multi-byte runs); single bytes
    /// remain the always-available fallback.
    #[must_use]
    pub(crate) fn with_symruns(mut self, runs: Vec<Vec<u8>>) -> Self {
        self.symrun_maxlen = runs.iter().map(Vec::len).max().unwrap_or(1).max(1);
        self.symruns = runs.into_iter().collect();
        self
    }

    pub(crate) fn vocab_len(&self) -> usize {
        self.word_to_id.len()
    }

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

    /// Greedily segment a maximal non-alphanumeric run into the fewest
    /// symbol tokens via longest-match against the run vocabulary.
    fn segment_symrun(&self, run: &[u8], mut emit: impl FnMut(Tok)) {
        let mut p = 0;
        while p < run.len() {
            let maxl = self.symrun_maxlen.min(run.len() - p);
            let mut best = 1;
            let mut l = maxl;
            while l >= 2 {
                if self.symruns.contains(&run[p..p + l]) {
                    best = l;
                    break;
                }
                l -= 1;
            }
            emit(Tok::Sym(run[p..p + best].to_vec()));
            p += best;
        }
    }

    /// Tokenize `content`, invoking `sink(tok, space_after)` per token.
    pub(crate) fn tokenize_into(&self, content: &[u8], mut sink: impl FnMut(Tok, bool)) {
        let n = content.len();
        let mut i = 0;
        // Consume a single trailing space as the side-channel bit.
        let take_space = |i: &mut usize| -> bool {
            if *i < n && content[*i] == b' ' {
                *i += 1;
                true
            } else {
                false
            }
        };
        while i < n {
            let b = content[i];
            if b.is_ascii_alphabetic() {
                let start = i;
                while i < n && content[i].is_ascii_alphabetic() {
                    i += 1;
                }
                let word = &content[start..i];
                let tok = match Self::classify_word(word) {
                    Some((lower, case)) if self.word_to_id.contains_key(&lower) => Tok::Word {
                        id: self.word_to_id[&lower],
                        case,
                    },
                    _ => Tok::Esc(word.to_vec()),
                };
                let sa = take_space(&mut i);
                sink(tok, sa);
            } else if b.is_ascii_digit() {
                i += 1;
                let sa = take_space(&mut i);
                sink(Tok::Digit(b), sa);
            } else {
                let start = i;
                while i < n && !content[i].is_ascii_alphanumeric() {
                    i += 1;
                }
                // Symbol-run tokens never carry space_after: a maximal
                // non-alnum run is followed by an alphanumeric byte.
                self.segment_symrun(&content[start..i], |t| sink(t, false));
            }
        }
    }

    pub(crate) fn tokenize(&self, content: &[u8]) -> Vec<Unit> {
        let mut out = Vec::new();
        self.tokenize_into(content, |t, sa| out.push((t, sa)));
        out
    }

    pub(crate) fn stats(&self, content: &[u8]) -> TokStats {
        let mut s = TokStats::default();
        self.tokenize_into(content, |t, sa| {
            match t {
                Tok::Word { .. } => s.words_in_vocab += 1,
                Tok::Esc(bytes) => {
                    s.esc_words += 1;
                    s.esc_letter_tokens += bytes.len() as u64;
                }
                Tok::Digit(_) => s.digits += 1,
                Tok::Sym(_) => s.sym_tokens += 1,
            }
            if sa {
                s.space_after += 1;
            }
        });
        s
    }

    /// Reconstruct the original bytes from a token+side-channel stream.
    pub(crate) fn detokenize(units: &[Unit], id_to_word: &[Vec<u8>]) -> Vec<u8> {
        let mut out = Vec::new();
        for (t, sa) in units {
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
                Tok::Esc(bytes) | Tok::Sym(bytes) => out.extend_from_slice(bytes),
                Tok::Digit(b) => out.push(*b),
            }
            if *sa {
                out.push(b' ');
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
        let runs: Vec<Vec<u8>> = ["[[", "]]", "]] ", ", "]
            .iter()
            .map(|s| s.as_bytes().to_vec())
            .collect();
        let tok = WordTok::from_words(&words).with_symruns(runs);
        (tok, words)
    }

    #[test]
    fn roundtrips_mixed_content() {
        let (tok, id_to_word) = vocab();
        let content = b"The cat ate 42 dogs!! [[Category|xQz]] THE\ndog, cat";
        let units = tok.tokenize(content);
        assert_eq!(WordTok::detokenize(&units, &id_to_word), content);
    }

    #[test]
    fn lone_space_becomes_side_channel() {
        let (tok, _) = vocab();
        let units = tok.tokenize(b"the cat");
        // "the"(sa=1) "cat"(sa=0) — the lone space is a bit, not a token.
        assert_eq!(units.len(), 2);
        assert!(units[0].1, "space after 'the' should be side-channel");
        assert!(matches!(units[1].0, Tok::Word { .. }));
    }

    #[test]
    fn symbol_run_merges_via_longest_match() {
        let (tok, _) = vocab();
        // "]] " is in the run vocab -> one token, not three.
        let units = tok.tokenize(b"dog]] cat");
        let syms: Vec<&Vec<u8>> = units
            .iter()
            .filter_map(|(t, _)| match t {
                Tok::Sym(b) => Some(b),
                _ => None,
            })
            .collect();
        assert_eq!(syms, vec![&b"]] ".to_vec()]);
    }

    #[test]
    fn case_is_a_side_channel() {
        let (tok, _) = vocab();
        let cases: Vec<Case> = tok
            .tokenize(b"the The THE")
            .into_iter()
            .filter_map(|(t, _)| match t {
                Tok::Word { case, .. } => Some(case),
                _ => None,
            })
            .collect();
        assert_eq!(cases, vec![Case::Lower, Case::Title, Case::Upper]);
    }

    #[test]
    fn mixed_case_and_oov_escape() {
        let (tok, id_to_word) = vocab();
        let content = b"caT xyz";
        let units = tok.tokenize(content);
        assert!(matches!(units[0].0, Tok::Esc(_)), "mixed-case -> escape");
        assert!(matches!(units[1].0, Tok::Esc(_)), "OOV -> escape");
        assert_eq!(WordTok::detokenize(&units, &id_to_word), content);
    }

    #[test]
    fn digits_are_individual() {
        let (tok, _) = vocab();
        let units = tok.tokenize(b"1987");
        assert_eq!(units.len(), 4);
        assert!(units.iter().all(|(t, _)| matches!(t, Tok::Digit(_))));
    }
}
