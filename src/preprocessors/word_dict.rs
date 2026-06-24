//! Word-canonicalizing dictionary transform (DRT-style).
//!
//! Runs right after case folding. Each entry is a whole lowercase word (a
//! maximal `a..=z` run); the transform replaces it with a short code drawn from
//! the bytes absent from the post-fold stream. The win is canonicalization and
//! context-reach extension: collapsing a frequent multi-byte word to one or two
//! bytes lets the fixed-order models span more words, and shortens the stream.
//!
//! Codes come in three tiers from the 74 free post-fold bytes (see [`code_pool`]):
//! 24 one-byte codes, then 34 lead bytes for two-byte codes (`lead + index`),
//! then 16 lead bytes for three-byte codes (`lead + index16`) — about 1.05M
//! slots in all. A byte is exactly one of: a one-byte code, a two-byte lead, a
//! three-byte lead, or literal data, so decoding is unambiguous with no escape.
//!
//! The embedded list ([`WORDS`], one word per line in code/frequency order) is
//! the **`L(D)` blob** — it ships in the binary and is mined from the corpus by
//! the `gen_word_dict` offline test. Code assignment is implicit in position
//! (line `i` takes the `i`-th code), so the file carries only words.

use std::collections::HashMap;

use super::Preprocessor;
use super::dictionary::code_pool;

/// The embedded word list, one lowercase word per line, in code/frequency order.
/// Empty (a lone newline or nothing) is an identity transform.
const WORDS: &[u8] = include_bytes!("words.dict");

const N_SINGLE: usize = 24;
const N_LEAD2: usize = 34;
const N_LEAD3: usize = 16; // 24 + 34 + 16 == code_pool().len() == 74
const SINGLE_END: usize = N_SINGLE;
const PAIR_END: usize = N_SINGLE + N_LEAD2 * 256;
const TRIP_END: usize = PAIR_END + N_LEAD3 * 65536;

/// Bytes of the code assigned to dictionary position `pos`.
pub(crate) fn code_for(pos: usize, pool: &[u8]) -> Vec<u8> {
    if pos < SINGLE_END {
        vec![pool[pos]]
    } else if pos < PAIR_END {
        let p = pos - SINGLE_END;
        #[allow(clippy::cast_possible_truncation)]
        let idx = (p % 256) as u8;
        vec![pool[SINGLE_END + p / 256], idx]
    } else {
        let q = pos - PAIR_END;
        #[allow(clippy::cast_possible_truncation)]
        let (b1, b2) = (((q >> 8) & 0xff) as u8, (q & 0xff) as u8);
        vec![pool[SINGLE_END + N_LEAD2 + q / 65536], b1, b2]
    }
}

/// Required word length for the code at `pos` to be a strict shrink.
pub(crate) const fn min_word_len(pos: usize) -> usize {
    if pos < SINGLE_END {
        2
    } else if pos < PAIR_END {
        3
    } else {
        4
    }
}

/// Word-replacement dictionary built from an ordered word list.
#[derive(Debug)]
pub(crate) struct WordDict {
    code_of: HashMap<&'static [u8], Vec<u8>>,
    is1: [bool; 256],
    is2: [bool; 256],
    is3: [bool; 256],
    rev1: Vec<Option<&'static [u8]>>,
    rev2: HashMap<u16, &'static [u8]>,
    rev3: HashMap<u32, &'static [u8]>,
    active: bool,
}

impl WordDict {
    /// Build from words in code order (`words[i]` takes the `i`-th code).
    pub(crate) fn from_words(words: &[&'static [u8]]) -> Self {
        let pool = code_pool();
        let mut code_of = HashMap::new();
        let mut is1 = [false; 256];
        let mut is2 = [false; 256];
        let mut is3 = [false; 256];
        let mut rev1 = vec![None; 256];
        let mut rev2 = HashMap::new();
        let mut rev3 = HashMap::new();
        for (pos, &w) in words.iter().enumerate().take(TRIP_END) {
            debug_assert!(w.len() >= min_word_len(pos), "word too short for its code");
            let code = code_for(pos, &pool);
            match code.as_slice() {
                [c] => {
                    is1[*c as usize] = true;
                    rev1[*c as usize] = Some(w);
                }
                [lead, idx] => {
                    is2[*lead as usize] = true;
                    rev2.insert((u16::from(*lead) << 8) | u16::from(*idx), w);
                }
                [lead, b1, b2] => {
                    is3[*lead as usize] = true;
                    rev3.insert(
                        (u32::from(*lead) << 16) | (u32::from(*b1) << 8) | u32::from(*b2),
                        w,
                    );
                }
                _ => unreachable!(),
            }
            code_of.insert(w, code);
        }
        Self {
            active: !code_of.is_empty(),
            code_of,
            is1,
            is2,
            is3,
            rev1,
            rev2,
            rev3,
        }
    }

    /// The shipped dictionary (parsed from the embedded [`WORDS`] blob).
    pub(crate) fn embedded() -> Self {
        let words: Vec<&'static [u8]> = WORDS
            .split(|&b| b == b'\n')
            .filter(|w| !w.is_empty())
            .collect();
        Self::from_words(&words)
    }
}

impl Preprocessor for WordDict {
    fn forward(&self, input: &[u8]) -> Vec<u8> {
        if !self.active {
            return input.to_vec();
        }
        let mut out = Vec::with_capacity(input.len());
        let mut i = 0;
        while i < input.len() {
            if input[i].is_ascii_lowercase() {
                let s = i;
                while i < input.len() && input[i].is_ascii_lowercase() {
                    i += 1;
                }
                if let Some(code) = self.code_of.get(&input[s..i]) {
                    out.extend_from_slice(code);
                } else {
                    out.extend_from_slice(&input[s..i]);
                }
            } else {
                out.push(input[i]);
                i += 1;
            }
        }
        out
    }

    fn inverse(&self, input: &[u8]) -> Vec<u8> {
        if !self.active {
            return input.to_vec();
        }
        let mut out = Vec::with_capacity(input.len());
        let mut i = 0;
        while i < input.len() {
            let b = input[i];
            if self.is1[b as usize] {
                out.extend_from_slice(self.rev1[b as usize].unwrap());
                i += 1;
            } else if self.is2[b as usize] {
                let key = (u16::from(b) << 8) | u16::from(input[i + 1]);
                out.extend_from_slice(self.rev2[&key]);
                i += 2;
            } else if self.is3[b as usize] {
                let key =
                    (u32::from(b) << 16) | (u32::from(input[i + 1]) << 8) | u32::from(input[i + 2]);
                out.extend_from_slice(self.rev3[&key]);
                i += 3;
            } else {
                out.push(b);
                i += 1;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_words() {
        let words: Vec<&'static [u8]> = vec![b"the", b"and", b"category"];
        let dict = WordDict::from_words(&words);
        let data = b"\x00the category and theory of \x00things";
        let fwd = dict.forward(data);
        assert!(fwd.len() < data.len(), "frequent words should shrink");
        assert_eq!(dict.inverse(&fwd), data);
    }

    #[test]
    fn empty_is_identity() {
        let dict = WordDict::from_words(&[]);
        let data = b"anything at all";
        assert_eq!(dict.forward(data), data);
        assert_eq!(dict.inverse(data), data);
    }

    #[test]
    fn three_tiers_assign_distinct_codes() {
        let pool = code_pool();
        assert_eq!(code_for(0, &pool).len(), 1);
        assert_eq!(code_for(SINGLE_END, &pool).len(), 2);
        assert_eq!(code_for(PAIR_END, &pool).len(), 3);
    }

    /// Generate the embedded `words.dict`: mine the top-`LZR_N` (default 2000)
    /// most frequent lowercase words from case-folded `LZR_CORPUS` (default
    /// `assets/enwik9`, the submission target), keeping each only if it is long
    /// enough to be a strict shrink at its position, and write them one per line
    /// in code order. Re-run with a larger `LZR_N` to grow the dictionary (e.g.
    /// for the online-neural arm) — no code change needed. Run:
    /// `LZR_N=2000 cargo test --release gen_word_dict -- --ignored --nocapture`
    #[test]
    #[ignore = "offline: generate the embedded words.dict from the corpus"]
    fn gen_word_dict() {
        use crate::preprocessors::casefold::CaseFold;
        use std::collections::HashMap;

        let corpus = std::env::var("LZR_CORPUS").unwrap_or_else(|_| "assets/enwik9".into());
        let n: usize = std::env::var("LZR_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2000)
            .min(TRIP_END);
        let Ok(raw) = std::fs::read(&corpus) else {
            return;
        };
        let folded = CaseFold.forward(&raw);
        drop(raw);

        let mut freq: HashMap<&[u8], u32> = HashMap::new();
        let mut i = 0;
        while i < folded.len() {
            if folded[i].is_ascii_lowercase() {
                let s = i;
                while i < folded.len() && folded[i].is_ascii_lowercase() {
                    i += 1;
                }
                *freq.entry(&folded[s..i]).or_insert(0) += 1;
            } else {
                i += 1;
            }
        }
        let mut words: Vec<(&[u8], u32)> = freq.into_iter().collect();
        words.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

        let mut out: Vec<u8> = Vec::new();
        let mut dict_bytes = 0usize;
        let mut k = 0usize;
        for (w, _f) in &words {
            if k >= n {
                break;
            }
            if w.len() >= min_word_len(k) {
                out.extend_from_slice(w);
                out.push(b'\n');
                dict_bytes += w.len();
                k += 1;
            }
        }
        std::fs::write("src/preprocessors/words.dict", &out).unwrap();
        println!(
            "wrote src/preprocessors/words.dict: {k} words, {dict_bytes} word bytes, {} file bytes (from {corpus})",
            out.len()
        );
    }
}
