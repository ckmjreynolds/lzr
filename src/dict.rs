//! Static word dictionary + reversible substitution transform.
//!
//! The 50 most frequent whole words in enwik8 are each replaced by a single
//! byte unused by the corpus; one further unused byte is an escape so the
//! transform round-trips on arbitrary input (a literal code/escape byte is
//! emitted as escape+byte). The dictionary ships in the binary (small L(D)).
//! See JOURNAL: dictionary preprocessing probe (2026-06-04).

/// Number of dictionary words.
pub(crate) const NW: usize = 50;
/// The dictionary words, most frequent first.
const WORDS: [&[u8]; NW] = [
    b"the",
    b"of",
    b"and",
    b"in",
    b"to",
    b"a",
    b"quot",
    b"is",
    b"The",
    b"lt",
    b"s",
    b"gt",
    b"as",
    b"by",
    b"for",
    b"that",
    b"was",
    b"with",
    b"id",
    b"on",
    b"amp",
    b"are",
    b"or",
    b"from",
    b"http",
    b"be",
    b"it",
    b"an",
    b"which",
    b"his",
    b"www",
    b"at",
    b"In",
    b"not",
    b"also",
    b"have",
    b"title",
    b"has",
    b"were",
    b"he",
    b"this",
    b"text",
    b"A",
    b"page",
    b"br",
    b"Category",
    b"but",
    b"one",
    b"revision",
    b"contributor",
];
/// Code byte for each word (unused in enwik8).
const CODES: [u8; NW] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11,
    0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x7f, 0xc0,
    0xc1, 0xdd, 0xdf, 0xee, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xfb, 0xfc,
    0xfd, 0xfe,
];
/// Escape byte (unused in enwik8): precedes a literal code/escape byte.
const ESCAPE: u8 = 0xff;

use std::collections::HashMap;

/// Build the word -> code-byte lookup.
fn forward() -> HashMap<&'static [u8], u8> {
    WORDS.iter().copied().zip(CODES).collect()
}

/// `is_special[b]` is true for code bytes and the escape byte — literal
/// occurrences of these in the input must be escaped to stay reversible.
fn special_table() -> [bool; 256] {
    let mut s = [false; 256];
    for &c in &CODES {
        s[c as usize] = true;
    }
    s[ESCAPE as usize] = true;
    s
}

/// Build the code-byte -> word reverse lookup.
fn reverse() -> [Option<&'static [u8]>; 256] {
    let mut r: [Option<&'static [u8]>; 256] = [None; 256];
    for (w, c) in WORDS.iter().copied().zip(CODES) {
        r[c as usize] = Some(w);
    }
    r
}

/// Replace dictionary words with their code bytes; escape any literal code or
/// escape byte. Reversible by [`untransform`].
pub(crate) fn transform(input: &[u8]) -> Vec<u8> {
    let map = forward();
    let special = special_table();
    let mut out = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        let b = input[i];
        if b.is_ascii_alphabetic() {
            let start = i;
            while i < input.len() && input[i].is_ascii_alphabetic() {
                i += 1;
            }
            let run = &input[start..i];
            // Letters are never code/escape bytes, so a literal run needs no escaping.
            if let Some(&code) = map.get(run) {
                out.push(code);
            } else {
                out.extend_from_slice(run);
            }
        } else {
            if special[b as usize] {
                out.push(ESCAPE);
            }
            out.push(b);
            i += 1;
        }
    }
    out
}

/// Inverse of [`transform`]: expand code bytes back to words, unescape literals.
pub(crate) fn untransform(t: &[u8]) -> Vec<u8> {
    let rev = reverse();
    let mut out = Vec::with_capacity(t.len() * 2);
    let mut i = 0;
    while i < t.len() {
        let b = t[i];
        if b == ESCAPE {
            out.push(t[i + 1]);
            i += 2;
        } else if let Some(w) = rev[b as usize] {
            out.extend_from_slice(w);
            i += 1;
        } else {
            out.push(b);
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_roundtrips_text() {
        let s = b"the quick brown fox is in the United States of America. The end.";
        assert_eq!(untransform(&transform(s)), s);
    }

    #[test]
    fn transform_roundtrips_all_bytes() {
        let s: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        assert_eq!(untransform(&transform(&s)), s);
    }

    #[test]
    fn transform_actually_shrinks_words() {
        let s = b"the of and in to the of and in to".repeat(10);
        assert!(transform(&s).len() < s.len());
    }
}
