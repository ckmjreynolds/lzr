//! Static word dictionary + reversible substitution transform.
//!
//! Frequent whole words in enwik8 are replaced before modeling: the top 49 by
//! single unused byte codes, the next 256 (length >= 3) by a two-byte code
//! (`ESC_WORD` + index). A second escape (`ESC_LIT`) precedes any literal code/
//! escape byte so the transform round-trips on arbitrary input. The dictionary
//! ships in the binary (small L(D)). See JOURNAL: dictionary preprocessing.

pub(crate) const NW1: usize = 49;
pub(crate) const NW2: usize = 256;
const WORDS1: [&[u8]; NW1] = [
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
];
const CODES1: [u8; NW1] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11,
    0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x7f, 0xc0,
    0xc1, 0xdd, 0xdf, 0xee, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xfb, 0xfc,
    0xfd,
];
const WORDS2: [&[u8]; NW2] = [
    b"contributor",
    b"timestamp",
    b"their",
    b"other",
    b"com",
    b"its",
    b"had",
    b"can",
    b"comment",
    b"first",
    b"been",
    b"more",
    b"username",
    b"such",
    b"all",
    b"used",
    b"This",
    b"who",
    b"they",
    b"most",
    b"some",
    b"two",
    b"United",
    b"into",
    b"space",
    b"many",
    b"time",
    b"than",
    b"only",
    b"language",
    b"American",
    b"Image",
    b"right",
    b"org",
    b"align",
    b"would",
    b"math",
    b"may",
    b"preserve",
    b"xml",
    b"after",
    b"html",
    b"when",
    b"between",
    b"about",
    b"States",
    b"sup",
    b"over",
    b"there",
    b"years",
    b"use",
    b"known",
    b"nbsp",
    b"these",
    b"called",
    b"name",
    b"sub",
    b"New",
    b"people",
    b"center",
    b"system",
    b"English",
    b"left",
    b"number",
    b"will",
    b"thumb",
    b"often",
    b"small",
    b"jpg",
    b"new",
    b"where",
    b"any",
    b"being",
    b"them",
    b"British",
    b"made",
    b"out",
    b"year",
    b"well",
    b"For",
    b"then",
    b"University",
    b"century",
    b"both",
    b"under",
    b"during",
    b"see",
    b"See",
    b"world",
    b"style",
    b"ndash",
    b"part",
    b"state",
    b"through",
    b"same",
    b"him",
    b"htm",
    b"example",
    b"film",
    b"World",
    b"including",
    b"John",
    b"However",
    b"later",
    b"French",
    b"form",
    b"government",
    b"like",
    b"links",
    b"work",
    b"War",
    b"three",
    b"became",
    b"List",
    b"while",
    b"before",
    b"History",
    b"her",
    b"each",
    b"very",
    b"because",
    b"minor",
    b"early",
    b"There",
    b"even",
    b"much",
    b"German",
    b"could",
    b"city",
    b"music",
    b"based",
    b"several",
    b"against",
    b"game",
    b"since",
    b"history",
    b"those",
    b"different",
    b"large",
    b"life",
    b"set",
    b"mdash",
    b"now",
    b"end",
    b"term",
    b"main",
    b"IPA",
    b"External",
    b"common",
    b"series",
    b"group",
    b"power",
    b"Some",
    b"found",
    b"what",
    b"still",
    b"high",
    b"country",
    b"usually",
    b"European",
    b"book",
    b"own",
    b"until",
    b"Roman",
    b"include",
    b"long",
    b"article",
    b"theory",
    b"second",
    b"however",
    b"ref",
    b"ISBN",
    b"area",
    b"National",
    b"last",
    b"using",
    b"day",
    b"population",
    b"computer",
    b"modern",
    b"law",
    b"They",
    b"England",
    b"order",
    b"point",
    b"These",
    b"around",
    b"major",
    b"did",
    b"God",
    b"war",
    b"York",
    b"political",
    b"way",
    b"considered",
    b"France",
    b"Greek",
    b"non",
    b"within",
    b"South",
    b"Europe",
    b"Germany",
    b"important",
    b"death",
    b"you",
    b"word",
    b"general",
    b"another",
    b"code",
    b"One",
    b"languages",
    b"human",
    b"International",
    b"popular",
    b"place",
    b"four",
    b"top",
    b"Kingdom",
    b"due",
    b"given",
    b"without",
    b"although",
    b"does",
    b"countries",
    b"make",
    b"line",
    b"sometimes",
    b"Church",
    b"though",
    b"information",
    b"should",
    b"times",
    b"player",
    b"His",
    b"following",
    b"case",
    b"After",
    b"version",
    b"members",
    b"among",
    b"public",
    b"water",
    b"edu",
    b"become",
    b"North",
    b"must",
    b"Christian",
    b"America",
    b"military",
    b"Republic",
    b"period",
    b"George",
    b"When",
    b"back",
    b"Union",
    b"REDIRECT",
];
const ESC_WORD: u8 = 0xfe;
const ESC_LIT: u8 = 0xff;

use std::collections::HashMap;

/// `word -> single-byte code` for the top NW1 words.
fn forward1() -> HashMap<&'static [u8], u8> {
    WORDS1.iter().copied().zip(CODES1).collect()
}

/// `word -> index` into WORDS2 (two-byte coded).
#[allow(clippy::cast_possible_truncation)]
fn forward2() -> HashMap<&'static [u8], u8> {
    WORDS2
        .iter()
        .enumerate()
        .map(|(i, &w)| (w, i as u8))
        .collect()
}

/// `is_special[b]` is true for single-byte codes and both escape bytes, which
/// must be escaped when they occur literally.
fn special_table() -> [bool; 256] {
    let mut s = [false; 256];
    for &c in &CODES1 {
        s[c as usize] = true;
    }
    s[ESC_WORD as usize] = true;
    s[ESC_LIT as usize] = true;
    s
}

/// `code -> word` reverse map for the single-byte codes.
fn reverse1() -> [Option<&'static [u8]>; 256] {
    let mut r: [Option<&'static [u8]>; 256] = [None; 256];
    for (w, c) in WORDS1.iter().copied().zip(CODES1) {
        r[c as usize] = Some(w);
    }
    r
}

/// Replace dictionary words with their codes (1 byte for the top NW1, else
/// `ESC_WORD + index`); escape literal code/escape bytes. Reversible.
pub(crate) fn transform(input: &[u8]) -> Vec<u8> {
    let f1 = forward1();
    let f2 = forward2();
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
            if let Some(&code) = f1.get(run) {
                out.push(code);
            } else if let Some(&idx) = f2.get(run) {
                out.push(ESC_WORD);
                out.push(idx);
            } else {
                out.extend_from_slice(run);
            }
        } else {
            if special[b as usize] {
                out.push(ESC_LIT);
            }
            out.push(b);
            i += 1;
        }
    }
    out
}

/// Inverse of [`transform`].
pub(crate) fn untransform(t: &[u8]) -> Vec<u8> {
    let rev1 = reverse1();
    let mut out = Vec::with_capacity(t.len() * 2);
    let mut i = 0;
    while i < t.len() {
        let b = t[i];
        if b == ESC_WORD {
            out.extend_from_slice(WORDS2[t[i + 1] as usize]);
            i += 2;
        } else if b == ESC_LIT {
            out.push(t[i + 1]);
            i += 2;
        } else if let Some(w) = rev1[b as usize] {
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
        let s = b"the quick brown fox is a contributor to the timestamp of references. The end.";
        assert_eq!(untransform(&transform(s)), s);
    }

    #[test]
    fn transform_roundtrips_all_bytes() {
        let s: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        assert_eq!(untransform(&transform(&s)), s);
    }

    #[test]
    fn transform_shrinks_words() {
        let s = b"contributor timestamp references contributor timestamp references".repeat(8);
        assert!(transform(&s).len() < s.len());
    }
}
