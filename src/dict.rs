//! Static word dictionary + reversible substitution transform.
//!
//! Frequent whole words in enwik8 are replaced before modeling: the top NW1 by
//! single byte codes, the next 256 (length >= 3) by a two-byte code (`ESC_WORD`
//! then index). The single-byte codes reuse byte values freed by escaping their
//! rare literal occurrences (`< RARE_THRESHOLD` in the corpus); `ESC_LIT`
//! precedes any literal code/escape byte so the transform round-trips on
//! arbitrary input. The dictionary ships in the binary (small L(D)). See
//! JOURNAL: dictionary preprocessing.

pub(crate) const NW1: usize = 173;
pub(crate) const NW2: usize = 256;
const WORDS1: [&[u8]; NW1] = [
    b"the",
    b"of",
    b"and",
    b"in",
    b"a",
    b"to",
    b"quot",
    b"is",
    b"s",
    b"lt",
    b"gt",
    b"as",
    b"for",
    b"by",
    b"that",
    b"was",
    b"with",
    b"id",
    b"on",
    b"amp",
    b"are",
    b"it",
    b"from",
    b"or",
    b"http",
    b"be",
    b"an",
    b"this",
    b"his",
    b"which",
    b"at",
    b"he",
    b"www",
    b"not",
    b"also",
    b"have",
    b"title",
    b"has",
    b"were",
    b"one",
    b"text",
    b"page",
    b"br",
    b"but",
    b"category",
    b"other",
    b"revision",
    b"contributor",
    b"their",
    b"timestamp",
    b"its",
    b"t",
    b"first",
    b"they",
    b"some",
    b"com",
    b"i",
    b"can",
    b"all",
    b"had",
    b"more",
    b"comment",
    b"most",
    b"new",
    b"been",
    b"such",
    b"td",
    b"username",
    b"many",
    b"who",
    b"d",
    b"used",
    b"there",
    b"de",
    b"image",
    b"two",
    b"b",
    b"time",
    b"space",
    b"after",
    b"no",
    b"see",
    b"when",
    b"united",
    b"into",
    b"these",
    b"language",
    b"only",
    b"world",
    b"e",
    b"may",
    b"c",
    b"z",
    b"than",
    b"states",
    b"american",
    b"th",
    b"align",
    b"right",
    b"n",
    b"org",
    b"would",
    b"math",
    b"history",
    b"html",
    b"m",
    b"preserve",
    b"people",
    b"xml",
    b"about",
    b"between",
    b"however",
    b"over",
    b"system",
    b"center",
    b"if",
    b"years",
    b"f",
    b"x",
    b"war",
    b"sup",
    b"use",
    b"name",
    b"state",
    b"known",
    b"sub",
    b"called",
    b"nbsp",
    b"during",
    b"px",
    b"english",
    b"number",
    b"left",
    b"will",
    b"jpg",
    b"often",
    b"so",
    b"u",
    b"small",
    b"city",
    b"list",
    b"g",
    b"thumb",
    b"up",
    b"where",
    b"year",
    b"while",
    b"any",
    b"being",
    b"both",
    b"out",
    b"university",
    b"then",
    b"under",
    b"century",
    b"p",
    b"them",
    b"well",
    b"made",
    b"british",
    b"government",
    b"film",
    b"r",
    b"since",
    b"national",
    b"part",
    b"style",
    b"through",
    b"later",
    b"because",
    b"early",
    b"three",
    b"like",
];
const CODES1: [u8; NW1] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0b, 0x0c, 0x0d, 0x11, 0x12, 0x13,
    0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x24, 0x2b, 0x40, 0x41,
    0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f, 0x50, 0x51,
    0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x5e, 0x60, 0x7e, 0x7f, 0x81, 0x84, 0x85,
    0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95,
    0x96, 0x97, 0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f, 0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7,
    0xb9, 0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf, 0xc0, 0xc1, 0xc2, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9,
    0xca, 0xcb, 0xcc, 0xcd, 0xcf, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd,
    0xde, 0xdf, 0xe1, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef, 0xf0,
    0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd,
];
const WORDS2: [&[u8]; NW2] = [
    b"ndash",
    b"example",
    b"same",
    b"him",
    b"each",
    b"book",
    b"life",
    b"game",
    b"general",
    b"htm",
    b"music",
    b"main",
    b"before",
    b"even",
    b"including",
    b"her",
    b"links",
    b"work",
    b"john",
    b"french",
    b"form",
    b"day",
    b"although",
    b"several",
    b"became",
    b"international",
    b"minor",
    b"much",
    b"series",
    b"very",
    b"what",
    b"high",
    b"south",
    b"computer",
    b"great",
    b"group",
    b"party",
    b"those",
    b"second",
    b"now",
    b"external",
    b"common",
    b"against",
    b"country",
    b"based",
    b"german",
    b"could",
    b"end",
    b"modern",
    b"large",
    b"law",
    b"power",
    b"set",
    b"different",
    b"another",
    b"church",
    b"north",
    b"article",
    b"mdash",
    b"area",
    b"term",
    b"long",
    b"you",
    b"order",
    b"theory",
    b"still",
    b"major",
    b"king",
    b"god",
    b"ipa",
    b"population",
    b"last",
    b"usually",
    b"until",
    b"death",
    b"found",
    b"human",
    b"president",
    b"political",
    b"free",
    b"using",
    b"european",
    b"east",
    b"own",
    b"roman",
    b"military",
    b"science",
    b"information",
    b"note",
    b"point",
    b"ref",
    b"redirect",
    b"non",
    b"include",
    b"west",
    b"public",
    b"isbn",
    b"four",
    b"she",
    b"around",
    b"way",
    b"code",
    b"kingdom",
    b"though",
    b"england",
    b"within",
    b"times",
    b"top",
    b"house",
    b"following",
    b"old",
    b"did",
    b"size",
    b"languages",
    b"among",
    b"due",
    b"word",
    b"york",
    b"line",
    b"central",
    b"considered",
    b"union",
    b"family",
    b"france",
    b"important",
    b"popular",
    b"man",
    b"greek",
    b"europe",
    b"place",
    b"sometimes",
    b"home",
    b"members",
    b"germany",
    b"given",
    b"case",
    b"water",
    b"without",
    b"countries",
    b"data",
    b"player",
    b"version",
    b"black",
    b"make",
    b"age",
    b"does",
    b"republic",
    b"island",
    b"should",
    b"company",
    b"systems",
    b"air",
    b"thus",
    b"control",
    b"empire",
    b"news",
    b"font",
    b"team",
    b"western",
    b"back",
    b"period",
    b"net",
    b"others",
    b"games",
    b"force",
    b"must",
    b"various",
    b"how",
    b"original",
    b"similar",
    b"college",
    b"school",
    b"single",
    b"become",
    b"edu",
    b"football",
    b"america",
    b"christian",
    b"class",
    b"generally",
    b"development",
    b"less",
    b"color",
    b"best",
    b"few",
    b"george",
    b"islands",
    b"written",
    b"groups",
    b"earth",
    b"standard",
    b"field",
    b"books",
    b"according",
    b"down",
    b"land",
    b"david",
    b"born",
    b"battle",
    b"began",
    b"led",
    b"official",
    b"software",
    b"rather",
    b"service",
    b"red",
    b"body",
    b"london",
    b"works",
    b"character",
    b"million",
    b"army",
    b"sea",
    b"short",
    b"type",
    b"support",
    b"social",
    b"economic",
    b"open",
    b"river",
    b"court",
    b"fact",
    b"charles",
    b"just",
    b"every",
    b"white",
    b"culture",
    b"off",
    b"late",
    b"current",
    b"either",
    b"special",
    b"story",
    b"show",
    b"press",
    b"december",
    b"published",
    b"named",
    b"society",
    b"natural",
    b"region",
    b"png",
    b"good",
    b"result",
    b"ancient",
    b"means",
    b"today",
    b"research",
    b"league",
    b"emperor",
    b"said",
    b"bgcolor",
    b"person",
    b"third",
    b"play",
    b"possible",
];
pub(crate) const ESC_WORD: u8 = 0xfe;
pub(crate) const ESC_LIT: u8 = 0xff;

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

/// Case transform: lowercase the stream and mark capitalization in-band, so the
/// model sees a single pooled lowercase alphabet (fewer distinct contexts → less
/// collision tax) and the uppercase byte range `A`..=`Z` is freed for a larger
/// dictionary. `CAP1` precedes a single uppercased letter, `CAPW` an all-caps
/// letter run; `CASE_ESC` escapes a literal marker byte so the transform
/// round-trips on arbitrary input. Markers are non-alphabetic so word/dict
/// detection still folds the lowercase run.
const CAP1: u8 = 0x0E;
const CAPW: u8 = 0x0F;
const CASE_ESC: u8 = 0x10;

#[allow(clippy::needless_range_loop)]
pub(crate) fn case_transform(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() + input.len() / 8);
    let mut i = 0;
    while i < input.len() {
        let b = input[i];
        if b.is_ascii_alphabetic() {
            let start = i;
            while i < input.len() && input[i].is_ascii_alphabetic() {
                i += 1;
            }
            let run = &input[start..i];
            let n_upper = run.iter().filter(|c| c.is_ascii_uppercase()).count();
            if n_upper == run.len() && run.len() >= 2 {
                out.push(CAPW);
                out.extend(run.iter().map(u8::to_ascii_lowercase));
            } else if n_upper == 0 {
                out.extend_from_slice(run);
            } else {
                for &c in run {
                    if c.is_ascii_uppercase() {
                        out.push(CAP1);
                        out.push(c.to_ascii_lowercase());
                    } else {
                        out.push(c);
                    }
                }
            }
        } else {
            if b == CAP1 || b == CAPW || b == CASE_ESC {
                out.push(CASE_ESC);
            }
            out.push(b);
            i += 1;
        }
    }
    out
}

/// Inverse of [`case_transform`].
pub(crate) fn case_untransform(t: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(t.len());
    let mut i = 0;
    while i < t.len() {
        let b = t[i];
        if b == CAP1 {
            out.push(t[i + 1].to_ascii_uppercase());
            i += 2;
        } else if b == CAPW {
            i += 1;
            while i < t.len() && t[i].is_ascii_alphabetic() {
                out.push(t[i].to_ascii_uppercase());
                i += 1;
            }
        } else if b == CASE_ESC {
            out.push(t[i + 1]);
            i += 2;
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
    fn case_roundtrips_text() {
        let s = b"The Quick BROWN fox, iPhone NASA McDonald's. UTF-8 \x0e\x0f\x10 done.";
        assert_eq!(case_untransform(&case_transform(s)), s);
    }

    #[test]
    fn case_roundtrips_all_bytes() {
        let s: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        assert_eq!(case_untransform(&case_transform(&s)), s);
    }

    /// Regenerate the dictionary tables on the *case-transformed* (lowercased)
    /// corpus, using every byte the transform frees (uppercase range, minus the
    /// markers). Run: `cargo test --release gen_dict_tables -- --ignored --nocapture`.
    #[test]
    #[ignore = "regenerates dictionary tables from assets/enwik8; run manually"]
    fn gen_dict_tables() {
        use std::collections::HashMap;
        // Bytes appearing fewer than this in the case-transformed corpus are
        // "rare": escaping their literal occurrences (via ESC_LIT, the same
        // mechanism that protects code bytes) frees their value as a single-byte
        // dictionary code. ~0.4% of the stream is escape overhead at 10k; in
        // return the single-byte dictionary roughly doubles.
        const RARE_THRESHOLD: u64 = 10_000;
        let bytes = std::fs::read("assets/enwik8").unwrap();
        let cased = case_transform(&bytes);
        let mut count = [0u64; 256];
        for &b in &cased {
            count[b as usize] += 1;
        }
        let avail: Vec<u8> = (0u8..=255)
            .filter(|&b| {
                count[b as usize] < RARE_THRESHOLD
                    && b != ESC_WORD
                    && b != ESC_LIT
                    && b != CAP1
                    && b != CAPW
                    && b != CASE_ESC
            })
            .collect();
        let mut freq: HashMap<&[u8], u64> = HashMap::new();
        let mut i = 0;
        while i < cased.len() {
            if cased[i].is_ascii_alphabetic() {
                let s = i;
                while i < cased.len() && cased[i].is_ascii_alphabetic() {
                    i += 1;
                }
                *freq.entry(&cased[s..i]).or_insert(0) += 1;
            } else {
                i += 1;
            }
        }
        let mut words: Vec<(&[u8], u64)> = freq.into_iter().collect();
        words.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        let n1 = avail.len();
        let w1: Vec<&[u8]> = words.iter().take(n1).map(|&(w, _)| w).collect();
        let w2: Vec<&[u8]> = words
            .iter()
            .skip(n1)
            .filter(|&&(w, _)| w.len() >= 3)
            .take(256)
            .map(|&(w, _)| w)
            .collect();
        let fmt = |ws: &[&[u8]]| -> String {
            ws.iter()
                .map(|w| format!("b\"{}\"", std::str::from_utf8(w).unwrap()))
                .collect::<Vec<_>>()
                .join(", ")
        };
        println!("pub(crate) const NW1: usize = {n1};");
        println!("pub(crate) const NW2: usize = {};", w2.len());
        println!("const WORDS1: [&[u8]; NW1] = [{}];", fmt(&w1));
        let codes: Vec<String> = avail.iter().map(|b| format!("0x{b:02x}")).collect();
        println!("const CODES1: [u8; NW1] = [{}];", codes.join(", "));
        println!("const WORDS2: [&[u8]; NW2] = [{}];", fmt(&w2));
    }

    // The dictionary now runs on a case-folded stream (its codes reuse the
    // uppercase byte range), so these test the coupled case + dictionary
    // pipeline — the only way the dictionary is actually used.
    #[test]
    fn transform_roundtrips_text() {
        let s = b"the quick brown fox is a contributor to the timestamp of references. The end.";
        assert_eq!(
            case_untransform(&untransform(&transform(&case_transform(s)))),
            s
        );
    }

    #[test]
    fn transform_roundtrips_all_bytes() {
        let s: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        assert_eq!(
            case_untransform(&untransform(&transform(&case_transform(&s)))),
            s
        );
    }

    #[test]
    fn transform_shrinks_words() {
        let s = b"contributor timestamp references contributor timestamp references".repeat(8);
        assert!(transform(&s).len() < s.len());
    }
}
