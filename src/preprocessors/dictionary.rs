//! Dictionary transform: replace frequent tokens with single-byte codes.
//!
//! Runs right after case folding. Each entry is a whole *token* — a maximal run
//! of letters (`a..=z`) or of non-letters — replaced by one reserved code byte.
//! The win is context extension: a frequent token collapsed to one byte lets the
//! fixed-order and word models reach further back in tokens.
//!
//! Codes are drawn from the bytes absent from the post-fold stream: the 48
//! census-free C0/high values (minus the two case-fold markers) plus the 26
//! `A..=Z` freed by folding — 74 in all. They are assigned **highest byte value
//! first**, so the most useful entry gets `0xFF`; reclaiming a code for another
//! purpose drops the least useful entry (the lowest assigned byte).
//!
//! Reversibility is clean with no escaping: code bytes are absent from the
//! post-fold stream by construction, so a code in the output is unambiguously a
//! replacement, never literal data.

use std::collections::HashMap;

use super::Preprocessor;

/// The shipped dictionary entries, **savings-descending** (most useful first, so
/// it takes the highest code byte and the least useful is the natural one to drop
/// for reclamation). Picked offline by the isolated-savings selection (the
/// `dict_isolated` `#[ignore]` test): the candidates with positive standalone
/// byte savings on enwik8, comments showing those savings. An empty list is an
/// identity transform.
const EMBEDDED: &[&[u8]] = &[
    b">\x0a      <",   // +21695
    b"the",            // +21689
    b"quot",           // +20926
    b">\x0a    <",     // +11652
    b">\x0a        <", // +10021
    b">\x0a      </",  // +5529
    b"id",             // +4092
    b"gt",             // +3336
    b"contributor",    // +3158
    b">\x0a    </",    // +2741
    b"http",           // +2042
    b";/",             // +1898
    b"www",            // +1577
    b"amp",            // +1280
    b"timestamp",      // +1157
    b" />\x0a      <", // +1127
    b"preserve",       // +1009
    b"revision",       // +990
    b"category",       // +929
    b"com",            // +861
    b"lt",             // +851
    b"align",          // +763
    b"comment",        // +540
    b"title",          // +527
    b"redirect",       // +347
    b">\x0a  </",      // +326
    b">\x0a  <",       // +142
    b"2;).  \x00",     // +21
];

/// The 74 code bytes, **descending** — assigned to entries in order.
pub(crate) fn code_pool() -> Vec<u8> {
    let mut pool: Vec<u8> = Vec::with_capacity(74);
    // census-free C0/high (minus 0x00/0x01 used by case folding) + A..=Z
    pool.extend(0xF1..=0xFF);
    pool.extend([0xDF, 0xDD, 0xC1, 0xC0, 0x7F]);
    pool.extend(b'A'..=b'Z');
    pool.extend(0x0B..=0x1F);
    pool.extend(0x02..=0x08);
    pool.sort_unstable_by(|a, b| b.cmp(a));
    pool
}

/// Maps whole tokens to codes (forward) and codes back to tokens (inverse).
#[derive(Debug)]
pub(crate) struct Dictionary {
    map: HashMap<Vec<u8>, u8>,
    rev: Vec<Option<Vec<u8>>>,
}

impl Dictionary {
    /// Build from entries in priority order; codes assigned highest byte first.
    pub(crate) fn new(entries: &[&[u8]]) -> Self {
        let pool = code_pool();
        let mut map = HashMap::with_capacity(entries.len());
        let mut rev = vec![None; 256];
        for (entry, &code) in entries.iter().zip(&pool) {
            map.insert((*entry).to_vec(), code);
            rev[code as usize] = Some((*entry).to_vec());
        }
        Self { map, rev }
    }

    /// The shipped dictionary.
    pub(crate) fn embedded() -> Self {
        Self::new(EMBEDDED)
    }

    fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl Preprocessor for Dictionary {
    fn forward(&self, input: &[u8]) -> Vec<u8> {
        if self.is_empty() {
            return input.to_vec();
        }
        let mut out = Vec::with_capacity(input.len());
        let mut i = 0;
        while i < input.len() {
            let letter = input[i].is_ascii_lowercase();
            let start = i;
            while i < input.len() && input[i].is_ascii_lowercase() == letter {
                i += 1;
            }
            match self.map.get(&input[start..i]) {
                Some(&code) => out.push(code),
                None => out.extend_from_slice(&input[start..i]),
            }
        }
        out
    }

    fn inverse(&self, input: &[u8]) -> Vec<u8> {
        if self.is_empty() {
            return input.to_vec();
        }
        let mut out = Vec::with_capacity(input.len());
        for &b in input {
            match &self.rev[b as usize] {
                Some(token) => out.extend_from_slice(token),
                None => out.push(b),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_pool_is_74_unique_descending() {
        let pool = code_pool();
        assert_eq!(pool.len(), 74);
        let mut sorted = pool.clone();
        sorted.dedup();
        assert_eq!(sorted.len(), 74, "codes must be unique");
        assert!(pool.windows(2).all(|w| w[0] > w[1]), "descending");
        assert_eq!(pool[0], 0xFF);
    }

    #[test]
    fn roundtrips_tokens() {
        let dict = Dictionary::new(&[b"the", b"page", b">\n  <"]);
        let data = b"the page >\n  < theory pages the";
        let fwd = dict.forward(data);
        assert!(fwd.len() < data.len(), "frequent tokens should shrink");
        assert_eq!(dict.inverse(&fwd), data);
    }

    #[test]
    fn empty_is_identity() {
        let dict = Dictionary::new(&[]);
        let data = b"anything at all";
        assert_eq!(dict.forward(data), data);
        assert_eq!(dict.inverse(data), data);
    }
}
