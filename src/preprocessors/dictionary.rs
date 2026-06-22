//! Free code-byte pool, plus the (retired) substring dictionary.
//!
//! [`code_pool`] enumerates the 74 byte values absent from the post-fold stream
//! (48 census-free C0/high, minus the two case-fold markers, plus the 26 `A..=Z`
//! freed by folding) — the codes any post-fold transform may emit unambiguously.
//! It is the shared pool now used by [`super::word_dict`].
//!
//! The longest-match substring [`Dictionary`] below was the earlier dictionary
//! form; it is superseded by the word dictionary (see the 2026-06-22 journal
//! entry) and kept only for the offline `dict_*` selection tests, hence
//! `#[cfg(test)]`.

#[cfg(test)]
use std::collections::HashMap;

#[cfg(test)]
use super::Preprocessor;

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

/// One trie node: the code if a dictionary entry ends here, plus byte-keyed
/// children (arena indices).
#[cfg(test)]
#[derive(Debug, Default)]
struct Node {
    code: Option<u8>,
    next: HashMap<u8, usize>,
}

/// Longest-match substring dictionary. Entries are arbitrary byte strings (they
/// may overlap and cross any boundary), so the forward pass walks a trie at each
/// position and replaces the longest matching entry with its code byte.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct Dictionary {
    nodes: Vec<Node>, // arena; node 0 is the root
    rev: Vec<Option<Vec<u8>>>,
    active: bool,
}

#[cfg(test)]
impl Dictionary {
    /// Build from entries in priority order; codes assigned highest byte first.
    pub(crate) fn new(entries: &[&[u8]]) -> Self {
        let pool = code_pool();
        let mut nodes = vec![Node::default()];
        let mut rev = vec![None; 256];
        for (entry, &code) in entries.iter().zip(&pool) {
            let mut node = 0;
            for &b in *entry {
                let nlen = nodes.len();
                let child = *nodes[node].next.entry(b).or_insert(nlen);
                if child == nlen {
                    nodes.push(Node::default());
                }
                node = child;
            }
            nodes[node].code = Some(code);
            rev[code as usize] = Some((*entry).to_vec());
        }
        Self {
            nodes,
            rev,
            active: !entries.is_empty(),
        }
    }

    /// The longest entry matching at `input[i..]`, as `(code, length)`.
    fn longest_match(&self, input: &[u8], i: usize) -> Option<(u8, usize)> {
        let mut node = 0;
        let mut best = None;
        for (k, &b) in input[i..].iter().enumerate() {
            let Some(&n) = self.nodes[node].next.get(&b) else {
                break;
            };
            node = n;
            if let Some(c) = self.nodes[node].code {
                best = Some((c, k + 1));
            }
        }
        best
    }
}

#[cfg(test)]
impl Preprocessor for Dictionary {
    fn forward(&self, input: &[u8]) -> Vec<u8> {
        if !self.active {
            return input.to_vec();
        }
        let mut out = Vec::with_capacity(input.len());
        let mut i = 0;
        while i < input.len() {
            let (byte, step) = match self.longest_match(input, i) {
                Some((code, len)) => (code, len),
                None => (input[i], 1),
            };
            out.push(byte);
            i += step;
        }
        out
    }

    fn inverse(&self, input: &[u8]) -> Vec<u8> {
        if !self.active {
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
