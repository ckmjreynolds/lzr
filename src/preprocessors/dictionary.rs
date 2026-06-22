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
    b"he ",                                                              // +50849
    b">\x0a    ",                                                        // +35796
    b"uot;",                                                             // +25417
    b"nd ",                                                              // +21225
    b"ion",                                                              // +15196
    b"f ",                                                               // +8677
    b"t;",                                                               // +6680
    b"d>\x0a    <revision>\x0a      <id>",                               // +5661
    b"    ",                                                             // +3869
    b"\x00category:",                                                    // +3656
    b"omment>\x0a      <text xml:space=\"preserve\">",                   // +3183
    b"</\x01timestamp>\x0a      <contributor>\x0a        <",             // +3123
    b"&gt;",                                                             // +2933
    b"/id>\x0a    ",                                                     // +2652
    b"text>\x0a    </revision>\x0a  </page>\x0a  <page>\x0a    <title>", // +2501
    b"sername>\x0a        <id>",                                         // +2445
    b">\x0a      <t",                                                    // +1788
    b"/id>\x0a      </contributor>\x0a      <",                          // +1678
    b"sion>\x0a      <id>",                                              // +1552
    b">\x0a        <",                                                   // +1049
    b"  <",                                                              // +951
    b"quot;",                                                            // +851
    b"ext xml:space=\"preserve\">",                                      // +734
    b">\x0a      <",                                                     // +509
    b">\x0a      </contributor>\x0a      <minor />\x0a      <comment>",  // +442
    b">\x0a      </contributor>\x0a      <",                             // +373
    b"category:",                                                        // +370
    b">\x0a      ",                                                      // +324
    b"text xml:space=\"preserve\"",                                      // +241
    b">\x0a      <text xml:space=\"preserve\">",                         // +232
    b"ision>\x0a      <id>",                                             // +222
    b"/comment>\x0a      <text xml:space=\"preserve\"",                  // +209
    b"(\x01u.s. c\x01ensus)|\x00",                                       // +188
    b"timestamp>",                                                       // +171
    b"comment>\x0a      <text xml:space=\"preserve\">",                  // +135
    b"contrib",                                                          // +122
    b".s. c\x01ensus)|",                                                 // +101
    b"uot",                                                              // +93
    b">\x0a  <page>\x0a    <title>",                                     // +64
    b">\x0a",                                                            // +39
    b">\x0a  ",                                                          // +9
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

/// One trie node: the code if a dictionary entry ends here, plus byte-keyed
/// children (arena indices).
#[derive(Debug, Default)]
struct Node {
    code: Option<u8>,
    next: HashMap<u8, usize>,
}

/// Longest-match substring dictionary. Entries are arbitrary byte strings (they
/// may overlap and cross any boundary), so the forward pass walks a trie at each
/// position and replaces the longest matching entry with its code byte.
#[derive(Debug)]
pub(crate) struct Dictionary {
    nodes: Vec<Node>, // arena; node 0 is the root
    rev: Vec<Option<Vec<u8>>>,
    active: bool,
}

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

    /// The shipped dictionary.
    pub(crate) fn embedded() -> Self {
        Self::new(EMBEDDED)
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
