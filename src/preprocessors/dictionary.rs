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

/// Unified greedy longest-match dictionary over arbitrary byte-string entries
/// (words AND phrases), using the same three-tier code scheme as [`super::word_dict`]
/// (so up to ~1.05M entries, not the 74 of [`Dictionary`]). Built for the prep-lab
/// phrase experiments; decode is byte-for-byte the word-dict scheme.
#[cfg(test)]
#[derive(Default)]
struct GNode {
    code: Option<Vec<u8>>, // code bytes emitted when an entry ends at this node
    next: HashMap<u8, usize>,
}

#[cfg(test)]
pub(crate) struct GDict {
    nodes: Vec<GNode>,
    is1: [bool; 256],
    is2: [bool; 256],
    is3: [bool; 256],
    rev1: Vec<Option<Vec<u8>>>,
    rev2: HashMap<u16, Vec<u8>>,
    rev3: HashMap<u32, Vec<u8>>,
    active: bool,
}

#[cfg(test)]
impl GDict {
    /// Build from entries in priority order (`entries[i]` takes the `i`-th code).
    pub(crate) fn new(entries: &[&[u8]]) -> Self {
        use super::word_dict::code_for;
        let pool = code_pool();
        let mut nodes = vec![GNode::default()];
        let mut is1 = [false; 256];
        let mut is2 = [false; 256];
        let mut is3 = [false; 256];
        let mut rev1: Vec<Option<Vec<u8>>> = vec![None; 256];
        let mut rev2: HashMap<u16, Vec<u8>> = HashMap::new();
        let mut rev3: HashMap<u32, Vec<u8>> = HashMap::new();
        for (pos, &e) in entries.iter().enumerate() {
            let code = code_for(pos, &pool);
            let mut node = 0;
            for &b in e {
                let nlen = nodes.len();
                let child = *nodes[node].next.entry(b).or_insert(nlen);
                if child == nlen {
                    nodes.push(GNode::default());
                }
                node = child;
            }
            nodes[node].code = Some(code.clone());
            match code.as_slice() {
                [c] => {
                    is1[*c as usize] = true;
                    rev1[*c as usize] = Some(e.to_vec());
                }
                [l, idx] => {
                    is2[*l as usize] = true;
                    rev2.insert((u16::from(*l) << 8) | u16::from(*idx), e.to_vec());
                }
                [l, b1, b2] => {
                    is3[*l as usize] = true;
                    rev3.insert(
                        (u32::from(*l) << 16) | (u32::from(*b1) << 8) | u32::from(*b2),
                        e.to_vec(),
                    );
                }
                _ => unreachable!(),
            }
        }
        Self {
            active: !entries.is_empty(),
            nodes,
            is1,
            is2,
            is3,
            rev1,
            rev2,
            rev3,
        }
    }

    fn longest(&self, input: &[u8], i: usize) -> Option<(&[u8], usize)> {
        let mut node = 0;
        let mut best: Option<(&[u8], usize)> = None;
        for (k, &b) in input[i..].iter().enumerate() {
            let Some(&n) = self.nodes[node].next.get(&b) else {
                break;
            };
            node = n;
            if let Some(code) = &self.nodes[node].code {
                best = Some((code, k + 1));
            }
        }
        best
    }
}

#[cfg(test)]
impl Preprocessor for GDict {
    fn forward(&self, input: &[u8]) -> Vec<u8> {
        if !self.active {
            return input.to_vec();
        }
        let mut out = Vec::with_capacity(input.len());
        let mut i = 0;
        while i < input.len() {
            if let Some((code, len)) = self.longest(input, i) {
                out.extend_from_slice(code);
                i += len;
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
                out.extend_from_slice(self.rev1[b as usize].as_ref().unwrap());
                i += 1;
            } else if self.is2[b as usize] {
                let key = (u16::from(b) << 8) | u16::from(input[i + 1]);
                out.extend_from_slice(&self.rev2[&key]);
                i += 2;
            } else if self.is3[b as usize] {
                let key =
                    (u32::from(b) << 16) | (u32::from(input[i + 1]) << 8) | u32::from(input[i + 2]);
                out.extend_from_slice(&self.rev3[&key]);
                i += 3;
            } else {
                out.push(b);
                i += 1;
            }
        }
        out
    }
}

/// Whole-word dictionary that may also absorb a single trailing space into a
/// word's code (so "the " can become one code). Matches *maximal* a-z runs like
/// [`super::word_dict`] (no sub-word fragmentation), then, if the run is followed
/// by a space and "run+space" is an entry, prefers that. Decode is the three-tier
/// scheme. Built for the prep-lab inter-word-space experiment.
#[cfg(test)]
pub(crate) struct WSDict {
    code_of: HashMap<Vec<u8>, Vec<u8>>,
    is1: [bool; 256],
    is2: [bool; 256],
    is3: [bool; 256],
    rev1: Vec<Option<Vec<u8>>>,
    rev2: HashMap<u16, Vec<u8>>,
    rev3: HashMap<u32, Vec<u8>>,
    active: bool,
}

#[cfg(test)]
impl WSDict {
    pub(crate) fn new(entries: &[&[u8]]) -> Self {
        use super::word_dict::code_for;
        let pool = code_pool();
        let mut code_of = HashMap::new();
        let mut is1 = [false; 256];
        let mut is2 = [false; 256];
        let mut is3 = [false; 256];
        let mut rev1: Vec<Option<Vec<u8>>> = vec![None; 256];
        let mut rev2: HashMap<u16, Vec<u8>> = HashMap::new();
        let mut rev3: HashMap<u32, Vec<u8>> = HashMap::new();
        for (pos, &e) in entries.iter().enumerate() {
            let code = code_for(pos, &pool);
            match code.as_slice() {
                [c] => {
                    is1[*c as usize] = true;
                    rev1[*c as usize] = Some(e.to_vec());
                }
                [l, idx] => {
                    is2[*l as usize] = true;
                    rev2.insert((u16::from(*l) << 8) | u16::from(*idx), e.to_vec());
                }
                [l, b1, b2] => {
                    is3[*l as usize] = true;
                    rev3.insert(
                        (u32::from(*l) << 16) | (u32::from(*b1) << 8) | u32::from(*b2),
                        e.to_vec(),
                    );
                }
                _ => unreachable!(),
            }
            code_of.insert(e.to_vec(), code);
        }
        Self {
            active: !entries.is_empty(),
            code_of,
            is1,
            is2,
            is3,
            rev1,
            rev2,
            rev3,
        }
    }
}

#[cfg(test)]
impl Preprocessor for WSDict {
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
                // Prefer "run + trailing space" if that is an entry.
                if i < input.len() && input[i] == b' ' {
                    let mut ws = input[s..i].to_vec();
                    ws.push(b' ');
                    if let Some(code) = self.code_of.get(&ws) {
                        out.extend_from_slice(code);
                        i += 1;
                        continue;
                    }
                }
                match self.code_of.get(&input[s..i]) {
                    Some(code) => out.extend_from_slice(code),
                    None => out.extend_from_slice(&input[s..i]),
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
                out.extend_from_slice(self.rev1[b as usize].as_ref().unwrap());
                i += 1;
            } else if self.is2[b as usize] {
                let key = (u16::from(b) << 8) | u16::from(input[i + 1]);
                out.extend_from_slice(&self.rev2[&key]);
                i += 2;
            } else if self.is3[b as usize] {
                let key =
                    (u32::from(b) << 16) | (u32::from(input[i + 1]) << 8) | u32::from(input[i + 2]);
                out.extend_from_slice(&self.rev3[&key]);
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
    fn wsdict_roundtrips_word_space() {
        let dict = WSDict::new(&[b"the ", b"the", b"category"]);
        let data = b"\x00the category the,the the end";
        let fwd = dict.forward(data);
        assert!(fwd.len() < data.len());
        assert_eq!(dict.inverse(&fwd), data);
    }

    #[test]
    fn gdict_roundtrips_words_and_phrases() {
        // mixes a phrase (crossing a space) and plain words; greedy longest-match.
        let dict = GDict::new(&[b"the", b" of the ", b"category"]);
        let data = b"\x00the category of the things the of the end";
        let fwd = dict.forward(data);
        assert!(fwd.len() < data.len());
        assert_eq!(dict.inverse(&fwd), data);
    }

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
