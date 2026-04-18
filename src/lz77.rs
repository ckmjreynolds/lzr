//! LZ77 encoder and decoder.
//!
//! A [`Sequence`] describes zero or more literal bytes followed by an optional
//! back-reference (match). Three match finders are provided; each compression
//! level in [`LEVELS`] selects one:
//!
//! * **Head** (L1) — a single `hash_bits`-sized hash table with no chain,
//!   mirroring `lz4 -1`. One lookup per byte, no history.
//! * **Chain** (L2–L6) — hash table plus a `WINDOW_SIZE`-entry chain; each
//!   chain step visits one older position with the same 3-byte hash. Walk is
//!   bounded by `max_chain` and short-circuits on `nice_length`.
//! * **`BTree`** (L7–L9) — `wabi_tree` B+tree keyed by a 16-byte prefix,
//!   scanning `max_scan` neighbours in each direction; `depth` packed
//!   positions per visited key.
//!
//! # Pipeline
//!
//! ```text
//! Encode:  Bytes → LZ77 Encoder  → {Arithmetic Coder | raw bytes}
//! Decode:  {Arithmetic Decoder | raw bytes} → LZ77 Decoder → Bytes
//! ```

use static_assertions::assert_eq_size;
use wabi_tree::OSBTreeMap;

/// Sliding-window size in bytes (matches the `u16` distance range).
const WINDOW_SIZE: usize = 65_536;

/// Minimum useful match length.
const MIN_MATCH: usize = 3;

/// Maximum match length (constrained by `match_len: u8`).
const MAX_MATCH: usize = 255;

/// Maximum literal run per sequence (constrained by `literal_len: u8`).
pub(crate) const MAX_LITERALS: usize = 255;

/// Minimum lookahead required in streaming mode before emitting tokens.
const LOOKAHEAD_MIN: usize = MAX_MATCH + MIN_MATCH;

/// Minimum compression level.
pub(crate) const MIN_LEVEL: u8 = 1;

/// Maximum compression level.
pub(crate) const MAX_LEVEL: u8 = 9;

/// Default compression level (matches the gzip convention).
pub(crate) const DEFAULT_LEVEL: u8 = 6;

/// Sentinel "no position" used by the hash-based finders.
const EMPTY_POS: u32 = u32::MAX;

/// Knuth's multiplicative hash constant (`floor(2^32 / phi)`).
const HASH_MUL: u32 = 2_654_435_761;

// ===========================================================================
// Level configuration
// ===========================================================================

/// Per-finder parameters. Each variant is paired with the corresponding
/// [`MatchFinder`] variant at encoder construction time. Hash tables are
/// sized as `1 << hash_bits`; the match search stops early on `nice_length`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum FinderConfig {
    /// Single hash-table lookup per position. No chain walk, no history.
    Head {
        hash_bits: u32,
    },
    /// Hash-table head followed by a bounded chain walk through older
    /// positions sharing the same hash.
    Chain {
        hash_bits: u32,
        max_chain: usize,
        nice_length: usize,
    },
    /// `wabi_tree` B+tree with 16-byte prefix keys and range scans. `depth`
    /// is the packed-position slots per key (1..=4); `max_scan` is neighbours
    /// inspected per direction.
    BTree {
        max_scan: usize,
        depth: usize,
        nice_length: usize,
    },
}

/// Per-level configuration of the encoder and the on-disk format.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LevelConfig {
    /// If `true`, tokens are arithmetic-coded (FORMAT.md §3). If `false`,
    /// tokens are written byte-aligned (FORMAT.md §2.4).
    pub(crate) entropy: bool,
    /// Whether the parser peeks one position ahead before committing a match.
    pub(crate) lazy: bool,
    /// Which match finder to use at this level.
    pub(crate) finder: FinderConfig,
}

const LEVELS: [LevelConfig; 9] = [
    // L1: raw, greedy, head-only hash (lz4 -1 class).
    LevelConfig {
        entropy: false,
        lazy: false,
        finder: FinderConfig::Head {
            hash_bits: 16,
        },
    },
    // L2: raw, greedy, shallow chain.
    LevelConfig {
        entropy: false,
        lazy: false,
        finder: FinderConfig::Chain {
            hash_bits: 16,
            max_chain: 8,
            nice_length: 16,
        },
    },
    // L3: raw, lazy, deeper chain.
    LevelConfig {
        entropy: false,
        lazy: true,
        finder: FinderConfig::Chain {
            hash_bits: 16,
            max_chain: 32,
            nice_length: 32,
        },
    },
    // L4: entropy, lazy, deeper chain.
    LevelConfig {
        entropy: true,
        lazy: true,
        finder: FinderConfig::Chain {
            hash_bits: 17,
            max_chain: 64,
            nice_length: 48,
        },
    },
    // L5: entropy, lazy, even deeper chain.
    LevelConfig {
        entropy: true,
        lazy: true,
        finder: FinderConfig::Chain {
            hash_bits: 17,
            max_chain: 128,
            nice_length: 64,
        },
    },
    // L6 (default): entropy, lazy, balanced chain.
    LevelConfig {
        entropy: true,
        lazy: true,
        finder: FinderConfig::Chain {
            hash_bits: 18,
            max_chain: 256,
            nice_length: 96,
        },
    },
    // L7: entropy, lazy, B+tree — widens search vs L6 without short-circuits.
    LevelConfig {
        entropy: true,
        lazy: true,
        finder: FinderConfig::BTree {
            max_scan: 256,
            depth: 4,
            nice_length: MAX_MATCH,
        },
    },
    // L8: entropy, lazy, deeper B+tree scan.
    LevelConfig {
        entropy: true,
        lazy: true,
        finder: FinderConfig::BTree {
            max_scan: 1024,
            depth: 4,
            nice_length: MAX_MATCH,
        },
    },
    // L9: entropy, lazy, widest B+tree scan (exhaustive, no short-circuit).
    LevelConfig {
        entropy: true,
        lazy: true,
        finder: FinderConfig::BTree {
            max_scan: 4096,
            depth: 4,
            nice_length: MAX_MATCH,
        },
    },
];

/// Returns the configuration for the given compression level.
///
/// # Panics
///
/// Panics if `level` is outside `1..=9`.
pub(crate) const fn level_config(level: u8) -> LevelConfig {
    assert!(level >= MIN_LEVEL && level <= MAX_LEVEL, "compression level must be 1..=9");
    LEVELS[(level - 1) as usize]
}

// ===========================================================================
// Types
// ===========================================================================

/// One LZ77 token: zero or more literal bytes followed by an optional
/// back-reference.
///
/// * `literal_len` — number of literal bytes that precede the match (0..=255).
/// * `match_len`   — length of the back-reference copy. **0** means no match;
///   valid matches are in the range `3..=255`.
/// * `match_distance` — how far back to look (1..=65 535). Ignored when
///   `match_len == 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Sequence {
    pub(crate) literal_len: u8,
    pub(crate) match_len: u8,
    pub(crate) match_distance: u16,
}

assert_eq_size!(Sequence, u32);

/// A [`Sequence`] together with the literal bytes it owns.
///
/// The valid literal region is `&self.literals[..self.seq.literal_len as usize]`.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Token {
    pub(crate) seq: Sequence,
    pub(crate) literals: [u8; 255],
}

// ===========================================================================
// Shared match-finder helpers
// ===========================================================================

/// Returns the length of the common prefix of `a` and `b`, up to `max`.
fn common_prefix_len(a: &[u8], b: &[u8], max: usize) -> usize {
    a.iter().zip(b.iter()).take(max).take_while(|(x, y)| x == y).count()
}

/// 3-byte multiplicative hash projected into `1 << (32 - shift)` buckets.
///
/// Caller must ensure `bytes.len() >= 3`.
#[inline]
fn hash3(bytes: &[u8], shift: u32) -> usize {
    let v = (u32::from(bytes[0]) << 16) | (u32::from(bytes[1]) << 8) | u32::from(bytes[2]);
    (v.wrapping_mul(HASH_MUL) >> shift) as usize
}

// ===========================================================================
// Match finders
// ===========================================================================

/// Dispatch wrapper over the three match-finder implementations. Static
/// dispatch via `match` keeps the hot path free of indirect calls.
enum MatchFinder {
    Head(HeadFinder),
    Chain(ChainFinder),
    BTree(BTreeFinder),
}

impl MatchFinder {
    fn new(cfg: FinderConfig) -> Self {
        match cfg {
            FinderConfig::Head {
                hash_bits,
            } => Self::Head(HeadFinder::new(hash_bits)),
            FinderConfig::Chain {
                hash_bits,
                max_chain,
                nice_length,
            } => Self::Chain(ChainFinder::new(hash_bits, max_chain, nice_length)),
            FinderConfig::BTree {
                max_scan,
                depth,
                nice_length,
            } => Self::BTree(BTreeFinder::new(max_scan, depth, nice_length)),
        }
    }

    fn insert(&mut self, pos: usize, buf: &[u8], base: usize) {
        match self {
            Self::Head(f) => f.insert(pos, buf, base),
            Self::Chain(f) => f.insert(pos, buf, base),
            Self::BTree(f) => f.insert(pos, buf, base),
        }
    }

    fn find_match(&self, pos: usize, buf: &[u8], base: usize) -> (usize, usize) {
        match self {
            Self::Head(f) => f.find_match(pos, buf, base),
            Self::Chain(f) => f.find_match(pos, buf, base),
            Self::BTree(f) => f.find_match(pos, buf, base),
        }
    }

    fn clear(&mut self) {
        match self {
            Self::Head(f) => f.clear(),
            Self::Chain(f) => f.clear(),
            Self::BTree(f) => f.clear(),
        }
    }
}

// ---------------------------------------------------------------------------
// HeadFinder (L1)
// ---------------------------------------------------------------------------

/// Hash-table head only — one u32 absolute position per hash bucket.
struct HeadFinder {
    head: Vec<u32>,
    shift: u32,
}

impl HeadFinder {
    fn new(hash_bits: u32) -> Self {
        Self {
            head: vec![EMPTY_POS; 1usize << hash_bits],
            shift: 32 - hash_bits,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn insert(&mut self, pos: usize, buf: &[u8], base: usize) {
        let buf_pos = pos - base;
        if buf_pos + 2 >= buf.len() {
            return;
        }
        let h = hash3(&buf[buf_pos..], self.shift);
        self.head[h] = pos as u32;
    }

    fn find_match(&self, pos: usize, buf: &[u8], base: usize) -> (usize, usize) {
        let available = base + buf.len();
        let remaining = available - pos;
        let max_len = MAX_MATCH.min(remaining);
        if max_len < MIN_MATCH {
            return (0, 0);
        }
        let buf_pos = pos - base;
        if buf_pos + 2 >= buf.len() {
            return (0, 0);
        }

        let h = hash3(&buf[buf_pos..], self.shift);
        let cand = self.head[h];
        if cand == EMPTY_POS {
            return (0, 0);
        }
        let cand_pos = cand as usize;
        let min_pos = pos.saturating_sub(u16::MAX as usize);
        if cand_pos < min_pos || cand_pos >= pos || cand_pos < base {
            return (0, 0);
        }

        let ci = cand_pos - base;
        let len = common_prefix_len(&buf[ci..], &buf[buf_pos..], max_len);
        if len < MIN_MATCH {
            return (0, 0);
        }
        (pos - cand_pos, len)
    }

    fn clear(&mut self) {
        self.head.fill(EMPTY_POS);
    }
}

// ---------------------------------------------------------------------------
// ChainFinder (L2–L6)
// ---------------------------------------------------------------------------

/// Hash table plus a `WINDOW_SIZE`-entry chain indexed by `pos % WINDOW_SIZE`.
/// Each chain slot stores the previous position that hashed to the same
/// bucket. Stale links (from positions since evicted by modular reuse) are
/// filtered by the in-window check and the `max_chain` bound.
struct ChainFinder {
    head: Vec<u32>,
    chain: Vec<u32>,
    shift: u32,
    max_chain: usize,
    nice_length: usize,
}

impl ChainFinder {
    fn new(hash_bits: u32, max_chain: usize, nice_length: usize) -> Self {
        Self {
            head: vec![EMPTY_POS; 1usize << hash_bits],
            chain: vec![EMPTY_POS; WINDOW_SIZE],
            shift: 32 - hash_bits,
            max_chain,
            nice_length,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn insert(&mut self, pos: usize, buf: &[u8], base: usize) {
        let buf_pos = pos - base;
        if buf_pos + 2 >= buf.len() {
            return;
        }
        let h = hash3(&buf[buf_pos..], self.shift);
        let prev = self.head[h];
        self.chain[pos % WINDOW_SIZE] = prev;
        self.head[h] = pos as u32;
    }

    fn find_match(&self, pos: usize, buf: &[u8], base: usize) -> (usize, usize) {
        let available = base + buf.len();
        let remaining = available - pos;
        let max_len = MAX_MATCH.min(remaining);
        if max_len < MIN_MATCH {
            return (0, 0);
        }
        let buf_pos = pos - base;
        if buf_pos + 2 >= buf.len() {
            return (0, 0);
        }

        let h = hash3(&buf[buf_pos..], self.shift);
        let min_pos = pos.saturating_sub(u16::MAX as usize);
        let cutoff = self.nice_length.min(max_len);

        let mut best_len = MIN_MATCH - 1;
        let mut best_dist = 0;
        let mut cand = self.head[h];

        for _ in 0..self.max_chain {
            if cand == EMPTY_POS {
                break;
            }
            let cand_pos = cand as usize;
            if cand_pos < min_pos || cand_pos < base || cand_pos >= pos {
                break;
            }
            let ci = cand_pos - base;
            let len = common_prefix_len(&buf[ci..], &buf[buf_pos..], max_len);
            if len > best_len {
                best_len = len;
                best_dist = pos - cand_pos;
                if best_len >= cutoff {
                    break;
                }
            }
            cand = self.chain[cand_pos % WINDOW_SIZE];
        }

        if best_len >= MIN_MATCH {
            (best_dist, best_len)
        } else {
            (0, 0)
        }
    }

    fn clear(&mut self) {
        self.head.fill(EMPTY_POS);
        self.chain.fill(EMPTY_POS);
    }
}

// ---------------------------------------------------------------------------
// BTreeFinder (L7–L9)
// ---------------------------------------------------------------------------

/// Builds a big-endian u128 key from the first 16 bytes at `buf[offset..]`.
///
/// Big-endian ensures numeric order equals lexicographic order, so range
/// queries on the B+tree naturally find lexicographically adjacent suffixes.
fn make_key(buf: &[u8], offset: usize) -> u128 {
    let avail = buf.len() - offset;
    let take = avail.min(16);
    let mut bytes = [0u8; 16];
    bytes[..take].copy_from_slice(&buf[offset..offset + take]);
    u128::from_be_bytes(bytes)
}

/// Returns the number of leading bytes two u128 keys share.
const fn common_prefix_u128(a: u128, b: u128) -> usize {
    let xor = a ^ b;
    if xor == 0 {
        return 16;
    }
    (xor.leading_zeros() / 8) as usize
}

/// Extracts up to 4 packed u32 positions from a u128 value.
///
/// Positions are packed newest-at-LSB: on each insert the value is shifted
/// left by 32 and the new position is `OR`ed into the low 32 bits.
#[allow(clippy::cast_possible_truncation)]
const fn unpack_position(packed: u128, index: usize) -> u32 {
    (packed >> (index * 32)) as u32
}

/// Packs a new position into a u128 value, shifting older positions up.
const fn pack_position(packed: u128, pos: u32) -> u128 {
    (packed << 32) | pos as u128
}

/// B+tree match finder using u128 prefix keys.
///
/// Each key maps to a u128 packing up to 4 recent absolute positions. Stale
/// entries (outside the sliding window) are filtered at match time by the
/// distance check; the tree is cleared at Sonnet boundaries.
struct BTreeFinder {
    index: OSBTreeMap<u128, u128>,
    depth: usize,
    max_scan: usize,
    nice_length: usize,
}

impl BTreeFinder {
    const fn new(max_scan: usize, depth: usize, nice_length: usize) -> Self {
        Self {
            index: OSBTreeMap::new(),
            depth,
            max_scan,
            nice_length,
        }
    }

    /// Records `pos` in the index under its 16-byte prefix key.
    #[allow(clippy::cast_possible_truncation)]
    fn insert(&mut self, pos: usize, buf: &[u8], base: usize) {
        let buf_pos = pos - base;
        if buf_pos + 2 >= buf.len() {
            return;
        }

        let key = make_key(buf, buf_pos);
        let pos32 = pos as u32;

        let packed = self.index.get(&key).copied().unwrap_or(0);
        self.index.insert(key, pack_position(packed, pos32));
    }

    /// Finds the best match at `pos`.
    fn find_match(&self, pos: usize, buf: &[u8], base: usize) -> (usize, usize) {
        let available = base + buf.len();
        let remaining = available - pos;
        let max_len = MAX_MATCH.min(remaining);

        if max_len < MIN_MATCH {
            return (0, 0);
        }

        let buf_pos = pos - base;
        let min_pos = pos.saturating_sub(u16::MAX as usize);
        let key = make_key(buf, buf_pos);
        let cutoff = self.nice_length.min(max_len);

        let mut best_len = MIN_MATCH - 1;
        let mut best_dist = 0;

        // Returns true when a "good enough" match is found so the caller
        // can short-circuit. Good enough means best_len >= cutoff.
        let try_packed = |packed: u128, best_len: &mut usize, best_dist: &mut usize| -> bool {
            for i in 0..self.depth {
                let cand = unpack_position(packed, i) as usize;
                if cand < min_pos || cand >= pos || cand < base {
                    continue;
                }
                let ci = cand - base;
                let len = common_prefix_len(&buf[ci..], &buf[buf_pos..], max_len);
                if len > *best_len {
                    *best_len = len;
                    *best_dist = pos - cand;
                    if *best_len >= cutoff {
                        return true;
                    }
                }
            }
            false
        };

        // Exact key match.
        if let Some(&packed) = self.index.get(&key) {
            if try_packed(packed, &mut best_len, &mut best_dist) {
                return (best_dist, best_len);
            }
        }

        // Forward range scan (keys > our key).
        let mut checked = 0;
        for (&candidate_key, &packed) in self.index.range((std::ops::Bound::Excluded(key), std::ops::Bound::Unbounded))
        {
            if checked >= self.max_scan {
                break;
            }
            if common_prefix_u128(key, candidate_key) < MIN_MATCH {
                break;
            }
            if try_packed(packed, &mut best_len, &mut best_dist) {
                return (best_dist, best_len);
            }
            checked += 1;
        }

        // Backward range scan (keys < our key).
        checked = 0;
        for (&candidate_key, &packed) in self.index.range(..key).rev() {
            if checked >= self.max_scan {
                break;
            }
            if common_prefix_u128(key, candidate_key) < MIN_MATCH {
                break;
            }
            if try_packed(packed, &mut best_len, &mut best_dist) {
                return (best_dist, best_len);
            }
            checked += 1;
        }

        if best_len >= MIN_MATCH {
            (best_dist, best_len)
        } else {
            (0, 0)
        }
    }

    fn clear(&mut self) {
        self.index.clear();
    }
}

// ===========================================================================
// Encoder
// ===========================================================================

/// Streaming LZ77 encoder.
///
/// Data is fed incrementally via [`feed`](Self::feed). Call
/// [`next`](Self::next) to pull tokens. Call [`finish`](Self::finish)
/// to signal end of input, then drain remaining tokens with `next`.
///
/// Parsing strategy (greedy vs lazy) and match-finder choice are
/// determined by the compression level passed to [`new`](Self::new).
pub(crate) struct Encoder {
    buf: Vec<u8>,
    base: usize,
    pos: usize,
    lit_start: usize,
    lit_len: usize,
    finder: MatchFinder,
    finished: bool,
    lazy: bool,
}

impl Encoder {
    /// Creates a new encoder at the given compression level (1–9).
    ///
    /// # Panics
    ///
    /// Panics if `level` is not in `1..=9`.
    pub(crate) fn new(level: u8) -> Self {
        let cfg = level_config(level);
        Self {
            buf: Vec::with_capacity(2 * WINDOW_SIZE + MAX_MATCH),
            base: 0,
            pos: 0,
            lit_start: 0,
            lit_len: 0,
            finder: MatchFinder::new(cfg.finder),
            finished: false,
            lazy: cfg.lazy,
        }
    }

    /// Appends input data to the encoder's internal buffer.
    ///
    /// # Panics
    ///
    /// Panics if called after [`finish`](Self::finish).
    pub(crate) fn feed(&mut self, data: &[u8]) {
        assert!(!self.finished, "cannot feed after finish");
        self.maybe_compact();
        self.buf.extend_from_slice(data);
    }

    /// Signals that no more input will be fed.
    pub(crate) const fn finish(&mut self) {
        self.finished = true;
    }

    /// Processes all remaining buffered data and returns the resulting tokens.
    pub(crate) fn flush(&mut self) -> Vec<Token> {
        self.finished = true;
        let mut tokens = Vec::new();
        while let Some(token) = self.next() {
            tokens.push(token);
        }
        self.finished = false;
        tokens
    }

    /// Resets encoder state for a new Sonnet (preserves level).
    pub(crate) fn reset(&mut self) {
        self.buf.clear();
        self.base = 0;
        self.pos = 0;
        self.lit_start = 0;
        self.lit_len = 0;
        self.finder.clear();
        self.finished = false;
    }

    /// Returns `true` if there are pending literal bytes not yet emitted.
    pub(crate) const fn has_pending_literals(&self) -> bool {
        self.lit_len > 0
    }

    /// Returns input bytes that have been fed but not yet emitted as tokens.
    pub(crate) fn unconsumed_input(&self) -> Vec<u8> {
        let start = self.lit_start - self.base;
        self.buf[start..].to_vec()
    }

    /// Returns the next token, or `None` if more data is needed.
    pub(crate) fn next(&mut self) -> Option<Token> {
        self.next_inner(MAX_LITERALS)
    }

    /// Like [`next`](Self::next), but caps the literal run at `max_lits`.
    pub(crate) fn next_capped(&mut self, max_lits: u8) -> Option<Token> {
        let cap = max_lits as usize;

        if self.lit_len > 0 && cap == 0 {
            return None;
        }

        if self.lit_len > cap && cap > 0 {
            return Some(self.emit_partial_literals(cap));
        }

        self.next_inner(cap)
    }

    fn next_inner(&mut self, max_lits: usize) -> Option<Token> {
        let available = self.base + self.buf.len();

        loop {
            if self.lit_len >= max_lits && self.lit_len > 0 {
                return Some(self.emit_literals());
            }

            if self.pos >= available {
                if self.finished {
                    break;
                }
                return None;
            }

            if !self.finished && available - self.pos < LOOKAHEAD_MIN {
                return None;
            }

            let (mut dist, mut mlen) = self.finder.find_match(self.pos, &self.buf, self.base);
            self.finder.insert(self.pos, &self.buf, self.base);

            if mlen >= MIN_MATCH {
                // Lazy matching (level-gated): peek at pos+1 for a strictly
                // longer match. If found, emit pos as a literal and use pos+1's
                // match instead. Looking further (lazy2+) was tested and gave
                // <= 0.2 pts of ratio for a 10-17% encode slowdown.
                if self.lazy {
                    let next_pos = self.pos + 1;
                    if next_pos < available && (self.finished || available - next_pos >= LOOKAHEAD_MIN) {
                        let (dist2, mlen2) = self.finder.find_match(next_pos, &self.buf, self.base);
                        if mlen2 > mlen {
                            self.pos = next_pos;
                            self.lit_len += 1;
                            self.finder.insert(self.pos, &self.buf, self.base);
                            dist = dist2;
                            mlen = mlen2;
                        }
                    }
                }

                for i in 1..mlen {
                    self.finder.insert(self.pos + i, &self.buf, self.base);
                }
                self.pos += mlen;
                return Some(self.emit_match(dist, mlen));
            }

            self.pos += 1;
            self.lit_len += 1;
        }

        if self.lit_len > 0 {
            Some(self.emit_literals())
        } else {
            None
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn emit_literals(&mut self) -> Token {
        debug_assert!(self.lit_len > 0 && self.lit_len <= MAX_LITERALS);
        let mut literals = [0u8; 255];
        let start = self.lit_start - self.base;
        literals[..self.lit_len].copy_from_slice(&self.buf[start..start + self.lit_len]);

        let token = Token {
            seq: Sequence {
                literal_len: self.lit_len as u8,
                match_len: 0,
                match_distance: 0,
            },
            literals,
        };
        self.lit_start = self.pos;
        self.lit_len = 0;
        token
    }

    #[allow(clippy::cast_possible_truncation)]
    fn emit_partial_literals(&mut self, count: usize) -> Token {
        debug_assert!(count > 0 && count <= self.lit_len);
        let mut literals = [0u8; 255];
        let start = self.lit_start - self.base;
        literals[..count].copy_from_slice(&self.buf[start..start + count]);

        let token = Token {
            seq: Sequence {
                literal_len: count as u8,
                match_len: 0,
                match_distance: 0,
            },
            literals,
        };
        self.lit_start += count;
        self.lit_len -= count;
        token
    }

    #[allow(clippy::cast_possible_truncation)]
    fn emit_match(&mut self, dist: usize, mlen: usize) -> Token {
        let mut literals = [0u8; 255];
        if self.lit_len > 0 {
            let start = self.lit_start - self.base;
            literals[..self.lit_len].copy_from_slice(&self.buf[start..start + self.lit_len]);
        }

        let token = Token {
            seq: Sequence {
                literal_len: self.lit_len as u8,
                match_len: mlen as u8,
                match_distance: dist as u16,
            },
            literals,
        };
        self.lit_start = self.pos;
        self.lit_len = 0;
        token
    }

    fn maybe_compact(&mut self) {
        if self.buf.len() <= 2 * WINDOW_SIZE {
            return;
        }
        let abs_keep_from = self.pos.saturating_sub(WINDOW_SIZE);
        if abs_keep_from <= self.base {
            return;
        }
        let keep_from = abs_keep_from - self.base;
        self.buf.drain(..keep_from);
        self.base += keep_from;
    }
}

// ===========================================================================
// Decoder
// ===========================================================================

/// LZ77 decoder that reconstructs raw bytes from a stream of [`Sequence`]
/// tokens.
pub(crate) struct Decoder {
    window: Vec<u8>,
    pos: usize,
}

impl Decoder {
    pub(crate) fn new() -> Self {
        Self {
            window: vec![0; WINDOW_SIZE],
            pos: 0,
        }
    }

    pub(crate) fn decode(&mut self, seq: Sequence, literals: &[u8], output: &mut Vec<u8>) {
        debug_assert_eq!(literals.len(), seq.literal_len as usize);

        for &b in literals {
            output.push(b);
            self.window[self.pos % WINDOW_SIZE] = b;
            self.pos += 1;
        }

        let match_len = seq.match_len as usize;
        if match_len > 0 {
            let dist = seq.match_distance as usize;
            debug_assert!(dist > 0, "match_distance must be > 0 when match_len > 0");
            for _ in 0..match_len {
                let src = self.pos.wrapping_sub(dist) % WINDOW_SIZE;
                let b = self.window[src];
                output.push(b);
                self.window[self.pos % WINDOW_SIZE] = b;
                self.pos += 1;
            }
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::bench::lipsum_bytes;

    fn token_literals(token: &Token) -> &[u8] {
        &token.literals[..token.seq.literal_len as usize]
    }

    fn roundtrip_codec(data: &[u8], level: u8) -> Vec<u8> {
        let mut enc = Encoder::new(level);
        enc.feed(data);
        enc.finish();

        let mut dec = Decoder::new();
        let mut decoded = Vec::new();

        while let Some(token) = enc.next() {
            dec.decode(token.seq, token_literals(&token), &mut decoded);
        }

        assert_eq!(data, &decoded[..]);
        decoded
    }

    #[test]
    fn decode_empty() {
        roundtrip_codec(b"", DEFAULT_LEVEL);
    }

    #[test]
    fn lipsum_roundtrip_compresses() {
        let data = lipsum_bytes(65_536);

        let mut enc = Encoder::new(DEFAULT_LEVEL);
        enc.feed(&data);
        enc.finish();

        let mut dec = Decoder::new();
        let mut decoded = Vec::with_capacity(data.len());
        let mut seq_count = 0_usize;
        let mut total_literal_bytes = 0_usize;

        while let Some(token) = enc.next() {
            seq_count += 1;
            total_literal_bytes += token.seq.literal_len as usize;
            dec.decode(token.seq, token_literals(&token), &mut decoded);
        }

        let encoded_size = seq_count * size_of::<Sequence>() + total_literal_bytes;

        assert_eq!(data, &decoded[..]);
        assert!(encoded_size < data.len());
    }

    #[test]
    fn larger_than_window() {
        let data = lipsum_bytes(128 * 1024);
        roundtrip_codec(&data, DEFAULT_LEVEL);
    }

    #[test]
    fn multi_feed_roundtrip() {
        let data = lipsum_bytes(4096);

        let mut enc = Encoder::new(DEFAULT_LEVEL);
        let mut dec = Decoder::new();
        let mut decoded = Vec::new();

        for chunk in data.chunks(100) {
            enc.feed(chunk);
            while let Some(token) = enc.next() {
                dec.decode(token.seq, token_literals(&token), &mut decoded);
            }
        }
        enc.finish();
        while let Some(token) = enc.next() {
            dec.decode(token.seq, token_literals(&token), &mut decoded);
        }

        assert_eq!(data, &decoded[..]);
    }

    #[test]
    fn next_capped_basic() {
        let data = lipsum_bytes(1024);
        let mut enc = Encoder::new(DEFAULT_LEVEL);
        enc.feed(&data);
        enc.finish();

        let mut dec = Decoder::new();
        let mut decoded = Vec::new();

        while let Some(token) = enc.next_capped(10) {
            assert!(token.seq.literal_len <= 10);
            dec.decode(token.seq, token_literals(&token), &mut decoded);
        }

        assert_eq!(data, decoded);
    }

    #[test]
    fn next_capped_zero_with_pending() {
        // Feed enough data for the encoder to accumulate literals, then call
        // next_capped(0) — should return None when lit_len > 0.
        #[allow(clippy::cast_possible_truncation)]
        let data: Vec<u8> = (0..500u32).map(|i| ((i * 137 + 43) % 256) as u8).collect();
        let mut enc = Encoder::new(DEFAULT_LEVEL);
        enc.feed(&data);

        // Process until the encoder stalls on insufficient lookahead.
        while enc.next().is_some() {}

        // The encoder should have pending literals from the lookahead zone.
        assert!(enc.has_pending_literals());
        // next_capped(0) with pending literals must return None.
        assert!(enc.next_capped(0).is_none());
    }

    #[test]
    fn unconsumed_input_basic() {
        let data = lipsum_bytes(2048);
        let mut enc = Encoder::new(DEFAULT_LEVEL);
        enc.feed(&data);

        let mut consumed = 0;
        while let Some(token) = enc.next() {
            consumed += token.seq.literal_len as usize + token.seq.match_len as usize;
        }

        let unconsumed = enc.unconsumed_input();
        assert_eq!(consumed + unconsumed.len(), data.len());
    }

    proptest! {
        #[test]
        fn roundtrip(data in prop::collection::vec(any::<u8>(), 0..4_096)) {
            roundtrip_codec(&data, DEFAULT_LEVEL);
        }

        #[test]
        fn all_levels_roundtrip(
            data in prop::collection::vec(any::<u8>(), 1..1_024),
            level in MIN_LEVEL..=MAX_LEVEL,
        ) {
            roundtrip_codec(&data, level);
        }

        #[test]
        fn sequence_invariants(data in prop::collection::vec(any::<u8>(), 0..4_096)) {
            let mut enc = Encoder::new(DEFAULT_LEVEL);
            enc.feed(&data);
            enc.finish();

            let mut total_bytes = 0_usize;

            while let Some(token) = enc.next() {
                prop_assert_eq!(token_literals(&token).len(), token.seq.literal_len as usize);
                prop_assert!(
                    token.seq.match_len == 0 || token.seq.match_len as usize >= MIN_MATCH,
                    "bad match_len: {}", token.seq.match_len,
                );
                if token.seq.match_len > 0 {
                    prop_assert!(token.seq.match_distance > 0);
                }
                let consumed = token.seq.literal_len as usize + token.seq.match_len as usize;
                prop_assert!(consumed > 0, "zero-byte token");
                total_bytes += consumed;
            }

            prop_assert_eq!(total_bytes, data.len());
        }

        #[test]
        fn streaming_roundtrip(data in prop::collection::vec(any::<u8>(), 0..4_096)) {
            let mut enc = Encoder::new(DEFAULT_LEVEL);
            let mut dec = Decoder::new();
            let mut decoded = Vec::new();

            let chunk_size = 100;
            for chunk in data.chunks(chunk_size) {
                enc.feed(chunk);
                while let Some(token) = enc.next() {
                    dec.decode(token.seq, token_literals(&token), &mut decoded);
                }
            }
            enc.finish();
            while let Some(token) = enc.next() {
                dec.decode(token.seq, token_literals(&token), &mut decoded);
            }

            prop_assert_eq!(&data[..], &decoded[..]);
        }

        #[test]
        fn capped_roundtrip(
            data in prop::collection::vec(any::<u8>(), 1..4_096),
            cap in 1u8..=255,
        ) {
            let mut enc = Encoder::new(DEFAULT_LEVEL);
            enc.feed(&data);
            enc.finish();

            let mut dec = Decoder::new();
            let mut decoded = Vec::new();

            while let Some(token) = enc.next_capped(cap) {
                prop_assert!(token.seq.literal_len <= cap);
                dec.decode(token.seq, token_literals(&token), &mut decoded);
            }

            prop_assert_eq!(&data[..], &decoded[..]);
        }
    }

    #[test]
    fn emit_partial_literals_via_capped() {
        // Feed random (incompressible) bytes so the encoder accumulates many
        // literals without finding matches, then use next_capped with a small
        // cap to trigger emit_partial_literals.
        let data: Vec<u8> = (0..200u8).map(|i| i.wrapping_mul(137).wrapping_add(43)).collect();
        let mut enc = Encoder::new(DEFAULT_LEVEL);
        enc.feed(&data);
        enc.finish();

        let mut dec = Decoder::new();
        let mut decoded = Vec::new();

        // Use cap=3 to force partial literal emission.
        while let Some(token) = enc.next_capped(3) {
            assert!(token.seq.literal_len <= 3);
            dec.decode(token.seq, token_literals(&token), &mut decoded);
        }

        assert_eq!(data, decoded);
    }

    #[test]
    fn streaming_next_returns_none_without_finish() {
        // Feed a small amount of data without calling finish; next() should
        // return None because there isn't enough lookahead.
        let mut enc = Encoder::new(DEFAULT_LEVEL);
        enc.feed(b"hello");
        assert!(enc.next().is_none());

        // Feed empty data — exercises pos >= available without finish.
        let mut enc2 = Encoder::new(DEFAULT_LEVEL);
        enc2.feed(b"");
        assert!(enc2.next().is_none());
    }

    #[test]
    fn next_capped_triggers_partial_emission() {
        // Accumulate more literals than the cap, then call next_capped to
        // trigger emit_partial_literals.
        let mut enc = Encoder::new(DEFAULT_LEVEL);
        // Use enough unique data that the encoder accumulates many literals.
        #[allow(clippy::cast_possible_truncation)]
        let data: Vec<u8> = (0..500u32).map(|i| ((i * 137 + 43) % 256) as u8).collect();
        enc.feed(&data);

        // Process some tokens with normal next() — leaves pending literals
        // when lookahead runs out.
        while enc.next().is_some() {}

        // Now lit_len > 0 from the lookahead boundary. Call next_capped with
        // a cap smaller than lit_len to trigger emit_partial_literals.
        enc.finish();
        if enc.has_pending_literals() {
            // This should trigger emit_partial_literals(5).
            let token = enc.next_capped(5);
            assert!(token.is_some());
            let t = token.unwrap();
            assert!(t.seq.literal_len <= 5);
        }

        // Drain the rest and verify roundtrip.
        let mut dec = Decoder::new();
        let mut decoded = Vec::new();

        // Re-do from scratch to get all tokens for verification.
        let mut enc2 = Encoder::new(DEFAULT_LEVEL);
        enc2.feed(&data);
        enc2.finish();
        while let Some(token) = enc2.next() {
            dec.decode(token.seq, token_literals(&token), &mut decoded);
        }
        assert_eq!(data, decoded);
    }

    #[test]
    fn maybe_compact_early_return() {
        // Feed > 2*WINDOW_SIZE bytes in one call, then feed again. The second
        // feed triggers maybe_compact with buf > 128K but pos=0, so
        // abs_keep_from=0 <= base=0, exercising the early return.
        let mut enc = Encoder::new(DEFAULT_LEVEL);
        let big = vec![0x42u8; 150_000];
        enc.feed(&big);
        // Second feed triggers maybe_compact on the large buffer.
        enc.feed(b"extra");
        enc.finish();
        let mut dec = Decoder::new();
        let mut decoded = Vec::new();
        while let Some(token) = enc.next() {
            dec.decode(token.seq, token_literals(&token), &mut decoded);
        }
        let mut expected = big;
        expected.extend_from_slice(b"extra");
        assert_eq!(expected, decoded);
    }

    #[test]
    fn compact_triggers_on_large_incremental_feed() {
        // Feed data in chunks totaling >128 KiB to trigger maybe_compact().
        let data = lipsum_bytes(200 * 1024);
        let mut enc = Encoder::new(DEFAULT_LEVEL);
        let mut dec = Decoder::new();
        let mut decoded = Vec::new();

        for chunk in data.chunks(4096) {
            enc.feed(chunk);
            while let Some(token) = enc.next() {
                dec.decode(token.seq, token_literals(&token), &mut decoded);
            }
        }
        enc.finish();
        while let Some(token) = enc.next() {
            dec.decode(token.seq, token_literals(&token), &mut decoded);
        }

        assert_eq!(data, &decoded[..]);
    }

    // -------------------------------------------------------------------
    // Finder-specific unit tests (targeted coverage for edge cases beyond
    // what the all_levels_roundtrip proptest already exercises).
    // -------------------------------------------------------------------

    #[test]
    fn head_finder_empty_returns_no_match() {
        let finder = HeadFinder::new(12);
        let buf = b"hello world";
        assert_eq!(finder.find_match(0, buf, 0), (0, 0));
    }

    #[test]
    fn head_finder_basic_match() {
        // Insert position 0 then find at position 6 with a shared prefix.
        let buf = b"abcdefabcdef";
        let mut finder = HeadFinder::new(12);
        finder.insert(0, buf, 0);
        finder.insert(1, buf, 0);
        finder.insert(2, buf, 0);
        let (dist, mlen) = finder.find_match(6, buf, 0);
        assert_eq!(dist, 6);
        assert!(mlen >= MIN_MATCH);
    }

    #[test]
    fn head_finder_short_buf_returns_no_match() {
        // buf_pos + 2 >= buf.len() path — too few bytes left for a 3-byte hash.
        let buf = b"ab";
        let finder = HeadFinder::new(12);
        assert_eq!(finder.find_match(0, buf, 0), (0, 0));
    }

    #[test]
    fn chain_finder_walks_multiple_candidates() {
        // Three identical 3-byte prefixes preceding the query position.
        let buf = b"abcZabcYabcXabcQextra";
        let mut finder = ChainFinder::new(12, 16, 32);
        // Insert every position strictly before the query.
        for i in 0..12 {
            finder.insert(i, buf, 0);
        }
        let (dist, mlen) = finder.find_match(12, buf, 0);
        assert!(mlen >= MIN_MATCH);
        assert!(dist == 4 || dist == 8 || dist == 12);
    }

    #[test]
    fn chain_finder_nice_length_shortcircuits() {
        // A highly repetitive buffer; nice_length=3 means the chain walk
        // stops at the first 3+ byte match without walking the full chain.
        let buf = [b'a'; 64];
        let mut finder = ChainFinder::new(12, 1000, 3);
        for i in 0..32 {
            finder.insert(i, &buf, 0);
        }
        let (dist, mlen) = finder.find_match(32, &buf, 0);
        assert!(mlen >= MIN_MATCH);
        assert!(dist > 0);
    }

    #[test]
    fn chain_finder_empty_returns_no_match() {
        let finder = ChainFinder::new(12, 8, 16);
        let buf = b"hello world";
        assert_eq!(finder.find_match(0, buf, 0), (0, 0));
    }

    #[test]
    fn chain_finder_clear_wipes_state() {
        let buf = b"abcabcabc";
        let mut finder = ChainFinder::new(12, 8, 16);
        for i in 0..7 {
            finder.insert(i, buf, 0);
        }
        finder.clear();
        assert_eq!(finder.find_match(6, buf, 0), (0, 0));
    }

    #[test]
    fn level_config_dispatches_correct_finder() {
        assert!(matches!(level_config(1).finder, FinderConfig::Head { .. }));
        for l in 2..=6 {
            assert!(matches!(level_config(l).finder, FinderConfig::Chain { .. }));
        }
        for l in 7..=9 {
            assert!(matches!(level_config(l).finder, FinderConfig::BTree { .. }));
        }
    }
}
