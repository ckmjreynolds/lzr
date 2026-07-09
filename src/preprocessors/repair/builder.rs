//! Space-efficient exact Re-Pair grammar construction.
//!
//! Implements the frequency-based Re-Pair builder of
//!
//! > Philip Bille, Inge Li Gørtz, Nicola Prezza. *Space-Efficient Re-Pair Compression.*
//! > Data Compression Conference (DCC) 2017. arXiv:1611.01479.
//! > (vendored at `docs/papers/bille-gortz-prezza-2017-space-efficient-repair.pdf`)
//!
//! **Theorem 1(i):** compute the exact Re-Pair grammar in O(n/ε) expected time using (1+ε)n + √n
//! words of working space *on top of the text*. The classic Larsson–Moffat layout instead needs
//! ~5n words (a doubly-linked sequence plus a per-occurrence priority queue); on gigabyte inputs
//! (enwik9) that is ~24–28 GB and does not fit in memory, which is why we use this algorithm.
//!
//! The idea that buys the space: never store per-occurrence linked lists or a per-occurrence heap.
//! The sequence is a single rewritable array (adjacency is array order; replaced cells become
//! *blanks*), and occurrences of a pair are located, on demand, by **sorting the live positions by
//! their character pair** (`sort_pairs`, paper §4.2 / Lemma 3) so each pair's occurrences form a
//! contiguous run. A small priority queue holds only the currently-frequent pairs.
//!
//! This module maps to the paper as follows:
//! - [`Builder::compute_repair`] → Algorithm 2 (`compute_repair`), the phase loop.
//! - [`Builder::substitution_round`] → Algorithm 1 (`substitution_round`): replace the max pair,
//!   decrement neighbour pair frequencies, then `synchronize`.
//! - [`Builder::synchronize`] → the `Q.synchronize` operation (paper §3.1): re-group a pair's
//!   position interval to discover the new pairs a substitution created and refresh the pair's own
//!   interval. The amortization invariant `F > L/2` (paper §3.3) triggers it (Algorithm 1, line 13).
//! - [`Builder::sort_pairs`] → `sort_pairs(T)`.
//! - [`Builder::compact_text`] → `compact_text(T)` (delete blanks between phases).
//!
//! **Adaptations for this crate's capped variant.** Symbol ids are capped at `vocab_cap` (passed in
//! per build; the byte-BPE tokenizer uses `256`), so working symbols fit in a `Vec<u16>` (2
//! bytes/cell) rather than the paper's `⌈log n⌉`-bit words — and we need no inline run-length blank
//! encoding, using a single [`BLANK`] sentinel (`u16::MAX`, free because ids are `< vocab_cap`). The
//! [`pair_key`] packing assumes `vocab_cap < 2^16`. Position arrays are `u32`.
//! The paper's separate high-/low-frequency queues are unified into one **seed-capped** queue: each
//! phase seeds the pairs whose frequency is at least a threshold τ chosen so at most
//! [`Builder::seed_cap`] pairs are seeded (bounding the `εn` space term), drains them in exact
//! max-first order, then re-sorts. Correctness is identical to strict greedy Re-Pair; the seed cap
//! only decides how much work each sort batches.

use std::collections::{BinaryHeap, HashMap};

/// Blank sentinel marking a cell vacated by a replacement. Free as a working value because ids are
/// `< vocab_cap ≤ 256`, well below `u16::MAX`.
const BLANK: u16 = u16::MAX;

/// Sort key that groups positions by the *current* live pair starting at `p`: the packed
/// `⟨text[i], text[j]⟩` pair (with `j` the next non-blank position), tie-broken by `p` for a stable
/// order. Blank cells and positions with no live successor sort to the end (`u64::MAX`). The 16-bit
/// symbols are packed with a `<< 16` shift into a `u64` so distinct pairs never alias. A free
/// function, not a method, so it can be called from a `sort_unstable_by_key` closure while `tp` is
/// mutably borrowed (only `text` need be borrowed). The key is transient (used only inside the sort
/// closure), so it never inflates the persistent `u32` position arrays.
fn pair_key(text: &[u16], p: u32) -> (u64, u32) {
    let i = p as usize;
    if text[i] == BLANK {
        return (u64::MAX, p);
    }
    let mut j = i + 1;
    while j < text.len() && text[j] == BLANK {
        j += 1;
    }
    if j >= text.len() {
        (u64::MAX, p)
    } else {
        ((u64::from(text[i]) << 16) | u64::from(text[j]), p)
    }
}

/// A queue entry for one currently-tracked pair: the paper's `⟨P_ab, L_ab, F_ab⟩`.
///
/// `tp[p .. p + l]` contains all `f` occurrences of the pair (plus possibly stale/blank entries, so
/// `l ≥ f`). The amortization invariant kept at `max()` time is `f > l / 2` (paper §3.3).
#[derive(Clone, Copy, Debug)]
struct Entry {
    /// Start of the pair's interval in `tp`.
    p: u32,
    /// Interval length (`≥ f`).
    l: u32,
    /// True occurrence count.
    f: u32,
}

/// A lazily-revalidated max-queue item: the pair and the frequency it had when pushed. Ordered so
/// [`BinaryHeap`] yields the highest frequency, ties broken by the smallest pair — the deterministic
/// merge order shared with the reference implementation.
#[derive(Clone, Copy, PartialEq, Eq)]
struct HeapItem {
    freq: u32,
    pair: (u16, u16),
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // Max-heap on frequency; on ties the lexicographically *smaller* pair is treated as greater.
        self.freq.cmp(&other.freq).then_with(|| other.pair.cmp(&self.pair))
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Working state for one space-efficient Re-Pair run over a single input.
struct Builder {
    /// Rewritable sequence; live cells hold ids `< vocab_cap` (`u16`, since the byte-BPE vocab is
    /// capped at 256), vacated cells hold [`BLANK`].
    text: Vec<u16>,
    /// Live positions (those currently starting a pair), sorted by pair per phase. Paper's `TP`.
    tp: Vec<u32>,
    /// Priority queue of tracked pairs: `⟨P,L,F⟩` per pair plus a lazy max-heap over `(F, pair)`.
    q: HashMap<(u16, u16), Entry>,
    heap: BinaryHeap<HeapItem>,
    /// Rules in creation order; rule `k` has id `first_rule_id + k`.
    rules: Vec<(u16, u16)>,
    /// Next rule id to assign; the build stops when it reaches [`Builder::vocab_cap`].
    next_id: u16,
    /// Vocabulary ceiling: symbol ids run `0..vocab_cap`, so the build stops once `next_id` reaches
    /// it. Passed in per build (`256` for the byte-BPE tokenizer) rather than a fixed constant.
    vocab_cap: u16,
    /// Upper bound on pairs seeded per phase (bounds the `εn` space term).
    seed_cap: usize,
    /// Count of entries driven dead (`f < 2`) by [`Builder::decrease`] since the last
    /// [`Builder::compact_q`]. Pruning is O(q); firing it only once dead entries reach a
    /// fraction of `q` (rather than every merge) keeps that scan amortized O(1) per merge.
    dead: usize,
    /// Reused neighbour-pair scratch for [`Builder::substitution_round`]. Held on the builder
    /// (like the heap rebuilds) so its capacity survives across the millions of rounds a build
    /// runs, rather than allocating and freeing a fresh `Vec` per round.
    touched: Vec<(u16, u16)>,
}

impl Builder {
    fn new(symbols: Vec<u16>, n_terminals: u16, vocab_cap: u16) -> Self {
        let n = symbols.len();
        // Seed at most ~n/128 pairs per phase (the `εn` term, ε ≈ 1/128), never fewer than 4096 so
        // small inputs finish in a single phase. A smaller cap trades more sort phases (time) for less
        // queue/heap memory, which (with the reused-capacity rebuilds below) is what keeps enwik9 under
        // the memory target.
        let seed_cap = (n / 128).max(4096);
        Self {
            // Move (not copy) the input in as the working text — avoids a second n-word buffer.
            text: symbols,
            tp: Vec::new(),
            q: HashMap::new(),
            heap: BinaryHeap::new(),
            rules: Vec::new(),
            next_id: n_terminals,
            vocab_cap,
            seed_cap,
            dead: 0,
            touched: Vec::new(),
        }
    }

    /// First live position strictly right of `i`, or `None`. Blanks are skipped (linearly — the
    /// paper's constant-time run-skip encoding is a time optimization not needed for correctness).
    fn succ(&self, i: usize) -> Option<usize> {
        let mut j = i + 1;
        while j < self.text.len() {
            if self.text[j] != BLANK {
                return Some(j);
            }
            j += 1;
        }
        None
    }

    /// First live position strictly left of `i`, or `None`.
    fn pred(&self, i: usize) -> Option<usize> {
        let mut j = i;
        while j > 0 {
            j -= 1;
            if self.text[j] != BLANK {
                return Some(j);
            }
        }
        None
    }

    /// The live pair starting at position `i`, or `None` if `i` is blank or has no live successor.
    fn pair_at(&self, i: usize) -> Option<(u16, u16)> {
        if self.text[i] == BLANK {
            return None;
        }
        let j = self.succ(i)?;
        Some((self.text[i], self.text[j]))
    }

    /// Paper's `sort_pairs(T)`: rebuild `tp` with every live position, sorted by character pair. In
    /// place (`sort_unstable` is O(1) extra space), so working space stays one position array.
    #[expect(clippy::cast_possible_truncation, reason = "positions < text.len() <= u32::MAX (build_grammar guard)")]
    fn sort_pairs(&mut self) {
        self.tp.clear();
        // Reserve exactly the upper bound so pushes never trigger a doubling realloc — the transient
        // 2× buffer would be several GB at enwik9 scale.
        self.tp.reserve(self.text.len());
        for i in 0..self.text.len() {
            if self.pair_at(i).is_some() {
                self.tp.push(i as u32);
            }
        }
        // Borrow-safe: `pair_key` reads only `text`, so `tp` can stay mutably borrowed by the sort.
        let text = &self.text;
        self.tp.sort_unstable_by_key(|&p| pair_key(text, p));
    }

    /// Push a fresh heap item for a pair's current frequency (lazy: stale items are filtered at pop).
    fn heap_push(&mut self, pair: (u16, u16), freq: u32) {
        self.heap.push(HeapItem {
            freq,
            pair,
        });
    }

    /// Rebuild the lazy heap from the live entries once stale items dominate, capping heap memory at
    /// O(tracked pairs). [`Builder::max_pair`] revalidates every item against `q`, so dropping the
    /// stale duplicates never changes which pair is selected.
    fn compact_heap(&mut self) {
        if self.heap.len() <= 4 * self.q.len().max(1024) {
            return;
        }
        self.rebuild_heap_from_q();
    }

    /// Refill the heap from the live queue entries, reusing its existing allocation (`clear` keeps
    /// capacity) so repeated compaction does not churn the allocator — the churn otherwise fragments
    /// the heap into gigabytes of retained-but-unused pages at enwik9 scale.
    fn rebuild_heap_from_q(&mut self) {
        let mut heap = core::mem::take(&mut self.heap).into_vec();
        heap.clear();
        heap.extend(self.q.iter().filter(|(_, e)| e.f >= 2).map(|(&pair, e)| HeapItem {
            freq: e.f,
            pair,
        }));
        self.heap = BinaryHeap::from(heap);
    }

    /// Drop dead entries (`f < 2`) from `q` once they dominate. Such entries are kept only transiently
    /// so the round that decremented them can `synchronize` their interval; afterwards they are dead
    /// weight. Without this the map keeps *every* pair touched during a phase (millions of dead
    /// entries at enwik9 scale) rather than only the live ones — the dominant avoidable allocation.
    /// A pair that later reappears is re-inserted fresh by `synchronize`, so pruning is safe.
    ///
    /// Gated on the `dead` count, not on `q.len()`: the live pairs alone can exceed any fixed
    /// `seed_cap`-based threshold (a large input drains in a single phase), so a `q.len()` gate fires
    /// this O(q) retain-and-rebuild on *every* merge — quadratic over the phase. Firing only once dead
    /// entries reach a quarter of `q` bounds the map to ~4/3 the live set and makes the scan amortized
    /// O(1) per merge (each scan reclaims ≥ ¼ of `q`).
    fn compact_q(&mut self) {
        if self.dead * 4 < self.q.len() {
            return;
        }
        self.q.retain(|_, e| e.f >= 2);
        self.dead = 0;
        // Heap items may now point at removed pairs; rebuild (reusing capacity) from the pruned map.
        self.rebuild_heap_from_q();
    }

    /// Return the highest-frequency tracked pair whose frequency is at least `floor`, or `None`.
    /// Revalidates lazily against `q` so stale heap items are discarded.
    fn max_pair(&mut self, floor: u32) -> Option<(u16, u16)> {
        while let Some(item) = self.heap.peek().copied() {
            match self.q.get(&item.pair) {
                Some(entry) if entry.f == item.freq => {
                    return (entry.f >= floor).then_some(item.pair);
                }
                // Stale (frequency changed or pair removed): drop and continue.
                _ => {
                    let _ = self.heap.pop();
                }
            }
        }
        None
    }

    /// Decrease a tracked pair's frequency by one (paper `Q.decrease`), refreshing the heap. The
    /// entry is *kept* even at frequency 0: its position interval is still needed so Pass 2's
    /// `synchronize` can discover the new pairs created at those positions (a removed entry would lose
    /// them). Stale low-frequency entries are simply skipped by [`Builder::max_pair`] and cleared when
    /// the phase ends.
    fn decrease(&mut self, pair: (u16, u16)) {
        if let Some(entry) = self.q.get_mut(&pair) {
            entry.f = entry.f.saturating_sub(1);
            if entry.f >= 2 {
                let freq = entry.f;
                self.heap_push(pair, freq);
            } else if entry.f == 1 {
                // Just crossed below the live floor (`2 → 1`): count it so `compact_q` can prune
                // once dead entries dominate. `synchronize` only ever inserts `f ≥ 2`, so this is
                // the only path that creates a prunable entry. A pair later revived by `synchronize`
                // is over-counted, which only makes pruning fire *earlier* — safe.
                self.dead += 1;
            }
        }
    }

    /// Paper's `Q.synchronize(ab)`: re-group the interval `tp[p .. p+l]` by *current* pair so the new
    /// pairs a substitution created are discovered, and `ab`'s own interval is refreshed to exactly
    /// its remaining occurrences. Every distinct live pair found is a new pair (a fresh id was just
    /// introduced) or `ab` itself, so upserting its `⟨P,L,F⟩` is safe (no interval belongs elsewhere).
    #[expect(clippy::cast_possible_truncation, reason = "run/index < text.len() <= u32::MAX (build_grammar guard)")]
    fn synchronize(&mut self, ab: (u16, u16)) {
        let Some(entry) = self.q.get(&ab).copied() else {
            return;
        };
        let (start, len) = (entry.p as usize, entry.l as usize);
        // Re-sort the interval by current pair (dead/blank positions to the end).
        let text = &self.text;
        self.tp[start..start + len].sort_unstable_by_key(|&p| pair_key(text, p));
        // Drop `ab`'s old entry; each contiguous run of a live pair below becomes a fresh entry.
        let _ = self.q.remove(&ab);
        let mut off = 0usize;
        while off < len {
            let pos = self.tp[start + off];
            let Some(pair) = self.pair_at(pos as usize) else {
                break; // reached the blank/dead tail
            };
            let mut run = 1usize;
            while off + run < len {
                let p2 = self.tp[start + off + run];
                if self.pair_at(p2 as usize) == Some(pair) {
                    run += 1;
                } else {
                    break;
                }
            }
            let f = run as u32;
            if f >= 2 {
                let _ = self.q.insert(
                    pair,
                    Entry {
                        p: (start + off) as u32,
                        l: f,
                        f,
                    },
                );
                self.heap_push(pair, f);
            }
            off += run;
        }
    }

    /// Paper's Algorithm 1: substitute every occurrence of the max pair `ab` with a fresh id.
    /// Returns `false` if the vocabulary cap is hit (no id available), signalling the build to stop.
    #[expect(
        clippy::many_single_char_names,
        reason = "a,b are the pair; x the new id; i,j text positions — standard Re-Pair notation"
    )]
    fn substitution_round(&mut self, ab: (u16, u16)) -> bool {
        if self.next_id >= self.vocab_cap {
            return false;
        }
        let (a, b) = ab;
        let x = self.next_id;
        self.rules.push((a, b));
        self.next_id += 1;
        let entry = self.q[&ab];
        let (start, len) = (entry.p as usize, entry.l as usize);

        // Pass 1 (Algorithm 1, lines 4–10): replace occurrences, decrement each neighbour pair's
        // frequency, and remember the neighbours so their intervals can be refreshed below. `touched`
        // is taken from the builder so its capacity is reused across rounds (restored at the end).
        let mut touched = std::mem::take(&mut self.touched);
        touched.clear();
        for off in 0..len {
            let i = self.tp[start + off] as usize;
            if self.text[i] != a {
                continue; // stale: blanked or consumed by an overlapping replacement
            }
            let Some(j) = self.succ(i) else {
                continue;
            };
            if self.text[j] != b {
                continue; // no longer `ab` here (overlap already consumed the `b`)
            }
            let x_pos = self.pred(i);
            let y_pos = self.succ(j);
            if let Some(xp) = x_pos {
                let xa = (self.text[xp], a);
                self.decrease(xa);
                touched.push(xa);
            }
            if let Some(yp) = y_pos {
                let by = (b, self.text[yp]);
                self.decrease(by);
                touched.push(by);
            }
            self.text[i] = x;
            self.text[j] = BLANK;
        }

        // Pass 2 (Algorithm 1, lines 11–14, applied symmetrically to both neighbours): refresh any
        // touched pair whose amortization invariant `F > L/2` was broken by the decrements. This both
        // restores the invariant `max_pair` relies on and discovers the new left-context pairs `xX`.
        // (The paper writes only the `xA` side; `By` needs the same treatment or a decremented right
        // neighbour can violate the invariant and hide a higher-frequency pair.)
        touched.sort_unstable();
        touched.dedup();
        for &pair in &touched {
            if let Some(e) = self.q.get(&pair).copied() {
                if e.f <= e.l / 2 {
                    self.synchronize(pair);
                }
            }
        }
        // Return the buffer (emptied) to the builder so the next round reuses its capacity.
        touched.clear();
        self.touched = touched;

        // Line 15–16: discover the new right-context pairs `Xy` inside `ab`'s interval, then drop `ab`.
        self.synchronize(ab);
        let _ = self.q.remove(&ab);
        true
    }

    /// Paper's `compact_text(T)`: drop blanks, shrinking the text to its live symbols.
    fn compact_text(&mut self) {
        let mut w = 0usize;
        for r in 0..self.text.len() {
            if self.text[r] != BLANK {
                self.text[w] = self.text[r];
                w += 1;
            }
        }
        self.text.truncate(w);
    }

    /// Visit each run of a distinct pair in the sorted `tp`, calling `f(pair, start_offset, freq)`
    /// for runs of length ≥ 2. Shared by both passes of [`Builder::seed_queue`].
    fn for_each_run(&self, mut f: impl FnMut((u16, u16), usize, u32)) {
        let mut off = 0usize;
        while off < self.tp.len() {
            let Some(pair) = self.pair_at(self.tp[off] as usize) else {
                break; // blank/dead tail
            };
            let mut run = 1usize;
            while off + run < self.tp.len() && self.pair_at(self.tp[off + run] as usize) == Some(pair) {
                run += 1;
            }
            if run >= 2 {
                f(pair, off, u32::try_from(run).unwrap_or(u32::MAX));
            }
            off += run;
        }
    }

    /// Seed the queue from the freshly sorted `tp`: keep the pairs whose frequency is at least the
    /// threshold τ that admits at most `seed_cap` pairs. Returns τ (`0` if nothing is mergeable); the
    /// drain then processes pairs while `max ≥ τ`. Two passes over the runs (τ via a bounded min-heap,
    /// then seeding) avoid materializing an entry for every distinct pair — that vector would be the
    /// largest non-`tp` allocation at scale.
    #[expect(clippy::cast_possible_truncation, reason = "offsets < tp.len() <= u32::MAX (build_grammar guard)")]
    fn seed_queue(&mut self) -> u32 {
        self.q.clear();
        self.heap.clear();
        // Pass 1: τ = the `seed_cap`-th largest run frequency (≥ 2). A size-capped min-heap keeps only
        // the top `seed_cap` frequencies, so its memory is O(seed_cap), not O(distinct pairs).
        let cap = self.seed_cap;
        let mut top: BinaryHeap<core::cmp::Reverse<u32>> = BinaryHeap::new();
        let mut any = false;
        self.for_each_run(|_, _, freq| {
            any = true;
            if top.len() < cap {
                top.push(core::cmp::Reverse(freq));
            } else if matches!(top.peek(), Some(&core::cmp::Reverse(min)) if freq > min) {
                let _ = top.pop();
                top.push(core::cmp::Reverse(freq));
            }
        });
        if !any {
            return 0;
        }
        let tau = if top.len() < cap {
            2
        } else {
            top.peek().map_or(2, |&core::cmp::Reverse(m)| m).max(2)
        };
        // Pass 2: seed every pair with frequency ≥ τ.
        let mut seeds: Vec<((u16, u16), u32, u32)> = Vec::new();
        self.for_each_run(|pair, off, freq| {
            if freq >= tau {
                seeds.push((pair, off as u32, freq));
            }
        });
        for (pair, p, f) in seeds {
            let _ = self.q.insert(
                pair,
                Entry {
                    p,
                    l: f,
                    f,
                },
            );
            self.heap_push(pair, f);
        }
        tau
    }

    /// Paper's Algorithm 2: the phase loop. Each phase sorts, seeds the frequent pairs, drains them
    /// in exact max-first order (creating rules), then compacts. Stops when no pair repeats or the
    /// vocabulary cap is reached.
    fn compute_repair(&mut self) {
        while self.next_id < self.vocab_cap {
            self.sort_pairs();
            let tau = self.seed_queue();
            if tau == 0 {
                break; // nothing repeats
            }
            while let Some(ab) = self.max_pair(tau) {
                if !self.substitution_round(ab) {
                    return; // cap hit mid-drain
                }
                self.compact_q();
                self.compact_heap();
            }
            self.compact_text();
        }
    }

    /// The surviving sequence: the live symbols in order.
    fn collect_sequence(&self) -> Vec<u16> {
        self.text.iter().copied().filter(|&s| s != BLANK).collect()
    }
}

/// Build the Re-Pair grammar of `symbols` (terminal ids in `0..n_terminals`) space-efficiently,
/// minting symbol ids up to `vocab_cap` (exclusive). Returns the rules in creation order (rule `k`
/// has id `n_terminals + k`) and the reduced sequence, exactly the shape
/// [`super::RepairTokenizer::forward`] consumes.
pub(super) fn build_grammar(symbols: Vec<u16>, n_terminals: u16, vocab_cap: u16) -> (Vec<(u16, u16)>, Vec<u16>) {
    debug_assert!(n_terminals <= vocab_cap, "caller must admit at least the terminals");
    if symbols.len() < 2 {
        return (Vec::new(), symbols);
    }
    let mut builder = Builder::new(symbols, n_terminals, vocab_cap);
    builder.compute_repair();
    let sequence = builder.collect_sequence();
    (builder.rules, sequence)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::*;

    /// A large vocabulary cap used to exercise the builder's full id range in tests (the production
    /// tokenizer caps at 256, but the builder is generic up to the `pair_key` packing limit, one below
    /// the `BLANK` sentinel).
    const TEST_CAP: u16 = 60_000;

    /// Map arbitrary bytes to compact terminal ids `0..t` (as `forward` does), returning the symbol
    /// vector and terminal count.
    fn terminals(data: &[u8]) -> (Vec<u16>, u16) {
        let mut present = [false; 256];
        for &d in data {
            present[usize::from(d)] = true;
        }
        let mut id = [0u16; 256];
        let mut t = 0u16;
        for b in 0..=255usize {
            if present[b] {
                id[b] = t;
                t += 1;
            }
        }
        (data.iter().map(|&d| id[usize::from(d)]).collect(), t)
    }

    /// Expand a grammar back to its terminal-id sequence (the inverse of Re-Pair), via an explicit
    /// stack. Proves the grammar is lossless without depending on any particular tie-break.
    fn expand(rules: &[(u16, u16)], sequence: &[u16], n_terminals: u16) -> Vec<u16> {
        let mut out = Vec::new();
        let mut stack = Vec::new();
        for &s in sequence {
            stack.push(s);
            while let Some(sym) = stack.pop() {
                if sym < n_terminals {
                    out.push(sym);
                } else {
                    let (l, r) = rules[usize::from(sym - n_terminals)];
                    stack.push(r);
                    stack.push(l);
                }
            }
        }
        out
    }

    /// True occurrence count of every pair by scanning the live text (test oracle).
    fn true_counts(b: &Builder) -> HashMap<(u16, u16), u32> {
        let mut c = HashMap::new();
        let mut prev: Option<u16> = None;
        for &s in &b.text {
            if s == BLANK {
                continue;
            }
            if let Some(p) = prev {
                *c.entry((p, s)).or_insert(0) += 1;
            }
            prev = Some(s);
        }
        c
    }

    /// Drive the builder step by step and assert every merge is a *maximum-frequency* pair — the
    /// defining property of Re-Pair. (The exact tie-break is implementation-defined: the paper defers
    /// some new pairs via the `F ≤ L/2` amortization trigger, so `Q.max()` returns *a* max pair, not
    /// necessarily the lexicographically smallest one — any choice yields a valid Re-Pair grammar.)
    fn assert_valid_repair(symbols: &[u16], t: u16) {
        let mut b = Builder::new(symbols.to_vec(), t, TEST_CAP);
        loop {
            if b.next_id >= TEST_CAP {
                break;
            }
            b.sort_pairs();
            let tau = b.seed_queue();
            if tau == 0 {
                break;
            }
            while let Some(ab) = b.max_pair(tau) {
                let counts = true_counts(&b);
                let max_freq = counts.values().copied().filter(|&c| c >= 2).max().unwrap_or(0);
                let chosen = counts.get(&ab).copied().unwrap_or(0);
                assert_eq!(chosen, max_freq, "merged {ab:?} at freq {chosen}, but max is {max_freq}");
                if !b.substitution_round(ab) {
                    return;
                }
            }
            b.compact_text();
        }
    }

    proptest! {
        /// The grammar must be lossless: expanding it reproduces the input exactly.
        #[test]
        fn lossless(data in prop::collection::vec(any::<u8>(), 0..2048)) {
            let (symbols, t) = terminals(&data);
            let (rules, sequence) = build_grammar(symbols.clone(), t, TEST_CAP);
            prop_assert_eq!(expand(&rules, &sequence, t), symbols);
        }

        /// Lossless on a tiny alphabet (heavy overlap, e.g. `aaaa`).
        #[test]
        fn lossless_small_alphabet(data in prop::collection::vec(0u8..3, 0..2048)) {
            let (symbols, t) = terminals(&data);
            let (rules, sequence) = build_grammar(symbols.clone(), t, TEST_CAP);
            prop_assert_eq!(expand(&rules, &sequence, t), symbols);
        }

        /// Every merge is a maximum-frequency pair — the builder computes a genuine Re-Pair grammar.
        #[test]
        fn valid_repair(data in prop::collection::vec(any::<u8>(), 0..2048)) {
            let (symbols, t) = terminals(&data);
            assert_valid_repair(&symbols, t);
        }

        /// Same, on the binary alphabet that maximizes overlap and `a == b` merges.
        #[test]
        fn valid_repair_binary(data in prop::collection::vec(0u8..2, 0..4096)) {
            let (symbols, t) = terminals(&data);
            assert_valid_repair(&symbols, t);
        }
    }

    #[test]
    fn brute_force_all_short_binary_valid_and_lossless() {
        // Exhaustively check every binary string up to length 18: each grammar is lossless and a
        // valid Re-Pair. Catches overlap/synchronize corner cases proptest might miss.
        for len in 1..=18usize {
            for bits in 0u32..(1u32 << len) {
                let data: Vec<u8> = (0..len).map(|k| u8::try_from((bits >> k) & 1).unwrap()).collect();
                let (symbols, t) = terminals(&data);
                let (rules, sequence) = build_grammar(symbols.clone(), t, TEST_CAP);
                assert_eq!(expand(&rules, &sequence, t), symbols, "not lossless: {data:?}");
                assert_valid_repair(&symbols, t);
            }
        }
    }

    #[test]
    fn known_small_cases_lossless() {
        for input in [b"".as_slice(), b"a", b"aa", b"aaaa", b"abab", b"abracadabra", b"mississippi"] {
            let (symbols, t) = terminals(input);
            let (rules, sequence) = build_grammar(symbols.clone(), t, TEST_CAP);
            assert_eq!(expand(&rules, &sequence, t), symbols, "input {input:?}");
        }
    }

    #[test]
    fn tiny_seed_cap_stays_valid_and_lossless() {
        // A tiny seed cap forces many sort/compact phases; the phase batching must still yield a
        // valid, lossless grammar.
        let data = b"the quick brown fox the quick brown fox the quick brown fox jumps".repeat(4);
        let (symbols, t) = terminals(&data);
        assert_valid_repair(&symbols, t); // uses default cap
        let mut builder = Builder::new(symbols.clone(), t, TEST_CAP);
        builder.seed_cap = 3;
        builder.compute_repair();
        assert_eq!(expand(&builder.rules, &builder.collect_sequence(), t), symbols);
    }

    #[test]
    fn compresses_repetitive_input() {
        let data = vec![0u16; 4096];
        let (rules, sequence) = build_grammar(data.clone(), 1, TEST_CAP);
        assert!(sequence.len() < data.len(), "repetitive input should shrink");
        assert_eq!(expand(&rules, &sequence, 1), data);
    }

    #[test]
    fn vocab_cap_is_respected() {
        // A tiny cap must bound the total symbol count: terminals + minted rules never exceed it.
        let data = b"the quick brown fox jumps over the lazy dog. ".repeat(64);
        let (symbols, t) = terminals(&data);
        let cap = 300u16;
        let (rules, _sequence) = build_grammar(symbols, t, cap);
        assert!(usize::from(t) + rules.len() <= usize::from(cap), "vocab exceeded the {cap} cap");
    }
}
