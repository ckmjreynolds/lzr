//! Bitwise logistic context-mixing byte codec — the v7 logit-domain mixer.
//!
//! Where [`crate::cmix`] blends per-order count distributions *linearly*, this
//! codec mixes in the **logit (stretch) domain**, the PAQ/lpaq design. Each
//! byte is coded as eight binary decisions, MSB-first. For each bit, every
//! model emits a probability that the bit is 1; those are mapped through
//! `stretch(p) = ln(p / (1-p))`, combined by a learned weighted sum, and mapped
//! back with `squash`. The mixer weights are trained online by gradient descent
//! on coding loss (`w += lr * (y - p) * stretch(p_k)`).
//!
//! Why logit-domain: a linear blend of agreeing predictors cannot exceed the
//! most confident input — if two models both say 0.9, the blend stays ~0.9.
//! In the logit domain their evidence *adds*, so agreement sharpens the
//! prediction past either input. That sharpening is where the bits are.
//!
//! Models: orders `0..=MAX_ORDER` (byte-history count contexts) plus an
//! optional **match model** (long-range exact repeats). The match model is
//! what separates the `lmatch` codec from the pure-mixer `lmix` codec — both
//! share this module; a `use_match` flag selects whether the match input is
//! live. When off, the match mixer input is held at a neutral `0`, so `lmix`
//! is bit-identical to the no-match design.
//!
//! Per-bit context: within a byte we track a tree node — start at 1, and after
//! each coded bit `node = (node << 1) | bit`, so `node` walks `1..=255` over
//! the eight decisions. Each order model is keyed on `(context_k, node)`.
//!
//! Determinism / `L(D) = 0`: every lookup is keyed on content (packed
//! `(context, node)`, or a verified history match), so the `HashMap` seed is
//! irrelevant; all float work runs the same code path on both sides; encoder
//! and decoder replay the same stream and rebuild byte-identical state. No
//! weights are shipped.
//!
//! Archive layout: a 64-bit length prefix, then the AC bitstream — each of the
//! `8 * len` bits coded against a two-symbol CDF `[0, c0, TOTAL]`.

use anyhow::Result;

use crate::ac::{AcDecoder, AcEncoder, TOTAL};
use crate::bits::{BitReader, BitWriter};
use crate::codec::{Codec, Decomposition};

/// Highest context order, in bytes. Models orders `0..=MAX_ORDER`.
const MAX_ORDER: usize = 6;
/// Number of order models.
const N_ORDERS: usize = MAX_ORDER + 1;
/// High-order byte contexts (longer than the `u64`-packable orders, so hashed
/// from the 16-byte `ctx_hi`). Orders past 16 can't be represented (the
/// register holds 16 bytes) and silently alias order-16. A quick-panel set
/// sweep was flat — [8,12,16] 1.706, [7,10,13,16] 1.705, [8,16,24] 1.709 (the
/// 24 aliases 16, a wasted arm), [7,9,12,16,24] 1.705 — so the gain from longer
/// byte context saturates by ~order-16; [8,12,16] is the simplest set with no
/// redundancy.
const HI_ORDERS: [usize; 3] = [8, 12, 16];
/// Number of high-order context models.
const N_HI: usize = HI_ORDERS.len();
/// Number of sparse (non-contiguous byte) context models.
const N_SPARSE: usize = 2;
/// Mixer inputs: orders, match, two words, high orders, word-trigram, sparse.
const N_INPUTS: usize = N_ORDERS + 4 + N_HI + N_SPARSE;
/// Index of the match model's mixer input.
const MATCH_IN: usize = N_ORDERS;
/// Index of the W0 (current partial word) mixer input.
const WORD0_IN: usize = N_ORDERS + 1;
/// Index of the W1 (previous word + current partial word) mixer input.
const WORD1_IN: usize = N_ORDERS + 2;
/// Base index of the high-order context mixer inputs (`HI_IN .. HI_IN + N_HI`).
const HI_IN: usize = N_ORDERS + 3;
/// Index of the W2 (previous two words + current partial word) mixer input.
const WORD2_IN: usize = N_ORDERS + 3 + N_HI;
/// Base index of the sparse-context mixer inputs.
const SPARSE_IN: usize = N_ORDERS + 4 + N_HI;
/// Log2 size of each per-context bit-predictor table. The tables are fixed-size
/// and direct-mapped (hash the key, tolerate collisions), so memory is bounded
/// regardless of input length — unlike a growing `HashMap`, which is unbounded
/// and blew past the 10 GB judging limit. 2^25 slots × 8 B = 256 MiB/table.
const CTX_BITS: u32 = 25;
/// Number of slots per context table.
const CTX_SIZE: usize = 1 << CTX_BITS;
/// Set associativity: slots per bucket. A key hashes to a bucket and may take
/// any slot in it, so up to `CTX_WAYS` colliding contexts coexist before any
/// eviction — fewer conflict misses than direct-mapped at the same memory.
/// 8 × 8 B = 64 B = one cache line, so scanning a bucket is one cache miss.
const CTX_WAYS: usize = 8;
/// Bits of the hash selecting the bucket (the rest of the table is the ways).
const BUCKET_BITS: u32 = CTX_BITS - CTX_WAYS.ilog2();
/// Mixer gradient-descent step size on coding loss. An enwik8 quick-panel
/// sweep bottomed out near 0.002 (0.05 → 2.059, 0.02 → 1.971, 0.004 → 1.936,
/// 0.002 → 1.933); below that the curve is flat.
const MIX_LR: f64 = 0.004;
/// Initial per-input mixer weight (before any online training).
const INIT_W: f64 = 0.3;
/// Number of mixer weight sets, selected by (previous byte, bit position).
const N_WSETS: usize = 256 * 8;
/// Floor on a bit-predictor's adaptive learning rate, so a well-observed
/// context still tracks local drift instead of freezing.
const RATE_FLOOR: f64 = 1.0 / 256.0;
/// Cap on a bit-predictor's observation count (caps the slowest rate).
const N_CAP: u8 = 255;
/// Clamp bounds on a probability before `stretch`, so the logit stays finite
/// (and the mixer never sees an infinite input → `NaN` weight).
const P_LO: f64 = 1e-6;
const P_HI: f64 = 1.0 - 1e-6;

/// Bytes of matching context required to seed a match. An enwik8 quick-panel
/// sweep favored short seeds — 8 → 1.839, 6 → 1.839, 5 → 1.831, 4 → 1.817,
/// 3 → 1.805, 2 → 1.807 — bottoming at 3 (earlier seeding catches repeats
/// sooner; the adaptive length-bucketed confidence map handles the higher
/// false-match rate a short seed brings).
const MIN_MATCH: usize = 3;
/// Log2 of the match hash-table size (entries map context-hash → last
/// position). 2^22 entries × 4 B = 16 MiB.
const MATCH_BITS: u32 = 22;
/// Match-length buckets for the adaptive confidence map (a longer verified
/// match predicts more reliably; the map learns how reliable each bucket is).
const MATCH_BUCKETS: usize = 64;
/// Empty slot sentinel in the match table. enwik9 (1e9) < `u32::MAX`, so a
/// real position never collides with it.
const MATCH_EMPTY: u32 = u32::MAX;

/// SSE/APM: number of interpolation knots spanning the stretch domain.
const APM_KNOTS: usize = 33;
/// SSE/APM: half-width of the stretch domain the knots span. Mixer logits past
/// `±APM_S_MAX` clamp to the extreme knot (probabilities already near 0/1).
const APM_S_MAX: f64 = 12.0;
/// SSE/APM: learning rate for nudging knots toward the observed bit. A quick-
/// panel sweep was flat around it (0.008 → 1.775, 0.02 → 1.773, 0.05 → 1.774,
/// 0.1 → 1.781).
const APM_RATE: f64 = 0.02;
/// SSE/APM: weight on the recalibrated probability when blending it with the
/// raw mixer probability (damps APM noise; the rest is the mixer's own output).
/// Quick-panel sweep: 0.5 → 1.777, 0.7 → 1.773, 0.85 → 1.774, 1.0 → 1.783, so
/// trusting the APM ~70% beats both the cautious blend and the pure map.
const APM_BLEND: f64 = 0.7;

/// Word model: rolling-hash seed for an empty word (between words / in markup).
const WORD_SEED: u64 = 0xcbf2_9ce4_8422_2325;
/// Word model: rolling-hash multiplier folding each letter into the word hash.
const WORD_MUL: u64 = 0x0000_0100_0000_01B3;
/// Word model: golden-ratio prime mixing context hashes with the byte node and
/// the previous word with the current one.
const WORD_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

/// Logit of a probability: `ln(p / (1-p))`, clamped to keep it finite.
#[inline]
fn stretch(p: f64) -> f64 {
    let p = p.clamp(P_LO, P_HI);
    (p / (1.0 - p)).ln()
}

/// Inverse of [`stretch`]: the logistic squashing function.
#[inline]
fn squash(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Fold a word-context hash together with the within-byte tree `node` into a
/// lookup key. `node < 512` so distinct `(ctx, node)` pairs stay distinct.
#[inline]
const fn word_key(ctx_hash: u64, node: usize) -> u64 {
    ctx_hash.wrapping_mul(WORD_MIX).wrapping_add(node as u64)
}

/// Combine the previous word's hash with the current partial word's (the W1
/// context).
#[inline]
const fn word_combine(prev: u64, cur: u64) -> u64 {
    prev.wrapping_mul(WORD_MIX).wrapping_add(cur)
}

/// Hash a context key into its bucket's base slot index plus a 16-bit checksum,
/// taken from adjacent well-mixed high bits of one multiplicative hash. Two keys
/// in the same bucket almost always differ in checksum, so a collision is
/// detected rather than silently blended.
#[inline]
#[allow(clippy::cast_possible_truncation)]
const fn locate(key: u64) -> (usize, u16) {
    let h = key.wrapping_mul(WORD_MIX);
    let bucket = (h >> (64 - BUCKET_BITS)) as usize;
    let check = (h >> (64 - BUCKET_BITS - 16)) as u16;
    (bucket * CTX_WAYS, check)
}

/// Read a hashed context's logit: the predictor's `stretch(p)` from the slot in
/// its bucket whose checksum confirms this key, else a neutral `0` (no slot
/// holds this context — don't trust the others' stats).
#[inline]
#[allow(clippy::needless_range_loop)]
fn ctx_logit(table: &[BitModel], key: u64) -> f64 {
    let (base, check) = locate(key);
    for s in 0..CTX_WAYS {
        let bm = table[base + s];
        if bm.check == check {
            return stretch(f64::from(bm.p));
        }
    }
    0.0
}

/// Update a hashed context toward bit `y`. If the bucket already holds this key,
/// update it; otherwise claim the least-trained slot (lowest count) with fresh
/// stats — so a colliding context evicts the cheapest entry to lose rather than
/// corrupting a well-trained one.
#[inline]
#[allow(clippy::needless_range_loop)]
fn update_ctx(table: &mut [BitModel], key: u64, y: f64) {
    let (base, check) = locate(key);
    let mut victim = base;
    let mut min_n = u8::MAX;
    for s in 0..CTX_WAYS {
        if table[base + s].check == check {
            table[base + s].update(y);
            return;
        }
        if table[base + s].n < min_n {
            min_n = table[base + s].n;
            victim = base + s;
        }
    }
    table[victim] = BitModel::fresh(check);
    table[victim].update(y);
}

/// A fresh fixed-size context table, every slot an untouched (neutral) model.
fn ctx_table() -> Vec<BitModel> {
    vec![BitModel::default(); CTX_SIZE]
}

/// Lookup key for a high-order context: hash the last `order` bytes (the low
/// `8 * order` bits of the 16-byte `ctx_hi` register) together with the
/// within-byte tree `node`.
#[inline]
#[allow(clippy::cast_possible_truncation)]
const fn hi_key(ctx_hi: u128, order: usize, node: usize) -> u64 {
    let masked = if order >= 16 {
        ctx_hi
    } else {
        ctx_hi & ((1u128 << (8 * order)) - 1)
    };
    let lo = masked as u64;
    let hi = (masked >> 64) as u64;
    let mixed = lo
        .wrapping_mul(WORD_MIX)
        .wrapping_add(hi.wrapping_mul(WORD_MUL));
    mixed.wrapping_mul(WORD_MIX).wrapping_add(node as u64)
}

/// One adaptive bit predictor: a probability of "next bit is 1", an observation
/// count that schedules the learning rate (fast while young, floored once
/// mature), and a key checksum used by the hashed context tables to detect
/// collisions. `check` fits free in the struct's existing 16-byte alignment
/// padding; the dense order-0 and match-bucket tables index directly and ignore
/// it.
#[derive(Clone, Copy, Debug)]
struct BitModel {
    /// Probability stored as `f32` (the AC quantizes to 24 bits anyway) so the
    /// whole struct is 8 bytes, halving every context table's memory.
    p: f32,
    n: u8,
    check: u16,
}

impl Default for BitModel {
    fn default() -> Self {
        Self {
            p: 0.5,
            n: 0,
            check: 0,
        }
    }
}

impl BitModel {
    /// A fresh predictor stamped with `check` (used when a slot is claimed for a
    /// new context after a collision eviction).
    const fn fresh(check: u16) -> Self {
        Self {
            p: 0.5,
            n: 0,
            check,
        }
    }

    /// Move the probability toward the observed bit and age the counter. The
    /// step is computed in `f64` and rounded back to the stored `f32`.
    #[inline]
    #[allow(clippy::suboptimal_flops, clippy::cast_possible_truncation)]
    fn update(&mut self, y: f64) {
        let rate = (1.0 / (f64::from(self.n) + 1.5)).max(RATE_FLOOR);
        let p = f64::from(self.p);
        self.p = (p + rate * (y - p)) as f32;
        if self.n < N_CAP {
            self.n += 1;
        }
    }
}

/// Stretch value of knot `i` — knots span `[-APM_S_MAX, APM_S_MAX]` evenly.
#[inline]
#[allow(clippy::cast_precision_loss)]
fn knot_stretch(i: usize) -> f64 {
    APM_S_MAX.mul_add(2.0 * i as f64 / (APM_KNOTS - 1) as f64, -APM_S_MAX)
}

/// Secondary symbol estimation (an adaptive probability map, lpaq's APM): a
/// learned recalibration of the mixer's probability. For each context it holds
/// a probability at each of `APM_KNOTS` knots across the stretch domain; a
/// query stretches the input, linearly interpolates the two surrounding knots,
/// and the observed bit nudges those two knots. Each row starts as the
/// identity map (`squash(knot_stretch)`), so an untrained APM passes its input
/// through unchanged.
#[derive(Debug)]
struct Apm {
    t: Vec<f64>,
}

impl Apm {
    fn new(n_ctx: usize) -> Self {
        let mut t = vec![0.0; n_ctx * APM_KNOTS];
        for row in t.chunks_exact_mut(APM_KNOTS) {
            for (i, slot) in row.iter_mut().enumerate() {
                *slot = squash(knot_stretch(i));
            }
        }
        Self { t }
    }

    /// Recalibrate logit `s` under context `ctx`. Returns the refined
    /// probability and the `(knot, frac)` coordinates needed to update later.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn refine(&self, s: f64, ctx: usize) -> (f64, usize, f64) {
        let s = s.clamp(-APM_S_MAX, APM_S_MAX);
        let pos = (s + APM_S_MAX) / (2.0 * APM_S_MAX) * (APM_KNOTS - 1) as f64;
        let i = (pos as usize).min(APM_KNOTS - 2);
        let frac = pos - i as f64;
        let base = ctx * APM_KNOTS;
        let p = self.t[base + i].mul_add(1.0 - frac, self.t[base + i + 1] * frac);
        (p, i, frac)
    }

    /// Nudge the two knots straddling a prior [`Apm::refine`] toward bit `y`,
    /// each weighted by how close the query landed to it.
    #[allow(clippy::suboptimal_flops)]
    fn update(&mut self, ctx: usize, i: usize, frac: f64, y: f64) {
        let base = ctx * APM_KNOTS;
        self.t[base + i] += APM_RATE * (1.0 - frac) * (y - self.t[base + i]);
        self.t[base + i + 1] += APM_RATE * frac * (y - self.t[base + i + 1]);
    }
}

/// Which model stages are active. Each `l*` codec is one fixed value; the
/// stack is cumulative — lmix ⊂ lmatch ⊂ lsse ⊂ lword ⊂ lhi. A flags struct is
/// the legitimate exception to the no-many-bools lint.
#[derive(Clone, Copy, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct Arms {
    use_match: bool,
    use_sse: bool,
    use_word: bool,
    use_hi: bool,
}

impl Arms {
    const LMIX: Self = Self {
        use_match: false,
        use_sse: false,
        use_word: false,
        use_hi: false,
    };
    const LMATCH: Self = Self {
        use_match: true,
        use_sse: false,
        use_word: false,
        use_hi: false,
    };
    const LSSE: Self = Self {
        use_match: true,
        use_sse: true,
        use_word: false,
        use_hi: false,
    };
    const LWORD: Self = Self {
        use_match: true,
        use_sse: true,
        use_word: true,
        use_hi: false,
    };
    const LHI: Self = Self {
        use_match: true,
        use_sse: true,
        use_word: true,
        use_hi: true,
    };
}

/// Online multi-order bit model with a logistic (logit-domain) mixer, an
/// optional long-range match model, and optional secondary estimation (SSE).
#[derive(Debug)]
struct Model {
    /// Order-0: one predictor per within-byte tree node (`1..=255`).
    o0: Vec<BitModel>,
    /// Orders `1..=MAX_ORDER`. `maps[k - 1]` is a fixed-size table indexed by
    /// `slot(key(k, node))` — see [`Model::key`] and [`slot`].
    maps: Vec<Vec<BitModel>>,
    /// Mixer weights, trained online by gradient descent on coding loss. One
    /// weight vector per selection context (the previous byte) — the blend can
    /// differ between markup, letters, digits, etc.
    w: Vec<[f64; N_INPUTS]>,
    /// Second mixer, selected by the byte two back; its output is averaged with
    /// the first so the blend is conditioned on two independent regimes.
    w2: Vec<[f64; N_INPUTS]>,
    /// Third mixer, selected by the byte three back.
    w3: Vec<[f64; N_INPUTS]>,
    /// Fourth mixer, selected by the current word's hash (a language regime,
    /// decorrelated from the byte selectors).
    w4: Vec<[f64; N_INPUTS]>,
    /// Rolling context — the last up-to-8 bytes, most recent in the low byte.
    ctx: u64,
    /// Active model stages (which optional arms contribute).
    arms: Arms,

    /// All bytes seen so far (warm + coded), indexed by the match pointer.
    hist: Vec<u8>,
    /// Match table: hash of the last `MIN_MATCH` bytes → the position that
    /// context last ended at. `MATCH_EMPTY` marks an unused slot.
    mtable: Vec<u32>,
    /// Adaptive confidence per match-length bucket: P(actual bit == predicted).
    mmap: Vec<BitModel>,
    /// Index in `hist` of the byte the match model predicts next (valid iff
    /// `mlen > 0`).
    mptr: usize,
    /// Current verified match length, in bytes.
    mlen: u32,

    /// SSE map keyed on the within-byte tree node (`0..=255`).
    apm: Apm,

    /// Fixed-size predictor tables: `wmaps[0]` keyed on the current partial word
    /// (W0), `wmaps[1]` on the previous word combined with the current one (W1).
    wmaps: Vec<Vec<BitModel>>,
    /// Rolling hash of the current partial word (`WORD_SEED` when empty).
    word_hash: u64,
    /// Length of the current partial word in letters (`0` between words).
    word_len: u32,
    /// Rolling hash of the most recently completed word (`WORD_SEED` initially).
    prev_word_hash: u64,
    /// Rolling hash of the word completed before `prev_word_hash` (W2 context).
    prev2_word_hash: u64,

    /// Per-high-order fixed-size predictor tables (`himaps[i]` for
    /// `HI_ORDERS[i]`), indexed by `slot` of the hashed `(context, node)`.
    himaps: Vec<Vec<BitModel>>,
    /// Rolling context — the last up-to-16 bytes, most recent in the low byte;
    /// source for the high-order context hashes.
    ctx_hi: u128,
    /// Sparse (non-contiguous byte) predictor tables.
    spmaps: Vec<Vec<BitModel>>,
}

impl Model {
    fn new(arms: Arms) -> Self {
        let Arms {
            use_match,
            use_sse,
            use_word,
            use_hi,
        } = arms;
        let (hist, mtable, mmap) = if use_match {
            (
                Vec::new(),
                vec![MATCH_EMPTY; 1usize << MATCH_BITS],
                vec![BitModel::default(); MATCH_BUCKETS],
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        let apm = Apm::new(if use_sse { 256 } else { 0 });
        let wmaps = if use_word {
            (0..3).map(|_| ctx_table()).collect()
        } else {
            Vec::new()
        };
        let himaps = if use_hi {
            (0..N_HI).map(|_| ctx_table()).collect()
        } else {
            Vec::new()
        };
        let spmaps = if use_hi {
            (0..N_SPARSE).map(|_| ctx_table()).collect()
        } else {
            Vec::new()
        };
        Self {
            o0: vec![BitModel::default(); 256],
            maps: (0..MAX_ORDER).map(|_| ctx_table()).collect(),
            w: vec![[INIT_W; N_INPUTS]; N_WSETS],
            w2: vec![[INIT_W; N_INPUTS]; N_WSETS],
            w3: vec![[INIT_W; N_INPUTS]; N_WSETS],
            w4: vec![[INIT_W; N_INPUTS]; N_WSETS],
            ctx: 0,
            arms,
            hist,
            mtable,
            mmap,
            mptr: 0,
            mlen: 0,
            apm,
            wmaps,
            word_hash: WORD_SEED,
            word_len: 0,
            prev_word_hash: WORD_SEED,
            prev2_word_hash: WORD_SEED,
            himaps,
            ctx_hi: 0,
            spmaps,
        }
    }

    /// Lookup key for sparse context `which` at `node`: a hash of two
    /// non-contiguous history bytes — {c1,c3} for 0, {c2,c4} for 1 — so the
    /// model can exploit patterns where the skipped byte is noise.
    #[inline]
    const fn sparse_key(&self, which: usize, node: usize) -> u64 {
        let sctx = if which == 0 {
            (self.ctx & 0xFF) | (((self.ctx >> 16) & 0xFF) << 8)
        } else {
            ((self.ctx >> 8) & 0xFF) | (((self.ctx >> 24) & 0xFF) << 8)
        };
        word_key(sctx, node)
    }

    /// Low `8 * k` bits of the rolling context (order-`k` byte history).
    const fn ctx_k(&self, k: usize) -> u64 {
        if k >= 8 {
            self.ctx
        } else {
            self.ctx & ((1u64 << (8 * k)) - 1)
        }
    }

    /// Packed lookup key for order `k` at within-byte tree `node`: the order's
    /// byte history shifted up by 9 bits, with `node` (`< 512`) in the low bits.
    /// For `k <= 6` the history is `<= 48` bits, so the pair packs into a `u64`
    /// losslessly before [`locate`] hashes it into a fixed-table slot.
    #[inline]
    const fn key(&self, k: usize, node: usize) -> u64 {
        (self.ctx_k(k) << 9) | node as u64
    }

    /// The match model's prediction at `node`, or `None` if there is no active
    /// match or the match's predicted byte has already diverged from the bits
    /// coded so far in this byte. Returns `(length bucket, predicted bit)`.
    #[inline]
    fn match_pred(&self, node: usize) -> Option<(usize, u8)> {
        if !self.arms.use_match || self.mlen == 0 {
            return None;
        }
        let pb = self.hist[self.mptr];
        let j = node.ilog2();
        // The bits coded so far (`node`) must be the top-`j` bits of the
        // predicted byte, else this candidate no longer applies this byte.
        let prefix = usize::from(pb) >> (8 - j);
        if node != (1usize << j) | prefix {
            return None;
        }
        let pred_bit = (pb >> (7 - j)) & 1;
        let bucket = (self.mlen as usize).min(MATCH_BUCKETS - 1);
        Some((bucket, pred_bit))
    }

    /// Predict the next bit at `node`. Fills `x` with each model's logit (a
    /// novel/inactive model contributes a neutral `stretch(0.5) = 0`) and
    /// returns `(p_code, p_mix, sse_coords)`: the probability to code against,
    /// the raw mixer probability the mixer trains on, and — when SSE is on —
    /// the APM `(ctx, knot, frac)` to update after the bit is known.
    #[allow(clippy::needless_range_loop, clippy::similar_names)]
    fn step_predict(
        &self,
        node: usize,
        x: &mut [f64; N_INPUTS],
    ) -> (f64, f64, Option<(usize, usize, f64)>) {
        x[0] = stretch(f64::from(self.o0[node].p));
        for k in 1..=MAX_ORDER {
            x[k] = ctx_logit(&self.maps[k - 1], self.key(k, node));
        }
        x[MATCH_IN] = match self.match_pred(node) {
            Some((bucket, pred_bit)) => {
                let pc = f64::from(self.mmap[bucket].p);
                let p1 = if pred_bit == 1 { pc } else { 1.0 - pc };
                stretch(p1)
            }
            None => 0.0,
        };
        let (w0, w1, w2) = if self.arms.use_word {
            let k0 = word_key(self.word_hash, node);
            let k1 = word_key(word_combine(self.prev_word_hash, self.word_hash), node);
            let ctx2 = word_combine(
                word_combine(self.prev2_word_hash, self.prev_word_hash),
                self.word_hash,
            );
            let k2 = word_key(ctx2, node);
            (
                ctx_logit(&self.wmaps[0], k0),
                ctx_logit(&self.wmaps[1], k1),
                ctx_logit(&self.wmaps[2], k2),
            )
        } else {
            (0.0, 0.0, 0.0)
        };
        x[WORD0_IN] = w0;
        x[WORD1_IN] = w1;
        x[WORD2_IN] = w2;
        for i in 0..N_HI {
            x[HI_IN + i] = if self.arms.use_hi {
                ctx_logit(&self.himaps[i], hi_key(self.ctx_hi, HI_ORDERS[i], node))
            } else {
                0.0
            };
        }
        for i in 0..N_SPARSE {
            x[SPARSE_IN + i] = if self.arms.use_hi {
                ctx_logit(&self.spmaps[i], self.sparse_key(i, node))
            } else {
                0.0
            };
        }
        let bitpos = node.ilog2() as usize;
        let sel1 = (((self.ctx & 0xFF) as usize) << 3) | bitpos;
        let sel2 = ((((self.ctx >> 8) & 0xFF) as usize) << 3) | bitpos;
        let sel3 = ((((self.ctx >> 16) & 0xFF) as usize) << 3) | bitpos;
        let sel4 = (((self.word_hash & 0xFF) as usize) << 3) | bitpos;
        let dot = |w: &[f64; N_INPUTS]| -> f64 { w.iter().zip(x.iter()).map(|(a, b)| a * b).sum() };
        let s =
            (dot(&self.w[sel1]) + dot(&self.w2[sel2]) + dot(&self.w3[sel3]) + dot(&self.w4[sel4]))
                / 4.0;
        let p_mix = squash(s);
        if self.arms.use_sse {
            let (p_apm, i, frac) = self.apm.refine(s, node);
            let p_code = APM_BLEND.mul_add(p_apm, (1.0 - APM_BLEND) * p_mix);
            (p_code, p_mix, Some((node, i, frac)))
        } else {
            (p_mix, p_mix, None)
        }
    }

    /// After bit `y` is coded at `node`: step the mixer weights down the
    /// coding-loss gradient (on its own output `p_mix`), update each model's
    /// bit predictor, and nudge the SSE map.
    #[allow(
        clippy::suboptimal_flops,
        clippy::needless_range_loop,
        clippy::similar_names
    )]
    fn step_update(
        &mut self,
        node: usize,
        x: &[f64; N_INPUTS],
        p_mix: f64,
        sse: Option<(usize, usize, f64)>,
        y: u8,
    ) {
        let err = f64::from(y) - p_mix;
        let bitpos = node.ilog2() as usize;
        let sel1 = (((self.ctx & 0xFF) as usize) << 3) | bitpos;
        let sel2 = ((((self.ctx >> 8) & 0xFF) as usize) << 3) | bitpos;
        let sel3 = ((((self.ctx >> 16) & 0xFF) as usize) << 3) | bitpos;
        let sel4 = (((self.word_hash & 0xFF) as usize) << 3) | bitpos;
        for (wk, &xk) in self.w[sel1].iter_mut().zip(x.iter()) {
            *wk += MIX_LR * err * xk;
        }
        for (wk, &xk) in self.w2[sel2].iter_mut().zip(x.iter()) {
            *wk += MIX_LR * err * xk;
        }
        for (wk, &xk) in self.w3[sel3].iter_mut().zip(x.iter()) {
            *wk += MIX_LR * err * xk;
        }
        for (wk, &xk) in self.w4[sel4].iter_mut().zip(x.iter()) {
            *wk += MIX_LR * err * xk;
        }
        let yf = f64::from(y);
        self.o0[node].update(yf);
        for k in 1..=MAX_ORDER {
            let key = self.key(k, node);
            update_ctx(&mut self.maps[k - 1], key, yf);
        }
        if let Some((bucket, pred_bit)) = self.match_pred(node) {
            self.mmap[bucket].update(f64::from(u8::from(y == pred_bit)));
        }
        if self.arms.use_word {
            let k0 = word_key(self.word_hash, node);
            let k1 = word_key(word_combine(self.prev_word_hash, self.word_hash), node);
            let ctx2 = word_combine(
                word_combine(self.prev2_word_hash, self.prev_word_hash),
                self.word_hash,
            );
            let k2 = word_key(ctx2, node);
            update_ctx(&mut self.wmaps[0], k0, yf);
            update_ctx(&mut self.wmaps[1], k1, yf);
            update_ctx(&mut self.wmaps[2], k2, yf);
        }
        if self.arms.use_hi {
            for i in 0..N_HI {
                let key = hi_key(self.ctx_hi, HI_ORDERS[i], node);
                update_ctx(&mut self.himaps[i], key, yf);
            }
            for i in 0..N_SPARSE {
                let key = self.sparse_key(i, node);
                update_ctx(&mut self.spmaps[i], key, yf);
            }
        }
        if let Some((ctx, i, frac)) = sse {
            self.apm.update(ctx, i, frac, yf);
        }
    }

    /// Hash of the `MIN_MATCH` bytes ending at `pos` (caller guarantees
    /// `pos + 1 >= MIN_MATCH`).
    #[inline]
    fn match_hash_at(&self, pos: usize) -> usize {
        let mut h = 0x9E37_79B9_7F4A_7C15u64;
        for &byte in &self.hist[pos + 1 - MIN_MATCH..=pos] {
            h = h
                .wrapping_mul(0x0100_0000_01B3)
                .wrapping_add(u64::from(byte));
        }
        h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (h >> (64 - MATCH_BITS)) as usize
    }

    /// True if the `MIN_MATCH` bytes ending at `ci` equal those ending at
    /// `pos` — a real context match, not a hash collision.
    #[inline]
    fn verify_match(&self, ci: usize, pos: usize) -> bool {
        (0..MIN_MATCH).all(|i| self.hist[ci - i] == self.hist[pos - i])
    }

    /// Fold byte `b` into the match state: extend or break the current match,
    /// append to history, and (if no match is active) seed a new one from the
    /// table, then record this position.
    #[allow(clippy::cast_possible_truncation)]
    fn advance_match(&mut self, b: u8) {
        if !self.arms.use_match {
            return;
        }
        if self.mlen > 0 {
            if self.hist[self.mptr] == b {
                self.mptr += 1;
                self.mlen = self.mlen.saturating_add(1);
            } else {
                self.mlen = 0;
            }
        }
        self.hist.push(b);
        if self.hist.len() < MIN_MATCH {
            return;
        }
        let pos = self.hist.len() - 1;
        let h = self.match_hash_at(pos);
        if self.mlen == 0 {
            let cand = self.mtable[h];
            if cand != MATCH_EMPTY {
                let ci = cand as usize;
                if ci < pos && self.verify_match(ci, pos) {
                    self.mptr = ci + 1;
                    self.mlen = MIN_MATCH as u32;
                }
            }
        }
        self.mtable[h] = pos as u32;
    }

    /// Fold byte `b` into the word state: extend the current word on a letter,
    /// or close it out (promoting it to `prev_word`) and reset on a boundary.
    fn advance_word(&mut self, b: u8) {
        if !self.arms.use_word {
            return;
        }
        if b.is_ascii_alphabetic() {
            self.word_hash = self
                .word_hash
                .wrapping_mul(WORD_MUL)
                .wrapping_add(u64::from(b));
            self.word_len += 1;
        } else {
            if self.word_len > 0 {
                self.prev2_word_hash = self.prev_word_hash;
                self.prev_word_hash = self.word_hash;
            }
            self.word_hash = WORD_SEED;
            self.word_len = 0;
        }
    }

    /// Roll all per-byte context state forward after byte `b` is known: match
    /// state, word state, and the two rolling context registers. Shared by the
    /// learn / encode / decode / residual paths so they evolve identically.
    fn advance(&mut self, b: u8) {
        self.advance_match(b);
        self.advance_word(b);
        self.ctx = (self.ctx << 8) | u64::from(b);
        self.ctx_hi = (self.ctx_hi << 8) | u128::from(b);
    }

    /// Replay a byte through predict+update without coding it — used to prime
    /// model state from the `warm` prefix on both sides identically.
    fn learn_byte(&mut self, b: u8) {
        let mut node = 1usize;
        let mut x = [0.0; N_INPUTS];
        for i in (0..8).rev() {
            let y = (b >> i) & 1;
            let (_p_code, p_mix, sse) = self.step_predict(node, &mut x);
            self.step_update(node, &x, p_mix, sse, y);
            node = (node << 1) | usize::from(y);
        }
        self.advance(b);
    }
}

/// Fill a two-symbol CDF `[0, c0, TOTAL]` where symbol 0 (bit = 0) holds mass
/// `1 - p1`. `c0` is clamped to `[1, TOTAL - 1]` to satisfy the AC's
/// strictly-increasing-CDF contract.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn fill_bit_cdf(p1: f64, cdf: &mut [u32; 3]) {
    let c0 = ((1.0 - p1) * f64::from(TOTAL)).round() as u32;
    cdf[1] = c0.clamp(1, TOTAL - 1);
}

/// Encode `measure` (preceded by `warm` priming) with the given model stages.
/// Shared by the `lmix` / `lmatch` / `lsse` / `lword` codecs.
#[allow(clippy::cast_possible_truncation)]
fn encode_impl(
    arms: Arms,
    component: &str,
    warm: &[u8],
    measure: &[u8],
) -> (Vec<u8>, Decomposition) {
    let mut writer = BitWriter::new();
    writer.write_bits(measure.len() as u64, 64);
    let mut model = Model::new(arms);
    for &b in warm {
        model.learn_byte(b);
    }
    {
        let mut enc = AcEncoder::new(&mut writer);
        let mut x = [0.0; N_INPUTS];
        let mut cdf = [0u32, 0, TOTAL];
        for &b in measure {
            let mut node = 1usize;
            for i in (0..8).rev() {
                let y = (b >> i) & 1;
                let (p_code, p_mix, sse) = model.step_predict(node, &mut x);
                fill_bit_cdf(p_code, &mut cdf);
                enc.encode(&cdf, usize::from(y));
                model.step_update(node, &x, p_mix, sse, y);
                node = (node << 1) | usize::from(y);
            }
            model.advance(b);
        }
        enc.finish();
    }
    let (buf, _pad) = writer.finish();
    let mut decomp = Decomposition::new();
    decomp.add(component, 8 * buf.len() as u64);
    (buf, decomp)
}

/// Decode an archive produced by [`encode_impl`] with the same stages.
fn decode_impl(arms: Arms, warm: &[u8], archive: &[u8]) -> Result<Vec<u8>> {
    let mut reader = BitReader::new(archive);
    let n = usize::try_from(reader.read_bits(64)?).expect("length prefix fits usize");
    let mut model = Model::new(arms);
    for &b in warm {
        model.learn_byte(b);
    }
    let mut out = Vec::with_capacity(n);
    let mut dec = AcDecoder::new(&mut reader);
    let mut x = [0.0; N_INPUTS];
    let mut cdf = [0u32, 0, TOTAL];
    for _ in 0..n {
        let mut node = 1usize;
        let mut b = 0u8;
        for _ in 0..8 {
            let (p_code, p_mix, sse) = model.step_predict(node, &mut x);
            fill_bit_cdf(p_code, &mut cdf);
            let y = u8::try_from(dec.decode(&cdf)?).expect("bit symbol 0/1 fits u8");
            model.step_update(node, &x, p_mix, sse, y);
            b = (b << 1) | y;
            node = (node << 1) | usize::from(y);
        }
        model.advance(b);
        out.push(b);
    }
    Ok(out)
}

/// The v7 logistic-mixing codec: bitwise multi-order context mixing in the
/// logit domain + AC, nothing else (no match model).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LmixCodec;

impl Codec for LmixCodec {
    fn name(&self) -> &'static str {
        "lmix"
    }

    fn encode_window(&self, warm: &[u8], measure: &[u8]) -> Result<(Vec<u8>, Decomposition)> {
        Ok(encode_impl(Arms::LMIX, "lmix", warm, measure))
    }

    fn decode_window(&self, warm: &[u8], archive: &[u8]) -> Result<Vec<u8>> {
        decode_impl(Arms::LMIX, warm, archive)
    }
}

/// [`LmixCodec`] plus a long-range match model as an extra mixer input.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LmatchCodec;

impl Codec for LmatchCodec {
    fn name(&self) -> &'static str {
        "lmatch"
    }

    fn encode_window(&self, warm: &[u8], measure: &[u8]) -> Result<(Vec<u8>, Decomposition)> {
        Ok(encode_impl(Arms::LMATCH, "lmatch", warm, measure))
    }

    fn decode_window(&self, warm: &[u8], archive: &[u8]) -> Result<Vec<u8>> {
        decode_impl(Arms::LMATCH, warm, archive)
    }
}

/// [`LmatchCodec`] plus secondary symbol estimation (SSE/APM) recalibrating
/// the mixer's output.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LsseCodec;

impl Codec for LsseCodec {
    fn name(&self) -> &'static str {
        "lsse"
    }

    fn encode_window(&self, warm: &[u8], measure: &[u8]) -> Result<(Vec<u8>, Decomposition)> {
        Ok(encode_impl(Arms::LSSE, "lsse", warm, measure))
    }

    fn decode_window(&self, warm: &[u8], archive: &[u8]) -> Result<Vec<u8>> {
        decode_impl(Arms::LSSE, warm, archive)
    }
}

/// [`LsseCodec`] plus the W0/W1 word-context models as extra mixer inputs.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LwordCodec;

impl Codec for LwordCodec {
    fn name(&self) -> &'static str {
        "lword"
    }

    fn encode_window(&self, warm: &[u8], measure: &[u8]) -> Result<(Vec<u8>, Decomposition)> {
        Ok(encode_impl(Arms::LWORD, "lword", warm, measure))
    }

    fn decode_window(&self, warm: &[u8], archive: &[u8]) -> Result<Vec<u8>> {
        decode_impl(Arms::LWORD, warm, archive)
    }
}

/// [`LwordCodec`] plus high-order hashed byte contexts (orders 8/12/16).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LhiCodec;

impl Codec for LhiCodec {
    fn name(&self) -> &'static str {
        "lhi"
    }

    fn encode_window(&self, warm: &[u8], measure: &[u8]) -> Result<(Vec<u8>, Decomposition)> {
        Ok(encode_impl(Arms::LHI, "lhi", warm, measure))
    }

    fn decode_window(&self, warm: &[u8], archive: &[u8]) -> Result<Vec<u8>> {
        decode_impl(Arms::LHI, warm, archive)
    }
}

/// Map a codec name to its model stages, so the residual analyzer can
/// reproduce any codec's model exactly.
pub(crate) const fn flags_for(name: &str) -> Option<Arms> {
    Some(match name.as_bytes() {
        b"lmix" => Arms::LMIX,
        b"lmatch" => Arms::LMATCH,
        b"lsse" => Arms::LSSE,
        b"lword" => Arms::LWORD,
        b"lhi" => Arms::LHI,
        _ => return None,
    })
}

/// Byte-class labels for residual attribution, indexed by [`byte_class`].
pub(crate) const CLASS_NAMES: [&str; 9] = [
    "lower", "upper", "digit", "space", "newline", "markup", "punct", "high", "other",
];

/// Bucket a byte into a residual class. `markup` covers the wiki/XML structural
/// characters; everything non-letter/digit/space splits into markup vs other
/// punctuation so we can see which kind of structure is expensive.
const fn byte_class(b: u8) -> usize {
    match b {
        b'a'..=b'z' => 0,
        b'A'..=b'Z' => 1,
        b'0'..=b'9' => 2,
        b' ' => 3,
        b'\n' => 4,
        b'<' | b'>' | b'&' | b';' | b'=' | b'[' | b']' | b'{' | b'}' | b'|' | b'/' => 5,
        0x21..=0x7e => 6,
        0x80..=0xff => 7,
        _ => 8,
    }
}

/// Per-class coding-cost attribution from running a model over `measure`.
#[derive(Clone, Debug)]
pub(crate) struct ResidualReport {
    /// Ideal coded bits (cross-entropy) spent per class.
    pub bits: [f64; 9],
    /// Bytes seen per class.
    pub count: [u64; 9],
}

impl ResidualReport {
    pub(crate) fn total_bits(&self) -> f64 {
        self.bits.iter().sum()
    }

    pub(crate) fn total_count(&self) -> u64 {
        self.count.iter().sum()
    }
}

/// Run a codec's model over `measure` (primed by `warm`) and attribute the
/// ideal coding cost (`-log2 P(actual bit)`, summed) to each byte's class. This
/// is the model's cross-entropy — what the archive costs, minus negligible AC
/// framing — so it shows where the bits actually go without needing the coder.
pub(crate) fn residual_report(arms: Arms, warm: &[u8], measure: &[u8]) -> ResidualReport {
    let mut model = Model::new(arms);
    for &b in warm {
        model.learn_byte(b);
    }
    let mut bits = [0.0; 9];
    let mut count = [0u64; 9];
    let mut x = [0.0; N_INPUTS];
    for &b in measure {
        let cls = byte_class(b);
        let mut node = 1usize;
        let mut cost = 0.0;
        for i in (0..8).rev() {
            let y = (b >> i) & 1;
            let (p_code, p_mix, sse) = model.step_predict(node, &mut x);
            let p = if y == 1 { p_code } else { 1.0 - p_code };
            cost -= p.max(1e-12).log2();
            model.step_update(node, &x, p_mix, sse, y);
            node = (node << 1) | usize::from(y);
        }
        model.advance(b);
        bits[cls] += cost;
        count[cls] += 1;
    }
    ResidualReport { bits, count }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrips_text(codec: &dyn Codec) {
        let measure = b"the quick brown fox jumps over the lazy dog. \
                        the quick brown fox jumps over the lazy dog again."
            .repeat(40);
        let (archive, decomp) = codec.encode_window(b"", &measure).unwrap();
        assert_eq!(decomp.total(), 8 * archive.len() as u64);
        let decoded = codec.decode_window(b"", &archive).unwrap();
        assert_eq!(decoded, measure);
        assert!(
            archive.len() < measure.len(),
            "archive {} not smaller than input {}",
            archive.len(),
            measure.len()
        );
    }

    fn roundtrips_with_warm(codec: &dyn Codec) {
        let warm = b"warm priming context that both sides share verbatim. ".repeat(8);
        let measure = b"warm priming context that both sides share, then new tail.".to_vec();
        let (archive, _) = codec.encode_window(&warm, &measure).unwrap();
        let decoded = codec.decode_window(&warm, &archive).unwrap();
        assert_eq!(decoded, measure);
    }

    fn roundtrips_all_bytes(codec: &dyn Codec) {
        let measure: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        let (archive, _) = codec.encode_window(b"", &measure).unwrap();
        let decoded = codec.decode_window(b"", &archive).unwrap();
        assert_eq!(decoded, measure);
    }

    fn roundtrips_empty(codec: &dyn Codec) {
        let (archive, _) = codec.encode_window(b"", b"").unwrap();
        let decoded = codec.decode_window(b"", &archive).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn lmix_roundtrips() {
        roundtrips_text(&LmixCodec);
        roundtrips_with_warm(&LmixCodec);
        roundtrips_all_bytes(&LmixCodec);
        roundtrips_empty(&LmixCodec);
    }

    #[test]
    fn lmatch_roundtrips() {
        roundtrips_text(&LmatchCodec);
        roundtrips_with_warm(&LmatchCodec);
        roundtrips_all_bytes(&LmatchCodec);
        roundtrips_empty(&LmatchCodec);
    }

    #[test]
    fn lsse_roundtrips() {
        roundtrips_text(&LsseCodec);
        roundtrips_with_warm(&LsseCodec);
        roundtrips_all_bytes(&LsseCodec);
        roundtrips_empty(&LsseCodec);
    }

    #[test]
    fn lword_roundtrips() {
        roundtrips_text(&LwordCodec);
        roundtrips_with_warm(&LwordCodec);
        roundtrips_all_bytes(&LwordCodec);
        roundtrips_empty(&LwordCodec);
    }

    #[test]
    fn lhi_roundtrips() {
        roundtrips_text(&LhiCodec);
        roundtrips_with_warm(&LhiCodec);
        roundtrips_all_bytes(&LhiCodec);
        roundtrips_empty(&LhiCodec);
    }

    #[test]
    fn lmatch_beats_lmix_on_long_repeat() {
        // A long verbatim repeat is exactly what the match model exists for:
        // after the first copy, the second should cost almost nothing.
        let unit = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. ";
        let measure = unit.repeat(200);
        let (a_mix, _) = LmixCodec.encode_window(b"", &measure).unwrap();
        let (a_match, _) = LmatchCodec.encode_window(b"", &measure).unwrap();
        assert!(
            a_match.len() < a_mix.len(),
            "match archive {} should beat mixer-only {}",
            a_match.len(),
            a_mix.len()
        );
    }
}
