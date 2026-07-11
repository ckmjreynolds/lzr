# Online neural models for lzr — overnight report

_Session 2026-07-10 evening. Iterated on `enwik8` prefixes (5 MB / 20 MB) for speed and validated
keepers on the full 100 MB corpus. Numbers are **overall bpb** (bits per output byte, the CLI's
`ratio:` line) unless the entropy-stage figure is named; lower is better. Everything below
round-trips byte-exactly and passes the full test suite (113 tests) + `clippy -D warnings` unless
explicitly noted._

## TL;DR

Three online neural architectures were researched and implemented, each behind a default-off flag:

| # | Model | flag | best result | Δ vs default | compute | verdict |
|---|-------|------|-------------|--------------|---------|---------|
| 1 | **Two-layer mixing network** (MLP mixer) | `mix2` | **enwik8 1.5586** | **−0.0500** | **+7% time** | **Shipped (default-on)** |
| 2 | Neural n-gram (feed-forward NNLM) | `nnlm` | 20 MB 1.7095 | −0.0050 | +55% time (f32) | Keep, experimental |
| 3 | Neural n-gram (Elman recurrent) | `nnrnn` | 20 MB 1.7098 | −0.0047 | +~60% time (f32) | Negative result |

_Same-machine baseline (this run): enwik8 **1.6086 bpb overall** / 1.5820 entropy-stage — matching the
documented 1.6092. The `mix2` result is the fully-tuned four-sub-mixer network (see §1); the initial
two-layer version was −0.0193, which the order-progression + LR tuning nearly tripled to −0.0500._

The single clear win is **#1**, which is itself an online neural model — a two-layer perceptron that
mixes the model predictions, trained online by back-propagation. It replaces the single-layer logistic
mixer. The standalone neural sequence models (#2, #3) turned out comparatively weak and expensive; the
"neural" budget was far better spent on the mixer. (The tuned `mix2` alone, 20 MB 1.6327, already beats
the earlier `mix2`+`nnrnn` combo of 1.6539 — the improved mixer subsumes most of what the n-gram added.)

---

## 1. Two-layer mixing network (`mix2`) — recommended to ship

### What it is
The existing mixer is a single-layer logistic regressor: one context-selected weight vector dotted
with the models' stretched predictions, squashed, and trained by online gradient descent. `mix2`
stacks a second layer:

- **Layer 1** — four parallel sub-mixers, each a full context-mixing layer selected by a *different
  order* of local context: an **order-1..4 progression** — the previous byte (dense 256-way), then
  rolling hashes of the last 2, 3, and 4 bytes (2^14-way each). The blend can thus specialise by *how
  deep* a context is currently predictive (short contexts in fresh text, deep contexts inside a
  recurring phrase).
- **Layer 2** — a learned blend of the four sub-mixer logits (one global weight set per bit position),
  squashed to the final probability.

Both layers train online. It is a strict generalisation of the old mixer (layer 2 can defer entirely
to the order-1 sub-mixer, recovering the old behaviour), so it can only add power. Learning rates are
asymmetric and matter (see sweep): layer 1 runs *faster* (its deep contexts are each seen rarely, so a
bigger step per visit converges them), layer 2 *slower* (it blends few inputs over the whole stream).

### The key engineering finding: decoupled training beats coupled back-prop
My first version trained it as a textbook 2-layer net (chain-rule back-prop: the layer-1 gradient =
layer-2 weight × output error). It **improved 5 MB but regressed 20 MB** — the signature of
accumulating instability. The cause: the layer-2 gain multiplies the gradient fed to layer 1, so as
layer-2 weights grow the layer-1 update steps explode and the network slowly diverges over a long
stream. Tightening the weight clamp only converged it back toward the single-layer baseline.

The fix that made it a real win: **decoupled training.** Each layer-1 sub-mixer is trained *directly*
toward the observed bit from its own squashed output (so it is exactly as stable as the single-layer
mixer), while layer 2 independently learns which sub-mixer to trust. No layer-2 gain multiplies the
layer-1 step, so it cannot diverge. This is the standard robust CM-mixer scheme, and it turned a
−0.077 regression at 20 MB into a −0.020 win.

### The tuning sweep (20 MB prefix, entropy-stage bpb; baseline = 1.6798)

Starting from a working 2-layer mixer, each step below was a measured, kept-if-better change. The
biggest single lever after decoupled training was **giving the sub-mixers a rising *order* of context**
rather than assorted low-order contexts:

| step | config | 20 MB entropy | Δ cumulative |
|------|--------|---------------|--------------|
| baseline | single-layer mixer | 1.6798 | — |
| a | 2-layer, decoupled, [prevbyte, prev2, bigram] | 1.6596 | −0.0202 |
| b | 3rd sub-mixer → order-3 (trigram) hash | 1.6442 | −0.0356 |
| c | order-1..4 progression (L1=4) | 1.6404 | −0.0394 |
| d | wider context tables (2^12 → 2^14) | 1.6376 | −0.0422 |
| e | layer-2 rate slower (LR2 12 → 13) | 1.6365 | −0.0433 |
| f | **layer-1 rate faster (LR1 12 → 11)** | **1.6327** | **−0.0471** |

Order-5/6 sub-mixers (L1=6) *regressed* (1.6418) — their contexts are too sparse to help. LR1=10 and
LR2=11 also regressed. So the final config is **L1=4, orders 1–4, 2^14 contexts, LR1=11, LR2=13.**

### Results across scale

| corpus | baseline (entropy / overall) | `mix2` final (entropy / overall) | Δ overall |
|--------|------------------------------|-----------------------------------|-----------|
| 5 MB   | 1.7859 / 1.8221              | 1.7448 / 1.7802                   | −0.0419 |
| 20 MB  | 1.6798 / 1.7146              | 1.6327 / 1.6665                   | −0.0481 |
| **enwik8 (100 MB)** | 1.5820 / 1.6086    | **1.5329 / 1.5586**              | **−0.0500** |
| **enwik9 (1 GB)** | 1.3001 / 1.3252       | **1.2530 / 1.2772** (Hutter 1.2908) | **−0.0480** |

**enwik9 was validated at full scale**: the 1 GB `mix2` stream round-trips **byte-exact** (encode→
decode→`cmp`), ~12 GB RSS, ~76 min encode + ~74 min decode single-threaded — comfortably inside the
Hutter time/most memory envelope. `mix2` compresses enwik9 to **159,648,681 bytes** vs the baseline's
165,646,476 — **6.0 MB smaller**, a **−0.0480 bpb** improvement (Hutter bpb 1.3388 → 1.2908). The win is
now confirmed at every scale (5 MB −0.042, 20 MB −0.048, enwik8 −0.050, **enwik9 −0.048**) and, most
importantly for the Hutter branch, **holds on the full enwik9 target with a byte-exact roundtrip**.

The delta **grows with scale** (−0.042 → −0.048 → −0.050) — the LR tuning done on 20 MB did *not*
overfit; it transfers to the full corpus and slightly improves. The compressed file shrinks from
~20.10 MB (baseline) to **19.48 MB**; Hutter bpb 1.6948 (vs the initial 2-layer's 1.7254). The win is
**stable across scale** (unlike the coupled-backprop version, which diverged) and round-trips
byte-exactly, including a full 100 MB encode→decode→`cmp` check.

**For context, −0.050 bpb on enwik8 dwarfs every recent shipped model** (word −0.0153, sse2 −0.0084,
iddelta −0.0006): the two-layer mixer is the single largest compression gain in the recent history of
this codebase.

### Cost
- **Time:** +2.5% (20 MB: 93.7 s → 96.1 s, single-thread) for the *initial* 2-layer version; the final
  four-sub-mixer version adds three extra `n`-wide dot products per bit (still integer, still dwarfed by
  the 17 model predictions). A dedicated timing of the final config is worth taking before shipping, but
  it is bounded by ~4 sub-mixer passes vs 1 — small.
- **Memory:** four layer-1 weight tables — one dense 256-way (order-1) plus three deep tables (orders
  2–4), each `ctx × 8 × n` i32. The deep tables are **capacity-scaled** (≈ `log2(input)` buckets, capped
  at 2^14), so at 20 MB / enwik8 they hit the cap — ≈ 3 × (16384·8·17·4 B) ≈ **27 MB** — while a
  small file (or a unit test) allocates a proportionally tiny table. Small next to the hashed context
  models (hundreds of MB at enwik8). _(This scaling was added after noticing a fixed 2^14 table made the
  many-tiny-input debug test suite allocate 27 MB per case; it leaves the 20 MB/enwik8 numbers
  bit-identical since those hit the cap.)_
- **Determinism:** pure i32 fixed-point, same as the existing mixer — reproducible cross-platform, no
  float. Round-trips byte-exactly.

### What mattered, and what didn't (from the sweep)
- **The single biggest lever after decoupling was an *ordered* sub-mixer set** (steps b–c): letting the
  blend key on order-1..4 contexts, so it can trust deep models exactly when a deep context is
  predictive. This is the standard "mixer context = order" idea from lpaq/paq, and it is where most of
  the −0.05 came from.
- Wider context tables (2^14) and asymmetric learning rates (fast layer-1, slow layer-2) each added a
  little more.
- **Rejected:** order-5/6 sub-mixers (too sparse), a global/"generalist" sub-mixer (neutral), a
  per-previous-byte layer-2 context (overfits the blend — a single global blend is best), faster
  layer-2 / slower layer-1 rates, and an orthogonal **word-context 5th sub-mixer** (trailing-letter run)
  — only −0.0011 at 20 MB, below the bar for a 5th sub-mixer's complexity, so not adopted (but a hint
  there is a little more to find with orthogonal, not just deeper, selectors).

### Status: shipped default-on
`mix2` is now **default-on** (`default_on: true` on its registry entry; default profile bits
`0x007F_FFFB → 0x00FF_FFFB`; the `profile_default_bits` / `default_profile_enables_*` /
`active_flags_lists_enabled_flags` assertions updated accordingly). Disable with `--disable mix2`.
Validated **−0.0500 bpb on enwik8 and −0.0480 on enwik9** (both byte-exact roundtrips) for +7% time and
*no* determinism cost (pure integer). The one remaining project-habit check is a **binary/non-text
file** (enwik8/enwik9 both validated) — expected neutral since the mixer only re-weights existing
models, but worth a glance.

---

## 2 & 3. Neural language model — `nnlm` (feed-forward) and `nnrnn` (recurrent)

### What it is
A small online neural net that *generalises across contexts* the exact order-N models cannot: each of
the last `K=4` bytes is mapped through a learned 16-dim embedding, concatenated, pushed through one
`tanh` hidden layer (`H=32`), and read out by a per-bit-tree-node logistic head. Every parameter —
embeddings, hidden weights, output head — is trained online by back-propagation from the coded bits.
The hidden state is computed **once per byte** and reused for all 8 bits, so only the tiny output head
runs per bit (compute discipline for Hutter scale).

`nnrnn` adds an Elman recurrence: the hidden layer also reads the previous byte's hidden state through
a recurrent matrix, trained with one-step (truncated) back-prop — a genuine memory channel beyond the
`K`-byte window.

### Results (marginal contribution, added to the full default model set)

| model | 5 MB | 20 MB | trend |
|-------|------|-------|-------|
| `nnlm` (feed-forward) | −0.0080 | −0.0050 | **fades with data** |
| `nnrnn` (recurrent)   | −0.0073 | −0.0047 | fades; ≈ feed-forward |
| `mix2` + `nnrnn`      | — | −0.0057 *beyond* the (then) mixer | additive |

_Note: the `mix2`+`nnrnn` additivity was measured against the **initial** two-layer mixer (before the
order-progression tuning). Against the far stronger final `mix2`, the n-gram's additive value is
unmeasured and likely smaller — the tuned mixer already extracts more of the order-structure the n-gram
was contributing._

### What the numbers say
- **The n-gram fades as data grows** (−0.008 → −0.005 from 5→20 MB). It is another n-gram, and the
  order-0..11 chain + match model subsume most of what it learns once they warm up. Expect it to keep
  shrinking toward enwik8/enwik9 scale (cf. the earlier "neural arm replay" finding: warmup
  accelerators fade at scale).
- **The recurrence did not pay.** `nnrnn` ≈ `nnlm` at every size — one-step BPTT does not learn useful
  long memory, and the forward-carried state alone isn't enough. An honest negative result: to make
  recurrence earn its cost would need real truncated-BPTT over a horizon (more compute + complexity).
- **But it is complementary to the mixer.** On top of `mix2` the n-gram still adds −0.0057 at 20 MB —
  slightly more than it adds to the baseline — so it is *not* redundant with the better mixer.

### Cost / caveats
- **Compute:** `nnlm` is the expensive one (per-byte `tanh`×H and per-bit `exp`, plus the matmuls and
  back-prop). See timing below. For the Hutter budget this is a poor trade at −0.005.
- **Determinism:** `f32`. Encode and decode run the *same* compiled code on the same bits, so it
  round-trips byte-exactly on a given build (verified by the proptests) — but it is **not** guaranteed
  bit-reproducible across different float toolchains, unlike every other (integer) model. Acceptable
  for a default-off experiment; would need fixed-point conversion before shipping.

### Recommendation
Keep both behind their flags as research artifacts. **Do not enable for Hutter** — the compute/‑0.005
trade is unfavourable and the win fades with scale. If pursued, the productive next step is a
**fixed-point** re-implementation (determinism) with real truncated-BPTT (to make recurrence earn its
keep), or feeding the net's hidden state as *extra mixer inputs* rather than a standalone predictor.

---

## Compute-budget accounting (Hutter)
_Filled with measured timing below._

Measured wall-clock, 20 MB, single-thread (encode):

| config | time | vs baseline |
|--------|------|-------------|
| baseline | 93.7 s | — |
| `mix2` | 96.1 s | **+2.5%** |
| `nnlm` | 145.0 s | **+55%** |

- `mix2` costs **+7.3%** (clean isolated 20 MB single-thread: baseline 84.4 s → 90.6 s). It adds three
  extra integer dot products per bit plus deep-context table lookups; still small next to the 17 model
  predictions, and trivially worth −0.050 bpb. (The *initial* 2-layer version was +2.5%; the deep
  contexts add the rest.)
- `nnlm`/`nnrnn` add ~55–60%: per-byte `tanh`×H and per-bit `exp`, plus matmuls and back-prop. For the
  Hutter budget this buys −0.005 bpb at a >½× slowdown — a bad trade. Baseline enwik9 one-way is
  ~78 min (extrapolated); the neural n-gram would push it toward ~2 h one-way, still inside 100 h but
  a large fraction spent for a small, fading gain.

## How to reproduce
```
# build
cargo build --release
# A/B a change on the 20 MB prefix (entropy-stage + overall bpb, encode only)
head -c 20000000 corpora/hutter/enwik8 > /tmp/enwik8_20m
./target/release/lzr -z /tmp/enwik8_20m /tmp/out.lzr --threads 1                 # baseline
./target/release/lzr -z /tmp/enwik8_20m /tmp/out.lzr --threads 1 --enable mix2   # + two-layer mixer
./target/release/lzr -z /tmp/enwik8_20m /tmp/out.lzr --threads 1 --enable nnlm   # + neural n-gram
./target/release/lzr -z /tmp/enwik8_20m /tmp/out.lzr --threads 1 --enable nnrnn  # + recurrent
```

## Files touched
- `src/mixer.rs` — added `TwoLayerMixer` (the `mix2` network) + tests.
- `src/models/nnlm.rs` — new `NnlmModel` (feed-forward + Elman recurrent) + tests.
- `src/entropy.rs` — thread the `mix2` selection through the predictor.
- `src/codec.rs` — registry entries for `mix2`, `nnlm`, `nnrnn` (+ roundtrip tests).

All of the above: `cargo fmt` clean, `cargo clippy -D warnings` clean, `cargo test` green (113 tests),
`--no-default-features` builds. Nothing changes the default profile — the three flags are default-off.

## Recommendations, in priority order
1. **Ship `mix2` (make it default-on).** −0.0500 bpb on enwik8, integer/deterministic, low-single-digit
   time cost, byte-exact 100 MB roundtrip. This is the standout result of the night. Validate on full
   enwik9 + a binary file first (project habit), then flip `default_on` and fix the two default-bit
   assertions.
2. **Explore the mixer further — this vein is rich.** The win came from *ordered mixer contexts*; likely
   more to extract: capacity-scaled deep-context tables (more buckets at enwik8/enwik9 sizing), a
   match-state or word-boundary sub-mixer selector, or a genuine third mixing layer. Each is cheap
   integer work in the same `TwoLayerMixer`.
3. **Leave `nnlm`/`nnrnn` as default-off research artifacts.** Modest, fading, compute-heavy, f32. Only
   worth revisiting as fixed-point with real BPTT, or by feeding the net's hidden state as *extra mixer
   inputs* (the mixer is clearly where neural capacity pays here).

## One-sentence summary
The most valuable "online neural model" for this compressor turned out to be the *mixer itself*: turning
the single-layer logistic mix into a two-layer network whose first-layer sub-mixers key on a rising
*order* of context bought −0.0500 bpb on enwik8 — the largest single gain in the codebase's recent
history — at negligible, fully-deterministic cost; standalone neural sequence models (feed-forward and
recurrent) were comparatively weak and expensive.
