# lzr → Hutter Prize: Status Review & Candidate Roadmaps

_Prepared 2026-07-24 from the `v0.1.0` branch state, the `hutter` branch JOURNAL.md (359 commits,
2025-04 → 2026-07-03), and current external scoreboards. Intended as a living document for
collaborative exploration; nothing here is a commitment._

---

## 1. Where the two branches stand

### 1.1 `v0.1.0` / `main` — the general-purpose library

The crates.io-published, composable byte-stream compressor. Clean layering (container → codec →
pipeline), strict lints, proptest reversibility, multi-threaded blocks, self-describing container
with Adler-32.

- **enwik8 1.5334 bpb, enwik9 1.2585 bpb** (default pipeline: casefold → entities → CM entropy).
- Entropy stage: order0–11 + sparse2 + match + word/xmltag/iddelta/prevword/prevword2 models,
  bit-history StateMaps, per-prev-byte mixer contexts, two-layer `mix2` network, SSE/SSE2 chain.
- enwik9 decode RSS 8.77 GB.

This branch is the *product*; it absorbs portable wins from the hutter line (mix2, bit-history,
prevword all originated as hutter-side ideas). It is **not** the prize vehicle: it pays container
overhead, is multi-threaded (prize requires single-core), and is ~0.13 bpb behind the hutter stack.

### 1.2 `hutter` branch — the prize vehicle (v9 era)

Nine architecture generations are recorded in JOURNAL.md. Summary of the eras and their verdicts:

| Era | Bet | Outcome |
|---|---|---|
| v1–v5 | Pre-trained MoE transformer → arithmetic coder | Works (95M teacher codes enwik8 at 1.08 L(C)) but dies on the **L(D) wall**: shipped weights are paid ~2×; the net-optimal size (~44M params) still lands net ~1.42–1.54 |
| v6 | Word/symbol tokenization + structural routing | BPE beats word tokens; routing a unit-error mirage |
| v7 | Deterministic cmix/PAQ-class stack | 1.4267 bpb enwik8 at L(D)≈0 — proved the deterministic path |
| v8 | Ternary BitNet transformer, train-GPU/ship-Rust | Net 1.4160 enwik9 (L(D)=0.16 kills it); **kernel work survives**: int8 `sdot`, ternary matvec, KV-cache, ~45× CPU-beats-GPU for batch-1 decode |
| v9 (current) | Deterministic CM + tiny frozen net + online "warming heads" + offline global transforms | **enwik9 net 1.1303** |

**Current standing best (2026-07-03): net 1.1303 bpb** = L(C) 1.1173 (139,661,944 B) +
L(D) 0.01305 (binary 815,504 B, counted 2×). Encode 12.3 h / decode ~11.9 h on the dev machine,
inside the ~41 h judging budget; coding RSS ~8.6 GB under the 10 GB cap.

The v9 stack, in pipeline order:

1. **Reorder** — greedy TF-IDF nearest-neighbour article ordering (encode-side only; decoder
   restores by in-content `<id>` sort, so the permutation ships for free). Worth ~−0.013 at enwik9.
2. **Dictionary** — static 4000-word DRT-style canonicalizing dictionary + suffix-array-mined
   substrings, shipped in the binary (~30 KB, clearly net-positive vs. any online variant).
3. **LZ factoring** of long repeats.
4. **Deterministic CM core** — order chain with bit-history StateMaps, per-model table sizing
   (HASH_BITS up to 27–28), match models at keys 4/8/12/16, sparse/skip contexts, indirect models,
   word models, field-aware numeric model.
5. **Mixing** — context-selected weights + regime selectors, an **M1 residual neural mixer**
   (cmix-topology MLP over the ensemble), SSE/APM chain.
6. **Frozen pretrained byte-MLP** (K=32 window, H=256, ~500 K params, q8 blob ~338 KB in-binary)
   plus **11 online "warming heads"** — per-context linear readouts over the frozen embedding,
   trained online on both sides, L(D)≈0. The heads are the branch's signature invention.

**Owed before any submission:** (a) decode gate on the 1.1173 archive itself; (b) a
submission-grade live-RAM measurement (instantaneous RSS ran 12–13 GB from allocator-arena
lag during reorder, though the live ledger is ~8.6 GB) and possibly an arena-conscious reorder.

---

## 2. The scoreboard, the rules, and the gap

**Prize record:** fx2-cmix (Kaido Orav & Byron Knoll, Sept 2024), **S = 110,793,128 B ≈ 0.886 bpb**.
**Prize line:** ≥1 % improvement required → **S ≤ 109,685,196 B ≈ 0.8775 bpb**. Payout ≈ 5,000 € per
1 % below the record (~470 k€ notionally remains in the fund).

**Constraints:** single CPU core, no GPU, ≤10 GB RAM, ≤100 GB disk, time ≤ 70,000/GB5 hours
(~41 h equivalent on the dev machine, both directions each), S counts the shipped binary twice
(`comp9a == decomp9` shape).

**Where 1.1303 sits** (LTCB, updated 2026-07-08, total-size column):

| Program | enwik9 total | bpb | Eligible? |
|---|---|---|---|
| nncp v3.2 | 107,261,318 | 0.858 | No (GPU) |
| cmix v21 | 108,244,767 | 0.866 | No (~32 GB RAM) |
| cmix-lex | 109,190,109 | 0.874 | **Unknown — watch item** (new on LTCB, would beat the prize line if eligible) |
| fx2-cmix | 110,351,665 | 0.883 | Yes — the record |
| fast-cmix / cmix-hp | ~113.7 M | ~0.910 | Yes |
| starlit | 114,951,433 | 0.920 | Yes |
| phda9 1.8 | 116,587,793 | 0.933 | Yes |
| paq8px_v206 | 125,099,359 | 1.001 | (RAM-configurable) |
| **lzr (hutter)** | **~141.3 M** | **1.1303** | Yes (pending RAM/decode gates) |

So lzr currently sits between lpaq-class (~1.15) and paq8-class (~1.00). The gap to the prize line
is **−0.253 bpb, a 22 % relative reduction** — three "class jumps": lpaq→paq8 (~−0.13),
paq8→fast-cmix (~−0.09), fast-cmix→fx2 (~−0.03). The journal's own path-to-prize analysis
(2026-06-29) framed the record as a *hardware wall*: the two better ratios (nncp, cmix) are both
ineligible, and single-core memory bandwidth caps an online f32 net at ~0.3–0.4 M params. Every
roadmap below is a bet on some way around that wall.

**Honest economics:** even a successful attempt pays out tens of k€ against a year-plus of work.
The real prizes are the LTCB leaderboard, the paper the journal is already a draft of, and the
general-purpose library inheriting the wins.

---

## 3. Ground already burned (do not retread)

The journal's negatives, condensed. Each was measured, most at full-enwik8 scale or better.

**Shipped-weights capacity:** any pretrained net large enough to matter loses to its own 2× L(D)
(v5/v8, multiple sizes; the 43.6 M MoE was the net optimum and still lost). Frozen-net scaling is
**spent on every axis**: quality (06-25), context K (optimum 32), width H (H=384 erodes at scale to
net-negative; H=512 hits a dead-embedding trainer wall), and capacity-through-heads (07-02).

**Online neural arms:** the LSTM arm's marginal collapsed to −0.0068 once the warming heads
existed (subsumption, 06-28); the selective-SSM arm wins per-FLOP standalone but **loses e2e**
(batched bake-offs don't predict batch-1-online marginals, 06-30); error-threshold update skipping
trades marginal for speed ~proportionally at enwik8 scale; LR warmup hurts; depth hurts online.

**Per-event offline foresight:** match-break side channels lose 5–11× their gain; the durable rule
is *offline foresight pays only as a global, amortized decision* (reorder, dictionary), never
per-symbol. Online/LRU dictionaries: eviction is poison, and even frozen-dynamic loses to the
shipped static dict (the decoder structurally lacks frequency foresight).

**Misc closed:** case folding on the dict'd stream (lexical, not structural); coder precision >12
bit; mode×c1 and mode×byte-class selectors; column/structure contexts; neural-embedding article
ordering (the K=32 net's embedding is stylistic, not topical — TF-IDF wins 4×); 2-opt on the greedy
order (−0.0010, not worth it *at the TF-IDF metric*); multi-candidate verified match acquisition
(neutral vs. discrete higher-order keys); word/symbol tokenization vs. BPE; post-hoc ternarization;
GPU in the codec (45× slower at batch 1); finer-than-word dictionary granularity.

**Main-branch negatives that transfer:** Re-Pair tokenization regresses against a strong CM;
static byte reordering regresses; rare-byte escape before Re-Pair is structurally blocked;
nnlm/nnrnn fade with data.

---

## 4. Roadmaps

Five roadmaps, roughly orthogonal — they compose. Each idea is tagged **[port]** (proven elsewhere
in the field, just not in lzr), **[under]** (published outside this field, never adequately applied
to it — the emphasis you asked for), or **[novel]** (no known prior application). Expected deltas
are guesses calibrated against the journal's own history of slice-vs-full erosion; every one needs
the standard gate ladder (8/20 MB slice → full enwik8 → enwik9 confirmation).

### Roadmap A — Chase the deterministic frontier (lpaq → paq8 → cmix class)

The certainty play. fx2-cmix, cmix, starlit, and phda9 are all **open source**; the single
highest-certainty resource nobody fully exploits is a systematic ablation-port of the record
holder's model list and transform suite. lzr's CM core has ~20 models; paq8px has hundreds
(most worthless for enwik, some not), cmix ~2,000 driven by an LSTM mixer.

- **A1. Port the phda9/starlit transform suite** [port] — number/date encodings, UTF-8 and
  link transforms, the parts of their Wikipedia-specific preprocessing the DRT dictionary doesn't
  already cover. Grunt work, known payoff class (~−0.01 to −0.03 cumulative).
- **A2. Model-zoo expansion from fx2-cmix's list** [port] — read `fx2-cmix`'s model set, port the
  ones lzr lacks (two-dimensional/word-gap contexts, more indirect variants, match extensions),
  gate each individually. The 06-23 sessions showed this vein pays in −0.005 increments and the
  journal's methodology is already tuned for it.
- **A3. Recurrent M1 mixer** [port] — flagged by the branch's own path-to-prize analysis as *the
  top un-built lever*. cmix's mixer is an LSTM; lzr's M1 is feed-forward. Adding recurrence to the
  mixer (not a standalone arm — the arm lesson doesn't apply to the mixer position, where nmix
  already earns its keep) is the one neural upgrade with a proven in-field precedent.
- **A4. Bigger key-12/16 finder tables** [port] — measured collision starvation at enwik9
  (94/98 % occupancy, 41/54 % collision writes). 28-bit tables cost +1.5 GB — currently doesn't
  fit; pairs with D2 (RAM reclamation) or B1 (make small tables act big).
- **Target: ~1.05–1.08 net.** High confidence, high labor, zero novelty. This alone does not reach
  the prize line — fx2-cmix itself is the asymptote of this road — but every other roadmap's
  marginal is measured on top of it.

### Roadmap B — Memory systems: leapfrog, don't chase ★ (the underapplied vein)

lzr's binding losses are now *table losses*: collisions, evictions, and exact-match-only
retrieval. The caching/streaming/sketching literature solved these problems and compression has
barely noticed. This is the roadmap where lzr can do something fx2-cmix doesn't.

- **B1. Frequency-gated admission (W-TinyLFU) for CM tables** [under] ★ — Einziger et al.'s
  admission policy (Caffeine cache): on collision, replace the incumbent only if the newcomer's
  sketch-estimated frequency is higher. PAQ-class tables use overwrite-on-collision or small
  checksum buckets; nobody uses principled admission. Directly attacks the measured key-12/16
  saturation at **zero RAM growth**, fully deterministic (both sides run the same sketch).
  Cheap to build (a count-min sketch + a comparison in the write path), gate on the saturated
  finders first. This is the single best effort-to-evidence ratio idea in this document.
- **B2. Sketch-backed ultra-high-order models** [under] ★ — Talbot & Osborne's Bloom/bloomier
  language models (SMT, 2007) proved n-gram stats survive lossy sketching at ~1–2 bytes/n-gram.
  Apply to CM: back orders 16/24/32 with count-min-sketch bit-histories instead of exact slots —
  an order-24 model in 256 MB instead of 2 GB. Nobody has put sketched contexts into a
  context-mixing coder; the mixer's job is precisely to discount noisy inputs, so the sketch's
  false-positive noise lands in the component best equipped to absorb it.
- **B3. Fuzzy match model via SimHash/LSH** [under] ★ — every match model in the field is
  *exact*-context retrieval; Wikipedia's redundancy is substantially *near*-duplicate (templates,
  infoboxes, boilerplate prose, category lists). SimHash the last-k-words context, retrieve a
  similar (not identical) history position, predict its continuation through a StateMap keyed on
  (similarity bucket, agreement history). Deterministic, L(D)≈0, decorrelated from every existing
  model by construction — and decorrelation is exactly what the head sweeps showed the stack is
  starved for. Risk: alignment slippage makes predictions noisy; the bit-history machinery exists
  to learn when to trust it.
- **B4. Wikilink-graph reorder** [under] — reorder 2.0 via the *link graph* rather than (or
  blended with) TF-IDF: articles that link to each other share vocabulary. Layered Label
  Propagation / BV-style graph clustering (Boldi et al., web-graph compression) is a published,
  fast, deterministic way to get a compression-friendly vertex order — never applied to Hutter
  article ordering (fx2-cmix uses t-SNE over content). Encode-side only, so it is free to be
  expensive and free to fail. Cheap alternative in the same slot: LSA/truncated-SVD over the
  existing TF-IDF vectors before the greedy chain (fixes the known failure mode of the
  net-embedding probe — topical vs. stylistic similarity).
- **B5. Suffix-automaton match arm (windowed)** [under] — replace hashed match finding with an
  exact online suffix automaton over a sliding ~200–400 MB window: unbounded-order longest-match
  with zero collisions, O(1) amortized. RAM is the constraint (~20 B/char naïvely); a sampled or
  windowed variant may fit post-D2. Lower priority than B1 (which fixes the same loss cheaper).
- **Target: −0.02 to −0.05 on top of A**, with B1/B2 also *funding* A4-class table growth by
  making RAM go further. Medium confidence, genuinely publishable either way.

### Roadmap C — Neural under the ledger ★ (zero-L(D) capacity)

The journal's negatives close *shipped* capacity and *continuous online* capacity. They do not
close the third quadrant: **capacity trained at decode time from the decoded prefix** — zero
L(D) at any size, deterministic by construction, bounded only by the time budget.

- **C1. Mid-stream deterministic retraining** [under] ★ — at byte N (say 50–100 MB), both sides
  pause and batch-train a replacement/augmented backbone on the already-decoded prefix using
  identical, single-threaded, integer-dominant arithmetic, then resume with warmed heads on the new
  embedding. This kills *both* documented killers at once: the L(D) tax that made H=384
  net-negative (its −0.0012 L(C) was real — the +0.0026 L(D) is what erased it), and the
  train/deploy distribution gap. nncp is the existence proof that train-during-compression wins
  big; it spends GPU-hours. The lzr twist is **discrete checkpointed retrains with v8's kernel
  inheritance** (int8 `sdot`, ternary matvec — already written, already fast) instead of continuous
  f32 BPTT. **First step is a feasibility probe, not a build:** measure the int8 kernel's
  MACs/s single-core, size the largest net trainable in ~5–8 h, and check H-collapse behavior
  under the int8 regime before committing. Kill criteria are crisp; upside is the only known
  route to "capacity that matters" inside the rules.
- **C2. Richer test-time-trained heads (TTT-style)** [under] — the warming heads are precisely
  test-time training (Sun et al. 2024), discovered independently. Mine that literature for head
  parameterizations one notch richer than a logistic readout (e.g., a rank-1 fast-weight update, a
  TTT-Linear-style inner reconstruction step) — capacity growth on the proven L(D)≈0 channel. The
  head *context* axis is saturated; the head *function class* axis was never swept.
- **C3. Int8 revival of the online arm — only if C1 fails** [port] — the arm's −0.0068 marginal
  was measured at f32 bandwidth. Int8 forward + f32 shadow updates ≈ 3–4× width at equal time.
  Probably still subsumed by heads; strictly a fallback.
- **Target: unknown — this is the moonshot lane.** C1 is the one idea here that could move lzr a
  full class on its own; it is also the likeliest to die in the feasibility probe. Budget it as a
  probe, not a bet.

### Roadmap D — Rule-edge engineering (small, near-certain, do alongside everything)

- **D1. Shrink the shipped binary** [port] — 815 KB → L(D) 0.01305 is now ~5 % of the total gap.
  `opt-level=z`, `panic=abort`, strip, no `clap`, then a self-extracting pack (tiny LZMA stub —
  standard practice among entries). A 300 KB effective binary is **−0.008 bpb for zero model risk**.
- **D2. Arena-conscious reorder + RAM reclamation** — closes the owed submission-grade RAM
  measurement AND funds A4/B5. (Run reorder in a scoped subprocess or bump-arena.)
- **D3. Spend the time budget deliberately** — encode is 12.3 h against ~41 h. That's ~3× headroom
  nobody is spending. A standing "cost ledger" (bpb per hour per GB per KB-of-binary) for every
  candidate turns ship/no-ship decisions mechanical.
- **D4. Submission hygiene** — the owed decode gate on the current archive; reproducible builds;
  judge-machine dry-run plan; track the **cmix-lex** LTCB entry (if it's an eligible entrant, the
  prize line moves before you get there).

### Roadmap E — Wildcards (cheap probes, high variance)

- **E1. Template/infobox grammar model** [novel] — after reorder, consecutive articles share
  MediaWiki template structure; a model keyed on (template name, field name) rather than byte
  context. The field-aware numeric model was this idea's first step and it shipped; the
  generalization to template fields is unexplored.
- **E2. Learned hash functions for context tables** [under] — Kraska-style learned indexes to
  reduce collision entropy for skewed context distributions. Riskier and stranger than B1; only
  worth it if B1 confirms collisions are the loss it appears to be.
- **E3. Encoder-side global model selection ("portfolio flag")** [novel] — encode with 2–3
  pipeline variants, ship the winner + a 2-bit flag. Bounded, cheap, only pays if variants have
  real corpus-level variance; the journal's two-state-lever skepticism applies.

---

## 5. Recommended sequencing

The roadmaps compose into one sensible program. A/D are the floor-raisers; B is the
differentiator; C is the capped-downside moonshot.

**Phase 0 — hygiene (days):** D1 binary shrink, D4 decode gate, D2 RAM work. Banks ~−0.008 net
and makes every later number submission-grade.

**Phase 1 — the frontier grind + first novelty (weeks):** A3 recurrent mixer, A1/A2 transform and
model ports in parallel with **B1 TinyLFU admission** (first novel result, cheapest probe) and
**B4/LSA reorder metric** (encode-side, zero risk). Gate: **net ≤ 1.08** before Phase 2.

**Phase 2 — memory systems (weeks):** B2 sketched high orders, B3 fuzzy match, A4 table growth
funded by D2/B1. In parallel: the **C1 feasibility probe** (kernel MACs/s → go/no-go memo) and a
C2 head-function sweep. Gate: **net ≤ 1.02**.

**Phase 3 — decision point:** with Phases 1–2 banked, lzr is paq8-class-plus with novel
components. If C1's probe was a go, build it — it is the only identified path from ~1.00 to the
0.8775 prize line that doesn't amount to out-tuning fx2-cmix at its own game. If C1 was a no-go,
the honest reframe is: continue A/B toward fast-cmix-class (~0.91), publish the novel components
(B1/B2/B3 are papers regardless of the prize), and keep the prize as a stretch rather than a plan.

**Standing experimental discipline** (all inherited from the journal, kept because they work):
slice → full-enwik8 → enwik9 gate ladder; same-process A/B for sub-0.001 marginals; marginals
measured over the *full* stack; frozen-vs-warming erosion signatures to predict scale transfer;
one enwik9 run per approval; every result journaled, negatives especially.

---

## 6. Sources

- [Hutter Prize official site](http://prize.hutter1.net/) — rules, record, payout formula
- [Large Text Compression Benchmark](https://www.mattmahoney.net/dc/text.html) — leaderboard (updated 2026-07-08)
- [Hutter Prize — Wikipedia](https://en.wikipedia.org/wiki/Hutter_Prize)
- fx2-cmix, cmix, starlit, phda9 — open-source entries (study targets for Roadmap A)
- Einziger, Friedman & Manes, *TinyLFU: A Highly Efficient Cache Admission Policy* (2017) — B1
- Talbot & Osborne, *Smoothed Bloom Filter Language Models* (ACL 2007); Cormode & Muthukrishnan, count-min sketch — B2
- Charikar, *SimHash* (2002) — B3
- Boldi, Rosa, Santini & Vigna, *Layered Label Propagation* (WWW 2011) — B4
- Sun et al., *Learning to (Learn at Test Time)* (2024) — C2 framing of the warming heads
- Kraska et al., *The Case for Learned Index Structures* (2018) — E2
- Bellard, *NNCP* — the train-during-compression existence proof behind C1
- `hutter` branch `JOURNAL.md` — all internal figures and negatives cited above
