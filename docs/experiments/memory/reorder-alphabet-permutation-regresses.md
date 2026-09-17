---
name: reorder-alphabet-permutation-regresses
description: "Static byte-alphabet reordering before the entropy coder REGRESSES enwik8 (both a class-clustering and a frequency-ranked permutation); native ASCII structure beats any permutation for a CM bit-tree. Stage was built, measured, and REMOVED."
metadata:
  node_type: memory
  type: project
  commit: dc002da
  branch: v0.1.0
  originSessionId: 2b12dfcf-da16-4ca8-af10-2c91c6fa8a25
---

Tested (2026-07-10, HEAD `dc002da`) the "reorder terminals to aid prediction" idea: a permutation of
the 256-byte alphabet placed between `entities` and `repair` so the entropy coder's MSB-first bit-tree
splits on structure early. Built as `src/preprocessors/reorder.rs` (feature `reorder`, default-off).
**Two permutations tried, both REGRESS. Stage REMOVED after the experiment.**

- **Class-clustering** (letters->0..25, then space/punct/brackets/symbols/digits/uppercase/controls),
  20 MB enwik8: 1.8147 -> **1.8158** (+0.0011).
- **Frequency-ranked** (most-frequent byte -> code 0; transmits a 256-byte permutation-table header),
  full enwik8: pure effect **+0.0188** (repair-off both sides: 1.6339 -> 1.6527). WORSE than the class
  order. Confound found: the 256-byte header contains every byte value, which starved Re-Pair to 0 rules
  (see below); isolate by disabling repair on both sides.

**Durable lesson — the "textbook-optimal" intuition is WRONG here.** Frequency->short-codes is optimal
for a *memoryless* coder, but lzr's CM bit-tree feeds on the **semantic structure of native ASCII**
(letters share high bits, digits contiguous 0x30-0x39, case is bit 5). Frequency ranking shreds that
structure, so it hurt *more* than the class order (which preserved some of it). A permutation is a
bijection, so it also **cannot merge CM contexts** ("The"/"the" stay distinct) — the only channel it
touches is the bit-tree, and the native ASCII layout is already near-optimal for it. No static
reordering helps at this operating point. Same baseline-saturation family as [[casefold-ceiling]].

**Bonus finding from the confound:** with the CM this strong, Re-Pair itself is net-harmful — see
[[repair-net-harmful-default-off]] (disabling it wins −0.0222 bpb on enwik8). So the reorder "win" first
seen (1.6561->1.6527) was just reorder accidentally disabling repair, minus reorder's own +0.0188 harm.

Anchored per [[anchor-memories-to-commits]]; superseded-code context in [[codebase-state-u8-refactor]].
