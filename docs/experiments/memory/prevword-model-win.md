---
name: prevword-model-win
description: "prevword + prevword2 word-context models SHIPPED default-on; enwik8 1.5457->1.5334, enwik9 1.2729->1.2585, decode RSS 8.77 GB"
metadata: 
  node_type: memory
  type: project
  originSessionId: e37f72ec-6a60-4908-8dad-3f8e84018e78
---

Two new word-context models added default-on on the v0.1.0 branch, each keyed via the lpaq-class
`BitHistory` state path (1 byte/slot, calibrates sparse word contexts) with an abstain guard, folded
with the current partial word + bit-tree node:
- `prevword` (src/models/prevword.rs, bit 26): keys on the **previous completed word** (word-bigram) —
  the cross-word signal the existing `word` run model (current partial word only) cannot see.
- `prevword2` (src/models/prevword2.rs, bit 27): keys on the **previous two** completed words (word
  trigram) — the continuation prevword leaves ambiguous ("united states"->"of").

New default bits **0x0CFF_FFFB** (was 0x00FF_FFFB before prevword). Measured cumulatively atop the
bit-history-states baseline (commit dc0f6c2 = enwik8 1.5457 / enwik9 1.2729), all byte-exact:
- enwik8: 1.5457 -> 1.5372 (prevword, -0.0085) -> 1.5334 (prevword2, -0.0037)
- enwik9: 1.2729 -> 1.2641 (prevword, -0.0088) -> 1.2585 (prevword2, -0.0055)
- Both deltas GREW at enwik9 vs enwik8 (more word n-grams recur at scale) — unusual; most recent adds
  shrink with scale.
- enwik9 decode peak RSS: prevword 8.52 GB, **prevword2 8.77 GB** (each model = one ~256 MB BitHistory
  table at enwik9; +0.25 GB matched prediction). Under the 10 GB Hutter ceiling. Encode ~7.3-7.7 GB.
  Measured via /usr/bin/time -l on the raw ./target/release/lzr binary (NOT `cargo run` — that measures
  cargo, not the worker). enwik9 decode is SLOW (~1 h single-thread through the full stack).
- mixer weights: prevword w=+0.021 (word/match territory), prevword2 w=+0.011.

**Why:** the word model was a known gap — `word`/[[word-xmltag-sse2-sweep]] only conditions on the
current partial word. **How to apply:** next word-model lever is a WRT-style dictionary preprocessor
(the big Hutter word lever, still untried), or a 3-word context (likely marginal — prevword2's delta is
already half prevword's on enwik8). If chasing memory back, each model's `BitHistory` width is
`hashed_bits(capacity)` (cap 2^28 = 256 MB at enwik9); a lower cap trades a little win for RSS. Anchor
per [[anchor-memories-to-commits]] once committed. Supersedes the enwik9 decode-RSS figure in
[[bit-history-states]] for this branch. UNCOMMITTED as of writing (working tree on v0.1.0).
