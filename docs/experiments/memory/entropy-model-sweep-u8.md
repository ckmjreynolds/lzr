---
name: entropy-model-sweep-u8
description: post-u8-refactor cumulative entropy-model sweep on enwik8; final default set + per-model deltas
metadata: 
  node_type: memory
  type: project
  originSessionId: 647a2a38-56ce-4efc-9f97-1ab70ec38bd8
---

Cumulative entropy-model sweep on enwik8 (block-size 100M = single block, `--threads 1`), run
2026-07-09 on branch `v0.1.0` atop commit 33a4aa1 (the u22→u8 "token" refactor). Method: add one
model at a time in registry order, keep it only if whole-stream bpb dropped.

Whole-stream bpb as each model was added (default pipeline casefold→entities→repair→entropy+sse):
- null (baseline) 4.3804
- +order0 4.2364 ✓ · +order1 3.0926 ✓ · +order2 2.3348 ✓ · +order3 1.9294 ✓ · +order4 1.7957 ✓
- +order5 1.7512 ✓ · +order6 1.7322 ✓ · +order7 1.7227 ✓
- +sparse2 1.7217 ✓ (marginal −0.0010) · +sparse24 1.7221 ✗ (regressed +0.0004, mixer weight went
  negative → dropped) · +match 1.6893 ✓ (biggest single win after the order chain, −0.0324)

Final default set (made `default_on` in the `FEATURES` registry): order0..order7, sparse2, match —
enwik8 → 1.6893 bpb, round-trips byte-identical.

`sparse24` (SparseModel positions 2+4) was **removed from the registry entirely**, not just left
off: an enwik9 confirmation (1 GiB single block) reproduced the regression (default 1.3742 bpb /
171,776,211 B vs +sparse24 1.3748 bpb / 171,844,185 B — costs ~68 KB, mixer weight −0.003). The
`SparseModel` type still supports arbitrary offsets; no feature uses multi-offset now.

Bit layout after removal (bit = registry index): every feature is default-on, so
`Profile::default().to_bits()` == 0x7FFF; known-bits mask is 0x7FFF (bit 15+ rejected). Same work also
made `repair` non-mandatory (`MANDATORY` now empty; every stage is optional).
(**Bit layout now stale:** order8..11 later extended the chain and Re-Pair was made **default-off**
(bit 2) — default is now `0x7FFFB` over 19 bits (all set except repair). See [[order8-model-result]],
[[repair-net-harmful-default-off]], [[codebase-state-u8-refactor]].) Format is pre-1.0 and
unstable, so reassigning these bits is fine. Note enwik9 1 GiB blocks are memory-hungry (~14–19 GB
peak RSS: hashed tables hit the 28-bit MAX_HASH_BITS cap at 1e9 bytes). See [[anchor-memories-to-commits]].
