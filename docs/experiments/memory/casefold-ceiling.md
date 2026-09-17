---
name: casefold-ceiling
description: "The total cost of representing case in lzr is ~0.0150 bpb (20MB enwik8 ceiling diagnostic) — below the ship bar, so replacing inline casefold with a fancier case model is not worth it"
metadata: 
  node_type: memory
  type: project
  commit: dc002da
  branch: v0.1.0
  originSessionId: 2b12dfcf-da16-4ca8-af10-2c91c6fa8a25
---

Diagnostic run 2026-07-10 (HEAD `dc002da`) to bound the payoff of any better case-handling scheme
(dedicated case-bit model, LSB-permutation, folded-context case model) before building one.

**Method (the reusable "ceiling" trick):** compress a fully lowercased enwik8 (`LC_ALL=C tr 'A-Z'
'a-z'`) and compare coded size to normal enwik8. The delta is *everything* the pipeline spends
representing case; any reversible scheme still owes `H(case|text)`, so the achievable win is only a
fraction of the delta.

**Result (first 20 MB, default pipeline):**
- normal: 4,536,779 B = 1.8147 bpb
- lowercased: 4,499,402 B = 1.7998 bpb
- **case budget = 37,377 B = 0.0150 bpb** (0.19% of input, ~0.83% of the coded stream)

**Verdict: don't build a fancier case scheme.** 0.0150 bpb is the whole ceiling, and the reachable
slice (after the irreducible `H(case|text)` — proper nouns, acronyms, markup are genuinely informative)
sits at/below the level of experiments already rejected here (neural arm −0.0053 rejected). Case cost
per byte tends to *shrink* with more data, so full enwik8 is ≤ this. Current inline shift/caps-lock
`casefold` already captures most of it. Consistent with the baseline-saturation pattern
([[neural-arm-replay-result]], [[order8-model-result]]). Related dead-end: [[reorder-alphabet-permutation-regresses]].
