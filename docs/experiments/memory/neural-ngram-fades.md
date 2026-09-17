---
name: neural-ngram-fades
description: FINDING — online neural n-gram (nnlm/nnrnn) is modest + fades + compute-heavy; not for Hutter
metadata: 
  node_type: memory
  type: project
  originSessionId: 698e98a9-12e5-4ec7-b794-bbf01fa4123f
---

Overnight session (branch v0.1.0, atop 4910b12) implemented an **online neural language model** in
`src/models/nnlm.rs` (`NnlmModel`), gated by flags `nnlm` (feed-forward) and `nnrnn` (Elman recurrent),
both **default-off**. Feed-forward: last K=4 bytes → learned 16-dim embeddings → tanh hidden (H=32) →
per-bit-tree-node logistic head; trained online by backprop; hidden computed once/byte, output head
per bit.

Verdicts (all measured, kept behind flags for reference — do NOT enable for Hutter):
- **Modest and FADING**: added to the full model set it gave −0.0080 bpb at 5 MB but only −0.0050 at
  20 MB — it is just another n-gram; the order-0..11 chain + match subsume it as they warm up. Same
  pattern as [[neural-arm-replay-result]] (warmup accelerators fade at scale).
- **Recurrence didn't pay**: `nnrnn` (Elman + 1-step truncated BPTT) ≈ `nnlm` at every size
  (20 MB 1.6751 vs 1.6748). One-step BPTT doesn't learn useful long memory. Honest negative result.
- **Compute-heavy**: +55% wall time (20 MB 93.7 s → 145 s) for −0.005 — a bad Hutter trade. Per-byte
  tanh + per-bit exp dominate.
- **f32**: round-trips byte-exactly on a given build (tests pass) but NOT guaranteed reproducible
  across float toolchains, unlike every other (integer) model. Would need fixed-point before shipping.

If revisited: fixed-point + real truncated-BPTT over a horizon, or feed the net's hidden state as
extra *mixer inputs* rather than a standalone predictor. The mixer upgrade [[two-layer-mixer-win]] was
the far better use of the "neural" budget.
