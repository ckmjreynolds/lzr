---
name: two-layer-mixer-win
description: mix2 two-layer mixing network — biggest recent enwik8 win (−0.050 bpb); config + why
metadata: 
  node_type: memory
  type: project
  originSessionId: 698e98a9-12e5-4ec7-b794-bbf01fa4123f
---

Overnight session (branch v0.1.0, atop commit 4910b12) built `mix2`, a **two-layer mixing network**
that replaces the single-layer logistic mixer. Measured **enwik8 1.6086 → 1.5586 bpb overall (−0.0500)** and **enwik9 1.3252 → 1.2772 (−0.0480)**
— the single largest compression gain in the codebase's recent history (word was −0.0153). +7% time
(clean 20 MB 84.4→90.6 s), pure i32 (no float), **byte-exact roundtrip validated on full enwik8 AND
enwik9** (1 GB, ~12 GB RSS, ~76 min encode / ~74 min decode single-thread). enwik9 baseline (current
model set) is 1.3252 overall / 1.3001 entropy. Lives in `src/mixer.rs` as `TwoLayerMixer`,
gated by the `mix2` flag (registry in `src/codec.rs`), **default-ON** (shipped; default profile bits
0x00FF_FFFB). Disable with `--disable mix2`. Full writeup: `NEURAL_MODELS_REPORT.md`.

Final config: **L1=4 sub-mixers over an order-1..4 context progression** (previous byte dense-256, then
rolling 2/3/4-byte hashes at 2^14 buckets each); layer-2 blends their logits with one global weight set
per bit position. LR1_SHIFT=11 (layer-1 *faster* — deep contexts seen rarely), LR2_SHIFT=13 (layer-2
*slower*). W_CLAMP=1<<19 safety rail.

Two decisive findings:
1. **Decoupled training is mandatory.** Textbook coupled backprop (layer-1 grad = layer-2 weight × err)
   *diverges over long inputs* (helped 5 MB, regressed 20 MB by +0.077) because the layer-2 gain
   amplifies the layer-1 step. Fix: train each layer-1 sub-mixer *directly* toward the bit from its own
   squashed output (each is then a standard stable mixer); layer 2 independently learns the blend.
2. **Sub-mixer context = rising *order* is where most of the win came from** (bigram→trigram alone was
   −0.015 more). Order-5/6 regressed (too sparse); global/per-prevbyte layer-2 context both worse.

The delta GROWS with scale (5 MB −0.042, 20 MB −0.048, enwik8 −0.050) — LR tuning on 20 MB did not
overfit. See [[entropy-model-sweep-u8]] for the prior model set this builds on.
