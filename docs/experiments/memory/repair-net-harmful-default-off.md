---
name: repair-net-harmful-default-off
description: "Re-Pair tokenization now REGRESSES enwik8 by ~0.0222 bpb (1.6339 off vs 1.6561 on) because the strengthened CM subsumes it; repair made default-off. The current deterministic CM is stronger than the hutter branch's deterministic part was."
metadata:
  node_type: memory
  type: project
  commit: fe3c25f
  branch: v0.1.0
  originSessionId: 2b12dfcf-da16-4ca8-af10-2c91c6fa8a25
---

Measured 2026-07-10 (full enwik8, single 1 GiB block, roundtrip-verified) while decoupling the
[[reorder-alphabet-permutation-regresses]] confound:

- **repair ON (was default):** 1.6561 bpb, 71 rules
- **repair OFF:** **1.6339 bpb** — disabling Re-Pair *wins* **−0.0222 bpb** (~276 KB on enwik8).

**Re-Pair is net-harmful at the current operating point.** This is bigger than order8-11 (−0.0050) and
approaches the mixer-context win (−0.0282). Cause: the CM has outgrown it. Re-Pair was justified back at
`dd7a90b`, but order8..11, per-previous-byte mixer weights, and SSE all landed *afterward*, and the
`match` model already captures the repeats — so Re-Pair's ~71 token boundaries now inject
less-predictable structure that disrupts the models more than the grammar shrinks the stream. Classic
"model outgrew the tokenizer" (same arc as byte-level NNs closing the gap to tokenized ones at scale;
and consistent with [[operating-point-beats-cost-calibration]]'s note that LZ77/tokens disrupt a strong
CM).

**Shipped this session:** `repair` `default_on: false` in the `FEATURES` registry (kept as opt-in
`--enable repair` for corpora where it still pays); default pipeline is now casefold -> entities ->
entropy. Also set `DEFAULT_BLOCK_SIZE = 1 << 30` (1 GiB) so enwik9 is a single block (max per-block CM
context, no boundary resets). New enwik8 default best: **1.6339 bpb**.

**User's framing (worth remembering):** the current codec's *deterministic* stack is now stronger than
the deterministic part of the `hutter` branch is/was. Why SOTA Hutter compressors still tokenize: it's a
*different* kind — external word-dictionary (WRT) for context reach + injected knowledge, or subword BPE
to cut neural-arm compute + extend context — not a self-built 256-cap Re-Pair grammar, which a strong CM
strictly outgrows.

**OPEN / TODO before removing repair entirely:** −0.0222 is enwik8-only. Validate `--enable repair` vs
off on **enwik9** and a non-text corpus (Silesia/E.coli) — repair may still help where the CM is weaker
or redundancy is higher. Default-off is safe now; deletion needs that breadth.
