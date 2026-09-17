---
name: context-selected-mixer-weights
description: "Context-selected mixer weights (per-previous-byte selector) landed -0.0305 bpb on v0.1.0 — a cheap, proven win"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3db5e992-1174-42cb-8e53-10d8fe1c3f4c
---

Ported the lpaq-class "context-selected mixer weights" lever from the [[hutter]] branch to `v0.1.0`
(2026-07-09). The mixer previously held one weight set per bit-position, global across context; now it
holds `MIX_CONTEXTS = 256` × bit-position weight sets, selected per bit by the **previous byte** (order-1
context, constant across a byte's 8 bits, so determinism/L(D) hold).

- `src/mixer.rs`: `Mixer::new(n, bit_positions, contexts)`, offset `(sel*bit_positions + bpos)*n`.
- `src/entropy.rs`: `sel = hist.last().unwrap_or(0)`, passed to `mixer.mix`.
- **−0.0305 bpb** on a 20 MB enwik8 slice (1.8452 → 1.8147), beating the journal's projected −0.022.
- **Confirmed on full enwik8: 1.6843 → 1.6561 bpb (−0.0282), byte-exact roundtrip** (a new default best
  on this branch). All tests + clippy/fmt/`--no-default-features` clean. ~112 KB extra memory.

Was item #1 of a review of proven Hutter levers not yet in `v0.1.0`. Other high-ROI next levers from
that review (not yet done): indirect "what-followed" context models (−0.047 on Hutter), word-context
arms (−0.053), longer-key match models, set-associative/checksummed StateMap tables. See
[[rare-byte-escape-before-repair-blocked]] for the preprocessor idea (#3) that was tried alongside and
proved a dead-end.
