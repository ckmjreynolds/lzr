---
name: rare-byte-escape-before-repair-blocked
description: "Rare-byte escaping as a standalone stage before Re-Pair is architecturally blocked — don't re-attempt it that way"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3db5e992-1174-42cb-8e53-10d8fe1c3f4c
---

Tried (v0.1.0 branch, 2026-07-09) adding an `escape` preprocessor stage just before `repair` to free
Re-Pair slots — the idea being Re-Pair's rule budget is `256 − distinct_bytes`, so escaping the rarest
bytes out of the terminal alphabet should hand Re-Pair more non-terminal slots. **It cannot work as an
independent stage. Confirmed dead-end — do not re-attempt this shape.**

**Why:** reversibility forces the stage to store each escaped byte's *value* in its self-describing
`(escaped, index)` mapping header. Those value-bytes land in the stage's output stream, so when Re-Pair
censuses that output every "escaped" byte is still present (count 1 in the header). Net alphabet change
is `−K + K(header) + 1(marker) = +1` distinct — Re-Pair *loses* a slot.

**Measured:** repair rules went 76→74 (down, digram disruption), the `ESCAPE_FRACTION` sweep was dead
flat 0.0002→0.03 (the `K ≤ present/2` cap binds, ~90 bytes escaped regardless), final bpb regressed
+0.0025–0.0034 on enwik8 slices. My own `escaping_reduces_distinct_bytes` unit test caught it raising
distinct by 1. Reverted to green.

**Deeper reason it was marginal anyway:** Re-Pair here is beneficial-digram-bound (~74–76 useful merges
on enwik) sitting in front of a strong context mixer, so freed slots wouldn't convert to bits — the same
"upstream factoring doesn't help a strong CM" pattern the [[hutter]] journal hits repeatedly.

**Only viable path (not pursued):** do the escaping *inside* the Re-Pair stage — escape its rarest
terminals during the build, store their values in Re-Pair's own grammar header (which flows to entropy,
never re-censused), and reuse the freed byte values as extra non-terminals. A `repair/` redesign, not a
stage, with uncertain payoff.

Measured atop the context-selected mixer-weights change (per-previous-byte selector, `MIX_CONTEXTS=256`
in mixer.rs) which landed −0.0305 bpb on a 20 MB enwik8 slice — both uncommitted on `v0.1.0` at time of
writing. See [[operating-point-beats-cost-calibration]] and [[calibration-scaffolding-status]] for other
removed/rejected preprocessor-side experiments on this project.
