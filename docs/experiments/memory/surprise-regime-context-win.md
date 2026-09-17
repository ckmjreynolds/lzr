---
name: surprise-regime-context-win
description: "FINDING (2026-09-12, v0.1.0 atop b8713a8): the coder's own recent coding cost, quantized, used as a mixer selector + SSE key ('surprise' flag) wins enwik8 1.5334->1.5248 (-0.0086) byte-exact; shipped default-on; the fast (~1-byte) timescale carries the signal"
metadata: 
  node_type: memory
  type: project
  commit: b8713a8 (uncommitted working tree on top; branch v0.1.0)
  originSessionId: 6ce2376c-5193-4dd4-9aaa-c9859107f092
  modified: 2026-09-12T17:26:12.279Z
---

**Idea:** use the model's *own internal state* as context — the exponentially decayed final per-bit
coding cost (`src/surprise.rs`, integer 1/16-bit LUT, deterministic on both sides) quantized to a
log2 level. No byte-history context can express "we are currently in an unpredictable stretch".

**Result on 20 MB enwik8 slice (base 4,101,781 B / 1.6407 bpb), cumulative sweep:**
- blend-layer selector + (bucket × node) APM: −4,861 B
- fifth sub-mixer keyed on bucket: −8,725 (dropped: <0.0005 extra once the keys below exist)
- order-1 sub-mixer keyed (prev byte × fast level): −13,299 ; order-1 APM keyed (fast level × prev
  byte × node): −12,961 ; **both together: −19,445** (they act on different components and stack)
- timescale: fast shift 3 (~1 byte) ≈ shift 2 > shift 4 ≫ shift 5; slow shift 5..8 flat (<0.0003).
  Final (shift 3/5): 4,077,743 B = −24,038 = −0.0096 bpb.
- 64-way (fast × slow) key on the mixer: −1,047 more, not worth it; on the APM it costs ~550 MB.

**Full enwik8:** 19,167,547 → 19,060,465 B (1.5334 → **1.5248**, −0.0086), byte-exact roundtrip,
+6% time (660→702 s), +36 MB RSS. Shipped **default-on** as flag bit 28 (default bits 0x1CFF_FFFB).
Not yet validated on enwik9 (needs user approval per CLAUDE.md).

**Why it works:** the fast level is essentially "how surprised was the coder by the last byte"; the
mixer can then trust deep contexts less right after a miss, and the APM re-calibrates confidence per
regime. It is the cheap cousin of PAQ's match-mismatch context, but applied to the *whole* mix.

**How to apply:** this "model-internal-state-as-context" family is open — next candidates are a few
sign bits of the selected mixer weight vector as an SSE key, and per-model recent-loss levels as
mixer selectors. Gate on the 20 MB slice as here (~2.5 min/run, 3 variants in parallel on 12 cores).
Related: [[two-layer-mixer-win]], [[word-xmltag-sse2-sweep]], [[anchor-memories-to-commits]].
