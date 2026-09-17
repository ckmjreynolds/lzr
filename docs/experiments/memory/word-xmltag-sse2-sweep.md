---
name: word-xmltag-sse2-sweep
description: "FINDING — word + xmltag models and sse2 APM-chain added default-on (enwik8 1.6339→1.6092); digit model neutral, default-off"
metadata: 
  node_type: memory
  type: project
  originSessionId: 30aef67a-710d-4a6e-b488-00e7e6d86638
---

SHIPPED as da1d9c3 (branch v0.1.0, atop fe3c25f). Full-enwik8 measurements, cumulative on the default profile. New models are `run.rs` (RunModel, char-class trailing-run keyed on bit-tree node; word=alpha, digit=digit) and `xmltag.rs` (nearest enclosing `<tag>` name within 64B). `sse2` is a flag chaining a second APM keyed on (prev-byte × node), blended `(refine*3+p)>>2` after the order-0 `sse` APM.

Cumulative enwik8 bpb (baseline = old default 1.6339):
- **word** (RunModel alpha): 1.6186 (−0.0153) — biggest win, KEPT default-on
- **+sse2** (order-1 APM chain): 1.6102 (−0.0084) — KEPT default-on
- **+digit** (RunModel digit): 1.6101 (−0.0001, ~1.8KB) — NEUTRAL, so **REMOVED entirely** (id low-digits near-random; real structure is cross-token delta a trailing-digit ctx can't see → pursuing an iddelta model instead)
- **+xmltag**: 1.6092 (−0.0010) — KEPT default-on
- **+iddelta** (iddelta.rs): 1.6086 (−0.0006, w=+0.007) — KEPT default-on. The "genuine" cross-token id model: predicts a number's digits from the PREVIOUS completed number (held in model state, reaches across pages). ~6x the trailing-digit model → confirms id/timestamp structure is cross-token (shared high-order prefix of near-monotonic seqs), not in a single number's digits. Uses global last-number (mixes fields); a per-field/per-length version might do better (untried).
- **New best enwik8 = 1.6086 bpb** (was 1.6561 before context-mixer weights; 1.6339 immediate baseline)

Default bitmask now 0x7FFFFB (23 features; all on except repair=bit2). Registry order appended after `match`: word(19), xmltag(20), sse2(21), iddelta(22). RunModel (run.rs) stays generic (char-class predicate); only word instantiates it (digit swept + removed). word/xmltag/sse2 shipped as da1d9c3; iddelta shipped as 4910b12.

Method notes (learned the hard way): full-enwik8 compress is **single-threaded** (default 1 GiB block = 1 block) → ~7min each, ~6.2GB RSS. Run ONE compression per background task; multi-run batch scripts >~15min get SIGTERM'd and 0-byte outputs (NOT the model — that scared me twice). Don't run concurrent enwik8 jobs (contention → killed processes). See [[codebase-state-u8-refactor]], [[entropy-model-sweep-u8]], [[order8-model-result]], [[context-selected-mixer-weights]], [[anchor-memories-to-commits]].

Next ideas not yet tried: word-bigram (predict word from previous whole word), 2-stage mixer, cross-token ID delta model.
