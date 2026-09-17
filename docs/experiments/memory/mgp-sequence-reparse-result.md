---
name: mgp-sequence-reparse-result
description: "lzr MGP (minimal-grammar-parse) sequence re-parse: implemented, opt-in --mgp, small but real bpb win, largest at large vocab"
metadata: 
  node_type: memory
  type: project
  originSessionId: 55b4ce22-75f9-42b6-b8dc-4a72317512b2
  commit: dd7a90b
  branch: v0.1.0
---

> **SUPERSEDED CODE (see [[codebase-state-u8-refactor]]):** the u8 refactor (commit 33a4aa1) DELETED
> `--mgp`/`mgp.rs`, `--auto`/`autotune.rs`, `--num-tokens`/`--min-match`, `EncodeOptions`, and `lz77`.
> None of them exist at HEAD, and `--auto` is NOT the CLI default any more. The *finding* (MGP is a
> real re-parse win that grows with vocab size and fades at the small-vocab operating point) is
> historical context; every flag/API reference below is stale.

**Code state:** branch `v0.1.0` @ commit `dd7a90b` — `--mgp`/`preprocessors/repair/mgp.rs` was committed here (with `--auto`), capturing the tree these numbers were measured against.

Implemented MGP (minimal-grammar-parse) in lzr (2026-07-07). Unlike the removed local grammar-prune
cost models ([[calibration-scaffolding-status]]), MGP does NOT change which rules exist — it re-parses
the top-level Re-Pair **sequence** into a minimum-cost cover of the same text over the fixed pruned
symbol set (Aho-Corasick-style trie of symbol expansions + shortest-path DP, EM-iterated 3 rounds with
a self-information cost proxy). Binary rules untouched, so no format/decoder change; encode-only,
opt-in via `--mgp` / `EncodeOptions::with_mgp()`, default OFF. Lives in
`src/preprocessors/repair/mgp.rs`, called in `RepairTokenizer::forward` after prune, before serialize.
Roundtrip-clean (proptests + CLI cmp on bible & enwik8-8m). ~7% encode-time overhead at 512 tokens.

**Measured bpb (container size / original, roundtrip-verified, LZ77 off):**
- bible.txt: 256 −0.43%, 384 −0.35%, 512 −0.99%, 1024 −2.06%, cost-stop −4.68% (1.8353→1.7495).
- enwik8-8m: 256 +0.02% (neutral), 384 −0.20%, 512 −0.15%, cost-stop −2.68% (2.3004→2.2388).

**Key finding:** MGP is a real, always-positive-or-neutral win, but the effect **grows with vocab
size** and **shrinks toward the small-vocab operating point** where `--auto` lives — at the smallest
vocab (256) it is marginal/neutral. This is consistent with [[operating-point-beats-cost-calibration]]:
once the CM dominates on a dense small alphabet, grammar improvement has little headroom. MGP's biggest
wins are at the cost-stop **default** vocab, where it recovers most of the LZ77-off-default regression
(bible cost-stop+MGP 1.7495 even beats the old LZ77-on default 1.8110).

**Best absolute** unchanged in ranking: bible still optimal at ~256 tokens (1.5905 base → 1.5836 MGP).
Operating-point selection is stable under MGP (256 still wins), so MGP shifts the whole curve down
rather than moving the optimum.

**Integrated (2026-07-07, at user's request):** MGP is now **always-on inside `--auto`** (added
`.with_mgp()` in `autotune::options_for`, so every sample/refine/final compress uses it), and **`--auto`
is now the CLI default** for a bare compression. Default `lzr in out` → auto+MGP (bible 4 MB → 256-token
cap, lz77 off, **1.5836 bpb**, roundtrip-clean). Opt-outs in `src/main.rs::run`: any explicit pipeline
flag (`--num-tokens`/`--min-match`/`--enable`/`--disable`) selects the manual path; `--no-auto` forces
manual with no flags; `--auto` forces the search regardless (conflicts_with no_auto). This kept all
existing cli.rs tests on the manual path unchanged (they all pass flags). NOTE: the auto default was set
"for now" — a provisional flip, revisit before shipping. The library `compress`/`compress_owned_with`
API defaults are unchanged; only the CLI default flipped.

Deferred follow-up: dead-rule GC (MGP can orphan rules that are still serialized) — not done, would need
downward-closed removal + vocab change.
