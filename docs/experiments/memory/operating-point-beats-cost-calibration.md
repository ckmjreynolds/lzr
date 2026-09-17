---
name: operating-point-beats-cost-calibration
description: "For lzr bpb, the pipeline operating point (small vocab + LZ77 off) beats any Re-Pair grammar cost-model tuning"
metadata: 
  node_type: memory
  type: project
  originSessionId: cd030ade-6952-42a5-8f0f-034e3feb82c3
  commit: dd7a90b
  branch: v0.1.0
---

> **SUPERSEDED CODE (see [[codebase-state-u8-refactor]]):** the u8 refactor (commit 33a4aa1) DELETED
> `lz77`, `--auto`/`autotune.rs`, `--num-tokens`, and `EncodeOptions`. Re-Pair is now fixed byte-BPE at
> `VOCAB_CAP=256`, so the small-vocab operating point below is **baked in** rather than searched. The
> *finding* (small dense vocab sharpens the CM; per-rule cost models fail because the benefit is global)
> still holds; every code/flag reference below is historical.

**Code state:** branch `v0.1.0` @ commit `dd7a90b` — `--auto`/`autotune.rs` was committed here (was uncommitted when these numbers were measured; this commit captures that exact tree).

Investigated (2026-07-07) how to improve lzr bpb via better Re-Pair grammar pruning. Two things established by measurement (roundtrip-verified, on bible.txt and an 8 MB enwik8 slice):

**Local grammar cost models fail.** Both an order-0 self-information prune cost and a full CM-realized per-symbol prune cost (trial-encode metering) produced *more* rules than the default varint-width prune and lost ~1-4% bpb. Root cause: any per-rule "does inlining save coded bits here" decision is local, but the benefit of a small vocabulary is **global** (a denser token alphabet sharpens the higher-order context models) — invisible to per-rule decisions. This is a fundamental limitation, not a tuning issue.

**The real win is the operating point, worth 8-12%:** a much smaller vocabulary than the width prune picks, AND LZ77 disabled.
- bible.txt: default (width, LZ77 on) 1.8110 bpb → best 1.5878 at ~280 tokens, LZ77 **off** (−12.3%).
- enwik8 8 MB: default 2.2680 → ~2.083 at ~500-1000 tokens, LZ77 off (~−8%).
- Mechanism: small dense vocab makes order2's mixer weight jump 0.015 → 0.535 (it was useless at the default). LZ77's match records inject less-predictable bytes that disrupt the CM, and the CM's own match model already captures the repeats — so LZ77 is net-negative at small vocab (it still helps at large vocab).
- The LZ77-on and LZ77-off objectives have *different* vocab optima, so a LZ77-off calibration signal cannot steer the LZ77-on pipeline.

**Built (2026-07-07):** `src/autotune.rs` — `compress_auto` / CLI `--auto` (+ `--auto-sample`). Searches the true container-size objective over (vocabulary cap ladder × LZ77 on/off) on a prefix sample, then one full compress at the winner. Default-unchanged, gated, tested, green. Exports `compress_auto`, `AutoReport`, `OperatingPoint`, `DEFAULT_SAMPLE_BYTES` (2 MB). See [[calibration-scaffolding-status]].

Validated (roundtrip-clean): bible 4 MB default 1.8110 → auto 1.5905 (−12.2%); enwik8 8 MB default 2.2680 → auto (2 MB sample) 2.1187 (−6.6%), auto (full 8 MB sample) 2.0782 (−8.4%, 576-token cap LZ77 off). The selector finds the true optimum when the sample is representative.

**Sample-undershoot fixed with a full-input refinement pass:** after the coarse 2 MB-sample search, `compress_auto` hill-climbs the vocabulary directly on the full input (bounded, `MAX_REFINE_EVALS=6`, LZ77 fixed to the sample's choice). enwik8-8m `--auto` now lands on 576-token cap → 2.0782 (matching the full-data optimum), up from 256 → 2.1187 without refinement. `AutoReport` reports `sample_evaluations` + `refine_evaluations`.

**LZ77 is now default-off** (`codec.rs` FEATURES `lz77` `default_on: false`). CAVEAT: this *regresses the plain default* (bible 1.8110→1.8353, enwik8-8m 2.2680→2.3004, ~+1.4%) because the bare default still uses the large cost-stop vocabulary, where LZ77 *helps*. LZ77-off only wins paired with a small vocab — so the plain default is now a mild regression unless also small-vocab (or use `--auto`). This tradeoff is open with the user.

Not yet validated on binary corpora (Silesia, E.coli) — LZ77 likely still helps there, so trusting `--auto` broadly and the LZ77-off default both need breadth first.
