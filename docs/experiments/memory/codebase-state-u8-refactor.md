---
name: codebase-state-u8-refactor
description: "What the u22->u8 byte-BPE refactor (commit 33a4aa1) REMOVED — lz77, --auto/autotune, --mgp, --num-tokens/--min-match, EncodeOptions, u22 uleb codec — so older memories naming them are stale"
metadata: 
  node_type: memory
  type: project
  commit: dc002da
  branch: v0.1.0
  originSessionId: 2b12dfcf-da16-4ca8-af10-2c91c6fa8a25
---

Orienting note so I stop citing machinery that no longer exists. The `u22 -> u8` byte-BPE refactor
(**commit 33a4aa1**, "Refactor u22 -> u8 tokens, blocking, multi-threading") deleted a large amount of
the earlier `dd7a90b`-era code. Verified gone at HEAD `dc002da` (`v0.1.0`):

- **LZ77 is REMOVED** (`src/preprocessors/lz77.rs` deleted) — it is not "default-off", it does not exist.
  The `match` entropy model captures repeats instead.
- **`--auto` / `src/autotune.rs` / `compress_auto` / `OperatingPoint` — REMOVED.**
- **`--mgp` / `src/preprocessors/repair/mgp.rs` / `.with_mgp()` — REMOVED.**
- **`--num-tokens`, `--min-match`, and `EncodeOptions` — REMOVED.** No per-run encode options survive;
  the decoder recovers everything from the stream.
- **`uleb128` u22 fixed-width codec — REMOVED.** Only the general `u64` codec remains (container profile
  field + Re-Pair rule count).

What replaced it: Re-Pair is now **byte-BPE** — symbol ids *are* byte values, terminals keep their
original byte, non-terminals take the block's free byte slots, grammar serialized as plain
`[R][(sym,left,right)]*R[sequence]`. It is **fixed at `VOCAB_CAP = 256`** with no per-run params — so the
"small-vocab operating point" that `--auto` used to search for is now effectively **baked in** as the
default behavior (the win from [[operating-point-beats-cost-calibration]] was adopted as fixed design,
not exposed as a knob).

CLI is now just: default compress / `.lzr` decompress, `-z`/`-d`, `--enable`/`--disable <feature>`,
`--list-features`. Default pipeline is **casefold -> entities -> entropy** — Re-Pair is now
**default-off** (it regresses the strengthened CM, see [[repair-net-harmful-default-off]]) and
`DEFAULT_BLOCK_SIZE` is **1 GiB** (`1 << 30`, so enwik9 is a single block). A `reorder` stage was
briefly added and removed this session ([[reorder-alphabet-permutation-regresses]]).

Supersedes the present-tense code claims in [[operating-point-beats-cost-calibration]] and
[[mgp-sequence-reparse-result]] (their *findings* still hold; their *code references* are historical).
See [[anchor-memories-to-commits]].
