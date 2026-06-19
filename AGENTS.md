# CLAUDE.md

Project-specific guidance for Claude Code working in this repo. Global Claude Code defaults still apply; this file only captures what's specific to **LZR**.

## What this is

A Hutter Prize attempt: a single Rust binary that compresses `enwik9` below the current record (~116 MB ≈ 0.928 bpb combined).

**Hutter scoring formula** (verbatim from `prize.hutter1.net/hrules.htm`):

- *Default (self-extracting archive)*: `S = length(comp9.exe) + length(archive9.exe)`. The compressor and the self-extracting archive both contain a copy of the model weights, so weights are paid for `2×`.
- *Relaxation (split compressor + bare archive)*: `S = length(comp9a.exe) + 2 × length(decomp9.exe) + length(archive9.bhm)`. **If `comp9a.exe == decomp9.exe`** (same binary used for both directions), the `2×` reduces to `1×`, giving `S = 2 × length(binary) + length(archive)`.

The favorable shape for us is **one binary serving both directions**: the binary appears twice in `S` (once as the compressor, once as the reduced-multiplier decompressor), so its bytes count `2×`. There is no submission shape where `L(D)` is counted only `1×`. Split binaries (`comp9a != decomp9`) are strictly worse (`3×` the binary cost).

**`L(D)` in bpb** = `8 × 2 × shipped-binary-bytes / 1e9` = `16 × binary-bytes / 1e9` (×8 bits/byte, ×2 penalty, over enwik9's 1 GB). Use the **actual built binary size** — code *and* embedded weights — not just the weight blob; the binary is the thing scored. The `lzr nn-test` / `compress` subcommands read `current_exe()` and report this. (Historical note: code prior to 2026-06-08 used `2 × bytes / 1e9`, which dropped the ×8 and counted only the blob, under-reporting `L(D)` ~11×.)

**Current branch is `v8-neural`** — a neural-first codec (below). Every earlier line is preserved on its own branch and is not carried here: `hutter` (v1 LZ77+PPM-D+RWKV, 1.985 bpb enwik8), `v2`–`v6` (mode-routed deterministic + neural-integration experiments), and **`v7`** (the deterministic cmix/PAQ stack, the standing best at **1.4267 bpb enwik8**, `L(D)≈0`). Their history and rationale live in [JOURNAL.md](JOURNAL.md); pull code from those branches when needed rather than keeping it here. The retired Python+MLX predecessor lives at `/Users/creynolds/Programming/lzr-train/`.

## Build

Always build via `./build.sh`, not a bare `cargo build`. The script runs fmt, clippy (default features + no-default-features) with `-Dwarnings`, release tests, and nightly coverage. It also enforces the **zero-threading-dependency** invariant via `cargo tree --features submission` — adding rayon, crossbeam, gemm, tokio, async-std (or anything that pulls them in transitively) is a regression. If `build.sh` fails, fix the root cause — do not bypass with `--no-verify` or `-A warnings`.

## Architecture constraints

- **Single binary**, **modular source** (a trait per concern so models and preprocessors are easy to add/remove — `src/{coder,mixer,fenwick,codec}.rs`, `src/models/`, `src/preprocessors/`). Direction is **file-extension-driven**, not subcommands: `lzr <in> <out>` encodes, `lzr <in>.lzr <out>` decodes. Compressor and decompressor are still the same executable — the `comp9a == decomp9` relaxation that reduces the decompressor multiplier from `2×` to `1×`. The binary still counts `2×` in total (once as the compressor, once as the reduced decompressor), but that is `1×` better than the `3×` a split-binary submission would pay.
- **Hutter judging machine limits** (the only hardware constraints actually guaranteed): single CPU core, no GPU, ≤ 10 GB RAM, ≤ 100 GB HDD, time ≤ `70,000 / Geekbench5` hours. The specific CPU model has changed before and may change again — do not hard-code assumptions.
- **Submission binary is strictly single-threaded.** No `std::thread`, no rayon, no parallel iterators.
- **Kernel strategy: architecture-neutral first.** Write clean scalar code and trust LLVM auto-vectorization with `RUSTFLAGS=-C target-cpu=native` (configured in `.cargo/config.toml`). v1's Python-side experiments showed clean scalar code reaching ~75 GOPS on Apple NEON where explicit `wide::i16x8` SIMD gave 2.9 GOPS — auto-vec is almost always the right tool. If a measurement justifies it, add an architecture-specific kernel behind a runtime dispatch (`is_x86_feature_detected!`) and keep the neutral path as the fallback.

## v8 design principles (current branch)

The v8 codec is a **pre-trained BitNet transformer over online-BPE tokens** driving an arithmetic coder. The split that makes it shippable:

- **Train on GPU, ship pure Rust.** Training (`examples/neural.rs`, behind the `neural` feature) uses `burn` on the wgpu/Metal GPU as an autodiff tensor engine — `burn` is heavy and threaded, so it is **never** in the submission build. It produces a weight blob. The submission side (`src/v8.rs`, default/submission build, **no `burn`, no threads**) `include_bytes!`s the blob and runs a hand-rolled scalar forward + range coder. Same binary compresses and decompresses.
- **GPU is for training and batched bpb estimation only — not the codec.** The autoregressive codec is single-stream (batch 1, one token at a time). On that workload the CPU `v8` path (KV-cache + auto-vec, no per-call launch overhead) **beats burn-GPU ~45×** — measured 2026-06-08: GPU ~3.8 ms/byte vs CPU ~0.085 ms/byte (enwik9 ETA ~1050 h vs ~24 h). burn-GPU's win (~12×) is on *batched* matmul, i.e. training and a teacher-forced single-pass bpb estimate. So: **run the actual codec, authoritative bpb, throughput, and the submission on the CPU `v8` path; use GPU for training and quick batched bpb estimates.** There is no reason to test or report GPU codec speed.
- **Backends are not bit-identical → no cross-backend AC.** burn (GPU) and `v8` (CPU) are different float implementations; per-step CDFs diverge, so a GPU-encoded archive does **not** decode on the CPU codec (and vice versa) — verified false. Each backend is internally deterministic and round-trips with itself; that self-consistency is the only guarantee the AC needs (the shipped binary encodes and decodes on the same CPU path).
- **Online-BPE tokenizer (`src/bpe.rs`).** Deterministic incremental BPE: sweep the data in 1% passes, merging top pairs up to a quota ramping to the target vocab (8K). It is a learning *curriculum*, fixed for the run — the merge list tokenizes the training data and ships in the blob so the decoder tokenizes identically. Shared verbatim with the trainer via `#[path]`. Caps to respect before scaling: the range coder's `CDF_TOTAL = 1<<16` floors each symbol at 1, so **vocab ≤ 65536**; `Bpe::encode` is O(merges × len), needing a near-linear tokenizer before enwik9-scale compress.
- **Ternary BitNet, QAT not post-hoc.** Weights are ternary (`{-1,0,1}` + per-tensor absmean scale) trained with quantization-aware training (fake-quant + straight-through estimator). Post-hoc quantization does not survive (v5: 1.25→3.04 bpb) — it must be trained in. The blob packs ternary at 2 bits/weight (1.6 b/w trit-packing is a known tightening). The **tied token embedding is ternary too** (it is the dominant parameter table at a large BPE vocab, and it trains fine — tracks fp); pos and the LayerNorm affines stay f32 (small).
- **Incremental KV-cache forward.** `step` processes one token, appending its K/V; past tokens' K/V are finalized by causality, making the per-byte cost linear, not O(t²). Learned **absolute** positions cannot slide (it would invalidate the cache), so the context **block-resets** at the `ctx` boundary — matches training's contiguous-from-0 windows. True sliding needs relative positions (ALiBi/RoPE), a deliberate future change.
- **Auto-vec kernels (see kernel strategy above).** The ternary matvec is a multi-accumulator f32 `dot` (`LANES=16`, `mul_add` → NEON `fmla.4s`) over weights unpacked to f32 in RAM (the on-disk blob stays 2-bit, so `L(D)` is unaffected). Reusable `Scratch` buffers avoid per-byte allocation. An int8 `sdot` matvec (~4×) is the next kernel lever but departs from pure-auto-vec f32.

### Per-turn reporting

When a turn changes the codec (and there is no test failure), report — `lzr nn-test` prints all of it:

- **Codec parameters and architecture**: model type/size, `d_model`/layers/heads/ffn/ctx, BPE vocab size, parameter count.
- **Encoder bpb on enwik8** — `L(C)` on a held-out slice (and `L(D)` / net).
- **CPU encode/decode throughput as an enwik9 ETA** (hours). CPU only — GPU codec speed is not reported (see above).

`v8.rs` also carries an integration test (`enwik8_random_point_roundtrip`): a reproducible random point ≥ 1 MB from either end of enwik8, online-BPE up to it, then a CPU encode→decode of 1000 tokens of the following bytes, byte-for-byte. It is pure-Rust (runs in `build.sh`) and skips cleanly if `assets/enwik8` is absent.

## Dependencies

- Pin runtime deps with `=` exact versions.
- **Submission binary deps stay minimal** — currently `anyhow` and `clap`. Anything that pulls in threading is forbidden in the submission path (the `build.sh` `cargo tree` gate enforces it).
- **`burn` is training-only**, gated behind the `neural` feature (`burn = { optional = true, default-features = false, features = ["std","wgpu","ndarray","autodiff"] }` — defaults pull 600+ image/video/web packages, always trim). It must never enter the default/submission build.
- **Do not add `safetensors`, `mlx-rs`, `bitnet-*` from crates.io.** The bitnet crates are stubs; mlx is Apple-only and the Python MLX work is retired; safetensors duplicates raw-binary functionality.
- Optional / training-only deps go behind a feature flag, never plain `[dependencies]`.

## Code style

- `#![deny(unsafe_code)]` at the crate root. If a kernel truly needs unsafe, scope it locally with `#[allow(unsafe_code)]` and a documented justification.
- Stable Rust only. Do not reach for nightly features (`portable_simd`, `core::intrinsics::*`) unless a measurement justifies it. Per-target SIMD, when warranted, uses stable `std::arch::*` with `#[target_feature]`.
- **Trust LLVM auto-vec first.** Clean scalar inner loops, `target-cpu=native`, only reach for intrinsics when a benchmark shows headroom.
- Don't add error handling, fallbacks, or validation for scenarios that can't happen.
- No code comments unless the *why* is non-obvious. Identifiers carry the *what*.
- Lint-clean (`-Dwarnings`) including `clippy::pedantic` and `clippy::nursery`. Per-lint exceptions are scoped to the function/block, never the crate root.

## Journal upkeep — `JOURNAL.md`

The journal is the paper-draft substrate. Follow the **Guidelines** block at the top of [JOURNAL.md](JOURNAL.md) precisely:

- **Reverse chronological**: new entries go immediately below the guidelines block.
- **Append-only**: never edit past entries. Corrections are new dated entries that reference the original by date.
- **Dates are when the work happened**, not when the entry is written.
- **Flow Who/When/Where/Why into prose** in the opening paragraph rather than using explicit `**Who:**` labels.
- **No double-equals in prose** — the renderer treats paired `=`+`=` as CriticMarkup highlight. Use a single `=` or rephrase. Backticks do not protect against this.

Write a journal entry whenever one of these lands: an experiment result (numbers > prose), an architectural decision with its reasoning, a negative result, an unexpected finding, or an external data point (paper, benchmark, hardware spec). Do not journal day-to-day plumbing, refactors, or conversation transcripts.

If a session produces a journal-worthy finding, propose the entry; CDR reviews before it lands.

## Keeping this file current — `CLAUDE.md`

Update this file when a durable project-level decision changes — branch strategy, build-script contract, deps policy, hardware target, architectural principles. Do **not** use it for transient state (in-progress tasks, current bpb numbers, experimental config) — that belongs in the journal or not at all.

Propose changes to this file rather than silently rewriting it; CDR reviews.
