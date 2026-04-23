# CLAUDE.md

Project-specific guidance for Claude Code working in this repo. Global Claude Code defaults still apply; this file only captures what's specific to **LZR**.

## What this is

A Hutter Prize attempt: a single Rust binary that compresses `enwik9` below **109,685,197 bytes** (99% of the 110,793,128-byte record = `L(C) + L(D) + L(A)`), using a byte-level transformer as the probability model for an arithmetic coder. The compressor and decompressor are the same executable, which triggers 1× scoring on `L(D)` rather than 2×.

Current history and rationale live in [JOURNAL.md](JOURNAL.md). The retired predecessor (Python + MLX feasibility work) lives at `/Users/creynolds/Programming/lzr-train/`.

## Build

Always build via `./build.sh`, not a bare `cargo build`. The script runs fmt, clippy with `-Dwarnings`, a `--no-default-features` check, release tests, and nightly coverage. If it fails, fix the root cause — do not bypass with `--no-verify` or `-A warnings`.

`cargo tree --features submission` must show **zero threading dependencies** (no rayon, crossbeam, gemm, etc.). Adding anything that pulls them in is a regression.

## Feature flags

- `submission` — default for release. Zero threading deps, no candle. This is what ships.
- `training` — adds `candle-nn` for autograd/AdamW. Training-only code goes behind `#[cfg(feature = "training")]`.
- `dev` — `submission + training`. Used locally for dev builds that need both sides.

The submission binary must work with no features enabled. Optional runtime deps go under `training` or later-added features, never plain `[dependencies]`.

## Architecture constraints

- **Single binary**, single `main.rs`, subcommands for `train` / `compress` / `decompress`. Do not split compressor and decompressor into separate binaries.
- **Weights are a raw binary file** embedded via `include_bytes!("../assets/weights.bin")`. No safetensors, no per-tensor metadata. Arch constants (`VOCAB`, `D_MODEL`, `N_LAYERS`, etc.) are compiled as `const`. A compile-time assertion that the weights length matches the expected packed size keeps weights and code in lockstep.
- **Byte-level tokenization** — no custom tokenizer, no BPE. The overhead of shipping a tokenizer is not justified.
- **`d_model` must be in {256, 512, 768, 1024}** — divisible by 256 for k-quants (Q4K, Q2K, Q6K) and SIMD lane pairs. `d_model = 384` is deprecated; k-quants fail on it at runtime.
- **Hutter judging machine limits** (the only hardware constraints that are actually guaranteed): single CPU core, no GPU, ≤ 10 GB RAM, ≤ 100 GB HDD, time ≤ `70,000 / Geekbench5` hours. The specific CPU model has changed before and may change again — do not hard-code assumptions about it.
- **Kernel strategy: architecture-neutral first.** Write clean scalar code and rely on LLVM auto-vectorization. If a specific target needs more, add an architecture-specific kernel behind a runtime dispatch (`is_x86_feature_detected!` or equivalent) and keep the neutral path as the fallback. Do not build the whole system around one ISA.
- **Submission binary is strictly single-threaded.** No `std::thread`, no rayon, no parallel iterators.

## Dependencies

- Pin runtime deps with `=` exact versions (e.g. `candle-nn = "=0.10.2"`). Dev-deps likewise.
- **Do not use `bitnet-core` / `bitnet-inference` / `bitnet-quant` / `bitnet-metal` from crates.io** despite their 1.0.0 labels — they are stubs and placeholder code. Hand-roll any BitNet kernels needed.
- **Do not add `safetensors`.** Raw binary only.
- **Do not add `mlx-rs`.** Candle is the chosen training backend; MLX is Apple-only and the Python MLX work is retired.
- Any dep that pulls in threading (candle-core, rayon, gemm, etc.) must be gated behind `training`.

## Code style

- `#![forbid(unsafe_code)]` in every crate root. If a kernel truly requires unsafe, discuss first.
- Stable Rust only. Do not reach for nightly features (`portable_simd`, `core::intrinsics::*`) unless a measurement justifies it. Per-target SIMD, when warranted, uses stable `std::arch::*` with `#[target_feature]`.
- **Trust LLVM auto-vec first.** On the Python-side experiments a clean scalar inner loop with `RUSTFLAGS=-C target-cpu=native` produced ~75 GOPS on Apple NEON, where explicit `wide::i16x8` SIMD gave 2.9 GOPS. Only reach for intrinsics when a benchmark shows auto-vec is leaving headroom, and when you do, keep the scalar path as the fallback.
- Don't add error handling, fallbacks, or validation for scenarios that can't happen.
- No code comments unless the *why* is non-obvious. Identifiers carry the *what*.

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

Update this file when a durable project-level decision changes — feature flags restructured, deps policy shifts, hardware target updated, new build-script contract, etc. Do **not** use it for transient state (in-progress tasks, current branch, experimental config) — that belongs in the journal or not at all.

Propose changes to this file rather than silently rewriting it; CDR reviews.
