# CLAUDE.md

Project-specific guidance for Claude Code working in this repo. Global Claude Code defaults still apply; this file only captures what's specific to **LZR**.

## What this is

A Hutter Prize attempt: a single Rust binary that compresses `enwik9` below the current record (~116 MB ≈ 0.928 bpb combined).

**Hutter scoring formula** (verbatim from `prize.hutter1.net/hrules.htm`):

- *Default (self-extracting archive)*: `S = length(comp9.exe) + length(archive9.exe)`. The compressor and the self-extracting archive both contain a copy of the model weights, so weights are paid for `2×`.
- *Relaxation (split compressor + bare archive)*: `S = length(comp9a.exe) + 2 × length(decomp9.exe) + length(archive9.bhm)`. **If `comp9a.exe == decomp9.exe`** (same binary used for both directions), the `2×` reduces to `1×`, giving `S = 2 × length(binary) + length(archive)`.

The favorable shape for us is **one binary serving both directions**: the binary appears twice in `S` (once as the compressor, once as the reduced-multiplier decompressor), so its bytes count `2×`. There is no submission shape where `L(D)` is counted only `1×`. Split binaries (`comp9a != decomp9`) are strictly worse (`3×` the binary cost). When projecting `L(D)` bpb in code or journal entries, always multiply shipped-binary-bytes by 2 before dividing by 1 GB.

Two branches:
- **`hutter`** — v1, ensemble of LZ77 + cross-stream PPM-D + 4M-param RWKV byte-level neural arm. Best result: **1.985 bpb on enwik8** (249 MB extrapolated to enwik9). Architecture: byte-class-routed (Lower/Upper/NonLetter), monolithic codec, ad-hoc bit accounting. **Saturated** — see 2026-05-10 journal entry.
- **`v2`** — clean-slate rewrite, mode-routed codec with eval-first discipline and per-component bit decomposition. Best deterministic result so far: **2.608 bpb on enwik8** (xml-lz-cp). No neural arm yet.
- **`v3`** — adds neural-arm integration onto the v2 mode-routed codec. Best result: 1.811 bpb on enwik9 with `nano_plus` AR transformer mixed on `token_oov_sep` (Phase 27).
- **`v4`** — neural-first pure-MoE codec. Sparse `n_experts=32` AR transformer + AC + uniform fallback; no LZ/PPM/classifier/tokenizer. Best result (E=32 at 60 K steps, int8ch-quantized at load): **L(C) 1.6098 + 2×L(D) 0.0524 = 1.6622 combined bpb on 1 MB enwik9.** Current development branch.

Current history and rationale live in [JOURNAL.md](JOURNAL.md). The retired Python+MLX predecessor lives at `/Users/creynolds/Programming/lzr-train/`.

## Build

Always build via `./build.sh`, not a bare `cargo build`. The script runs fmt, clippy (default features + no-default-features) with `-Dwarnings`, release tests, and nightly coverage. It also enforces the **zero-threading-dependency** invariant via `cargo tree --features submission` — adding rayon, crossbeam, gemm, tokio, async-std (or anything that pulls them in transitively) is a regression. If `build.sh` fails, fix the root cause — do not bypass with `--no-verify` or `-A warnings`.

## Architecture constraints

- **Single binary**, single `main.rs`, subcommands. Compressor and decompressor are the same executable — this is the `comp9a == decomp9` relaxation that reduces the decompressor multiplier from `2×` to `1×`. The binary still counts `2×` in total (once as the compressor, once as the reduced decompressor), but that is `1×` better than the `3×` a split-binary submission would pay.
- **Hutter judging machine limits** (the only hardware constraints actually guaranteed): single CPU core, no GPU, ≤ 10 GB RAM, ≤ 100 GB HDD, time ≤ `70,000 / Geekbench5` hours. The specific CPU model has changed before and may change again — do not hard-code assumptions.
- **Submission binary is strictly single-threaded.** No `std::thread`, no rayon, no parallel iterators.
- **Kernel strategy: architecture-neutral first.** Write clean scalar code and trust LLVM auto-vectorization with `RUSTFLAGS=-C target-cpu=native` (configured in `.cargo/config.toml`). v1's Python-side experiments showed clean scalar code reaching ~75 GOPS on Apple NEON where explicit `wide::i16x8` SIMD gave 2.9 GOPS — auto-vec is almost always the right tool. If a measurement justifies it, add an architecture-specific kernel behind a runtime dispatch (`is_x86_feature_detected!`) and keep the neutral path as the fallback.

## v2 design principles (current branch)

- **Eval infrastructure first.** Per-component bit decomposition (`src/eval.rs`) is the substrate for every architectural decision. Every codec implements `crate::codec::Codec` and reports a `Decomposition` whose components must sum to `8 * archive_bytes` exactly — caught at panel time.
- **Bit-budget discipline.** Before adding any predictor or codec component, predict in writing which decomposition cells move and by how much. Measure. If the prediction is wrong, the architecture is wrong, not the predictor.
- **Mode-routed codec.** A streaming `Classifier` (`src/classifier.rs`) is a Moore FSM tagging each byte as `Content`/`TagStructure`/`AttrValue`. Encoder and decoder run the same classifier deterministically — no signaling, zero bits overhead. Per-mode codecs handle the bytes their classifier reports.
- **Per-phase codec layering.** Each new architectural step lands as a new codec (`xml`, `xml-ppm`, `xml-lz-ppm`, `xml-lz-word`, `xml-lz-ord3`, `xml-lz-cp`, `xml-lz-ppmc`). They sit side by side and can be A/B-tested via `--codec <name>`. Old codecs stay shippable for regression checks.
- **Sub-8-bit AC framing.** A bit-level arithmetic coder (`src/ac.rs`) with chunked uniform-CDF emission for >8-bit raw-bit values. All codec components share one AC stream so framing is consistent. Two latent AC bugs found and fixed under v2's varied bench conditions (see 2026-05-11 entry); both would surface in v1 too if exercised the same way.

## Dependencies

- Pin runtime deps with `=` exact versions.
- **Submission binary deps stay minimal.** v2 currently runs on `anyhow` and `clap`. Anything that pulls in threading is forbidden in the submission path.
- **Do not add `safetensors`, `mlx-rs`, `bitnet-*` from crates.io.** The bitnet crates are stubs; mlx is Apple-only and the Python MLX work is retired; safetensors duplicates raw-binary functionality.
- Optional / training-only deps go behind a feature flag, never plain `[dependencies]`. v2 currently has no such deps; when a neural arm is added, gate it.

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
