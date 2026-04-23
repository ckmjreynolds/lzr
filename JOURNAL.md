# LZR Journal

Record of experiments, architectural decisions, results, and external data points for the Hutter Prize project.

## Guidelines (Instructions for Claude)

- **Purpose**: Who did, what, when, where, and why along with the results and conclusions in suitable detail to later write a paper or blog post if asked. This is not that paper or blog post but entries should be kept professional if concise. For brevity, the user can be referenced as "CDR" and "Who" is always "CDR" or "CDR/Claude" unless you're referencing external material in which case the author of that material should be referenced along with the appropriate link(s).
- **Format**: Journal entries: `## YYYY-MM-DD — <short title>`. Append `HH:MM` only when it disambiguates multiple entries on the same day or when the exact time actually matters for the finding. For work that spans multiple days, use a date range: `## YYYY-MM-DD → YYYY-MM-DD — <short title>`.
- **Dates reflect when the work happened**, not when the entry was written.
- **Order**: reverse chronological — newest entry immediately below this guidelines block, older entries further down.
- **Avoid double-equals in prose** — the renderer interprets paired double-equals as highlight delimiters (CriticMarkup-style). Use a single `=` or rephrase. Backticks do not protect against this.
- **Append-only**: never edit past entries. Corrections are new entries that reference the old one by date.

---

## 2026-04-18 → 2026-04-21 — Python feasibility retrospective

Over three days in the now-retired `lzr-train` repo (Python + MLX on an Apple M3 Pro), CDR/Claude tested the core bet for a Hutter Prize attempt: whether a small byte-level transformer, trained to memorize `enwik9` and used as the probability model of an arithmetic coder, can beat the current record of 110,793,128 bytes combined (compressor + decompressor + archive). Three days were budgeted for a go/no-go signal before committing to the Rust submission path; the findings below informed the architecture now planned in this repo (`lzr`).

### Hutter Prize rules (verified 2026-04-21 against the Hutter FAQ)

Submission scores `S = L(C) + L(D) + L(A)` where `C`, `D`, `A` are compressor, decompressor, archive; `S` must be ≤ 99% of the prior winner (≤ 109,685,197 bytes). If `C ≠ D`, `L(D)` is doubled. A single binary avoids the 2× factor. **However: even when the compressor and decompressor are the same binary, both `L(C)` and `L(D)` still appear in the sum, so embedded weights are counted twice.** Runtime limits on the judging machine (2021 spec: AMD Ryzen 7 Zen 2, AVX2/FMA, no AVX-512/VNNI): single core, no GPU, ≤ 10 GB RAM, ≤ 100 GB HDD, time ≤ `70,000 / Geekbench5` hours ≈ 53 hr per program. Training time is unbounded (offline).

### Compression — positive signals

- Byte-level transformer memorization works and scales sub-linearly in compute: 4× training data needed ~2.5× compute for equivalent memorization quality.
- MEDIUM config (19M params) on 1 MB reached 0.037 train bpb and 0.008 codec bpb on the memorized data; arithmetic-coder roundtrip was bit-identical.
- Post-training quantization preserves memorization through int4 (~1.4× bpb penalty vs fp32). int6 is free. int3 collapses.
- Codec bpb is meaningfully better than validation-sample bpb (~4× on memorized data): the codec's sliding window grows contiguously, while eval samples random windows.

### Compression — open questions and negative results

- Validation bpb regime changes around ~16 MB of training data. Below, the model pure-memorizes (train bpb ≪ val bpb). Above, capacity forces generalization (train 0.85 / val 1.54 bpb on MEDIUM-16MB at 15k steps). Whether memorization scales to the full 1 GB at a submission-viable model size is the central open question.
- BitNet (ternary QAT) works at MEDIUM-1MB (0.063 train bpb, bit-identical roundtrip) but hits a hard capacity ceiling at MEDIUM-4MB (plateaus at 0.67 train bpb, 17.5× worse than fp32 at the same size). Escape would require a LARGER BitNet (~80M params), untested.
- Training `seq_len` must equal `cfg.context_length`. Training TINY at seq=256 with ctx=1024 yielded 0 train bpb but 5.81 codec bpb on 64 KB — RoPE positions beyond the trained length were out-of-distribution. Retraining at seq=1024 dropped codec bpb to 0.0155.

### Performance — what works and what does not

- Candle's `Tensor` API is unusable for submission-time inference: at batch=1 single-token decode, per-op allocation/reshape/shape-checking dominates, yielding ~3% of theoretical peak. MEDIUM fp32 measured 43 ms/byte (510 days on 1 GB); Q8_0/Q4_0 at 7.4 ms/byte (86 days). Candle stays only as training and correctness-reference code, not in the submission hot path.
- Hand-rolled scalar ternary matvec with `RUSTFLAGS=-C target-cpu=native` auto-vectorizes to 75–80 GOPS on Apple NEON, giving 0.285 ms/byte for the matmul portion of MEDIUM (3.3 days on 1 GB, matmul-only). Full-forward per-byte estimate is ~0.37 ms/byte.
- Explicit `wide::i16x8` SIMD was *slower* than plain scalar (2.9 GOPS vs 75 GOPS): manual lane-by-lane unpacking defeated LLVM's NEON generation. Lesson: trust auto-vec; reach for intrinsics only when measurement shows headroom.
- Budget gap: 0.37 ms/byte × 1 GB ≈ 103 hr/side, 2–3× over the 53 hr Zen 2 budget. MEDIUM alone does not fit. Closing the gap requires some combination of SMALL config (≈3×), AVX2 intrinsics over auto-vec (≈1.3–1.5×), and shorter context (≈1.3× on attention). AVX-512 VNNI would be 4× but Zen 2 lacks it.
- Cross-framework numerics are not bit-identical: MLX-GPU and candle-CPU produce ~1 ULP logit drift, which occasionally tips an arithmetic-coder bucket. Encode and decode must run in the same implementation (planned: both in Rust).
- `bitnet-*` crates on crates.io (`bitnet-core`, `bitnet-inference`, `bitnet-quant`, `bitnet-metal`) are unusable despite 1.0.0 version labels — SIMD modules are stubs, CPU backends are TODOs, repos contain `*_broken.rs` / `*_backup.rs` files. Hand-rolling the kernel is the only path.

### Architectural decisions (carried into the Rust rewrite)

- Single Rust crate, single binary (compressor and decompressor are the same executable — 1× scoring).
- Feature-gated build: `submission` (default; zero threading deps, no candle) vs `training` (candle-nn for autograd + AdamW) vs `dev` (submission + training).
- Raw binary weight layout with `include_bytes!` into the submission binary — no safetensors, no per-tensor metadata. Architecture constants compiled as `const`; a compile-time assertion that the embedded weights' length matches the expected packed size guarantees the weights file and the code stay in lockstep.
- Target `d_model ∈ {256, 512, 768, 1024}`. `d_model = 384` deprecated: not divisible by 256, so k-quants (Q4K, Q2K, Q6K) fail, and it wastes SIMD lane pairs.
- Byte-level tokenization stays; custom tokenizer considered and dropped (decompressor overhead not worth the weight savings).
- Training on candle (Metal on Apple, CUDA if cloud needed) rather than MLX Python or `mlx-rs`, for single-codebase portability across dev and cloud hardware.

### Submission-size math

Record is 110.8 MB combined. At 1 bpb the archive alone is 125 MB — already over. The 2× weights count makes small+aggressively-quantized models decisively favored:

| Config                     | Weights  | 2×      | Archive budget | Required bpb on 1 GB |
|----------------------------|---------:|--------:|---------------:|---------------------:|
| TINY int4                  |   0.5 MB |    1 MB |       109.8 MB |             0.88     |
| SMALL int4                 |     3 MB |    6 MB |       104.8 MB |             0.84     |
| MEDIUM ternary             |     4 MB |    8 MB |       102.8 MB |             0.82     |
| MEDIUM int4                |    10 MB |   20 MB |        90.8 MB |             0.73     |
| LARGE ternary (hypothetical) |  16 MB |   32 MB |        78.8 MB |             0.63     |

For reference: fx2-cmix (record holder) ≈ 0.88 bpb effective; NNCP (best neural) ≈ 0.75–0.80 bpb. Memorization of 1 MB at 0.04 bpb does not extrapolate to 1 GB — at ~10 MB weight capacity vs ~900 MB of enwik9 entropy, the model must predict patterns, not recall bytes. **Primary target: MEDIUM fp32-trained → int4 PTQ, needing ≤ 0.73 bpb on 1 GB.** Stretch: LARGE BitNet if the 4 MB capacity ceiling lifts at ~80M params.

### Conclusion

Go. The core bet is viable: memorization is real, scales sub-linearly, survives int4 quantization, and hand-rolled kernels close enough to the runtime budget that a SMALL/MEDIUM configuration is plausible. Every promising configuration projects into the 95–115 MB range on 1 GB — genuinely competitive with the 110.8 MB record, not obviously winning. The Rust rewrite is justified both by the ~30× candle-CPU overhead measured above and by the single-binary submission requirement.

## 2026-01-31 → 2026-04-11 — Classical LZ77 (abandoned)

Over roughly ten weeks in this repo (`lzr`), CDR/Claude built and twice rewrote a classical LZ77-based compressor before abandoning the approach in favor of the transformer bet described in the entry above. The crate name was reserved on crates.io a year earlier (2025-04-26, commit `df3a722` "Release 0.0.0 (reserving name)"); the repo then went dormant for about nine months while CDR detoured into building `wabi_tree`, an order-statistic tree crate originally conceived as the LZR matchfinder's core data structure. Active work on LZR itself resumed 2026-01-31, when CI and project scaffolding were copied back in from `wabi_tree` and the classical LZ77 implementation began in earnest.

### What was built

A byte-oriented LZ77 format (`.lzr`) with a 1 MiB sliding window, a 4-byte header (`LZR` magic + version byte), variable-length frames carrying either literal bytes or distance+length back-references, and an Adler-32 footer checksum. At its largest, the source tree included modules for `adler32`, `buffer`, `cli`, `cursor`, `decode`, `encode`, `error`, `format`, `matchfinder` (~834 LOC), `options`, `packed`, `ringbuf`, plus roundtrip tests and encode/decode benches. The format spec is preserved in the git history (`docs/FORMAT.md`) if ever needed for reference.

### Timeline of resets

- **2026-02-15** — first working roundtrip: "'bout to let Claude run" (`2665adb`) followed by "Works but slow" (`7de0b70`). ~1100 lines of encoder/decoder/format across the day.
- **2026-02-18** — first reset: "Reset, format solidified, removed original dumb implementation" (`9aa8277`). ~3200 LOC deleted; format spec retained, re-implementation began.
- **2026-03-17** — "Changing format" (`1cde682`). Substantial matchfinder (~834 LOC) plus ring-buffer rewrite in a single commit.
- **2026-04-11** — second reset: "Reset/rethink" (`bff2be5`). ~3400 LOC deleted across `encode`/`decode`/`matchfinder`/`format`/`wild` modules, returning to an empty `main.rs`. This is the pivot point: checkpoints from here through 2026-04-18 are scaffolding for the new direction, after which work moved into the `lzr-train` Python repo.

### Why abandoned

Classical LZ77 dictionary coders bottom out well above the Hutter Prize threshold regardless of matchfinder tuning. The record is held by fx2-cmix / cmix at ≈0.88 bpb on enwik9 using context-mixing and neural-network statistical models; pure LZ77 on text typically lands in the 1.5–2+ bpb range. No amount of work inside the LZ77 design space closes that gap. The bet shifted to a statistical compressor using a transformer as the probability model for an arithmetic coder — the path evaluated in the feasibility work above.
