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

## 2026-04-23 — TL1/TL2 LUT kernels; TINY training smoke test; arch bumped to ~11M

CDR/Claude ported `bitnet.cpp`'s TL1 (ARM NEON) and TL2 (x86_64 AVX2) lookup-table matvec kernels into `src/bitnet.rs`, ran an aborted 1M-parameter training smoke test to exercise the pipeline end-to-end, and bumped `src/arch.rs` to a ~11M configuration as the target for the first real training run. The I2_S direct-SIMD kernel stays as the scalar correctness oracle and the fallback on architectures without a LUT path.

### Why LUT kernels, why now

The 2026-04-21 feasibility retrospective sized the Rust budget gap at 2–3× over the 53 hr Zen 2 ceiling for MEDIUM (≈20M params). Matvec dominates wall time at the ≥10M scale where memorization stops being trivial, so kernel throughput is the highest-leverage runtime lever before hitting the ceiling. Microsoft's published TL1/TL2 numbers show ~2× kernel throughput over I2_S direct-SIMD, which translates roughly 1:1 into end-to-end speedup at matvec-dominated scales.

### Why TL1 on ARM, TL2 on x86 — answering "why not TL2 on NEON"

The table-lookup primitive shape differs by ISA. ARM's `vqtbl1q_u8` is a 16-entry lookup across 16 lanes — a perfect fit for K=2 (9 valid entries fit in the 16 slots). x86's `_mm256_shuffle_epi8` (`pshufb`) is also 16-entry per 128-bit lane, but K=3 packs 27 valid entries and needs a paired `pshufb` + blend across two 16-byte sub-LUTs; the extra blend pays back because each 6-bit index encodes 1.5× more weights than a 4-bit K=2 index. TL2-on-NEON is feasible via `vqtbl4q_u8` (64-entry table across 4 registers) but burns 4× the register budget for the LUT; TL1-on-AVX2 is feasible but leaves 1.5× weights-per-shuffle on the table. Matching `bitnet.cpp` arch-by-arch is the right call.

### Design: I2_S on disk, LUT in RAM

On-disk weight format stays I2_S so trained checkpoints don't need to know which LUT kernel will consume them. At `Weights::from_bytes` time, a one-shot `repack_i2s_to_lut` builds the arch-preferred LUT buffer; the I2_S bytes are then dropped on LUT-supporting architectures, saving `PACKED_WEIGHTS_LEN` of dead RSS. `lut_supports(out_dim, in_dim)` is a compile-time `const` assertion run against every tensor shape in `weights.rs` — the build fails if any matrix slips outside the kernel's supported range. Current bounds: `in_dim ≤ 512` for TL1 and `in_dim ≤ 768` for TL2, both set by `i16` accumulator overflow (worst-case `|acc| ≤ 63 · in_dim` for TL1 and `32 · in_dim` for TL2).

Activation precision reduces by 1 bit on TL1 (`x_q >> 1`, range `[-64, 63]`) and 2 bits on TL2 (`x_q >> 2`, range `[-32, 31]`), so partial-sum magnitudes `|Σ wᵢ·xᵢ|` with `K ∈ {2, 3}` ternary weights stay in i8 range. The row-end scale compensates with a ×2 (TL1) or ×4 (TL2) factor. Training quantization remains 8-bit absmax; this creates a small train/inference mismatch that will need a QAT-side `--activation-bits` knob at scale-up.

### Kernel structure

- **TL1 NEON**: 16-output-row tiles. Packing co-locates 16 output rows at each activation-pair position (2 pairs per 16-byte block, nibble-interleaved). One shared 16-entry LUT per pair is broadcast across 16 lanes by a single `vqtbl1q_s8` → 32 MAC-equivalents per shuffle. Uses baseline NEON only (no dotprod dependency).
- **TL2 AVX2**: 32-output-row tiles. Two 16-byte sub-LUTs per triple (entries 0..15 and 16..31, with 27..31 zero-padded). Paired `vpshufb` + MSB-sentinel routing (`idx | 0x80` on the low path when `idx ≥ 16`; `idx − 16` yields negative → zero return on the high path when `idx < 16`) → OR-combine yields 32 i8 partial sums for 96 MAC-equivalents per pair of shuffles. AVX2 path carries a `// UNVERIFIED ON X86` header — the M3 dev box can't run it; first execution will be on CI or the judging machine. Cross-compile to `x86_64-unknown-linux-gnu` is clean.
- **LUT build hoisted to `Scratch.lut`** in `model.rs::step`, built once per `quantize_into` and reused across the matvecs that share the activation (Q/K/V share one build, w1/w3 share another) — 4 LUT builds per layer per token instead of 7.

### Results at TINY scale (D=256, L=2, ~1M params)

End-to-end encode: 0.99 s → 0.97 s per 16 KiB (median of 5 runs) after TL1 + LUT hoisting. ~2% speedup. The modest gain is expected at TINY because matvec is only ~10% of wall time at this scale; the projected gain at ≥10M is ~1.7–1.9× end-to-end. 34 unit tests green (including random-input diff-tests of TL1 NEON vs TL1 scalar bit-exactly, and worst-case-bound assertions of TL1 and TL2 vs I2_S scalar). `./build.sh` clean on submission and dev feature sets; x86_64 cross-compile clean.

### Training smoke test: pipeline end-to-end works; eval harness is noisy

A brief 12 K-step TINY training run (2026-04-23, `training.log`) confirmed the candle-nn training loop, Metal backend, checkpoint format, and codec roundtrip all work end-to-end. Loss descended 5.55 → ~2.1 nats in ~2500 steps, LR cosine schedule ramped cleanly, all checkpoints roundtripped OK.

The run surfaced an eval-side methodology issue that had been invisible in previous smoke tests: codec bpb at each checkpoint is measured on a randomly-offset 16 KiB window of enwik9, so neighboring checkpoints with near-identical train loss show bpb 2.46 vs 3.78 purely from content variance (XML boilerplate vs rare-article prose in different offsets). This makes `--stop-after-stale-ckpts` plateau detection fire on noise rather than on model quality. Fix is deferred until the first real training run and will consist of: a fixed panel of ~5 eval offsets reporting the mean, plus a held-out eval-loss pass over a fixed ~1 MB slice for plateau detection. The TINY run itself was aborted — its purpose was to test the code, not to train a usable model.

### Arch bumped to ~11M for next run

`src/arch.rs` updated: `D_MODEL = 512`, `N_LAYERS = 6`, `N_HEADS = 8`, `HEAD_DIM = 64`, `MLP_MULT = 1`, `CONTEXT_LEN = 256`. Approximately 11.01 M ternary weights plus 158 K f32 parameters (token embedding, per-row scales, RMSNorm weights). Packed weight file size: ~3.23 MiB. Every tensor has `in_dim = 512`, squarely inside both TL1 (≤ 512) and TL2 (≤ 768) kernel bounds.

Held `MLP_MULT = 1` rather than the conventional `4×` (or `SwiGLU`'s `2.67×`) so `w2`'s `in_dim` stays at 512. Moving to a wider MLP requires widening the LUT-kernel accumulators from `i16` to `i32` (or adding periodic `i16 → i32` flush), which is a kernel change not worth making before a shape-sensitive benchmark says it's needed. Six layers at `D = 512` with narrow MLP is unconventional but still produces a reasonable transformer at the target capacity; the trade-off is explicit and reversible.

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
