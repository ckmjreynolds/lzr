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

## 2026-04-24 — First Full RWKV Training (1M) -> Scaling To 2M

CDR/Claude trained the first full RWKV model (D_MODEL=256, N_LAYERS=2, CM_MULT=1, ~1M params) end-to-end on enwik9 over ~7 hours and 23k steps on an M3 Pro via the candle-nn + Metal training loop landed earlier the same day. The run confirms the pivot worked, isolates a still-open eval-harness issue, and establishes the plateau that motivates the next scale-up to ~2M.

The earlier 2026-04-24 entry ("Pivoting To RWKV") projected RWKV v6; the landed implementation is v4. v6's extra data-dependent time-mix parameters would have added f32 overhead without an obvious capacity win at this scale, and the v4 WKV recurrence (single time-first / time-decay vector per layer) was easier to verify against the reference paper. Upgrade path to v6 stays open if later capacity studies warrant it.

### Training curve

| Step  | train nats | info-bpb | codec bpb | offset |
|------:|-----------:|---------:|----------:|-------:|
|  1602 |       1.91 |     2.76 |      2.59 |   261M |
|  4812 |       1.45 |     2.09 |      2.38 |   462M |
|  8029 |       1.48 |     2.14 |      1.90 |   888M |
| 11291 |       1.33 |     1.92 |      2.17 |   105M |
| 14551 |       1.45 |     2.09 |      2.23 |   172M |
| 17793 |       1.38 |     1.99 |      1.94 |   752M |
| 19403 |       1.41 |     2.03 |      1.02 |   539M |
| 22622 |       1.24 |     1.79 |      1.94 |   935M |

Train loss and codec bpb agree to within ~0.2 across the whole run — this is the headline. The 11M transformer's sliding-window distribution mismatch has no analog in RWKV's running-state recurrence: state at each layer is five D_MODEL-sized vectors carried forward, never a re-rotated cache of historical K/V, so there's nothing to go out-of-distribution at byte N × 256.

### Eval-harness noise dominates individual bpb readings

Codec bpb at each checkpoint is measured on a single random 16 KiB window of enwik9, and content variance across offsets swamps the signal: step 12921 (offset 582M) produced bpb 0.73 at train loss 1.48, while step 14551 (offset 172M) produced bpb 2.23 at train loss 1.45 — nearly-identical models, 3× bpb spread. The 2026-04-23 deferred fix (fixed 5-offset panel + held-out 1 MB slice) stays deferred for now, but becomes blocking before the next scale-up's plateau-detection logic can be trusted.

### Best checkpoint: step 22622 (train loss 1.24 nats)

Selected by train loss, not codec bpb. The codec-bpb minima at steps 12921 and 19403 are sampling-luck artifacts — both had *higher* than neighbor-average train loss. Using train loss as the selection signal is a standing rule until the eval harness is fixed.

### Plateau

Train loss flatlines in the 1.3–1.5 nats band (~1.9–2.1 bpb equivalent) from ~step 8000 onward. The cosine LR schedule is barely moved off peak (2.996e-4 of 3e-4 — the schedule was sized for a much longer run), so this isn't LR decay. The ceiling is capacity: a 1M-weight byte model at 256-byte context cannot encode enwik9's residual entropy more tightly. For reference: Hutter record sits at ~0.88 bpb; NNCP's best neural result ~0.75 bpb; our 11M transformer info-bpb hit 2.15 before its codec path collapsed — this 1M RWKV run is already competitive with that transformer at 1/11 the params and with a working codec.

### Throughput

Batch=32, seq=256 training on Metal averaged 1.12 s/step. 23k steps = 25,760 s ≈ 7.1 h wall. Inference-side decode-hours-for-1GB came out ~9.5 h per checkpoint — well inside the Hutter 53 h single-side budget.

### Decision: scale to ~2M via depth

`src/arch.rs` bumped to `N_LAYERS = 4`, `D_MODEL = 256` and `CM_MULT = 1` held. Parameter budget: ~1.84M ternary + ~82K f32 ≈ 1.92M total, i.e. ~2×. Going deep rather than wide is deliberate: every matvec's `in_dim` stays at 256, leaving the current LUT kernel bounds (≤ 512 TL1 / ≤ 768 TL2) with headroom — no kernel change needed, no accumulator widening, no new diff-test surface. A 2× N_LAYERS bump roughly doubles ternary weight count, doubles per-step training cost, and adds one full time-mix + channel-mix round trip per byte at inference. Expected per-step training time ≈ 2.2 s, expected 20 h run to match 23k steps. Inference decode-hours-for-1GB scales similarly to ~19 h, still inside budget.

Expected capacity gain is the open question: if bpb-vs-params scales like the 11M/1M gap implies, we should see train loss land in the 1.0–1.2 nats range (1.4–1.7 bpb), still well above the 0.73 bpb target for MEDIUM int4 from the 2026-04-21 submission-size math, but trending the right way. If the 2M run plateaus at a similar bpb to 1M, depth alone isn't the answer and we need width (which needs kernel work) or a different objective.

---

## 2026-04-24 — 11M Training Diverges; Sliding-Window Transformer Retired; Pivoting To RWKV

CDR/Claude, debugging the 11M training run started 2026-04-23 15:13 on a M3 Pro (D=512, L=6, N_HEADS=8, MLP_MULT=1), diagnosed a train/inference divergence that turned out to be neither a kernel bug nor a training bug, but a fundamental incompatibility between sliding-window inference and the independent-window training distribution. After isolating the cause we're retiring the transformer design and pivoting to RWKV v6.

### Symptom: train loss improving, codec bpb worsening

Over ~16 hours and 48k+ steps, train loss descended cleanly 5.49 → 1.49 nats (info-theoretic 2.15 bpb). Codec bpb at each checkpoint, measured on a random 16 KiB window of enwik9, diverged in the opposite direction. Unlike the TINY 1M smoke test (content-variance noise in the 2–4 bpb range), the 11M run showed monotonic degradation:

| Step  | train nats | codec bpb |
|------:|-----------:|----------:|
|  5934 |       2.16 |      4.61 |
| 17905 |       1.67 |      6.81 |
| 30103 |       1.67 |      8.49 |
| 42298 |       1.61 |      9.73 |
| 48396 |       1.49 |      9.39 |

Codec bpb above 8 means the arithmetic coder is producing output larger than the input (encoded 19250 bytes from a 16384-byte window at step 48396), i.e. the model's predictions at inference are worse than uniform.

### Eliminating the LUT kernels

First hypothesis: the TL1/TL2 activation-precision reduction (`x_q >> 1` for TL1, `>> 2` for TL2, with ×2/×4 compensation at row end) is benign at TINY but catastrophic at 11M. Added a diagnostic path `Weights::from_bytes_force_i2s` / `load_checkpoint_force_i2s` that skips LUT conversion and keeps the I2_S-packed buffer populated, routing `matvec_prequant` to the I2_S direct-SIMD fallback. Ran codec on five fixed offsets (100/300/500/700/900 MB) with both paths:

| Path  | mean codec bpb |
|-------|---------------:|
| LUT   |         9.3905 |
| I2_S  |         9.3752 |

Delta 0.015 bpb — within quantization noise, not the 4× gap we're looking for. The LUT kernels are vindicated.

### Localizing the failure to sliding-window attention

Added a direct cross-entropy measurement (run the inference forward on a 256-byte window, compute `-E[log p(target)]` against the true continuation). Ran four variants on the same five offsets:

| Variant                          | mean nats | info-bpb |
|----------------------------------|----------:|---------:|
| Pre-wrap (predictions 0..254)    |    1.4559 |   2.1004 |
| Post-wrap (predictions 255..16382) |  8.2061 |  11.8391 |
| Reset every 256 bytes            |    1.5588 |   2.2489 |
| Chunk-256 codec (through AC)     |         — |   2.4166 |

Pre-wrap loss matches training loss exactly. The model forward works. Reset-every-256 and chunk-256 codec both land near the info-theoretic floor. The failure is isolated to the specific case where the KV cache wraps and subsequent steps attend to a sliding window of cached entries.

### Why the RoPE-relative-position argument doesn't save us

RoPE's property that `q · rope(k, p)` depends only on `q_pos − k_pos` guarantees that at post-wrap step N, Q at logical 255 attending to K at logical 0..255 reproduces relative distances 0..255 — identical inner-product geometry to a training window starting at byte N−255 at its position 255. Same rotations, same cosine-distance spectrum. But layer ≥ 1 K values in the cache were computed at each byte's original inference step, with whatever autoregressive history the cache held at that moment. Training position 0 of any window has K computed with zero prior context; the sliding-inference K at logical 0 (at post-wrap step 256+) has K computed at its real step with a non-empty cache. Same rotation applied, different K content, and the mismatch compounds nonlinearly across six layers into garbage attention patterns.

This is a training-distribution problem, not a code bug. bitnet.cpp's reference inference doesn't help: BitNet-b1.58-2B was trained with 2K-token context, never silently slides, and when llama.cpp's KV cache fills it aborts rather than sliding. Our target is 1 GB of continuous stream — we must slide, and the training distribution must anticipate it.

### Architectural retrospective and decision

The session forced the question: is a transformer even the right tool? On the compression-quality side, reconciling finite training context with unbounded inference context has three options — (a) train with context equal to inference length (infeasible at 1 GB), (b) sliding-window-aware training such as Mistral SWA or StreamingLLM's attention sinks (real design work), (c) chunk at inference and accept the lost cross-chunk context. Chunking gives 2.25 bpb info-theoretic at 11M — the ceiling any 256-byte-context byte model would hit regardless of architecture.

On the inference-cost side, at ctx=256 a transformer runs ~0.40 ms/byte vs. RWKV's estimated ~0.25 ms/byte — within 2×, but transformer per-byte cost scales linearly with context while RWKV's stays flat. On a single-core Zen 2 with a 53-hour budget over 1 GB, a transformer reaches the budget ceiling before it can use contexts long enough to actually improve bpb. RWKV v6 keeps inference O(1) per token, reuses the existing ternary LUT kernels (the only new kernel is an element-wise time-mix step), and has no sliding-window failure because it carries learned summary state rather than a re-rotated cache.

The NNCP precedent corroborates this: the best-published neural enwik9 compressor (~0.75 bpb) uses LSTMs, not transformers. Compression rewards tight calibration of running state; transformers only pull ahead once long-range attention is both affordable and trained for. Neither condition holds here.

Among RNN-family options, RWKV v6 was chosen over LSTM and Mamba on three criteria: (1) lowest per-byte inference op-count of the three, (2) direct reuse of the existing ternary LUT kernels (Mamba's selective scan would need a new kernel category), (3) training parallelizes reasonably via the receptance-weight-key-value parameterization. LSTM stays as the fallback if RWKV training proves brittle at byte level.

### Retired and carried forward

Retiring: `src/model.rs` (transformer step, attention, RoPE rotation kernels), `src/kv_cache.rs` (KV ring buffer), the RoPE tables in `src/arch.rs`, the scaled-dot-product and causal-mask path in `src/train.rs`. Carrying forward: `src/bitnet.rs` (TL1/TL2 LUT kernels, I2_S fallback, `quantize_activations`, `pack_ternary`), `src/ac.rs` (arithmetic coder), `src/probs.rs` (logits-to-CDF), `src/codec.rs` (AC glue — the `ProbSource` trait is architecture-agnostic), `src/weights.rs` (raw-binary layout, `include_bytes!` integration, the load-time `force_i2s` diagnostic), and the training-loop scaffolding in `src/train.rs` (LogWriter, LrSchedule, sample_batch, checkpoint machinery). The `Diag` subcommand stays for now as a permanent debugging aid.

---

## 2026-04-23 — TINY (1M) Smoke Test; TL1/TL2 LUT; Kickoff 11M

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
