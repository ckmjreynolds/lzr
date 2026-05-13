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

## 2026-05-13 — Phase 23G: Adding `p2_len` Coarse Warm Feature for the MLP — 1.875 bpb on Enwik9

Layer the same coarse-warm-feature pattern Phase 23F validated, one slot deeper in the token history. `p2_len` (length of the token two back) feeds a sibling `(p2_len_bucket, prefix, bit_pos)` feature in the `dict_id` MLP.

### What landed

1. **Extended `PredCtx`** with `p2_len: u8`. Same six construction sites updated — the simple-shift case becomes `p2_len: ctx.p1_len`; the LZ-match-shift case picks `dict.entry(lookahead[length_us - 2].id).lower.len()` when `length_us >= 2`, else `ctx.p1_len`. Decode side mirrors this from `copied[len_match - 2]`.
2. **New 5th MLP feature** `(p2_len_bucket, prefix, bit_pos)` with the same 4-bit cap as `p1_len`. `N_MLP_ID_FEATS` 4 → 5. Memory: 128 MiB → 160 MiB (+32 MiB).

### Results

| Config | enwik8 e2e bpb | enwik9 e2e bpb | enwik9 bytes | Δ vs Phase 23F |
|---|---:|---:|---:|---:|
| Phase 23F | 2.1141 | 1.8770 | 234,619,206 | — |
| **Phase 23G** | **2.1114** | **1.8754** | **234,426,423** | **-0.0027 / -0.0016 / -188 KiB** |

Roundtrip verified. Encode 673 s on enwik9 vs Phase 23F's 682 s (within noise — adding one cheap feature is rounding error against the rest of the codec). Memory +32 MiB, peak still ~5.4 GiB.

### Diminishing returns, but still positive

| Feature added | enwik8 Δ | enwik9 Δ | Cumulative enwik9 from 23D |
|---|---:|---:|---:|
| Phase 23F: `p1_len_bucket` | -0.0058 | -0.0022 | -0.0022 |
| Phase 23G: `p2_len_bucket` | -0.0027 | -0.0016 | -0.0038 |

The second length feature delivers ~70% of the first one's enwik9 gain. That's consistent with the intuition: `p1_len` captures what the *immediate* previous token was; `p2_len` adds context one step further out, which is correlated with `p1_len` (because of the Word/Separator alternation, lengths tend to cluster pairwise) but still adds independent signal.

### Phase 23G cumulative on enwik9

| Phase | enwik9 bytes | enwik9 bpb | Δ vs Phase 19k |
|---|---:|---:|---:|
| Phase 19k | 249,191,955 | 1.9935 | — |
| Phase 22 | 237,120,859 | 1.8970 | -0.0965 |
| Phase 23A | 235,510,295 | 1.8841 | -0.1094 |
| Phase 23B | 235,124,739 | 1.8810 | -0.1125 |
| Phase 23C | 235,066,968 | 1.8805 | -0.1130 |
| Phase 23D | 234,905,363 | 1.8792 | -0.1143 |
| Phase 23F | 234,619,206 | 1.8770 | -0.1165 |
| **Phase 23G** | **234,426,423** | **1.8754** | **-0.1181** |

Cumulative from `xml-tok` Phase 16 baseline (2.0667): **-0.1913 bpb / -19.1 MiB on enwik9**. Hutter ratio: **2.137× target**.

### Open portfolio

Still on the table from 23F's "open feature space" list:
- `p1_class`, `p2_class`: 3-valued token class. Even denser cells than length buckets.
- `(p1_len, p2_len)` joint: pairs the two existing length features. Natural cells ~2^28; should stay warm.
- `recency_bucket`: bytes since last LZ-match. New information not derivable from `p3..p1`.
- `page_offset_bucket`: byte position within current page.

Following the 23F → 23G ratio (~0.7×), the next portfolio addition should land between -0.0010 and -0.0016 bpb on enwik9. Cheap enough to stay in this groove until the diminishing-returns curve flattens.

---

## 2026-05-13 — Phase 23F: Coarse Warm Feature `(p1_len, prefix, bit_pos)` for the MLP — 1.877 bpb on Enwik9

Direct test of the Phase-23E lesson — "MLP features need observation density per slot, not deeper word-history order". The minimal experiment: add a **single coarse warm feature** built from a small-cardinality input (`p1`'s byte length, capped at 16 buckets) and see whether it does what the deep Order-4 feature couldn't.

### What landed

1. **Extended `PredCtx`** with a `p1_len: u8` field — the byte length of `p1`'s emitted token, capped at 255 (0 when `p1` is `None`). Updated all six `PredCtx` construction sites (encode/decode/prewarm × simple-shift and LZ-match-shift) to populate it from either the just-emitted `token.len()` or `dict.entry(last_id).lower.len()`.
2. **Added `id_mlp_features`** as a superset of `id_lr_features`. New 4th feature: `(p1_len_bucket, prefix, bit_pos)` where `p1_len_bucket = min(p1_len, 15)`. The bucket clips to 4 bits so the natural cell count is 16 × 65 K × 16 ≈ 2^24 — at K=20 each slot averages ~16 contexts, so SGD sees ~10⁴ observations per slot at enwik9 scale. Dense, the way 23E's hashed Order-4 wasn't.
3. **Bumped `N_MLP_ID_FEATS` 3 → 4**. MLP memory: 96 MiB → 128 MiB (+32 MiB, half of Phase 23E's +64 MiB).

### Results

| Config | enwik8 e2e bpb | enwik9 e2e bpb | enwik9 bytes | Δ vs Phase 23D |
|---|---:|---:|---:|---:|
| Phase 23D | 2.1199 | 1.8792 | 234,905,363 | — |
| **Phase 23F** | **2.1141** | **1.8770** | **234,619,206** | **-0.0058 / -0.0022 / -280 KiB** |
| Phase 23E (reverted) | 2.1203 | 1.8792 | 234,901,166 | +0.0004 / 0.0000 |

Roundtrip verified. Encode time on enwik9: 682 s vs Phase 23D's 654 s (+4 %). Decode: 379 s vs 360 s (+5 %). Total enwik9 encode+decode: 17.7 min — comfortably inside the Hutter budget. Memory peak unchanged at ~5.4 GiB (the +32 MiB is rounding error against the 2 GiB Order-2 anchor).

### Why this is the right confirmation of the 23E lesson

Side-by-side memory and bpb for the two feature-engineering experiments on top of Phase 23D:

| Experiment | Feature kind | Natural cells | K=20 obs/slot (enwik9) | Δ memory | Δ enwik8 bpb | Δ enwik9 bpb |
|---|---|---:|---:|---:|---:|---:|
| 23E (Order-4 + Wiki × Order-3) | Deep + sparse | ~2^73 / ~2^60 | ~10⁻⁴ | +64 MiB | +0.0004 | 0.0000 |
| 23F (`p1_len_bucket`) | Coarse + dense | ~2^24 | ~10⁴ | +32 MiB | **-0.0058** | **-0.0022** |

Half the memory, a clean positive instead of a noisy zero. The hypothesis from the 23E journal lands: SGD needs the per-slot observation count to dominate the per-slot context variance, otherwise gradient updates are noise. Order-4 fails on both axes; `p1_len_bucket` wins on both.

### What `p1_len_bucket` is actually measuring

The feature is "given the byte length of the previous token, what is the dict-id bit distribution for the next token?". Concrete examples this should resonate with:

- Length-1 separators (a single `\n`, ` `, `>`, `:`) are followed by very different `prev_id` → `next_id` distributions than length-3+ separators (` -- `, `\n  `, `''`, etc.).
- Length-2/3 short words (`is`, `of`, `the`) tend to be followed by longer content words; length-12+ words are usually nouns inside templates and tend to be followed by `]]` / `|` / `}}` patterns.
- OOV-flag patterns: `p1_len = 0` (the new-OOV-this-page case) has a very different downstream distribution than `p1_len ≥ 1` (a dict-known token).

None of these signals are *new information* in the strict sense — the dict id `p1` already encodes the length and class of the token. But they're new **conditioning axes** for the MLP, which lets the model partition the space along directions the Zipfian-mass-on-id partition can't reach for tail tokens.

### Phase 23F cumulative on enwik9

| Phase | enwik9 bytes | enwik9 bpb | Δ vs Phase 19k |
|---|---:|---:|---:|
| Phase 19k | 249,191,955 | 1.9935 | — |
| Phase 20l | 241,105,534 | 1.9288 | -0.0647 |
| Phase 21 | 237,259,434 | 1.8981 | -0.0954 |
| Phase 22 | 237,120,859 | 1.8970 | -0.0965 |
| Phase 23A | 235,510,295 | 1.8841 | -0.1094 |
| Phase 23B | 235,124,739 | 1.8810 | -0.1125 |
| Phase 23C | 235,066,968 | 1.8805 | -0.1130 |
| Phase 23D | 234,905,363 | 1.8792 | -0.1143 |
| **Phase 23F** | **234,619,206** | **1.8770** | **-0.1165** |

Cumulative from `xml-tok` Phase 16 baseline (2.0667): **-0.1897 bpb / -19.0 MiB on enwik9**. Hutter ratio: **2.139× target**.

### Open feature space

Phase 23F validates the path. Other coarse warm features that fit the same recipe:

- `p2_len_bucket`: length of token two back. Pairs with `p1_len_bucket` for a joint feature `(p1_len, p2_len, prefix, bit_pos)`.
- `p1_class`, `p2_class`: 3-valued token class (none / word / separator). Tiny cardinality, very warm.
- `recency_bucket`: bytes since last LZ-match, capped at 64. New information not present in `p3..p1`.
- `page_offset_bucket`: byte position within current page, log-bucketed. Captures "early-in-page-templates vs. mid-page-prose" structure.

Each is a small, well-bounded experiment. The next phase should pick one and measure; if positive, add the rest as a portfolio.

---

## 2026-05-13 — Phase 23E (Negative): New MLP Features (Order-4, Wiki × Order-3) Don't Pay — Reverted

Phase 23D ended with the observation that the MLP shared all three features with the sparse-LR, so the MLP could only add a nonlinearity, not new information. The Phase 23E experiment tested the obvious counter-move: give the MLP **its own additional features** the LR doesn't have.

### What was tried

1. **Extended `PredCtx` with `p4: Option<u32>`** — the 4th-most-recent dict id. Shifted through all six `PredCtx` construction sites in encode/decode/prewarm; PredCtx::NONE got a `p4: None` field. The Order-4 generalization of the existing length-1/2/3 LZ-match shift logic landed clean.
2. **Added an `id_mlp_features` helper** returning 5 hashes. First three identical to `id_lr_features` (Order-3 hashed, Wiki × Order-2, bias); two new:
   - **Order-4 hashed**: `(p4, p3, p2, p1, prefix, bit_pos)`. Phrase-level signal.
   - **Wiki × Order-3**: `(wiki, p3, p2, p1, prefix, bit_pos)`. Wiki sub-mode + trigram.
3. **Bumped `N_MLP_ID_FEATS` 3 → 5** keeping K=20, H=8. MLP memory: 96 MiB → 160 MiB.

The LR stayed on its 3-feature shape (its linear nature means more features mostly just pay for redundant linear combinations of existing ones).

### Results

| Config | enwik8 e2e bpb | enwik9 e2e bpb | enwik9 bytes | Δ vs Phase 23D |
|---|---:|---:|---:|---:|
| Phase 23D | 2.1199 | 1.8792 | 234,905,363 | — |
| Phase 23E | 2.1203 | 1.8792 | 234,901,166 | **+0.0004 / 0.0000** |

Enwik9 difference is 4,197 bytes out of 234.9 MB — measurement noise. Roundtrip OK in both runs. Encode time grew +8 % (654 s → 708 s) and decode +10 %; memory +64 MiB. Net result: the new features cost runtime + memory without any compression gain.

### Why this is the right negative

Hashed Order-4 has a natural cell count near 2^73 (16 × 65 K × 65 K × 65 K × 16 × prefix). Hashed into K=20 (1 Mi slots), the collision rate is ~10^16 distinct contexts per slot — even when one underlying 4-token sequence is genuinely warm, the slot's gradient is dominated by ~10^16 unrelated sequences also colliding. The SGD update is noise on top of noise.

The Phase-23A features worked because:
- **Order-3 hashed**: natural cell count ~2^57, collisions ~10^11, but Zipfian token distribution concentrates mass on a tiny working set — many slots see one dominant 3-gram and ~0 contributions from the long tail.
- **Wiki × Order-2**: wiki-cardinality is 5, so the bigram-conditioned space is only 5× the bigram one — much denser per slot than Order-4.

Order-4 doesn't have the Zipf concentration that makes Order-3 hashing work; the conditional distribution `p(bit | p4, p3, p2, p1)` is too sparse even at the enwik9 scale of ~200 M tokens.

### Reverted

`src/xml_tok_route.rs` restored to the Phase 23D state at commit `ade0239`. Journal entry stays as a guardrail: the MLP can only learn from features that have **observation density per slot**, not just from features with high naturally-conditional information. Future feature additions should target either coarser conditioning (denser cells) or genuinely new context dimensions (e.g., recency-since-last-LZ-match, page-position, token length) rather than deeper word-history orders.

---

## 2026-05-13 — Phase 23D: Small Online MLP Joins the `dict_id` Mixer — 1.879 bpb on Enwik9

Step B of the A → B → C neural-arm path CDR sketched: a two-layer online MLP, the smallest architecture that adds a `tanh` nonlinearity and a learned hidden dimension on top of the Phase-23A sparse-LR.

### What landed

1. **`crate::neural::OnlineMLP<M, H>`** in `src/neural.rs`. Per-feature embedding tables (`H` floats per slot) summed into an `H`-dim hidden vector → `tanh` → linear scalar logit. Online SGD via the standard hidden-output backprop, with output weights initialised at `1/H` (not zero — otherwise the embedding tables get no gradient because `tanh'(0) · 0 = 0`, the dead-tanh problem).
2. **Wired as the 5th `dict_id` mixer arm** (`N_MIX_ID` 4 → 5). The mixer arms are now `[Order-1, Order-2, Wiki, SparseLR, MLP]`. Shares the same 3-feature shape as the sparse-LR — hashed Order-3, wiki × Order-2, `(bit_pos, prefix)` bias — so the MLP's value-add is the learned nonlinearity, not different features.
3. **Sizing**: `K=20` (1 Mi slots/feature), `H=8` hidden, `lr=0.02`. Memory: 3 × 1 Mi × 8 × `f32` = **96 MiB** for embeddings, plus 8 output weights + 1 bias.

Per-bit work: `O(M · H) = O(24)` for forward + same for backward. Pure scalar `f32` adds, `mul_add`s, one `tanh`, one `exp`. Bit-exact deterministic on a single machine, CPU-only — no matmul library, no GPU primitives. Determinism and runtime trivially satisfy CDR's two constraints (no GPU at test time; Hutter time budget).

### Results

| Config | enwik8 e2e bpb | Δ vs Phase 23C |
|---|---:|---:|
| Phase 23C | 2.1242 | — |
| **Phase 23D** | **2.1199** | **-0.0043** |

| Config | enwik9 e2e bpb | enwik9 bytes | Δ vs Phase 23C |
|---|---:|---:|---:|
| Phase 23C | 1.8805 | 235,066,968 | — |
| **Phase 23D** | **1.8792** | **234,905,363** | **-0.0013 / -158 KiB** |

Encode time: 654 s on enwik9 vs Phase 23C's 566 s (+15.5 %). Decode: 360 s vs 276 s (+30 %). The decode hit is bigger because decode pays the full forward+backward (the SGD update keeps state in lock-step with the encoder). Both ratios are well inside the Hutter time budget — total enwik9 encode+decode is 16.9 minutes versus the budget's hours-on-the-judging-machine.

Roundtrip verified.

### Why the gain is smaller than expected

Pre-build prediction was -0.05 to -0.15 bpb. Actual: -0.0013 bpb on enwik9. Two reasons:

1. **Feature reuse.** The MLP shares the same 3 hashed features as the Phase-23A sparse-LR. The LR already extracts what a linear model can from those features; the MLP can only add interactions among them through its nonlinearity. With M=3 the interaction space is small.
2. **Diminishing returns at the dict_id layer.** Phases 23A/B/C have already pulled the mixer-arm portfolio to a fairly tight bound on what hashed-feature predictors can extract. cmix's gains come from much larger MLPs (~50 K params) eating dozens of features.

Both observations suggest the next probe should be **new features for the MLP**, not bigger weights. Candidates: `(bit_pos, prefix, wiki, prev_class)` (token class as wiki cross), a hashed `(p4, p3, p2, p1)` Order-4 feature, or a `(byte-offset-since-LZ-match)` recency feature. None of these are in the LR's feature set.

### Phase 23D cumulative on enwik9

| Phase | enwik9 bytes | enwik9 bpb | Δ vs Phase 19k |
|---|---:|---:|---:|
| Phase 19k | 249,191,955 | 1.9935 | — |
| Phase 20l | 241,105,534 | 1.9288 | -0.0647 |
| Phase 21 | 237,259,434 | 1.8981 | -0.0954 |
| Phase 22 | 237,120,859 | 1.8970 | -0.0965 |
| Phase 23A | 235,510,295 | 1.8841 | -0.1094 |
| Phase 23B | 235,124,739 | 1.8810 | -0.1125 |
| Phase 23C | 235,066,968 | 1.8805 | -0.1130 |
| **Phase 23D** | **234,905,363** | **1.8792** | **-0.1143** |

Cumulative from `xml-tok` Phase 16 baseline (2.0667): **-0.1875 bpb / -18.8 MiB on enwik9**. Hutter ratio: **2.14× target**.

### Memory note

CDR confirmed peak resident size during enwik9 encode + verify is ~5.4 GiB. The biggest single consumer is `token_id_bit_o2` (Order-2 count, K=29 → 2 GiB); next-largest is the corpus + decoded-buffer pair held by `run_compress` for the verify step (2 GiB combined; collapses to ~1 GiB on the judging machine where compress and decompress run as separate invocations). Headroom against the 10 GiB Hutter cap is ample for now, but future MLP-arm growth (larger `K` or `H`) and step C (pretrained transformer) need to budget against that 2 GiB Order-2 anchor.

---

## 2026-05-13 — Phase 23C: Bumped `dict_id` LR K=22 → K=24 — 1.881 bpb on Enwik9

One-line change to test the hypothesis that the Phase-23A `dict_id` sparse-LR was still saturation-limited at K=22 (4 Mi slots/feature) on the enwik9 scale.

| Config | enwik8 e2e bpb | enwik9 e2e bpb | enwik9 bytes |
|---|---:|---:|---:|
| Phase 23B (K=22) | 2.1243 | 1.8810 | 235,124,739 |
| **Phase 23C (K=24)** | **2.1242** | **1.8805** | **235,066,968** |
| Δ | -0.0001 | -0.0005 | -57,771 bytes |

Memory: 48 MiB → 192 MiB (16 Mi slots × `f32` × 3 features). Runtime: encode 566 s vs Phase 23B's 559 s (+1.3 %); decode +3 %. Roundtrip OK.

Tiny win — flat on enwik8 (within noise), measurable on enwik9 (the larger corpus warms the bigger table enough that hashed Order-3 collisions hurt less). The cost is +144 MiB of decode-time memory, comfortably under the 10 GiB Hutter cap. Keeping it because it's a clean positive at no runtime price; not a result worth chasing further at the dict_id layer.

The diminishing-returns shape (Phase 23A -0.0129 bpb at K=22, Phase 23C another -0.0005 at K=24) is exactly the saturation curve one would expect: the per-feature gradient signal is bounded by observation density, and K=22 already captures most of it on enwik9.

---

## 2026-05-13 — Phase 23B: Sparse-LR on `lz_flag` and `token_oov_bit` Mixers — 1.881 bpb on Enwik9

Follow-up to Phase 23A. The same `SparseLR` primitive applied to the two remaining mixed streams — `lz_flag` (was this token an LZ match?) and `token_oov_bit` (was the non-match token an OOV?). Both are 1-bit-per-token decisions already mixing Order-1 + Order-2 count predictors; the sparse-LR adds the same shape of hashed cross-features that worked at the `dict_id` layer.

### What landed

1. **`flag_lr_features(ctx)`** in `src/xml_tok_route.rs` — a shared 3-feature shape for the two binary streams:
   - **Hashed Order-3**: `(p3, p2, p1)`. Natural cell count ~2^48; SGD with K=20 collisions.
   - **Wiki × Order-2**: `(wiki, p2, p1)`.
   - **Wiki × Order-1**: `(wiki, p1)`. Denser, captures the wiki-unigram interaction when p2 lands cold.
2. **Two independent `SparseLR<3>` predictors**, one per stream, K=20 (1 Mi slots × `f32` per feature). 24 MiB total: 4 MiB × 3 features × 2 streams. Same `lr = 0.02` as Phase 23A.
3. **Mixer arities bumped**: `N_MIX_LZ_FLAG` 2 → 3, `N_MIX_TOKEN_OOV` 2 → 3.

The two LRs share their feature shape because their available context is identical, but the per-stream weights differ — the LZ-match decision and the OOV decision are correlated but not equivalent.

### Results

| Config | enwik8 e2e bpb | Δ vs Phase 23A |
|---|---:|---:|
| Phase 23A | 2.1274 | — |
| **Phase 23B** | **2.1243** | **-0.0031** |

| Config | enwik9 e2e bpb | enwik9 bytes | Δ vs Phase 23A |
|---|---:|---:|---:|
| Phase 23A | 1.8841 | 235,510,295 | — |
| **Phase 23B** | **1.8810** | **235,124,739** | **-0.0031 / -376 KiB** |

The enwik8 and enwik9 deltas are identical to four decimals (-0.0031). That's consistent with the binary streams being relatively scale-invariant — the OOV/hit and match/non-match ratios stabilize quickly and don't sharpen the same way the dict_id stream does as the corpus grows. Encode time on enwik9: 559 s (≈ Phase 23A's 555 s); the two extra LR observers add negligible per-token cost.

### Phase 23B cumulative on enwik9

| Phase | enwik9 bytes | enwik9 bpb | Δ vs Phase 19k |
|---|---:|---:|---:|
| Phase 19k | 249,191,955 | 1.9935 | — |
| Phase 20l | 241,105,534 | 1.9288 | -0.0647 |
| Phase 21 | 237,259,434 | 1.8981 | -0.0954 |
| Phase 22 | 237,120,859 | 1.8970 | -0.0965 |
| Phase 23A | 235,510,295 | 1.8841 | -0.1094 |
| **Phase 23B** | **235,124,739** | **1.8810** | **-0.1125** |

Cumulative from `xml-tok` Phase 16 baseline (2.0667): **-0.1857 bpb / -18.6 MiB on enwik9**. Hutter ratio: **2.14× target**.

### Why the gain is smaller than 23A

Phase 23A landed -0.0129 bpb on enwik9; Phase 23B added another -0.0031. Two reasons the second deployment is smaller:

1. **Stream size.** The `dict_id` stream is ~17 bits/hit-token; `lz_flag` and `token_oov_bit` are 1 bit/token each. Even a large relative improvement on a small stream is a small absolute one.
2. **Diminishing context-feature returns.** The Phase 21 mixer note "the per-bigram cells starved because the OOV/hit ratio is dominated by unigram `prev_id` stats" still applies — the higher-order signal is genuinely thinner here than at the dict_id layer. The LR weights end up assigning most of their mass to the wiki × Order-1 feature (the densest, most general one), with smaller marginal contributions from the Order-3 and wiki × Order-2 features.

Both observations argue for landing the gain and moving on rather than pushing K higher on these particular streams.

### Next probes still in the LR family

- **Bump dict_id LR K from 22 → 24** (192 MiB total): the dict_id stream is large enough that the Order-3 feature might still be saturation-limited at K=22.
- **Add a sparse-LR arm to `case_pattern_global`** or the OOV byte streams (`oov_word_letter`, `oov_sep_byte`) — neither currently uses a mixer; landing them would require adding a mixer layer first.

After those, step B (small online MLP) remains the next architectural step.

---

## 2026-05-13 — Phase 23A: Sparse-LR as Fourth `dict_id` Mixer Arm — 1.884 bpb on Enwik9

CDR set the next major lever as a neural arm with an explicit A → B → C path (sparse-LR → small online MLP → pretrained-and-embedded transformer), under two hard constraints: no GPU at test time, and the Hutter time budget (70,000 / Geekbench5 hours). Step A is the cheapest move that still exercises the integration path — a sparse logistic regression as a 4th arm of the Phase-22 `dict_id` mixer. It lives entirely in CPU scalar `f32` + one `exp` per bit, so determinism and runtime are trivially satisfied; the only open question was whether it would buy real bits.

### What landed

1. **`crate::neural::SparseLR<M>`** in `src/neural.rs`. `M` categorical features, each backed by a hashed weight table of `1 << K` `f32` slots. Forward: `P(0) = sigmoid(bias + Σ w_i[hash_i mod 2^K])`. Online SGD on log-loss: `w_i += lr · (target − P(0))` for each active weight, plus the bias. Pure scalar; no matmul, no transcendentals beyond `exp`.
2. **Wired as the 4th `dict_id` mixer arm** (`N_MIX_ID` 3 → 4). Features:
   - **Hashed Order-3**: `(p3, p2, p1, prefix, bit_pos)`. The natural cell count is ~2^57 — a direct count table is impossible; hashed SGD at K=22 (4 Mi slots × `f32` = 16 MiB) averages noisy updates within and across collisions.
   - **Wiki × Order-2**: `(wiki, p2, p1, prefix, bit_pos)`. The Phase-22 wiki predictor stopped at Order-1 because its count table couldn't afford the cross; the LR can.
   - **`(bit_pos, prefix)` bias**: a per-position marginal that lets the LR carry an honest intercept when both higher-order features land in cold slots.
3. **K=22, learning rate 0.02**. Total LR memory: 48 MiB (3 features × 16 MiB) plus 4 MiB for the now-4-arm mixer table. The `LogitMixer<4>` cold weight is `1/4`, so a never-trained sparse-LR arm contributes a neutral `sigmoid(0) = 0.5` that the mixer ignores until SGD warms it.

### Results

| Config | enwik8 e2e bpb | Δ vs Phase 22 |
|---|---:|---:|
| Phase 22 (3-arm mix) | 2.1564 | — |
| **Phase 23A (4-arm with sparse-LR)** | **2.1274** | **-0.0290** |

| Config | enwik9 e2e bpb | enwik9 bytes | Δ vs Phase 22 |
|---|---:|---:|---:|
| Phase 22 | 1.8970 | 237,120,859 | — |
| **Phase 23A** | **1.8841** | **235,510,295** | **-0.0129 / -1.61 MiB** |

Roundtrip verified. Encode time on enwik9: 555 s (vs ~600 s typical for Phase 22 — the additional per-bit work is well inside noise of the rest of the codec). Decode: 253 s.

### Why this works

The base `dict_id` predictors are count-table Order-1, Order-2, and Order-1-within-wiki-sub-mode. Phase 21's mixer already learns *which base wins per bit position*, but every base is unigram or bigram on `prev_id`. The sparse-LR adds **Order-3 word context** and **wiki × bigram** — signals the count tables literally cannot afford to materialize (a count-Order-3 with K=22 would have 0.16 obs/cell on enwik8; the LR happily learns gradients from a handful of updates per slot because there's no Laplace floor to drown the signal). The mixer then learns to weight the LR arm up exactly when the higher-order context is informative and down when it's noise.

The relative enwik8-vs-enwik9 gain (-0.029 vs -0.013 bpb) is consistent with this story: enwik8 is small enough that the LR's hashed Order-3 is one of the few ways to extract trigram signal at all; on enwik9 the count tables get warmer and the relative win shrinks but stays solidly positive.

### Phase 23A cumulative on enwik9

| Phase | enwik9 bytes | enwik9 bpb | Δ vs Phase 19k |
|---|---:|---:|---:|
| Phase 19k | 249,191,955 | 1.9935 | — |
| Phase 20l | 241,105,534 | 1.9288 | -0.0647 |
| Phase 21 | 237,259,434 | 1.8981 | -0.0954 |
| Phase 22 | 237,120,859 | 1.8970 | -0.0965 |
| **Phase 23A** | **235,510,295** | **1.8841** | **-0.1094** |

Cumulative from `xml-tok` Phase 16 baseline (2.0667): **-0.1826 bpb / -18.3 MiB on enwik9**. Hutter ratio: **2.15× target** (target 109,685,197 bytes).

### Constraint check

| Constraint | Budget | Phase 23A actual |
|---|---|---|
| GPU at test time | none | none — scalar `f32` + `exp` only |
| Encode + decode time, enwik9 | ≪ 20 h (well under 70,000 / Geekbench5 hours on the judging machine) | 808 s total ≈ 0.22 h |
| RAM | 10 GiB | ~50 MiB sparse-LR + existing tables (≪ budget) |
| Determinism encode ↔ decode | bit-exact required | bit-exact (same machine, same SGD schedule, same hash) |

### What this opens up

Step A was scoped to validate the integration path with the smallest possible move. The result is large enough that the same primitive deserves wider deployment before climbing to step B (small online MLP). Likely next probes, before going neural:

- **Sparse-LR on `lz_flag` / `token_oov_bit`** — both have mixers but their arms are Order-1 + Order-2 count only. Hashed cross-features (wiki × Order-2, prev-id × prev-prev-class) at SGD-amortized cost are a free probe.
- **Larger K on the dict_id LR** — K=22 is roughly where Order-3 cells average ~1 obs each on enwik9; K=24 (256 MiB total) might still help on the longer run.
- **An additional LR arm for sep/wordsep word-letter prediction** — the OOV byte streams have the same shape and Phase 22 confirmed structural splitting alone is too sparse.

Step B (small online MLP) remains the next architectural step, but the size of step A's win argues for harvesting the easy gains in the LR family first — each is a one-day probe with bit-budget discipline already wired.

---

## 2026-05-13 — Phase 22: 5-Mode Wiki Sub-Mode Classifier + Wiki-Conditioned dict_id Predictor — 1.897 bpb on Enwik9

CDR's roadmap put "Wikitext-syntax parser" as the next major lever after Phase 21's mixing infrastructure. A full parser is multi-week work; this phase scoped a narrower step: **promote the recon's 5-mode `WikiFineClassifier` to a first-class module and use it as a third predictor in the `dict_id` mixer**. The expectation: structural sub-mode (e.g. `LinkTarget` vs `Plain`) has different `dict_id` distributions that Order-1 / Order-2 alone don't capture.

### What landed

1. **5-mode `WikiFineClassifier`** in `src/wiki_classifier.rs`. Splits the old 3-mode classifier's `Link` into `LinkTarget` (before the `|`) and `LinkDisplay` (after); splits `Template` into `TemplateName` and `TemplateArg`. Same Moore-FSM construction, same zero-signaling-bits property. Promoted from a private helper inside `recon.rs`.
2. **Wiki-conditioned `dict_id` bit predictor** (`token_id_bit_wiki`, K=26 = 64 MiB) as the third arm of the `dict_id` mixer. Context = `(wiki_sub_mode, prev_id, prefix, bit_pos)`.

### Results

Mixer arity bumped to `N_MIX_ID = 3` for the `dict_id` stream while keeping `N_MIX_LZ_FLAG = N_MIX_TOKEN_OOV = 2` (per-stream const generics now).

| Config | enwik8 e2e bpb | Δ vs Phase 21 |
|---|---:|---:|
| Phase 21 (no wiki in mix) | 2.1576 | — |
| Phase 22a (3-mode wiki, K=25) | 2.1564 | -0.0012 |
| Phase 22 (5-mode wiki, K=26) | 2.1564 | -0.0012 |

| Config | enwik9 e2e bpb | enwik9 bytes | Δ vs Phase 21 |
|---|---:|---:|---:|
| Phase 21 | 1.8981 | 237,259,434 | — |
| **Phase 22 (5-mode wiki in mix)** | **1.8970** | **237,120,859** | **-0.0011 / -138 K bytes** |

The mixer correctly down-weights the wiki predictor on `Plain` mode (where `prev_id` alone is enough) and up-weights it on structural sub-modes; what's left is a small but real -0.001 bpb. The 5-mode classifier is the right substrate going forward — it carries the structural information later parser-based codecs will need — but the dict_id stream alone doesn't surface much of its value.

### Negative finding: per-wiki-sub-mode OOV byte model

Tried splitting the Order-4 OOV word letter model into 5 per-wiki-sub-mode sub-models (`Order1Ctx<5 × 27^4, 26>`, 276 MiB). Rationale: page names in `LinkTarget` have very different letter distributions than prose in `LinkDisplay`. Result on enwik8 e2e: **2.1610 / +0.0046 bpb regression**. 1.5 M OOV letters spread across 2.66 M cells = 0.56 obs/cell — too sparse no matter how the wiki mode partitions. The single global Order-4 model is already at the OOV-byte-volume saturation point; structural splitting is the wrong axis here. Reverted.

### Why this matters less than expected

The Phase 22 result reframes what the wiki sub-mode is for. **At the dict-id and OOV-byte streams, wiki sub-mode adds little signal beyond what Order-1/Order-2 already capture.** The bytes are mostly token-id boundaries and English letters; the structural distinction lives at a higher layer — *which tokens appear in `[[...]]` at all*, not *which bits encode the chosen token*. A real wikitext parser would intercept the structural framing (`{{`, `}}`, `[[`, `]]`, `|`) entirely and encode them as small structural codes plus per-stream content, not as bytes interleaved through the same dict.

That's a multi-week build. The 5-mode classifier is the substrate it would need. For now Phase 22 commits the modest in-place gain.

### Phase 22 cumulative on enwik9

| Phase | enwik9 bytes | enwik9 bpb | Δ vs Phase 19k |
|---|---:|---:|---:|
| Phase 19k | 249,191,955 | 1.9935 | — |
| Phase 20l | 241,105,534 | 1.9288 | -0.0647 |
| Phase 21 | 237,259,434 | 1.8981 | -0.0954 |
| **Phase 22** | **237,120,859** | **1.8970** | **-0.0965** |

Cumulative from `xml-tok` Phase 16 baseline (2.0667): **-0.1697 bpb / -16.9 MiB on enwik9**. Hutter ratio: **2.16×** target.

### Tables not pursued in Phase 22

| Idea | Where | Result | Reason |
|---|---|---|---|
| Per-wiki-sub-mode Order-4 OOV word letter | step b | +0.0046 e2e enwik8 | 5 × 531 K cells / 1.5 M OOV letters = 0.56 obs/cell |
| 3-mode → 5-mode at K=25 | step a' | wash | 1.67× context expansion at fixed K starves cells |
| Wiki-conditioned `lz_flag` predictor | (skipped) | n/a | Phase-22's small wiki signal on `dict_id` suggests lz_flag wouldn't move much either |

The realistic next step toward Hutter territory is **#3 from the roadmap — a neural arm** (LSTM / small transformer / pretrained-and-embedded). The wiki parser is the cleanest structural lever once we have a richer base predictor stack to feed.

---

## 2026-05-13 — Phase 21: PAQ-Style Logit Mixing Replaces the Router — 1.898 bpb on Enwik9

CDR asked for at least five proposals to push toward Hutter territory and indicated I could rewrite anything. The shortlist was (1) PAQ-style logit mixing, (2) Wikitext-syntax parser, (3) on-line / pretrained neural arm, plus minor extensions. We proceeded with #1 first because it's the substrate everything else slots into and the Phase 18 router was a known sub-optimal proxy for it.

### The mixer primitive

`src/mixer.rs` (`LogitMixer<N>`) blends `N` binary predictors in logit space. Per-context weights `w_i ∈ ℝ` are updated by online SGD on log-loss after each observed bit. Compared to the Phase-18 router (which paid `⌈log₂ N⌉` bits per token plus a `BitPredictor` table to compress the routing decision), the mixer pays **zero signaling bits**: the decoder runs the same weight-update schedule and arrives at the same prediction without any side-channel. Encoder and decoder mirror the observe path exactly, so weight state stays in lock-step.

Determinism: the implementation uses `f32` and the IEEE-754 ops in `stretch`/`squash` (`ln`, `exp`). On a single machine these are bit-exact between encode and decode, which is all the codec needs. The five built-in unit tests cover stretch/squash domain, cold-start uniform behavior, per-context learning convergence, and per-context independence.

### Step 2 — `dict_id` mixer (the big lever)

The Phase-18 router selected between `token_id_bit` (Order-1) and `token_id_bit_o2` (Order-2) per token, emitting 1 router bit per dict-id hit. Replaced with a `LogitMixer<2>` keyed on `(bit_pos, prefix)` — the mixer learns per-bit-position which base predictor wins. Memory: `K=18` slots × 2 weights × 4 B = 2 MiB. Learning rate swept on enwik8 e2e:

| LR | enwik8 e2e bpb |
|---:|---:|
| 0.01 | 2.1633 |
| **0.02** | **2.1625** |
| 0.05 | 2.1632 |
| 0.10 | 2.1665 |

LR=0.02 settled. Confirmed on enwik9 e2e: **1.9034 bpb / 237.9 MB — -0.0254 bpb / -3.18 MiB vs Phase 20l (1.9288 / 241.1 MB)**. The router bits disappear and the per-bit mixing produces sharper predictions than per-token routing on warm-vs-cold bigram transitions.

Mixer context experiments:

| Mixer ctx | K | enwik8 e2e bpb | Δ |
|---|---:|---:|---:|
| `(bit_pos, prefix)` | 18 | 2.1625 | baseline (selected) |
| `(bit_pos, prefix, prev_id)` | 18 | 2.1693 | +0.0068 — context too sparse |

Adding `prev_id` to the mixer context expanded the address space ~65 K× without growing the table, leaving every slot starved. The base predictors already carry `prev_id` / `prev_prev_id`; the mixer's job is **per-bit-position** weighting, not per-bigram. Confirmed.

### N=3 with Order-3 — not yet ready at enwik8 scale

Added `token_id_bit_o3` (K=26 = 64 MiB) and bumped N=3. On enwik8 e2e: 2.1630 vs 2.1625 N=2 — essentially noise (+0.0005). The Order-3 table at enwik8 scale has ~0.16 obs per cell on average; the mixer correctly weights it to near-zero, so it neither helps nor hurts. Removed the field for now; should re-test at enwik9 scale (where Phase 19j showed Order-2 K=28 helped) once Phase 22+ settle, since this is the cleanest place to integrate richer base predictors.

### Step 5b — `lz_flag` mixer

Reinstated the Phase-19f Order-1 `lz_flag` predictor (K=20 = 1 MiB) alongside the Phase-20l Order-2 (K=23 = 8 MiB) and a `LogitMixer<2>` keyed on `prev_id` (K=10 = 1024 slots). Result on enwik8 e2e: **2.1599 bpb / -0.0026 bpb / -29 K bytes** over step 2. Small but clean — the lz_flag stream is ~8 % of the archive, and the mixer trims another 0.3 % off that.

### Step 5c — `token_oov` bit mixer (rescues a Phase-20m negative result)

Phase 20m had tested Order-2 alone on the `token_oov_bit` and it regressed +0.003 bpb on enwik8 — the per-bigram cells starved because the OOV/hit ratio is dominated by unigram `prev_id` stats. The mixer fixes this exactly: it learns per-`prev_id` whether to trust Order-2 (warm bigram) or Order-1 (cold bigram).

Added `token_oov_bit_o2` (K=23 = 8 MiB) and `token_oov_mixer` (K=10, prev-id-keyed). enwik8 e2e: **2.1576 bpb / -0.0023 bpb / -29 K bytes** over step 5b. The mixer turns a -0.003 bpb regression into a +0.0023 win — a 0.005 bpb swing — by routing usage to the warm bigrams while keeping Order-1 dominant on cold ones.

### End-to-end on enwik9

| Phase | enwik9 bytes | enwik9 bpb | Δ vs Phase 20l |
|---|---:|---:|---:|
| Phase 20l baseline | 241,105,534 | 1.9288 | — |
| Phase 21 step 2 (`dict_id` mixer) | 237,921,727 | 1.9034 | -0.0254 |
| **Phase 21 final (+ lz_flag + token_oov mixers)** | **237,259,434** | **1.8981** | **-0.0307** |

The full archive drops **3.67 MiB on enwik9** and the codec is now at **2.16× the Hutter target** (down from 2.20× after Phase 20).

Cumulative from xml-tok Phase 16 (2.0667 bpb): **-0.1686 bpb / -16.8 MiB** on enwik9.

Encode time on enwik9 grew slightly: 489 s (Phase 20l) → 494 s (Phase 21). The mixer adds three lookup-and-update steps per emitted bit but each is `O(N)` with `N=2`, so the per-token cost increase is small.

### Methodology notes

- **The mixer rescues the panel-vs-e2e divergence pattern** seen all through Phases 19–20. Phase 19f Order-1 lz_flag was +0.059 bpb on panel and -0.011 on e2e; the mixer's adaptive weighting handles the cold→warm transition the same way that pattern manifests *within* a panel window. Future predictor additions can rely on the mixer to down-weight them when cold rather than tuning around panel quirks.
- **Mixers compose with Order-N upgrades from earlier phases.** The mixer replaces the *routing decision*, not the underlying predictors — every bit-budget gain from Phases 18-20 (Order-2 table K=29, Order-4 OOV word, Order-2 OOV sep, Order-2 lz_flag) still applies.
- **A small mixer LR (0.02) and a single (bit_pos, prefix) context is robust.** Initial Phase-21 instinct was to enrich the mixer context with `prev_id` for "per-bigram weight learning"; that regressed because the base predictors already carry `prev_id`. The mixer's job is *which base to trust*, which has a much smaller learnable surface than the bases themselves.

### What's left and what's next

Per the 5-suggestion roadmap, Step 1 of the plan (mixing infrastructure) is now live. Three streams (`dict_id`, `lz_flag`, `token_oov_bit`) are mixed; the multi-symbol OOV byte streams (`oov_word_letter`, `oov_sep_byte`) are not — adapting `LogitMixer` to multi-symbol mixing is a structural change that's deferred. Next step in the plan: **Wikitext-syntax parser** (proposal #4), independent of #1 and the biggest non-neural lever available.

### Tables not pursued

| Idea | Where | Result | Reason |
|---|---|---|---|
| Order-3 in mix at enwik8 | step 4 | +0.0005 | Order-3 table cold at enwik8 scale; revisit on enwik9 |
| `(bit_pos, prefix, prev_id)` mixer ctx | step 2 | +0.007 | mixer context too sparse |
| LR=0.05, LR=0.10 | step 3 | +0.0007, +0.004 | too fast, weights oscillate |
| LR=0.01 | step 3 | +0.0008 | too slow, weights under-converged at enwik8 scale |

---

## 2026-05-13 — Phase 20: Order-N Byte Modeling on OOV Streams + Order-2 lz_flag — 1.929 bpb on Enwik9

CDR returned and said "continue." The Phase 19k summary had handed off three candidates (PPM-D escape, OOV stream Order-2, mixing layer). With no further direction the cheapest first step was a fresh bit decomposition of the panel: where do the bits actually live in `xml-tok-route` Phase 19k? The answer reframed the next moves.

### The decomposition (Phase 20a/b)

Panel bench at 20 × 256 KiB measure windows, Phase 19k codec state:

| Component | Bits | % of archive |
|---|---:|---:|
| `lz_match_ac` (combined) | 7,129,694 | 52.36% |
| `token_hit_ac` | 3,711,050 | 27.22% |
| `token_oov_ac` | 2,029,907 | 14.89% |
| `case_ac` | 426,139 | 3.13% |
| `router_ac` | 245,226 | 1.80% |
| `tag_ac` | 65,301 | 0.48% |
| smaller streams | <30 K | <0.3% |

I split `lz_match_ac` into four sub-components (flag, bucket CDF, within-bucket uniform raw bits, length CDF) to see where the LZ bits really go:

| Sub-component | Bits | % of archive |
|---|---:|---:|
| `lz_uniform_ac` (raw within-bucket offset bits) | 3,483,993 | 25.56% |
| `lz_length_ac` | 1,423,485 | 10.44% |
| `lz_bucket_ac` | 1,127,462 | 8.27% |
| `lz_flag_ac` | 1,094,754 | 8.03% |

**The largest single uncompressed bit pool in the archive is `lz_uniform_ac` at 25.56%** — the within-bucket position of the LZ offset, encoded as raw bits through a uniform CDF for AC framing consistency.

### Phase 20c — `LZ_MATCH_NCTX` sweep, panel-window starvation

Doubling `LZ_MATCH_NCTX` from 1024 to 16384 on the panel **regressed** +0.024 bpb. The split decomposition showed `lz_bucket_ac` stayed flat (21-symbol CDF already saturated at 1024 rows × 21 syms = 21 K cells with ~350 K matches/panel) while `lz_length_ac` went up by 128 K bits — the 256-symbol length CDF has 16 K rows × 256 syms = 4 M cells, with only ~10 K matches per 256 KiB measure window. Cells went cold. Phase 20n later re-confirmed this on enwik8 end-to-end (NCTX=4096 regressed +0.016 bpb), so `LZ_MATCH_NCTX = 1024` is the saturation point even at enwik9 scale.

### Phase 20d — BitPredictor for the offset residual, abandoned

The 1/x distribution of LZ offsets says the MSB of the within-bucket residual should be biased toward 0 (lower half of each octave is ~58% likely). Replaced `encode_uniform_bits(rel, bucket)` with a `BitPredictor` keyed on `(bucket, bit_pos, prefix, prev_id)`, K=22 = 4 M slots.

- Panel: +0.007 bpb (small regression)
- enwik8 e2e: +0.026 bpb / +323 K bytes (worse, the cold-context warmup outweighs the bias signal)

Stripped `prev_id` and shrank to K=20 (`(bucket, bit_pos, prefix)` only):

- enwik8 e2e: +0.0004 bpb / +4,799 bytes (essentially a wash)

The theoretical maximum from the 1/x bias is ~0.022 bits/bit × ~3.5 M residual bits ≈ 77 K bits / panel = 0.7% of archive = 0.016 bpb. The signal is too weak to overcome per-context warmup overhead. Abandoned — `lz_uniform_ac` is treated as entropy-floor going forward.

### Phase 20e/f — Order-1 and Order-2 dict_id table size

The `ID_BIT_O2_K` curve from Phase 19j said K=27→28 gave -0.0040 on enwik9. The next steps:

| K | Order-1 table | Order-2 table | enwik8 e2e bpb | Δ vs prior |
|---:|---:|---:|---:|---:|
| baseline (Phase 19k) | K=25 (256 MiB) | K=28 (1 GiB) | 2.2299 | — |
| 20e: K=29 | K=25 | K=29 (2 GiB) | 2.2293 | -0.0006 |
| 20e: K=30 | K=25 | K=30 (4 GiB) | 2.2289 | -0.0010 (vs K=28) |
| 20f: K=26 | K=26 (512 MiB) | K=29 | 2.2283 | -0.0016 |
| 20f: K=27 | K=27 (1 GiB) | K=29 | 2.2280 | -0.0019 |

Settled on Order-1 K=27 + Order-2 K=29 (3 GiB total table). Diminishing returns past that; K=30 Order-2 added only -0.0004 over K=29 at the cost of doubling RAM. The per-doubling gain dropped from -0.0040 (K=27→28 on enwik9) to -0.0006 (K=28→29 on enwik8) — saturation as expected.

### Phase 20g/i/j/k — Order-N byte modeling on OOV streams (the real win)

`token_oov_ac` is the byte stream when a token isn't in the dict — `encode_length` + the per-character emission. Word characters used `Order1Ctx<27, 26>` (previous letter only, 729 cells); separator bytes used `Order1Ctx<257, 256>` (previous byte, ~66 K cells). Both Order-1 over the immediate preceding character.

English bigram statistics are extremely peaked (`th`, `he`, `in`, `er`, `an`...). Order-2 modeling — context = `prev_prev_char * NCTX + prev_char` over `(LETTER_NCTX, LETTER_NCTX)` for words and `(BYTE_NCTX, BYTE_NCTX)` for separators — is trivial memory-wise (76 KiB and 64 MiB respectively) and the per-cell observation count should saturate quickly on any realistic corpus.

| Phase | OOV word ctx | OOV sep ctx | enwik8 e2e bpb | Δ vs prior |
|---|---|---|---:|---:|
| baseline (Phase 20f, Order-1 both) | O1 | O1 | 2.2280 | — |
| **20g word**: Order-2 word letter | **O2** (729 cells) | O1 | 2.2207 | -0.0073 |
| **20g full**: + Order-2 sep byte | O2 | **O2** (66 K cells, ~64 MiB) | 2.2073 | -0.0207 |
| **20i**: Order-3 word | **O3** (19 K cells) | O2 | 2.2010 | -0.0270 |
| **20j**: Order-4 word | **O4** (531 K cells, ~55 MiB) | O2 | 2.1997 | -0.0283 |
| 20k: Order-5 word | O5 (14 M cells, ~1.5 GiB) | O2 | 2.2027 | +0.003 from O4 — regression |

Order-2 OOV separator (Phase 20g sep step) gave the single largest gain — **-0.0134 bpb** on its own — because the panel-level OOV breakdown turns out to be heavily separator-weighted (rare-byte sequences, URL fragments inside templates, accented letters). Word letter Order-2→3→4 then compounded: each new order dropped another 0.005–0.013 bpb on enwik8 e2e. Order-5 (14 M cells, 1.5 GiB) regressed because OOV-letter volume on enwik8 is ~1.5 M letters → 0.1 obs/cell on average, way too sparse. Order-4's 531 K cells at ~3 obs/cell is the sweet spot; enwik9 (~15 M OOV letters) would warm Order-5 better but probably not break even on warmup overhead vs Order-4.

### Phase 20l — Order-2 `lz_flag`

The `lz_flag` is binary (LZ match / no-match) emitted at every Content-token boundary. Order-1 (Phase 19f) saw `(prev_id)`; Order-2 sees `(prev_prev_id, prev_id)`. Binary streams converge fast even on cold bigrams (only two outcomes per cell), and the bigram catches multi-token structure the unigram misses — e.g., after `[[ X` for the same `X`, match likelihood differs sharply from after `; X` or `the X`.

| Config | enwik8 e2e bpb |
|---|---:|
| 20j (Order-1 lz_flag, K=20) | 2.1997 |
| 20l (Order-2 lz_flag, K=23 = 8 MiB) | 2.1983 |

-0.0014 bpb / -17 K bytes on enwik8. Small but real and clean.

### Phase 20m — Order-2 `token_oov_bit`, negative finding

Applied the same Order-2 upgrade to the `token_oov_bit` (the per-token hit-vs-OOV decision). Result on enwik8 e2e: **+0.0029 bpb / +37 K bytes — regression**. The reason: hit-vs-OOV is dominated by unigram `prev_id` statistics, not by bigram structure. After common prev tokens (`the`, `of`, `in`...) P(OOV) is ~0.04 and saturates almost immediately at Order-1. Splitting into bigrams just halves the observations per cell with no extra signal to extract. Reverted.

### End-to-end on enwik9 (the canonical measurement)

| Config | enwik9 bytes | enwik9 bpb | Δ vs Phase 19k |
|---|---:|---:|---:|
| Phase 19k baseline | 249,191,955 | 1.9935 | — |
| Phase 20j (without Order-2 lz_flag) | 241,839,904 | 1.9347 | -0.0588 |
| **Phase 20l (with Order-2 lz_flag) — final** | **241,105,534** | **1.9288** | **-0.0647** |

The full archive drops **8.09 MiB on enwik9**. The codec is now at **2.20× the Hutter target** (down from 2.27× after Phase 19k). Cumulative from the original xml-tok Phase 16 baseline (2.0667): **-0.1379 bpb**.

Time cost: encode 489 s, decode 201 s on enwik9 (Apple Silicon, single-threaded, default features). Memory peak ≈ 4 GiB.

### Bit-budget tracking — what just moved

The Order-N OOV byte models are doing exactly what the dict-id Order-2 routing did one level up: spending memory (55 MiB + 64 MiB on enwik8 e2e) to model a stream of high-entropy bytes against the bigram/trigram structure of natural English. The lesson generalizes: **wherever an emission stream still uses Order-0 or Order-1 byte-level conditioning, English's higher-order entropy floor (Order-3 ≈ 2.7 bits/letter, Order-4 ≈ 2.5) is leaving bits on the table.** Specifically out-of-scope for the lz_match path (offsets/lengths) where the symbols aren't English text and the bigram structure is the dict-id bigram, not letter bigrams.

### What's left in this layer

- **OOV separator Order-3**: 257^3 = 17 M direct rows = 17 GiB, over the 10 GiB budget. Would need hash-based addressing (BitPredictor-style with `(p3, p2, p1, bit_pos, prefix)` context). Probable -0.005 to -0.010 bpb on enwik9 based on Phase 20g/20i word scaling.
- **`lz_offset_residual` modeled per-prev_id**: failed at K=22 because the within-octave 1/x bias is too small relative to per-context warmup cost. Could revisit with a much coarser context (just `(bucket, bit_pos)` over ~400 cells, fully saturated) but the theoretical ceiling is 0.016 bpb total — bounded.
- **Cost-aware LZ matching**: currently emit a match whenever one exists at `MIN_MATCH = 3`. In warm bigram zones a 3-token match can cost more bits than 3 token emissions; skipping such matches deterministically (both sides compute the predicted cost) could save bits. Complex but a fresh direction.
- **Page-conditional models**: track current page section (body / talk / redirect) and condition predictors. Recon (`run_recon`) already classifies pages; the wiring would be modest.

The OOV/lz_flag wins are the cleanest extraction of the per-stream-Order-N pattern. Past this, gains come from architectural changes (e.g., mixing instead of routing) or external knowledge (grammar, semantics). The 1.93 bpb floor on enwik9 with a fully deterministic byte-level codec, no neural arm, no preprocessing, is now within reach of the v1 4M-parameter RWKV result (1.985 on enwik8 → ~1.98 on enwik9 extrapolated). The next branch question is whether v2 should pursue a neural arm, a richer non-neural model, or a different mode-routed split.

### Tables not pursued

| Idea | Phase | Result | Reason |
|---|---|---|---|
| `LZ_MATCH_NCTX = 4096`/16384 | 20c/20n | -0.024 panel / +0.016 e2e | lz_length cells go cold |
| `BitPredictor` LZ offset residual K=22 (w/ prev_id) | 20d | +0.026 e2e | 1/x signal too weak vs warmup |
| `BitPredictor` LZ offset residual K=20 (no prev_id) | 20d2 | +0.0004 e2e | wash |
| `ID_BIT_O2_K = 30` | 20e | -0.0004 vs K=29 | memory not worth marginal gain |
| `ID_BIT_K = 28` | (not measured) | n/a | already at saturation curve at K=27 |
| Order-5 OOV word | 20k | +0.003 from O4 | 14 M cells too sparse for OOV volume |
| Order-2 `token_oov_bit` K=23 | 20m | +0.003 e2e | OOV/hit dominated by unigram |
| Order-3 hashed OOV sep (64 K rows × 256 syms, FNV-mix trigram) | 20o | +0.018 e2e | ~1 M OOV sep bytes / 16 M cells = 0.06 obs/cell — sparse no matter how the trigram maps |

---

## 2026-05-16 — Phase 19: Routing Saturates at N=2; Order-1 Upgrades Compound to 2.006 bpb on Enwik9

CDR asked for a systematic sweep of three extensions to Phase 18's cost-aware routing — stickiness sweep, additional predictors (Order-3 hashed, wiki-sub-mode-conditional, class-based), and (later) Order-1 conditioning on the other Order-0 emission streams the codec was using. The framing was explicit: "test them all and report back," with grammar-aware and wide-context-window approaches called out as candidates of interest.

Two major findings, in the order they emerged:

### Finding 1: The routing layer saturates at N=2 with (Order-1, Order-2)

Phase 18 settled `xml-tok-route` at N=2 with the Order-1 and Order-2 `dict_id` predictors. Phase 19's first hypothesis was that adding more predictors to the routing stack would let the encoder route per-token to whichever predictor wins, compounding the gain. Tested predictors:

- **Order-3** — `(prev_prev_prev_id, prev_prev_id, prev_id, prefix, bit_pos)` context, 2^26 = 64 M-slot table.
- **WikiMode** — `(wiki_sub_mode, prev_id, prefix, bit_pos)` context using the 3-mode `wiki_classifier` from Phase 12, 2^25 slots.
- **Class** — `(class_of_prev_prev_id, prev_id, prefix, bit_pos)` where class is the 3-bit log-scale bucket of `prev_prev_id`'s dict slot. "Order-2 with coarsened `prev_prev_id`" — meant to be more saturated than full Order-2 on cold bigrams.

Configurations measured end-to-end on enwik9 (the only honest scale for this codec, per the Phase 17 lesson):

| Config | enwik9 bpb | Δ vs N=2 baseline (2.0392) |
|---|---:|---:|
| N=2 (O1, O2) baseline | 2.0392 | — |
| N=2 (O1, O3) | (e2e enwik8 +0.008 → e2e enwik9 ~+0.005) | regression |
| N=2 (O1, Wiki) | (e2e enwik8 +0.018) | regression |
| N=3 (O1, O2, O3) | 2.0519 | +0.013 |
| N=3 (O1, O2, Wiki) | (e2e enwik8 +0.016) | regression |
| N=3 (O1, O2, Class) | (e2e enwik8 +0.017) | regression |
| N=4 (O1, O2, O3, Wiki) | 2.0584 | +0.019 |

Every addition regressed. The mechanism: the router stream costs ~1 raw bit per token going from N=2 (1-bit router) to N=3/N=4 (2-bit router), and at ~0.06 effective bits / token under the adaptive router predictor, the overhead is ~12 MB on enwik9 — comparable to or larger than the savings any of the new predictors deliver. The new predictors overlap heavily with Order-2 because all four condition on `prev_id`; the tokens where Order-3/Wiki/Class win are also the tokens where Order-2 wins, so the marginal value at the routing layer is small.

**Stickiness sweep** (Phase 19a) on enwik8 e2e at fine granularity around the Phase-18 default (`ROUTE_STICKINESS_BITS = 1.0`):

| Stickiness | enwik8 e2e bpb |
|---:|---:|
| 0.0 | 2.2554 |
| 0.25 | 2.2540 |
| 0.5 | 2.2535 |
| 0.75 | 2.2533 |
| 1.0 | 2.2532 |
| 1.5 | 2.2543 |
| 2.0 | 2.2550 |

Optimum at 0.75–1.0, U-shaped. Kept at 1.0 (within noise of 0.75 and matches the enwik9 result we already had).

The conclusion from Phase 19a–e: **the routing architecture saturates at N=2 for `dict_id` emission**. Adding more predictors at this layer is exhausted at enwik9 scale.

### Finding 2: Order-1 conditioning on other Order-0 emission streams compounds

The bit decomposition on enwik9 has the dict_id at ~46% of bits, but `lz_match_ac` is another ~20% — dominated by the `lz_flag` emission (the no-match flag fires at every Content-token boundary, ~250 M times on enwik9). The Phase 16 `lz_flag` was an `Order0<2>` — a single global P(no-match) prior of ~10%. That gives ~0.46 bits per Content boundary, with **zero context conditioning**.

Replacing it with a `BitPredictor` keyed on `prev_id` turns it into Order-1: `P(no-match | prev_id)` instead of a global average. Same idea applied to the remaining Order-0 emissions on the codec's hot paths:

| Phase | Change | enwik8 e2e bpb | enwik9 e2e bpb | Δ on enwik9 |
|---|---|---:|---:|---:|
| baseline xml-tok-route N=2 | (O1, O2) Order-0 lz_flag, Order-0 token_oov, Order-0 lz_offset_bucket, Order-0 lz_length | 2.2532 | 2.0392 | — |
| **19f**: Order-1 `lz_flag` | `BitPredictor` on `prev_id`, replaces `Order0<2>` | 2.2426 | 2.0287 | **-0.0105** |
| **19g**: + Order-1 `token_oov` | Same shape, replaces the second `Order0<2>` | 2.2394 | 2.0216 | **-0.0176** |
| **19h**: + Order-1 `lz_offset_bucket` + `lz_length` | `Order1Ctx<1024, ...>` hashed on `prev_id` | 2.2340 | 2.0062 | -0.0330 |
| **19i**: + `ID_BIT_O2_K` = 26 → 27 (Order-2 table 256 → 512 MiB) | doubles slots, halves collisions on the long bigram tail | 2.2324 | 1.9997 | -0.0395 |
| **19j**: + `ID_BIT_O2_K` = 27 → 28 (Order-2 table 1 GiB) | another 2× | 2.2316 | 1.9957 | -0.0435 |
| **19k**: + `ID_BIT_K` = 24 → 25 (Order-1 table 128 → 256 MiB) | same idea for the Order-1 predictor | 2.2299 | **1.9935** | **-0.0457** |

The full enwik9 archive drops from **254,898,774 → 249,191,955 bytes (5.4 MiB smaller)**. Cumulative vs the original xml-tok (Phase 16 stack from 2026-05-13): **2.0667 → 1.9935 = -0.0732 bpb**. The codec is now under **2.0 bpb** on enwik9 for the first time, at **2.27× the Hutter target** (down from 2.32× after Phase 18, 2.35× after Phase 17, 2.41× after Phase 16).

The pattern from Phase 17/18 holds: panel results understate the gain when the change depends on adaptive-state saturation. Each Order-1 upgrade was modestly positive on enwik8 e2e and ~2× larger on enwik9.

**Phase 19i** bumped `ID_BIT_O2_K` from 26 to 27 (Order-2 table from 256 MiB to 512 MiB). On enwik8 e2e the delta was -0.0016 (within noise); on enwik9 it was **-0.0065**, matching the "saturation grows with corpus" pattern. **Phase 19j** scaled once more to K=28 (1 GiB table) for another -0.0040 on enwik9. K=29 (2 GiB) is the natural next test but past the point of meaningful return (the gains are halving with each doubling of the table; this is collision-vs-data-density limited).

### Class-based, Wiki-fine, and a methodology note

The class-based predictor (Phase 19e) was the most architecturally distinct of the alternates. It was meant to exploit "Order-2 with coarsened prev_prev_id" — the bet that the 8-bucket class is more saturated than full prev_prev_id and would win on cold bigrams that Order-2 misses. It regressed by **+0.017 bpb on enwik8 e2e**. The likely failure mode: a 3-bit class strips too much information from `prev_prev_id` for the savings on cold bigrams to outweigh the loss on warm ones, AND the routing overhead at N=3 swamps whatever marginal gain remains. A finer class (say 5 bits) might have done better, but the routing overhead pattern says the answer is no.

The 3-mode `WikiClassifier` was likely too coarse to win as the third predictor. The 5-mode `WikiFineClassifier` from `recon.rs` (split LinkTarget vs LinkDisplay, TemplateName vs TemplateArg) might have done better, but the routing-layer saturation finding suggests adding it as predictor #3 wouldn't help on enwik9 anyway. The honest move was to recognize the saturation and pivot to the Order-1 upgrade pattern that turned out to be the real win.

**Methodology note**: the divergence between panel and end-to-end was again the key signal. Order-1 lz_flag's full panel result on enwik8 was 2.440 bpb — **+0.059 worse than the N=2 baseline of 2.381**. End-to-end on enwik8 it was 2.2426 — **-0.0106 better than the N=2 e2e baseline of 2.2532**. The panel was wildly wrong. Per the rule of thumb from Phase 17/18 (panel-vs-e2e divergence > 0.02 bpb means the change interacts with predictor training volume), this is the steepest divergence we've seen. The lz_flag Order-0 predictor was so good after a few thousand observations that the panel's small measure windows just sample it at the saturated regime; the Order-1 BitPredictor on prev_id with `2^20` slots needs hundreds of thousands of observations to populate, so the panel sees the cold-context cost. The enwik9 result confirms what e2e enwik8 hinted at.

### What's left in this layer

I think the dict_id and LZ paths are now mostly squeezed. The remaining Order-0 emissions are small bit budgets:

- `tag_dict` (`Order0<33>`): ~3.3 K bits per panel window. Total budget on enwik9 is < 1 MB. Even halving doesn't move the needle.
- `tag_byte` (`Order0<256>`): same magnitude.
- `attr_byte` (`Order0<256>`): even smaller.
- `case_pattern_global` (`Order0<4>`): fires only on OOV words; small share.

OOV byte streams (`oov_word_letter`, `oov_sep_byte`) are already Order-1 on the previous *letter/byte*. Upgrading to Order-2 would buy a few thousand bits per panel window; ~5 MB on enwik9. Worth trying as a follow-up but not transformative.

The architectural next move is back to the bit-predictor table contents, not the predictor topology:

1. **Wider Order-2 table** (Phase 19i partial result): scaling `ID_BIT_O2_K` from 27 to 28 (1 GiB) is worth measuring; the marginal improvement at 26 → 27 was small (-0.0016 bpb) but at enwik9 scale the long tail of bigrams might benefit more.
2. **PPM-D-style explicit escape on dict_id**: the routing layer is one way to gate predictors; PPM-D's deterministic context-count escape is another. The two are equivalent in the strong-saturation limit, but PPM-D doesn't pay a router bit when the high-order context has no observations — which is roughly half of all tokens for Order-3.
3. **Different bit budget**: the bit_predictor `RESCALE_THRESHOLD = 4094` halves counts when total exceeds that. For super-hot contexts the halving fires frequently and the predictor underweights confidence. Tuning this could be free.

### Code state

`xml-tok-route` now has eight predictors plumbed (Order-1, Order-2, Order-3, Wiki, Class, plus the router_bit, lz_flag_bit, token_oov_bit) but the routing stack is reconfigured to N=2 (Order-1, Order-2). The unused predictors stay in the source as dead-code-marked fields so the dispatch table can be flipped to alternate configurations without re-plumbing. The five `PredKind` variants are all wired into `p_zero_for` — adding/dropping a predictor from the active stack is a one-line `PREDICTOR_KINDS` change.

The `lz_flag_bit`, `token_oov_bit`, `lz_offset_bucket_o1`, and `lz_length_o1` upgrades **replaced** their Order-0 predecessors rather than being dispatched optionally — they're strict wins everywhere we measured. The original `Order0<2>` lz_flag and `Order0<2>` token_oov stay in `Models` as dead-code-marked placeholders for traceability.

---

## 2026-05-15 — Phase 18: Cost-Aware Predictor Switching (New Best, −0.028 bpb on Enwik9)

CDR's idea: instead of mixing predictions cmix-style, maintain N predictors in lock-step and have the encoder emit a per-token "router" bit selecting whichever predictor minimizes that specific token's emission cost. Switching is **strictly more flexible** than PPM-D escape (which is a special case where the router decision is deterministic from context-count) and can exploit the encoder's per-token hindsight in a way the mixer can't — the mixer's weights are computed from past data, not from the actual token about to be emitted.

Implementation lands as `xml-tok-route` with N = 2 predictors over the 18-bit `dict_id` emission:

- **Order-1**: existing `(prev_id, prefix, bit_pos)` bit predictor (xml-tok's `token_id_bit`).
- **Order-2**: new `(prev_prev_id, prev_id, prefix, bit_pos)` predictor, 2²⁶ slots (~256 MiB table).

Per Content hit-token: encoder computes the 18-bit cost under each predictor (no AC state mutation — just `predict_p_zero` rollouts), adds a `ROUTE_STICKINESS_BITS = 1.0` penalty to non-current predictors to suppress flip-flop on near-ties, picks the cheaper. A 1-bit router via a `BitPredictor` keyed on `(prev_id, current_predictor)` encodes the choice; the predictor learns `P(stay)` and collapses to ~0.06 bits/token at scale. Both predictors observe every token regardless of which path was chosen, so encoder and decoder maintain identical state.

### Result scales with corpus size

| Scale | xml-tok bpb | xml-tok-route bpb | Δ |
|---|---:|---:|---:|
| enwik8 full panel (20 × 256 KiB measure, fresh predictor per window) | 2.381 | 2.390 | +0.009 (regression) |
| enwik8 end-to-end (`encode_window(b"", &enwik8)`) | 2.2576 | 2.2532 | -0.004 |
| **enwik9 end-to-end** (`encode_window(b"", &enwik9)`) | **2.0667** | **2.0392** | **-0.0275** |

Archive size on enwik9 drops from **258,332,159 → 254,898,774** bytes (3.43 MiB smaller). New v2/v3 best, putting the codec at **2.32× the Hutter target** (109,685,197 bytes). Encode time 5m 57 s (essentially unchanged from xml-tok's 5m 50 s — the per-token Order-2 lookup is amortized in the same hash table the Order-1 predictor uses). Decode 2m 13 s, 25% slower than xml-tok's 1m 46 s because the decoder also maintains both predictors and runs the router decode loop.

### Mechanism: the saturation effect that hurt Phase 17 now helps

Phase 17's failure mode was that as the corpus grew, the bit-level `dict_id` predictor accumulated enough per-context observations to predict rare-but-page-repeated tokens nearly as cheaply as the cache path could — the cache's amortization win shrank to zero by enwik9 scale. Phase 18 turns that mechanism around: the Order-2 predictor needs training volume to be competitive at all, and at enwik9 scale common bigrams (`(of, the)`, `(United, States)`, `(in, a)`) accumulate enough observations to beat Order-1 by 1–3 bits per token, while cold bigrams stay where the encoder routes them to Order-1 (Order-2 returns near-uniform with no observations).

Panel-vs-end-to-end calibration table from the decomposition:

| Component (per-window avg, full panel) | xml-tok | xml-tok-route | Δ |
|---|---:|---:|---:|
| token_hit_ac | 195,922 | 186,079 | **-9,843** |
| router_ac | 0 | 12,412 | +12,412 |
| Others | (same) | (same) | ~0 |

On the panel, Order-2 saves 9,843 bits/window but the router costs 12,412 — Order-2 just isn't trained enough on 256 KiB of measure window to pay back the routing overhead. At enwik8 end-to-end (100 MB measure), Order-2 saturation begins; at enwik9 (1 GB), Order-2 wins on enough tokens to net 27 K bits per MB.

### Methodology lesson (reversed from Phase 17)

Phase 17's lesson was: **panel can overstate** gain for codec changes that compete with adaptive-state cost reduction. Phase 18 adds the mirror image: **panel can understate** gain for codec changes that depend on adaptive-state saturation. The rule of thumb generalizes to: panel-vs-end-to-end divergence is the diagnostic — when they disagree by >0.02 bpb, the change is interacting with predictor training volume, which means the panel decision is unreliable.

End-to-end enwik8 was used as the **first reliable signal** (panel said +0.009; e2e said -0.004), then enwik9 confirmed the trend. The same two-stage calibration (panel → e2e enwik8 → e2e enwik9) should be the default discipline going forward.

### Extension path: N > 2 predictors

The codec is structured so adding a third or fourth predictor is small. `N_PREDICTORS`, `ROUTER_BITS = log2(N_PREDICTORS)`, and the `which: usize` parameter in `predict_id_cost` / `encode_dict_id_with` / `decode_dict_id_with` are the only places where N appears explicitly. The natural next candidates:

1. **Order-0 (page-conditioned)**: a per-page-reset predictor (similar in spirit to Phase 17's cache but without explicit indexing). Could capture topical drift inside long pages.
2. **Co-occurrence model**: keyed on a hash of recent dict ids in the page rather than a fixed-window order.
3. **Length-conditioned**: separate predictor when the previous token is a long separator (sentence break heuristic).

The risk of more predictors: as N grows, the encoder's stickiness penalty per non-current predictor (currently 1 bit) compounds, and the router cost per token grows as `log2(N)`. Empirically the right N is the one where each added predictor's savings cover its share of the router-bit growth — that's a per-predictor decision that needs the same panel → e2e enwik8 → e2e enwik9 calibration ladder.

### Tunable parameter audit

- `ROUTE_STICKINESS_BITS = 1.0`: encoder switches only if the new predictor saves at least 1 bit. Untuned. A sweep (0.0, 0.5, 1.0, 1.5) on enwik8 end-to-end is the natural next experiment if we want to squeeze another 0.005-0.01 bpb out of Phase 18 before adding predictor #3.
- `ID_BIT_O2_K = 26`: Order-2 table at 64 M slots (~256 MiB). Memory-vs-collision trade-off; the actually-populated set after 200 M tokens is ~50 M, so 26 has ~30% headroom. Could shrink to 24 to free 192 MiB without much accuracy loss.
- `ROUTER_BIT_K = 18`: 256 K-slot router predictor. Likely oversized for the `(prev_id, current_predictor)` context space; 16 (~64 K slots) would be enough.

---

## 2026-05-14 — Phase 17: Page-Local Cache (Negative at Enwik9, +0.008 bpb)

Following the Phase-0 recon, the candidate Proposal C — a per-page ring-buffer cache of dictionary ids reset at every `<page>` boundary — looked like the most promising novel angle. The recon showed mean `avg_in_page = 2.89` across enwik9's top-1000 word types (with mid-rank outliers like "td" at avg 76 per page), suggesting substantial in-page repetition to amortize over. The implementation lands as `xml-tok-pc`: an additive layer over `xml-tok` that emits a `cache_flag` bit (BitPredictor on `prev_id`) before each Content hit-token, and on `cache_flag=1` emits a recency-relative position via a second BitPredictor whose context drops `prev_id` to converge on the MRU-dominant rel distribution.

### Cache-size sweep on enwik8 full panel

The position-cost / hit-rate trade-off has a clear sweet spot well below the natural "fits a typical page" choice:

| `PAGE_CACHE_SIZE` | bpb (full panel, enwik8) | Δ vs xml-tok |
|---:|---:|---:|
| 4096 (12-bit pos) | 2.391 | +0.010 (regression) |
| 1024 (10-bit pos) | 2.372 | -0.009 |
|  512 (9-bit pos)  | 2.356 | -0.025 |
|  256 (8-bit pos)  | 2.345 | -0.036 |
|  128 (7-bit pos)  | **2.339** | **-0.042** |
|   64 (6-bit pos)  | 2.339 | -0.042 |
|   32 (5-bit pos)  | 2.345 | -0.036 |

Smaller caches win because position cost drops faster than hit rate falls. The recon's mean `avg_in_page = 2.89` means re-occurrences cluster in a short recency window; a 128-slot cache catches almost all of them at 7 bits raw position cost, vs. a 4096-slot cache paying 12 bits raw to retain barely more hits.

### Panel vs. end-to-end vs. enwik9: the gain disappears at scale

The chosen 128-slot configuration was then measured at three scales:

| Scale | xml-tok bpb | xml-tok-pc bpb | Δ |
|---|---:|---:|---:|
| enwik8 full panel (20 × 256 KiB measure, fresh predictor per window) | 2.381 | 2.339 | **-0.042** |
| enwik8 end-to-end (`encode_window(b"", &enwik8)`) | 2.2576 | 2.2433 | -0.014 |
| **enwik9 end-to-end** (`encode_window(b"", &enwik9)`) | **2.0667** | **2.0745** | **+0.008** |

The cache *regresses* on the actual Hutter target.

### Mechanism: bit-predictor saturation eats the cache's headroom

The page cache was designed to amortize the **first-occurrence cost** of page-concentrated rare tokens — the bit predictor's `(prev_id, prefix, bit_pos)` table is data-starved on those tokens early in the corpus, so the first few mentions of "Einstein" inside the Einstein page each cost 12–18 bits and the cache could replace those with 5–7-bit cache positions.

As the corpus grows, the bit predictor accumulates per-context observations. At enwik9 scale (200 M Content hit-tokens × 18 bits per id emission = 3.6 G bit-level observations spread across 16 M predictor slots, ~225 observations per slot average), the predictor's marginal cost on rare-but-page-repeated tokens drops from ~15 bits to ~6 bits. The cache, fixed at 1 (flag) + ~5 (position) ≈ 6 bits per hit, no longer beats it — and the `cache_flag` overhead paid on **every** hit-token (the predictor learns to predict `0` for hot tokens, but never with full confidence) becomes the dominant cost.

Quick BoE on enwik9: 250 M hit-tokens × ~0.07 bit/flag = 17.5 MB of cache_flag overhead. Cache savings at 30% hit rate × ~1.5 bits avg/hit = 14 MB. Net **-3.5 MB**, matching the observed +1 MB archive size regression.

### Methodology note: the panel's third blind spot

The panel-survey discipline calibrated against Order-2 word context (Phase 3) revealed that the panel can **overstate** gains for predictors that need lots of training data, because each panel window starts with fresh model state. Phase 17 reveals the opposite failure mode: the panel can also **overstate** gains for codec changes that compete with a predictor whose marginal cost falls with training volume. Either direction, the rule is the same — extrapolation from panel to end-to-end is **not** justified for changes whose ROI depends on model state. The panel is a comparator for **local** effects (same model state, two codec paths); architectural decisions like "add a cache" need end-to-end validation on the actual target corpus.

Added to the rule of thumb: **a panel gain greater than ±0.02 bpb that depends on shared adaptive state should always be re-measured end-to-end on enwik8 before committing the design; gains that survive enwik8 should be re-measured on enwik9 before claiming.**

### What to do with the code

`xml-tok-pc` is left in the tree as an experimental side-path codec, addressable via `--codec xml-tok-pc`. It's bit-identical to xml-tok minus the cache layer (removing the cache-flag emission and falling back to Path A recovers xml-tok). The recon module (`src/recon.rs`) is the more durable artifact — it lands the byte-share-per-wiki-sub-mode and per-token page-locality measurements we'll want for any subsequent architectural decision.

### Implication for next phase

The take-away is that **the bit predictor itself is the bottleneck**, not its inputs. At enwik9 scale, a 16 M-slot bit predictor with `(prev_id, prefix, bit_pos)` context is roughly saturated on the head distribution but still under-trained on the tail. The path forward is to make the predictor **stronger** rather than route around it:

1. **PPM-D-style word context with escape** (Phase 18 candidate, Proposal A from the recon). Order-2 word with explicit escape to Order-1, calibrated against the saturated bit-predictor's per-token cost. Standard cmix machinery, predicted **0.05–0.15 bpb**, lower variance than Phase 17.
2. **Mixer over multiple predictors**. Run Order-1 and Order-2 in parallel, combine via logistic regression on context features. Even more cmix-like, predicted **0.10–0.20 bpb** but adds significant complexity.
3. **Different bit-predictor shape**. Replace the FNV-hashed (prev_id, prefix, bit_pos) with something closer to a probabilistic suffix tree, so the predictor degrades to lower orders smoothly when high-order contexts are starved. Less standard, predicted **0.05–0.15 bpb** with significant implementation cost.

Phase 18 will run with option 1.

---

## 2026-05-13 — Case-CDF zero-mass AC desync (bug fix)

End-to-end compression of enwik8 through xml-tok crashed reproducibly at byte 74_825_535 with `index out of bounds` in `tok_lz.rs:64` (`stream[pos]` with `pos == stream.len()`). The same call shape — `codec.encode_window(b"", &corpus_bytes)` — works on the panel (4 MiB warm + 256 KiB measure per window) and on the 8 KiB unit test; it only fails at sustained encode beyond ~75 MB. CDR asked for a single end-to-end enwik9 run after Phase 3 was reverted; the bug surfaced because the panel never exercises a continuous AC stream long enough for this CDF degenerate case to land.

### Symptom

The matcher's `at(pos)` is only reached after the decoder reads `lz_flag=1` (a match record); it then decodes `(offset, length)` from the AC. If `length > offset` the source slice `[stream_len - offset ..]` doesn't cover all `length` ids and `at()` reads past the end. Encoder side never produces `length > offset` (its `find_match` caps `cap = stream.len() - cand_pos = offset`), so the only way to get there is **the decoder read a phantom match where the encoder wrote no match**, then proceeded to decode garbage `length` and `offset` values from misaligned AC bits.

### Diagnosis

Reproduced via a new test (`roundtrip_enwik8_large_chunk`) encoding the first 80 MB of enwik8 in one shot, then a temporary diagnostic test that logged `(low, high, code/pending, bits)` for both encoder and decoder at every Content-mode `lz_flag` boundary. The first divergence in `(low, high)` localized to **content_iter 8_876_004** (encoder wrote `flag=0`, decoder read `flag=1`). The state immediately before that iter showed delta `dec_bits - enc_bits = 10`, which violates the AC-bit-accounting identity `delta = 32 + pending` (a decoder-startup-offset of 32 plus current encoder pending must be ≥ 32 always). Some earlier operation had emitted bits that don't correspond to any renormalization step.

Adding per-call AC tracing under `cfg(test)` and gating on the iter just before the divergence dumped this encoder log:

```
[enc] sym=1 cdf=[65418,65418/65536] pre low=0x7ddbeef0 high=0xd449caa3 pending=1
       post-subdivide low=0xd421f400 high=0xd421f3ff
       post-renorm    low=0x00000000 high=0xffffffff pending=0 bits=+23
```

`cdf[symbol] == cdf[symbol+1]` — **the CDF asked the AC to encode a symbol with zero probability mass**. The subdivide produces `new_high < new_low` (the invariant the AC relies on); renormalization then thrashes through 23 iterations emitting nonsense bits, eventually landing at the AC initial state `(0, 0xffffffff)`. The decoder, faithfully renormalizing on its own intact state, reads 23 fewer bits during the same iter, and is permanently misaligned thereafter.

### Root cause: `case_cdf_from_counts` doesn't rescale

The bad CDF is from `case_cdf_from_counts(&dict.entry(id).case_counts, &mut cdf)` — the 5-entry, 4-symbol CDF for per-token case-pattern encoding. Construction:

```rust
let total = counts[0] + counts[1] + counts[2] + counts[3] + 4;     // Laplace +1 per symbol
let mut acc = 0u64;
for (slot, &c) in out.iter_mut().zip(counts.iter()).take(4) {
    *slot = ((acc * u64::from(TOTAL)) / total) as u32;
    acc += u64::from(c) + 1;
}
out[4] = TOTAL;
```

Mass for symbol `i` is `(counts[i] + 1) * TOTAL / total`, floored. **When `total > TOTAL = 65536` and `counts[i] + 1 < total / TOTAL`, mass floors to zero.** All the codec's other adaptive models (`Order0`, `Order1Ctx`, `BitPredictor`) defend this by rescaling counts (halve with `.max(1)`) when their running totals approach `TOTAL/2`. The per-dict-entry case counts had no such rescale.

For a hot dict entry like "the" (id ≈ 50): by byte 75 MB, AllLower count is on the order of 10⁶, TitleCase count is much smaller (sentence-start "The" is rare relative to mid-sentence "the"), AllUpper and Mixed are nearly zero. With `total ≈ 10⁶` and `TOTAL = 65536`, any case-pattern count below ~15 hits zero mass. The first time the encoder happens to see the rare pattern (e.g. an `AllUpper` "THE" somewhere in the corpus), it asks the AC to encode a zero-mass symbol and the desync starts.

### Fix

`case_counts_inc(counts: &mut [u32; 4], idx: usize)` increments the bucket and, when the raw sum exceeds `CASE_COUNT_RESCALE = 1 << 15`, halves all four counts with `.max(1)`. Both encoder and decoder go through this helper at every case observation (encode, decode, prewarm, OOV bootstrap), so the rescale fires at exactly the same logical points on both sides. All seven previous `dict.entry_mut(id).case_counts[i] += 1` sites were replaced.

A defensive `debug_assert!(cdf[symbol + 1] > cdf[symbol])` was added at the top of `AcEncoder::encode` so any future model bug that produces a zero-mass CDF for an encoded symbol fails loudly at the point of violation instead of silently corrupting the AC state. Debug-only because the AC's CDF contract already requires strictly-increasing CDFs and we don't want to pay the check in release.

### Regression test

`xml_tok::tests::roundtrip_enwik8_large_chunk` (under `#[ignore]` because it's a 50 s test) encodes 80 MB of enwik8 in one shot and verifies the roundtrip. Without the fix, the test panics; with the fix, it passes.

### First full-corpus result: enwik9 at 2.067 bpb

With the fix in place, `lzr compress --corpus assets/enwik9 --codec xml-tok` runs end-to-end. First full-enwik9 result for the v2/v3 stack:

- **Archive:** 258,332,159 bytes (246.36 MiB) on 1,000,000,000 bytes input
- **bpb:** **2.0667**
- **Encode time:** 5m 50.2 s (2.72 MiB/s, single thread)
- **Decode time:** 1m 46.5 s (8.96 MiB/s, single thread)
- **Roundtrip:** OK
- **RAM:** ~2.4 GiB peak (well within the 10 GiB judging-machine budget)

The 3.3× decode-vs-encode speedup comes from the matcher side: encode runs `find_match` (hash-chain walk, MIN_MATCH verification, extension comparison) at every Content-token boundary; decode just looks up `matcher.at(src_start + k)` for the offset/length the encoder shipped. Wall-clock relevant to Hutter scoring is encode+decode combined since both run on the judging machine: 7m 36 s here on a 2026 dev box.

The full-corpus result is **0.314 bpb lower** than the v3 quick-panel result of 2.381 bpb on enwik8. The panel structurally underestimates a streaming codec's bpb because each window only has 4 MiB of warm prefix and 256 KiB of measure — the dictionary, the bit-level predictor's hash table, the case prior, and the LZ matcher's stream all keep filling with new structure right through the measure window. End-to-end on a 1 GB corpus, those tables saturate and most subsequent tokens are dict hits at near-Shannon cost. The panel's value is local-effect detection for codec deltas; the full encode is the only honest absolute bpb.

Distance to target: the Hutter target is 109,685,197 bytes = 0.878 bpb. Current is 258 MB / 2.067 bpb; closing the 149 MB gap means roughly halving archive size. Still far from neural-class results (cmix hits ~1.1 bpb on enwik9) but a meaningful baseline for the next phase's improvements to attribute against.

### What this changes

The bug is in the case-pattern model alone; quick-panel bpb numbers from Phases 13–16 and v3 Phases 1–3 are unaffected because the panel's 256 KiB measure windows never accumulate enough per-entry case counts to trip the threshold. End-to-end enwik8/enwik9 compression results were previously unobtainable and are now unblocked. Two general lessons:

1. **Every adaptive count source needs a rescale or it eventually breaks.** The discipline is uniform across `Order0`/`Order1Ctx`/`BitPredictor`; case_counts was a one-off that escaped the convention. Audit any future per-entry stat the same way.
2. **The panel discipline doesn't catch streaming-scale bugs.** Two latent AC bugs in v2 were found this way; this is a third. End-to-end full-corpus roundtrip is a category of test the panel can't substitute for. The new `roundtrip_enwik8_large_chunk` regression test fills that gap for xml-tok; the analogous test belongs in every codec that adds streaming state.

---

## 2026-05-12 22:30 — v3 Phase 3: Order-2 Word Context (Negative Result, +0.079 bpb)

Following the panel-survey finding that the Order-1 asymptotic floor is 1.518 bpb (and the codec at 2.381 bpb has 0.86 bpb of headroom), the next architectural step was to fold `prev_prev_id` into the bit-level predictor's context — Order-2 word. Literature suggests Order-1 → Order-2 cuts conditional entropy 20–30%; with the calibrated 5–10× discount on tokenization-stack predictions, expected codec gain was **0.05–0.15 bpb**, target **~2.23–2.33 bpb**. The change implemented cleanly: ~50 LOC threading `prev_prev_id` alongside `prev_id` through `id_bit_ctx`, `encode_token`, `decode_token`, `observe_token` and the encode/decode/prewarm main loops.

Result: **2.460 bpb full panel, Δ +0.079 vs Phase 1's 2.381.** Regression of 0.079 bpb. Reverted.

### Why it failed: context starvation at panel scale

`id_bit_ctx`'s new context is `(prev_prev_id, prev_id, prefix, bit_pos)`. With a 262 K dictionary, the `(prev_prev_id, prev_id)` space is ~70 billion potential pairs. The actually-observed pairs over a panel run (warm + measure ≈ 1.2 M tokens) are bounded by ~1 M unique bigrams. With 18 bits emitted per id, each unique (pp, p, prefix, bit_pos) context sees on average **well under 1 observation** across the entire window.

The `BitPredictor` returns `P(bit=0) ≈ 0.5` (cost = 1 bit) for any context with no observations. Tokens whose (prev_prev, prev) pair was rare in warm therefore pay **~18 bits per id** — vs Order-1's ~5 bits per id for the same token. The decomposition tells the story:

| component | Phase 1 (Order-1) | Phase 3 (Order-2) | Δ |
|---|---:|---:|---:|
| token_hit_ac    | ~45,200 | ~49,664 | **+4,464** |
| token_oov_ac    | ~27,460 | ~27,452 | 0 |
| lz_match_ac     | ~80,200 | ~80,119 | 0 |
| case_ac         |  ~5,380 |  ~5,413 | 0 |

The regression is concentrated in `token_hit_ac` (+4,464 bits/window avg). Order-2 didn't help hot tokens (those use the same context-prefix-collapse trick that already worked at Order-1) and severely penalized rare-pair tokens.

### Methodology note: the survey's fourth blind spot

The panel survey measures **asymptotic** entropy floors. Order-2 word has a lower asymptotic floor than Order-1 (probably 1.0–1.2 bpb). But the codec doesn't operate at the asymptote — it has Laplace smoothing on a finite-capacity hash table with bounded training data per context.

For Order-1 word on the panel: ~30 K unique prev_ids × ~18 bit positions × ~32 typical prefix values ≈ **17 M contexts**, of which maybe 1–2 M are populated by the 250 K tokens per window. Each populated context sees ~5–10 observations on average — enough for the Laplace prior to wash out and approach the true conditional probability.

For Order-2 word on the panel: ~1 M unique (prev_prev, prev) pairs × 18 bit positions × ~32 prefixes ≈ **600 M contexts**, of which maybe ~5 M are populated. Each populated context sees ~0.05 observations — Laplace dominates, predictions are near 0.5, and the model is structurally worse than Order-1 for this corpus size.

**Rule of thumb (added to calibration discipline)**: Order-N word context requires ~10× more training tokens per increment of N to reach the same effective per-context observation count. At panel scale (~1.2 M tokens/window), the ladder caps out somewhere between Order-1 and Order-2. To unlock Order-2 cleanly would require either:

- **More training data per context**: corpus-scale training (run the whole prewarm over a 100 MB pre-pass before measure), or pre-shipped Order-2 statistics.
- **PPM-style escape mechanism**: try Order-2 first; if context has <K observations, escape to Order-1. Pays one bit of escape signaling on cold contexts. Standard in PAQ/cmix.
- **Smaller alphabet for prev_prev**: collapse `prev_prev_id` to a coarse class (e.g., top-1024 ids + "other") to reduce the context space by ~256×.

### What this means for next moves

Order-2 word is genuinely blocked at panel scale. The path to closing the 0.86 bpb headroom to the Order-1 floor goes through:

1. **LZ improvements** (50% of bits per window). Cost-aware decision per xml-lz-cp Phase 7, lazy parse, larger MAX_MATCH. Estimated 0.05–0.15 bpb.
2. **OOV reduction** (17% of bits per window). Per-byte models conditional on token-prefix-so-far instead of a single Order-1-over-bytes model. Or sub-word fallback for OOVs (a token's OOV bytes fed through BPE on the byte stream).
3. **PPM-D-style multi-order word model**. Order-2 with escape to Order-1 with escape to Order-0. ~300 LOC; gain depends on escape calibration. Estimated 0.10–0.25 bpb.

xml-tok stays at Phase 1's 2.381 bpb. Phase 3 lands as an unreverted code path **deliberately not committed** — the journal entry is the record. The next phase will target the LZ slice (Phase 4 candidate: cost-aware LZ on tokens).

---

## 2026-05-12 21:30 — v3 Phase 1: Dict Cap 65 K → 262 K (Δ −0.031 bpb; Survey Over-Predicted by 10×)

Following the v3 capability-survey methodology, the first architectural change informed by measurement: widen the dictionary cap from 65 K to 262 K, switch id type from u16 to u32 throughout, emit 18 bits per id (up from 16). Pre-measurement prediction (from the survey, biased toward floor): **0.3–0.5 bpb**. Actual on the full 20-window enwik8 panel: **2.381 bpb (Δ −0.031 vs Phase 16's 2.412)**.

The result is positive but ~10× below the predicted range. The discrepancy is a survey methodology error, not a codec failure.

### What the survey measured vs. what the codec does

The survey walks the first 16 MiB of enwik8 from byte 0 and computes OOV rate against a top-N-frequency dictionary built from that same walk. On a fresh walk, ~7% of bytes land outside the top-64 K most-frequent forms; bumping to 256 K drops OOV to 0%. The survey reads this as 0.527 bpb of available headroom.

The bench panel doesn't walk from byte 0. Each panel window has a 4 MiB warm prefix that runs through the same tokenizer-and-dict-insertion path as the measure window. **By the time measure starts, the dict already contains every token that appeared in the warm prefix**, including most of the rare-ish tokens the survey was counting as "OOV at top-64 K." The panel's effective OOV rate is much smaller than the survey's, and it's dominated by first-occurrence-in-corpus tokens (proper nouns specific to the article in this window) — those are OOV at any dictionary size because they haven't been seen yet.

The dict-cap bump only helps when the prewarm-trained dict would actually have been 64 K-clipped during prewarm. Measured: roughly 540 bits of OOV-bits saved per 64 KiB quick-panel window, i.e. ~0.04 bpb. Plus a small win on the few rare-but-not-novel tokens that now fit in dict instead of going through OOV.

### Decomposition (quick panel, per 64 KiB window avg)

| component | v2 Phase 16 | v3 Phase 1 | Δ |
|---|---:|---:|---:|
| token_hit_ac    | ~45,000 | ~45,200 | +200 |
| token_oov_ac    | ~28,600 | ~27,460 | −1,140 |
| token_class_ac  |     ~56 |     ~38 | −18 |
| case_ac         |  ~5,400 |  ~5,380 | −20 |
| lz_match_ac     | ~80,000 | ~80,200 | +200 |
| tag_ac          |    ~444 |    ~468 | +24 |
| attr_ac         |    ~123 |    ~124 | — |

The headline saving is concentrated in OOV bits (about 1 % of total per window). Hit bits stay flat: the extra 2 bits per id are mostly absorbed by the bit-level predictor's high-bit-collapse on hot tokens (those high bits are 0 for nearly all hot ids). The codec basically pays for the extra bits where the dict actually expanded — minor.

### Calibration: the survey's third blind spot

The survey assumed static dictionary; the codec runs adaptive with prewarm. For predictions to track:

1. **OOV rate should be measured under panel conditions (prewarm + measure)**, not on a fresh walk from byte 0. The fresh-walk number over-counts OOV by a large factor — most "first-N-occurrence" tokens hit dict during prewarm in any realistic bench.
2. **The Order-1 floor of 1.584 bpb is an asymptotic-data floor.** Adaptive models on a 4 MiB warm prefix don't reach it because per-context observation counts are small for the long tail.
3. **The remaining 0.83 bpb between Phase 16 (2.412) and the survey's floor (1.584) is the gap I want to close**, but most of it is *not* available from dictionary-cap surgery — it's available from richer context modeling (Order-2 word, sparse contexts) and from the things the survey didn't measure (per-token case bits, LZ overhead).

Prediction calibration updated: for survey-derived headroom predictions, divide the floor-vs-current gap by ~5–10× to get the actual gain achievable from a single targeted change. The survey's value is in showing the **direction** is correct (word tokens > BPE; conditional models > Order-0); the **magnitude** of any single change is much smaller than the asymptotic delta.

### Implementation notes

Mechanical refactor: every `u16` dict-id type became `u32`; `id_bit_ctx` takes u32 prev/prefix; bit loop emits `ID_BITS = 18` iterations; `ID_BIT_K` bumped from 22 to 24 (4 M → 16 M context slots = 64 MiB) to keep collision rate near zero at the widened context space. `tok_lz::TokenMatcher` stream is now `Vec<u32>`. Length encoding stays at 16 bits (separator tokens don't exceed 64 KiB).

All 7 xml-tok roundtrip tests pass including 4 MiB-warm + 8 KiB-measure on real enwik8. 129 total tests green.

### Status & next move

xml-tok v3 lands at 2.381 bpb full panel — v3's new best, 0.227 bpb under xml-lz-cp. The headline result is positive but the predicted-vs-actual ratio is informative: dict-cap surgery on the Phase 16 architecture is mostly tapped out. The next architecturally larger lever per the survey is **Order-2 word context** (fold `prev_prev_id` into `id_bit_ctx`). Literature suggests Order-1 → Order-2 cuts conditional entropy by ~20–30%; with my calibration-corrected expectation (~5× discount), realistic gain is **0.05–0.15 bpb**, landing xml-tok around **2.23–2.33 bpb**.

The survey should be re-run as a "panel survey" — same tokenization sweep but measuring entropy under panel-style prewarm conditions — to give predictions that actually match what the codec sees. Adding that to the queue.

---

## 2026-05-12 20:00 — v3 Capability Survey: Order-1 Word Floor is 1.584 bpb; The 65 K Cap Was Wrong

After a strategic conversation about whether incremental gains on the xml-tok stack could close the gap to the Hutter target, CDR raised three pointed questions: is the 65 K dictionary size the right choice, is letter/non-letter splitting the right unit, and was the adaptive-online commitment premature given that the Hutter scoring allows shipping a pre-computed vocab? Rather than guess further, branch `v3` was created from `v2` and a capability-survey subcommand was built to measure the entropy floor of each tokenization scheme on real corpus data before committing more codec engineering.

### Method

`lzr survey --corpus assets/enwik8 --bytes N` walks the first `N` bytes of the corpus and computes Order-0 and Order-1 entropy for every row in the survey, expressed as bits-per-byte-of-original so comparisons are direct. Schemes:

- **Bytes** at Order-0, Order-1, Order-2 (dense 256, 256×256, sparse 256×256×256).
- **Word tokens** (letter / non-letter alternating runs, lowercased word keys) at dict caps 8 K, 16 K, 32 K, 64 K, 128 K, 256 K. Top-N most-frequent keys are in-dict; the rest pay full byte-cost OOV.
- **BPE tokens** trained on a 4 MiB prefix (for tractability) at vocab sizes 4 K, 8 K, 16 K, 32 K, 64 K. Pre-split on letter / non-letter runs (mirroring xml-tok). Fast BPE implementation using a lazy max-heap over the pair-count table to avoid the `O(|pairs|)` find-max scan per merge.

OOV cost for word tokens is accounted as the full 8-bpb byte spelling, treated as an upper bound. Order-1 conditional entropy is `Σ P(prev) · H(curr | prev)` over the observed corpus.

### Result (16 MiB of enwik8)

```
scheme    params         n_tokens   Ord-0 bpb  Ord-1 bpb   notes
--------  ------------  ---------   ---------  ---------   -----------------------
bytes     Order-0        16.8 M       5.090       —          —
bytes     Order-1        16.8 M         —        3.875       —
bytes     Order-2        16.8 M         —        3.066       —
word      dict=8 K        4.3 M       3.675      3.124      22.2% OOV
word      dict=16 K       4.5 M       3.327      2.697      15.6% OOV
word      dict=32 K       4.6 M       3.080      2.379      10.8% OOV
word      dict=64 K       4.7 M       2.881      2.111       7.0% OOV  ← xml-tok cur
word      dict=128 K      4.8 M       2.637      1.795       2.8% OOV
word      dict=256 K      4.8 M       2.476      **1.584**   0.0% OOV
BPE       vocab=4 K       6.8 M       3.501      2.232       —
BPE       vocab=8 K       6.2 M       3.284      2.079       —
BPE       vocab=16 K      5.8 M       3.108      1.955       —
BPE       vocab=32 K      5.5 M       2.976      1.864       —
BPE       vocab=64 K      5.4 M       2.912      1.810       —
```

Reference points: xml-tok Phase 16 lands at 2.412 bpb on the full 20-window panel. xml-lz-cp at 2.608. v1 deterministic floor at 2.056.

### Five findings, in order of impact on architectural choices

**1. The 65 K dictionary cap was wrong by ~0.5 bpb of pure-entropy headroom.** At 64 K we pay 7% of corpus bytes as full-byte-cost OOV (~1.17 MB of 16 MiB). At 256 K, OOV vanishes. The Order-1 floor drops from 2.111 to **1.584 bpb** — a 0.527 bpb improvement that's available *just by changing the dictionary size cap*. The 16-bit packaging convenience cost us real bits.

**2. Word tokenization beats BPE at every comparable operating point.** BPE 64 K = 1.810 vs Word 128 K = 1.795 (with 2.8% OOV) vs Word 256 K = 1.584 (no OOV). BPE never reaches the floor that word tokenization hits at 256 K. The reason is that BPE splits common words into multiple sub-tokens, paying the Order-1 conditional-entropy tax once per sub-token rather than once per word; whole-word tokens preserve more semantic coherence and ride lower in the Zipfian distribution. **On English Wikipedia, BPE is not the winning move.**

**3. xml-lz-cp at 2.608 outperforms byte Order-2 (3.066) by 0.46 bpb.** Confirms that the LZ matcher + mode-routed classifier in xml-lz-cp adds genuine structure beyond byte Order-2 alone — not pure adaptive-Order-2 in another guise.

**4. The Order-1 word floor is 1.584 bpb; xml-tok Phase 16 is at 2.412.** Headroom of **0.83 bpb** on the current architectural choice if we approach the conditional-entropy floor. That's a meaningful gap and a clear next-phase target: chase the floor.

**5. Pre-computed vs adaptive is not the binding question.** The Order-1 entropy floor of a tokenization is a property of the tokenization itself, not of how IDs are assigned. Adaptive Order-N models converge to the same floor given enough training data; pre-computed vocabs reach it from token 1. The 4 MiB warm prefix already provides ~150 K training tokens — enough for the adaptive model to converge on hot tokens before measure starts. **The choice is about cold-start cost, not asymptotic compression.** Shipping a vocab is justified only if the cold-start cost on the first few thousand tokens of the corpus matters, which it might in tight panels but is amortized over enwik9.

### Methodology indictment

This survey would have prevented two negative-result phases (12 wiki sub-mode, 14 bit-level rewrite) and probably reshaped Phase 13 substantially. The pattern across the v2 build is that bit-budget discipline was applied *within* an architecture choice but never *across* architecture choices. Each phase predicted which decomposition cells would move; no phase predicted whether the next architecture would have a meaningfully better entropy floor.

Going forward: **any new architectural commitment runs the capability survey first.** ~500 LOC of analyzer, runs in 15 minutes, produces falsifiable numbers. The cost of not doing it has been measured: ~3 phases of work that landed wrong-side-of-target because we didn't know the floor.

### Architectural decision

Branch `v3` is currently survey-only — no codec changes. The clear next move is **bump the dictionary cap from 65 K to 262 K (18-bit ID encoding) on the existing xml-tok stack**. Expected gain: most of the 0.527 bpb gap to the 256 K Order-1 floor. Implementation: small change to `id_bit_ctx` and the bit-level emission loop; no model restructure.

After that, the remaining headroom (~0.5 bpb between the realized Order-1 dict-256 K and the pure floor of 1.584) lives in:
- **Order-2 word** context (`prev_prev_id`): closes another fraction of the floor gap.
- **Better OOV handling on enwik9**: even at 256 K cap on 16 MiB, OOV vanishes — but enwik9 has ~3× the unique-type count of enwik8; the OOV story will return at corpus scale.
- **Per-position context refinements** that the survey can't measure directly: skip contexts, indirect contexts, match models per order.

The 1.584 bpb floor is what we can achieve from a *single* Order-1 word predictor at infinite training. To push lower requires either higher-order context (Order-2, Order-3) or mixing multiple correlated predictors via PAQ-style adaptive regression — exactly the cmix recipe discussed earlier.

### Status

`v3` branched off `v2`. New code:
- `src/survey.rs` — orchestration + byte-level + word-token entropy.
- `src/bpe.rs` — fast BPE trainer (lazy max-heap) + tokenizer + tests.
- `src/main.rs` — `survey` subcommand.

129 tests pass (4 new BPE tests). Build clean.

Next session: decide whether to bring the 65 K → 262 K dict-cap change back into v2 directly, or continue iterating on v3 with the survey as the guiding measurement. Either way, the survey replaces the "what should we do next" guesswork with measured headroom.

---

## 2026-05-12 19:00 — Phase 16: LZ on the Token Stream — 2.412 bpb, −0.196 vs xml-lz-cp

CDR/Claude added a hash-chain LZ77 matcher operating on the u16 token stream emitted by xml-tok's online dictionary, mirroring `src/lz.rs`'s byte-level design but indexing on token IDs instead of bytes. Pre-measurement prediction (biased-toward-floor per the Phase 15 calibration note): **2.40–2.48 bpb**. Actual on the canonical 20×256 KiB enwik8 panel: **2.412 bpb** — right at the floor of the predicted range. v2's new best, **0.196 bpb under xml-lz-cp** (2.608), and **0.122 bpb under Phase 15's Order-1 word model** (2.534).

### Mechanism

New `src/tok_lz.rs` (~200 LOC): `TokenMatcher` with a hash-chain over a 1 M-token (`WINDOW_SIZE = 2^20`) sliding window, hashed on 3-token triplets into a 256 K-bucket hash table. `MIN_MATCH = 3` tokens, `MAX_MATCH = 258`, `CHAIN_DEPTH = 32` (same shape as `src/lz.rs`).

The matcher's stream contains every token id emitted to Content (warm + measure), in emission order. After each id is pushed, the 3-token window ending at the new position is hashed and inserted at the head of its bucket's chain. Tokens that didn't get a dictionary id (capacity-full corner case) break any match passing through that position.

`xml_tok.rs` changes:
- Three new models: `lz_flag: Order0<2>`, `lz_offset_bucket: Order0<21>`, `lz_length: Order0<256>`. Bucket-relative low bits emit through a shared `UniformCdfCache` (same construction as `xml_lz_cp`).
- Encode loop's Content branch first calls `lookahead_tokens(buf, pos, dict)` — walks forward up to `MAX_MATCH` tokens, stopping at an OOV (no dict id) or a separator containing `<` (the next byte transitions out of Content). If `>= MIN_MATCH` lookahead tokens exist, hash the first three and run `matcher.find_match()`.
- One `lz_flag` bit is emitted per Content-token boundary regardless. The bit's adaptive Order-0 cost converges to `H(match_rate)` ≈ 0.15 bpb when matches are rare and ≈ 1 bit when they're 50/50; in practice it lands at ~0.05 bpb.
- Match record: `flag=1 | bucket | raw bucket bits | length token`. Per matched token, emit only the case pattern (via per-token `case_counts`); the token ID itself is *not* emitted — the decoder reads it from `matcher.stream[stream_len - offset + k]`.
- Decode mirrors: read flag; if 1, decode offset+length, copy ids from history, emit each id's bytes (with case if it's a word per `dict.entry(id).lower[0]`), push each id to the matcher.
- Prewarm threads `matcher` through and pushes every observed warm token's id so the matcher's history is identical at the start of measure on both sides.

### Result (full panel, 20 × 256 KiB enwik8)

| codec | mean bpb | best window | Δ vs xml-lz-cp |
|---|---:|---:|---:|
| xml-lz-cp (Phase 7)                | 2.608 | 2.012 (W3 quick / W4 full at offset 19280128) | — |
| xml-tok Phase 15 (Order-1 word)    | 2.534 | 2.199 | −0.074 |
| **xml-tok Phase 16 (LZ on tokens)** | **2.412** | **2.016** | **−0.196** |

Best window: **2.016 bpb** at offset 19280128 — within 0.04 bpb of v1's deterministic floor (2.056) on a single window. All 20 panel windows improve over Phase 15.

Quick-panel Phase 15 → Phase 16 deltas:

| window | Phase 15 | Phase 16 | Δ |
|---|---:|---:|---:|
| W1 | 2.629 | 2.545 | −0.084 |
| W2 | 2.578 | 2.468 | −0.110 |
| W3 | 2.451 | 2.167 | **−0.284** |
| W4 | 2.589 | 2.564 | −0.025 |
| W5 | 2.607 | 2.442 | −0.165 |

Window 3 (the LZ-friendly repetition-heavy window) sees the biggest gain — the Order-1 word model alone couldn't capture phrase-level repetition the way LZ-on-tokens does.

### Decomposition

Quick-panel components (avg per 64 KiB window):

| component | Phase 15 | Phase 16 |
|---|---:|---:|
| `token_hit_ac`    | ~134 k | ~45 k |
| `token_oov_ac`    |  ~29 k |  ~29 k |
| `token_class_ac`  |     ~56 |     ~40 |
| `case_ac`         |   ~5.4 k |   ~5.3 k |
| **`lz_match_ac`** |       — |  **~80 k** |
| `tag_ac`          |    ~444 |    ~444 |
| `attr_ac`         |    ~123 |    ~123 |

The shift is dramatic and clean: `token_hit_ac` collapses from 134 k to 45 k bits (matched tokens no longer go through individual hit emission) while `lz_match_ac` picks up 80 k bits. The 80 k > (134 k − 45 k) = 89 k difference is the LZ overhead price — about 9 k bits/window of fixed-cost overhead (flag bits, bucket headers, length tokens) on top of the per-matched-token case bits.

Per-matched-token cost: roughly **~9 bits/match-record-amortized** (close to Phase 15's per-hit cost), but the savings come from the *fixed-overhead amortization* over long matches. A 10-token match costs ~30 fixed bits + 10 × ~1.5 case bits = ~45 bits = **4.5 bits/token**, vs Phase 15's ~8 bits/hit. The longer the match, the bigger the per-token saving.

### Prediction calibration

| phase | predicted | actual | error |
|---|---|---:|---:|
| Phase 13 (naive tokenization)  | 2.1–2.3 | 2.826 | +0.55 |
| Phase 14 (bit-level rewrite)   | gain    | null  | +0.30 |
| Phase 15 (Order-1 word)        | 2.0–2.2 | 2.534 | +0.33 |
| Phase 16 (LZ on tokens)        | **2.40–2.48** | **2.412** | **+0.00** |

First on-target prediction since the v2 rewrite. The "bias toward floor by 0.3 bpb" rule absorbed the systematic optimism from prior phases. Going forward I'll keep the floor-bias for tokenization-stack predictions and remove it once a phase outside this stack lands close to its un-biased estimate.

### Status

xml-tok at 2.412 bpb is v2's new best, **0.196 bpb under xml-lz-cp** and within 0.36 bpb of v1's deterministic floor. The token-level codec stack is now competitive in absolute terms — Phase 16 is the first v2 phase that meaningfully closes the v2→v1 gap. Phase 17 candidates, roughly in order of expected gain:

- **Order-2 word** (fold `prev_prev_id` into bit context): ~0.05–0.10 bpb additional. Cheap. Same hashing pattern as Phase 15.
- **Cost-aware LZ decision** (compute match-cost vs literal-cost per position, pick cheaper): ~0.02–0.06 bpb additional. Phase 16 takes any match ≥ MIN_MATCH; a cost-aware pass would reject marginal short matches whose amortized cost exceeds the Order-1 hit cost.
- **Lazy parse for LZ** (look 1 token ahead; if the match starting at `pos+1` is longer than the one at `pos`, prefer that): ~0.01–0.03 bpb. Standard LZ improvement that xml-lz-cp's Phase 7 already uses; cheap to port.
- **Hybrid byte-LZ on the OOV path**: when an OOV token's lowercase form has a substring match in the byte history, emit a byte-LZ reference instead of per-letter emission. Larger code change; gain uncertain.

125 tests pass; xml-tok roundtrips green including 4 MiB-warm + 8 KiB-measure on real enwik8.

---

## 2026-05-12 18:00 — Phase 15: Order-1 Word Model — v2's New Best (2.534 bpb full panel)

CDR/Claude folded `prev_id` into the bit-level dict-id context to convert the Phase-14 Order-0 model into an Order-1 word model. Pre-measurement prediction (60% confidence): 2.0–2.2 bpb. Actual on the 20×256 KiB full panel: **2.534 bpb** — above the predicted range but the first v2 phase to beat xml-lz-cp's 2.608 baseline. **Δ = −0.074 bpb, the new v2 best.**

### Mechanism

One-line change to the bit-context hash:

```rust
fn id_bit_ctx(prev_id: Option<u16>, prefix: u16, bit_pos: u32) -> u64 {
    let prev_field = prev_id.map_or(u64::MAX, u64::from);
    let h = fnv_mix(FNV_OFFSET, prev_field);
    let h = fnv_mix(h, u64::from(prefix));
    fnv_mix(h, u64::from(bit_pos))
}
```

Plus threading `last_token_id: Option<u16>` through the encode/decode/prewarm loops (reset to `None` on each Content-run entry and on every tag/attr-mode transition). `encode_token` / `decode_token` / `observe_token` now return the assigned id so the caller can pass it as the next iteration's `prev_id`. `K_BITS` bumped from 20 to 22 (4 M slots, 16 MiB) to handle the richer `(prev_id, bit_pos, prefix)` context space.

The OOV-length predictor stays Order-0 (passes `None` to `id_bit_ctx`) — token length is independent of `prev_id`.

### Result (full panel, 20 × 256 KiB)

| codec | mean bpb | best window | Δ vs xml-lz-cp |
|---|---:|---:|---:|
| xml-lz-cp (Phase 7) | 2.608 | 2.012 (W3) | — |
| xml-tok Phase 14 (Order-0)        | ~2.83 (quick) | — | +0.32 (quick) |
| **xml-tok Phase 15 (Order-1 word)** | **2.534** | **2.199 (W4)** | **−0.074** |

Quick-panel comparison shows where Phase 15 wins and loses:

| window | Phase 14 | Phase 15 | Δ |
|---|---:|---:|---:|
| W1 | 2.825 | 2.629 | −0.196 |
| W2 | 2.779 | 2.578 | −0.201 |
| W3 | 2.964 | 2.451 | **−0.513** |
| W4 | 2.797 | 2.589 | −0.208 |
| W5 | 2.797 | 2.607 | −0.190 |

Window 3 is the most repetitive panel window (xml-lz-cp achieves 2.012 bpb on it via long LZ matches). It's also where Phase 15 saw the biggest improvement (−0.513 bpb from Phase 14) — repetitive content has highly predictable next-token distributions conditional on previous tokens, which is exactly what Order-1 exploits. xml-lz-cp still wins this single window by 0.44 bpb, but Phase 15 wins 3 of the other 4 quick-panel windows and the full-20-window mean.

### Decomposition shift

Avg `token_hit_ac` dropped from ~150k bits/window (Phase 14) to ~134k (Phase 15). Per-hit cost: ~10 → ~8 bits. That matches the literature claim that word-level Order-1 conditional entropy is ~30% below Order-0 — concretely, 10 → 8 bits is a 20% reduction, slightly below the literature's 30–40% because:
- Heavy tail of unique `prev_id`s starves rare contexts (mitigated only partially by the global Laplace fallback via the BitPredictor's hash slot).
- The bit-level predictor's hash slot count (4 M) doesn't perfectly separate all `(prev_id, bit_pos, prefix)` contexts.
- The Order-1 conditional model still pays the same OOV tax when novel tokens appear.

### Why it landed above the predicted range

My prediction was 2.0–2.2 bpb based on word-level Order-1 reaching ~6–7 bits/token. Actual is ~8 bits/token. Two contributors I underweighted:

- **Separator-class tokens.** Separator tokens make up ~50% of token emissions and have less Order-1 structure than words ("United" → "States" is high-mass; "," → "the" is moderate but the alternation prior already captures it). Conditioning on `prev_id` helps less for the separator half.
- **OOV path overhead stays.** Phase 15 didn't touch the OOV path. ~10–20% of tokens are still OOV at the average cost of ~20 bits each. Order-1 doesn't move this.

### Predictions discipline

Two phases ago I predicted 2.1–2.3 bpb for naive tokenization (got 2.83). One phase ago I predicted Fenwick/bit-level would move the needle (got null). This phase I predicted 2.0–2.2 bpb (got 2.53). Calibration: my predictions are systematically optimistic on tokenization-stack phases by ~0.3–0.6 bpb. The pattern: each phase exploits the structure it targets *less* than literature numbers because (a) OOV tax in finite-window panels stays high, (b) the conditional model has fewer effective observations per context than a corpus-scale Markov model would, and (c) I keep forgetting to weight by the separator-token half.

For Phase 16, I'll bias my prediction toward the lower end of the headroom estimate by 0.3 bpb to absorb this pattern.

### Status

xml-tok at 2.534 bpb is v2's new best. xml-lz-cp at 2.608 drops to second. Still 0.48 bpb behind v1's deterministic floor of 2.056 — but with a different, complementary mechanism (word-level Order-1) than v1's byte-level LZ77 + PPM-D stack. Phase 16 candidate: LZ on the token stream — catch phrase-level repeats (window 3-style) on top of the Order-1 word model. Realistic gain estimate (with the bias correction): **0.05–0.15 bpb additional**, putting xml-tok at ~2.40–2.48 bpb.

121 tests pass; 7 of them xml-tok roundtrips including 4 MiB-warm + 8 KiB-measure on real enwik8.

---

## 2026-05-12 17:00 — Phase 14: Bit-Level Dict-Id Encoding (Null Result; Phase 13 Was Already At Shannon)

CDR/Claude replaced Phase 13's 2-byte-chunked dict-id encoding (`Order0<256>` high + `Order1Ctx<256, 256>` low) with a bit-level `BitPredictor` (`K_BITS=20`, 4 MiB of count pairs), emitting each 16-bit dict id MSB-first as 16 individual bit-level AC emissions conditioned on `(bit_pos, prefix_so_far)`. Same treatment for the OOV length encoding. The Phase 13 diagnosis was that the chunked encoding was paying ~8.7 bits/hit against a presumed Shannon floor of 4–5 bits, with the tail's under-trained `(high, low)` rows dragging the average. That diagnosis was wrong.

### Result (quick panel, 5 × 64 KiB)

| codec | mean bpb |
|---|---:|
| xml-lz-cp baseline                             | 2.506 |
| xml-tok Phase 13 (chunked 2-byte)              | 2.826 |
| xml-tok Phase 14 (bit-level, `K_BITS=20`)      | 2.832 |
| xml-tok Phase 14 (bit-level, `K_BITS=22`)      | 2.832 (byte-identical to K=20) |

Bumping the bit-predictor table from 1 M slots to 4 M slots produced **byte-identical archives** on every panel window — confirming there are no hash collisions left to recover and the bit predictor is operating at its information-theoretic limit for the contexts it sees.

### What went wrong with the Phase 13 prediction

Shannon entropy of a Zipfian distribution over an N=65,536 word alphabet (with a few high-mass head tokens and a long ~10 k tail) computes to roughly:

```
H ≈ 0.3 × 4 + 0.3 × 10 + 0.4 × 16 = 10.6 bits/token
```

— matching the measured 10 bits/hit almost exactly. The "5 bits/hit Shannon" figure in the Phase 13 entry was an artifact of weighting only the head; the tail's 16-bit-per-token contribution dominates. **Phase 13's Order-0 model was already at Shannon for its own distribution**, and Phase 14 is the equivalent operating point.

### Implications

- **Order-0 over a 65 k Zipfian alphabet has H ≈ 10 bits/token, full stop.** No reparameterization of the same Order-0 distribution (chunked, Fenwick, bit-level, PAQ-style mixing) can go below that floor.
- **The real architectural lever is the Order-N step.** To beat 10 bits/token requires conditional context — P(curr_token | prev_token) is structurally much sharper for English. Cmix's word model and related work show the Order-1 conditional cuts the Order-0 entropy by 30–40%. Target: ~6–7 bits/token, ~2.0 bpb on the panel — finally below xml-lz-cp.
- **Bit-level substrate is the right scaffolding for Order-1.** A dense Order-1 word model on 65 k × 65 k contexts would need ~17 GB. Sparse hashed bit-level with context `(prev_id, bit_pos, prefix)` packs the same expressive power into a 16 MiB `BitPredictor` — and a future Phase 15 can simply extend the existing `id_bit_ctx` to fold `prev_id` into the FNV mix. Phase 14's bit-level code lands as the substrate for that move.

### Status

xml-tok lands at 2.832 bpb (vs Phase 13's 2.826 — within rounding); xml-lz-cp remains v2's best at 2.608 bpb. Phase 14 is a null-result commit that clears a false hypothesis off the table and sets up Phase 15 (Order-1 word via bit-level conditional context) as the actual path past the Order-0 Shannon floor.

---

## 2026-05-12 16:30 — Phase 13: Word-Level Tokenization MVT (Negative Result, Diagnostic)

CDR proposed an architecturally distinct direction: tokenize Content into alternating word (`[a-zA-Z]+`) and separator (`[^a-zA-Z]+`) runs, build a u16-indexed dictionary on the fly via first-occurrence emission (no shipped dictionary, encoder/decoder grow it in lockstep), and run downstream modeling on the 16-bit token alphabet. CDR additionally flagged the case-folding move — store only the lowercase form in the dictionary and emit a 4-symbol case-pattern code so that "The"/"the"/"THE" share one slot instead of fragmenting three slots. Pre-measurement prediction: 2.1–2.3 bpb on the quick panel (vs xml-lz-cp's 2.506). The dimensional analysis was sound — word-level Shannon on enwik9 sits at ~1.5–1.8 bpb — so the question was whether a minimum-viable token codec could close ~30% of that gap.

It can't, at least not with Order-0 over the dictionary alphabet.

### Implementation

`src/tokenizer.rs` (run-length helpers + case classifier/applier; ~220 LOC) and `src/xml_tok.rs` (codec; ~1100 LOC) land Phase 13. The codec:

- Tokenizes Content into alternating word/separator runs.
- Dictionary capacity 65,536 entries; key is the lowercase byte sequence for words, raw bytes for separators.
- Token emission per token: 1-bit class (Order-1Ctx on prev class — forced alternation, near-zero cost after warm-up); 1-bit hit/OOV flag; on hit, dict_id via 2-byte chunked encoding (Order-0 for high byte, Order-1Ctx on high byte for low byte); on miss, 16-bit length + per-byte emission (Order-1Ctx over lowercase letters for words, over byte values for separators).
- Case pattern per word: 4-symbol (`AllLower`/`TitleCase`/`AllUpper`/`Mixed`). On hit, modeled by per-token case stats (Laplace +1 on the dictionary entry's case counts). On miss, modeled by a global Order-0. `Mixed` escapes to a per-letter uppercase bitmask.
- Per-token case stats from day one: each `DictEntry` carries `case_counts: [u32; 4]`; both encoder and decoder update them after every occurrence so the state stays in lockstep.
- Tag/Attr modes unchanged from xml-lz-cp (Phase 7).
- LZ disabled — the MVT isolates the tokenization gain.

7 roundtrip tests pass including a 4 MiB-warm + 8 KiB-measure window on real enwik8.

### Result (quick panel, 5 × 64 KiB)

| codec | mean bpb | Δ vs xml-lz-cp |
|---|---:|---:|
| xml-lz-cp (Phase 7 baseline) | 2.506 | — |
| xml-tok (Phase 13)           | **2.826** | **+0.320** |

Decomposition averaged across windows:

| component | bits/window | bpb contribution |
|---|---:|---:|
| token_hit_ac      | 150,485 | 2.30 |
| token_oov_ac      |  28,603 | 0.44 |
| case_ac           |   5,434 | 0.08 |
| token_class_ac    |      56 | 0.001 |
| tag_ac            |     461 | 0.007 |
| attr_ac           |     123 | 0.002 |
| ac_finish + framing + padding | ~40 | ~0.001 |
| **total**         | ~185,000 | **2.83** |

Two of CDR's design moves were validated cleanly even though the headline bpb regressed:

- **Forced alternation collapses class-bit cost.** `token_class_ac` is **56 bits per 64 KiB window** — essentially zero. The Order-1Ctx-on-previous-class model learns that Word follows Separator and vice versa to within Laplace's residual, exactly as predicted.
- **Case folding is cheap.** `case_ac` is **0.08 bpb** — well below the 0.6 bpb global-Order-0 case-entropy upper bound. Per-token case stats are responsible: "United" learns ~100% TitleCase quickly, "the" learns its real ~10% TitleCase / 90% AllLower split, and the per-entry Laplace prior collapses pattern entropy to near zero for words with consistent case.

### Why it loses

The dominant term is `token_hit_ac` at ~2.30 bpb. Average cost-per-hit is **8.7 bits**, against a Shannon target of ~4–5 bits for a Zipfian-distributed Order-0 over the dictionary alphabet. The bimodal nature of the bit distribution explains the gap:

- **Head (IDs ≈ 0–256)**: hot tokens like "the", "of", "and" get ~4 bits each — close to Shannon. The Order-0 on the high byte concentrates almost all mass on `high=0`; the conditional Order-1Ctx low-byte model picks up the within-page Zipfian.
- **Tail (IDs ≈ 256–65535)**: each (high≠0) bucket sees few observations because the dictionary is sparse beyond the head. The high-byte Order-0 has thin mass on `high=k` for k>0, and the conditional low-byte Order-1Ctx row for `high=k` is essentially uniform. Tail hits cost **12–16 bits each**.

Token counts per 64 KiB window are ~20,000, of which ~25–30% are tail hits. The tail's ~15-bit cost dominates the average: a head-heavy estimate of 5 bits/token would predict ~100,000 bits/window for the hit path; the tail drag bumps that to 150,000.

The structural problem is that **Order-0 over a 65,536-symbol alphabet, naively chunked as two Order-0 byte models, can't represent a heavy-tailed distribution efficiently**. The chunked decomposition forces each (high, low) row to be modeled independently, and tail rows starve.

### What would close the gap

Three architectural fixes, in increasing order of complexity:

1. **Frequency-binned ID assignment**. Reassign IDs by observed frequency periodically (in lockstep on encode and decode). Hot tokens settle into low IDs over time — concentrates mass on (high=0) more tightly than first-occurrence. Probably gives ~0.1 bpb.
2. **True Order-0 over the full alphabet via a Fenwick tree**. O(log N) per encode and observe; CDF queries don't materialize the full 65k table. This gets each hit token to its true `-log₂ P(id)` cost. Probably brings the codec to ~2.5 bpb — roughly at xml-lz-cp parity but without LZ.
3. **Order-1 word model (sparse, conditional on previous token ID)**. Captures "United" → "States", "of" → "the". This is where word-level codecs actually beat byte-level LZ77 — the per-token entropy drops from ~5 bits (Order-0 Shannon) to ~3–4 bits (Order-1 conditional). Target: 1.8–2.1 bpb.

### Lessons for the next attempt

- **Don't bench MVT-quality model architectures against polished baselines and conclude "tokenization doesn't work."** The architecture that gives word-level codecs their win is the *conditional* model. Order-0 is to word-level codecs what Order-0 byte was to byte-level codecs (3.4 bpb on enwik8) before LZ77 and Order-N arrived.
- **The 8.7-bits/hit figure is the load-bearing measurement.** If a Phase-14 Order-1-word codec gets the hit cost down to ~4 bits and OOV stays at ~1.4 bpb, the codec lands near 1.9 bpb — closer to Shannon than anything v2 has built.
- **Case folding lands as predicted on its own merits**: 0.08 bpb cost, ~40-50% effective dictionary expansion (the recon estimate). The DictEntry case-counts mechanism is reusable for any future tokenized codec.

Phase 13 lands as the negative-finding commit; tokenizer and codec stay in the tree as the substrate for Phase 14's Order-1 word model.

### Status

xml-lz-cp remains v2's best at 2.608 bpb on the full panel. Two negative findings in two days (wiki sub-mode, naive tokenization) but each one tightened the architectural map — Phase 12 told us where the sub-mode signal isn't worth carrying (dense Order-3 contexts); Phase 13 tells us where word-level codecs need conditional structure (Order-0 isn't enough). Both lessons feed Phase 14 directly.

---

## 2026-05-12 — Phase 12: Wiki Sub-Mode Literal Models (Negative Result)

CDR/Claude tested whether splitting the Content literal models (`letter`, `nonletter`) by an in-band wiki sub-mode (Plain / Link / Template) would close some of the v2 → v1 gap. Hypothesis came from a recon analyzer (`analyze-wiki`) that walked the first 16 MiB of enwik8 through the production XML classifier and found that **76% of all Content bytes live inside `[[..]]` internal-link spans** (template spans are another 4.5%, headings 0.3%, the leftover 19% is Plain prose). The intuition: the alphabet inside `[[..]]` is constrained — link targets are mostly letters with a small set of separators — so a Link-specific Order-3-letter model trained only on link-internal bytes should concentrate mass better than a single model trained on everything. Pre-measurement bpb prediction: 0.075–0.10 bpb on the panel.

### Recon: wiki sub-mode byte coverage on enwik8

`analyze-wiki --bytes 16777216` on `assets/enwik8`:

| layer | bytes | % of total | % of Content |
|---|---:|---:|---:|
| Content | 16,380,045 | 97.63 | — |
| TagStructure | 376,532 | 2.24 | — |
| AttrValue | 20,639 | 0.12 | — |
| Plain Content | 3,117,479 | 18.58 | 19.03 |
| Inside `[[..]]` | 12,471,947 | 74.34 | 76.14 |
| Inside `{{..}}` | 740,805 | 4.42 | 4.52 |
| Inside `==..==` | 49,814 | 0.30 | 0.30 |

The Link share is dominated by image links with thumb captions (`[[Image:Foo.jpg|thumb|right|300px|Caption...]]`), which average ~71 bytes per span across 175,651 spans. Bracket counts via `grep -oF '[[' | wc -l` confirm 175,652 `[[` openers and 175,659 `]]` closers — the FSM's depth tracking is correct.

### Implementation

New `WikiClassifier` (`src/wiki_classifier.rs`): a Moore-style sub-mode FSM run alongside `Classifier` inside Content. Three sub-modes (Plain / Link / Template); Heading dropped after the recon showed it at 0.3%, not worth line-start tracking. The FSM uses a one-byte memory (`prev`) rather than a one-byte lookahead so decode can advance the FSM without seeing the next byte — paired tokens (`[[`, `]]`, `{{`, `}}`) fire on the *second* byte of the pair. Depth-tracked nesting (templates nest heavily); Link takes priority over Template when both depths are active.

New codec `xml-wiki-lz-cp` (`src/xml_wiki_lz_cp.rs`): clone of Phase-7 xml-lz-cp with the Content letter and non-letter Order-1Ctx models replicated three times (one per sub-mode). Memory cost: letter ~30 MB × 3 = 90 MB, non-letter ~64 MB × 3 = 192 MB, total ~282 MB per window run. Within Hutter's 10 GB cap.

Two cuts:
- **Phase 12a**: split `lz_flag`, `is_letter`, `letter`, `nonletter` all by sub-mode.
- **Phase 12b**: only split `letter` and `nonletter`; keep `lz_flag` and `is_letter` shared, after 12a's decomp showed the LZ-flag split alone added ~566 bits/window with no offsetting gain.

### Results (quick panel, 5 × 64 KiB)

| codec | mean bpb | Δ vs xml-lz-cp |
|---|---:|---:|
| xml-lz-cp (Phase 7 baseline) | 2.506 | — |
| xml-wiki-lz-cp (Phase 12a, full split) | 2.517 | +0.011 |
| xml-wiki-lz-cp (Phase 12b, literal-only split) | 2.518 | +0.012 |

All 5 windows regress consistently — not panel noise. Phase 12b's failure to recover any of Phase 12a's loss means the regression is dominated by the literal-path split, not the LZ-flag split. The literal-path split itself is a clean negative.

### Diagnosis

The Order-3 letter context (`LETTER_NCTX = 52³ = 140,608` rows) already implicitly carries the wiki sub-mode signal: trigrams like `[[U` or `e]]` only appear in link-adjacent positions, so the prior-letter conditioning is already specialized to wiki structure where structure matters. Adding an explicit sub-mode tag is redundant *and* fragments the model — per-sub-mode training counts drop to roughly the sub-mode's byte share (19% / 76% / 4.5%), and Laplace smoothing pays the dilution cost on under-sampled contexts. The 4.5% template share is especially harmful: 1/3 of the table allocated to a slice that sees ~0.2 MiB of training in a 4 MiB warm prefix.

### Implications

- **Sub-mode-as-context-replication is the wrong mechanism for high-order models.** When the base context is already order-3, adding an orthogonal categorical dimension (sub-mode) hurts more than it helps because the dense context dilutes faster than the sub-mode signal sharpens.
- **A correctly-sized wiki specialization would attack different bytes**: the LOW-order positions where letter context is sparse — the first letter after `[[`, the first byte after `|`, the byte after a doubled `=`. These are the positions where the trigram context is "junk" (separators or unknown) but the sub-mode would be informative. The current Order-3 model wastes mass on these positions because its 3-letter history is dominated by non-letter framing.
- **The LZ matcher is already absorbing the bulk of wiki redundancy.** 65–70% of all archive bits in xml-lz-cp are `lz_match` bits, and the link/template framing repeats are captured there at long-match cost. The remaining literal residue is what's left after LZ peels off the easy gains, and that residue's distribution is closer to "novel content one-off" than "wiki-specific token". Sub-mode-conditioning a model fit to novel-content residue doesn't help.

### Status

Phase 12 lands on `v2` as a negative-finding commit. xml-lz-cp remains v2's best at 2.608 bpb on the full panel. The wiki sub-classifier and the analyzer stay in the tree — both will be reusable if a later phase wants to attack the sparse-context positions (e.g., a separate small-table predictor that fires only when letter context is empty AND sub-mode is non-Plain, mixed against the base predictor instead of replacing it).

---

## 2026-05-11 — v2 Clean-Slate Rewrite; Eval-First Discipline; Mode-Routed Codec Stack

CDR/Claude spent the day on a from-scratch v2 rewrite of the codec on a new branch (`v2` off `hutter`), motivated by the architectural critique at the end of the 2026-05-10 4M-scale-up entry: v1's "locally-greedy" iteration had reached a saturation point where mixer parameterization and neural scale weren't moving the needle, and the structural choices (byte-class routing, monolithic codec, ad-hoc bit accounting) had become the binding constraints. v2 inverts the build order: eval infrastructure and per-component bit decomposition first, then a mode classifier, then increasingly capable codecs layered on top. Eight build phases shipped in one session; the best operating point is **2.608 bpb on the enwik8 20×256KiB panel (xml-lz-cp)**, 0.55 bpb behind v1's deterministic floor of 2.056 — a real gap, but with the architecture and the audit table to know exactly where it sits.

### Method

v2 is committed on `v2` branch off `hutter`, single wipe commit followed by incremental phase commits. All v1 work (`src/`, weights, checkpoints) is intact on `hutter` for reference and rollback. v2 preserves `assets/` (the corpora), `JOURNAL.md`, `CLAUDE.md`/`AGENTS.md`, licenses, and `build.sh` (modified to drop the v1 `training`/`dev` feature plumbing). `Cargo.toml` rewritten with just `anyhow` and `clap` as runtime deps — no neural, no training, no candle. enwik8 (first 100 MB of enwik9) generated from the corpus for fast dev cycles; canonical Hutter measurement still happens on enwik9.

The meta-discipline is the bit-budget: every architectural decision is preceded by a written prediction of which decomposition cells move and by how much, then measured against that prediction. The bench panel emits a per-component decomposition CSV (`offset,measured_bytes,archive_bytes,bpb,bits_*`) for every codec. The eval harness asserts that the sum of named components equals exactly `8 * archive_bytes` — bits that don't sum to that are caught at panel time, not silently leaked.

### Phase-by-phase progression

| phase | commit | codec | mean bpb on enwik8 |
|------:|--------|-------|-------------------:|
| 0 | `67ed146` | null (sanity)              | 8.000 |
| 1 | `be97567` | classifier-stats (measure) | 8.000 |
| 2 | `ff8ec6d` | xml (templated dict)       | 7.830 |
| 3 | `2ce9229` | xml-ppm (AC + Order-1)     | 3.794 |
| 4 | `ec4710b` | xml-lz-ppm (LZ77)          | 2.707 |
| 5 | `e04a67c` | xml-lz-word (letter/non-letter split) | 2.642 |
| 6 | `267920e` | xml-lz-ord3 (Order-3/Order-2 dense) | 2.620 |
| 7 | `13e1096` | xml-lz-cp (cost-aware LZ) | **2.608** |
| 8 | `8104ecf` | xml-lz-ppmc (PPM-C order 3/2) | 2.617 |

The big steps are Phases 3 and 4 (the AC + Order-1 model and the LZ77 pre-pass). Subsequent phases tighten the literal residue and the match decision, with diminishing returns concentrated in the 0.05-0.10 bpb range per phase.

### Mode classifier and the empirical mode-distribution finding

A three-mode Moore FSM (`Content`, `TagStructure`, `AttrValue`) drives per-byte routing. Encoder and decoder run the same classifier deterministically — no signaling, zero bits of overhead. The `classifier-stats` codec attributes 8 bpb to per-mode buckets, giving an empirical mode-distribution on the panel before any compressor lands:

| mode          | % of corpus | extrapolated to 100 MB |
|---------------|------------:|-----------------------:|
| Content       | 97.51       | 97.5 MB |
| TagStructure  |  2.37       |  2.4 MB |
| AttrValue     |  0.13       |  0.1 MB |

This was a Phase-1 finding that immediately bounded Phase 2's ceiling: even a perfect XML codec saves at most ~2.5% of the corpus. The bit-budget discipline replaced an intuition-driven guess (where I'd have over-invested in XML schema fidelity) with a measured ceiling. Phase 2 shipped a 32-entry hardcoded dictionary that hit 100% on the panel (zero `tag_raw` entries) and saved 0.170 bpb global — almost exactly the predicted ceiling. No further XML codec work warranted.

### Two latent AC bugs surfaced under Phase 4

The LZ77 pre-pass exercised AC paths the short-codec tests didn't, and two long-standing latent bugs came out:

1. **`u32` cast truncation at initial state.** The encode/decode `(range * hi) / TOTAL` computation truncated `2^32` to `0` when `range = u32::MAX + 1` (the initial state) and `hi = TOTAL`. Result: `low + 0 - 1` underflow. Fixed by doing all arithmetic in `u64` and casting the result *after* the `-1`, when it's bounded by `u32::MAX`.
2. **Degenerate uniform CDFs for >8-bit alphabets.** With `TOTAL = 2^16`, a `2^22`-symbol uniform CDF can't give every symbol mass ≥ 1 — most consecutive entries are equal, violating the AC's strictly-increasing-CDF contract. Replaced offset bucket-relative-bits encoding with chunked emission: each 1-8 bit chunk is one AC symbol against a `2^chunk` uniform CDF. Cost stays at exactly `n_bits` for an `n`-bit value (equivalent to raw bit emission) but routed through the same AC stream so framing stays consistent.

Both were silent on small inputs and triggered immediately under the 4 MiB warm + 256 KiB measure regime. v1's matcher window and panel sizes happened to avoid them by accident.

### Architectural findings: where things ceiling out

The interesting phases are 6, 7, 8 — each gave a small win and a clear ceiling.

**Phase 6** (Order-3 letter + Order-2 non-letter dense): +0.022 bpb only, because LZ absorbs ~80% of bytes via matches and the literal residue skews toward non-letter bytes (punctuation, whitespace, digits — the bytes that don't repeat in long enough runs). Higher-order *letter* context helps less when most literals aren't letters. Dense Order-2 non-letter on a 256×256 table is the same model v1 also had — no new ground.

**Phase 7** (cost-aware LZ): +0.012 bpb. The mechanism works correctly — the bit-decomposition shows 0.126 bpb migrating from `lz_match` into `content_ac` and the net is positive. Net is small because the Order-3 letter model is the cap.

**Phase 8** (PPM-C without exclusions): swept the order configuration 2/1, 3/2, 4/3, 5/4. Best at 3/2 = 2.617 bpb. **Essentially tied with Phase 6 dense (2.620)**. Higher orders monotonically hurt — each escape pays `log2((T+K)/K)` bits, and without the exclusion mechanism (PPM-D's signature feature) lower-order CDFs still cover symbols handled by higher orders, wasting mass on impossible alternatives. The negative result is the architectural finding: **PPM-D's exclusion mechanism, not just multi-order escape, is what gives v1's PPM-D its wins**.

### Standing relative to v1

```
codec                                    mean bpb (enwik8 panel)
----------------------------------------+-----------------------
v1 LZ77 + cross-stream PPM-D + 4M neural             1.985
v1 deterministic floor (LZ77 + PPM-D)                2.056
v2 best (xml-lz-cp)                                  2.608   <- v2 operating
v2 Phase 6 dense Order-3/Order-2                     2.620
v2 Phase 8 PPM-C order 3/2                           2.617
v2 deterministic byte-stream baseline (null+xml)     7.830
```

v2 is **0.55 bpb behind v1's deterministic floor**. The gap is concentrated in two cells of the audit table:
- `content_ac` (0.875 bpb in xml-lz-cp): the literal residue after LZ. v1's PPM-D goes deeper here.
- `lz_match` (1.731 bpb): v1's combination of better Content prediction and probably slightly better LZ parsing keeps matches shorter.

The path to close most of this is well-defined: implement PPM-D's exclusion bookkeeping (~100 LOC on top of `src/ppm.rs`), raise non-letter order to 8 (v1's choice), add cost-DP optimal-parse LZ. Estimated cumulative gain: 0.2-0.35 bpb, putting v2 around 2.25-2.40 bpb — still some gap to v1's full deterministic floor, but with cleaner attribution and a proven path.

### What v2 demonstrates regardless of the absolute number

- **Bit-budget discipline works.** Phase 1's mode distribution correctly predicted Phase 2's ceiling. Phase 6's diminishing-return prediction matched the result. Phase 8's saturation was caught without surprise. Compare to v1's 4M neural scale-up, where 4× neural parameters bought ~0.06 bpb at the ensemble — a result that should have been predictable from the standalone-neural-vs-deterministic gap on the 1M baseline.
- **Per-component attribution gives a real audit table.** Every commit ships a decomposition that says where bits went and by how much each component moved. The "lost 0.012 bpb to a refactor" failure mode that haunts v1 doesn't happen here.
- **Latent correctness bugs surface under varied conditions.** Two AC bugs that v1's parameters avoided showed up in v2's bench. Both are now fixed.

### Next levers (if and when v2 resumes)

- **PPM-D with exclusions.** ~100 LOC on top of `src/ppm.rs`. Track an "excluded symbols" set across the fallback chain; rebuild CDFs minus excluded entries at each lower order. Estimated 0.10-0.20 bpb gain on top of current PPM-C.
- **Optimal-parse LZ with cost-DP.** Replace the greedy/lazy heuristic in `xml-lz-cp` with a proper DP over a small lookahead window. Estimated 0.05-0.10 bpb.
- **Order-8 non-letter PPM** (once exclusions work). Matches v1's parameters.
- **Different sub-mode classification within Content** — prose / wiki-markup / numeric / URL splits, with per-sub-mode codecs. Theoretical upside but each sub-mode is a small slice; the wins compound rather than multiply.

A neural arm is *not* on this list. The 2026-05-10 finding was that even a 4M neural barely moved the v1 ensemble (4× params → 0.06 bpb global). The bit budget for v2 should be spent on deterministic improvements until the deterministic floor is genuinely hit; only then does a neural residual model become worthwhile, and at that point it should target the specific residual rather than predicting raw bytes.

---

## 2026-05-10 → 2026-05-11 — 4M Neural Scale-Up; Mixer Saturation at 2 Arms; Order-2 as Third Arm

CDR/Claude scaled the neural arm 4× (1M → 4M, `N_LAYERS=2 → 8`, `D_MODEL=256` held, `CM_MULT=1` held) and ran a 24h training pass on enwik9. Best checkpoint landed at step 9056 / info-bpb 1.79 — a ~4% improvement on the neural arm in isolation. The improvement did not flow through the ensemble: 4M + 2-arm logistic at w=0.5 landed at 1.990 bpb (vs 1.994 at 1M), and three mixer-side experiments at 4M scale (single-global adaptive, per-class adaptive, fixed logistic at w=0.6) all converged to 1.988 ± 0.001. Adding Order-2 dense as a third arm via a 2-simplex adaptive log-linear mix landed at 1.985 — small but uniform across all five offsets. The dominant finding is architectural rather than numerical: **with two correlated predictors, the mixer's parameterization is saturated; predictor diversity is the binding constraint, not mixer cleverness**.

### 4M training run

`D_MODEL=256, N_LAYERS=8, CM_MULT=1`, identity tokenizer (vocab=256, no merges), seed 1, batch 32, seq 256, bptt_chunk 32, peak LR 3e-4. Tokenized 1 GB → 1B tokens in 7.3 s; trained at ~750 steps/h on M3 Pro Metal (4× slower per step than 1M, consistent with the ternary parameter count). RSS stable at 1.8 GB.

| step  | train loss (nats) | info-bpb |
|------:|------------------:|---------:|
|  1007 |             1.474 |     2.13 |
|  2014 |             1.395 |     2.01 |
|  3015 |             1.323 |     1.91 |
|  4021 |             1.281 |     1.85 |
|  5028 |             1.265 |     1.83 |
|  6035 |             1.252 |     1.81 |
|  7042 |             1.247 |     1.80 |
|  8049 |             1.245 |     1.80 |
|  **9056** |         **1.242** | **1.79** |
| 10064 |             1.249 |     1.80 |
| 11071 |             1.255 |     1.81 |
| 12077 |             1.260 |     1.82 |

Best at step 9056 / ~12h in. Plateaued and began the same alternating-stale oscillation seen on the 1M run, plateau-detector pattern matches. Killed at step 12077 / ~16h, archived as `BEST-step9056.ckpt` under `checkpoints/keep/4M-byte-2026-05-10/`.

Standalone neural info-bpb: 4M improves over 1M by 0.07 bpb (1.86 → 1.79). The 4× parameter cost bought ~4% in neural quality — at the lower end of the expected scaling-law gain at this scale.

### Mixer-side saturation at 2 arms (4M)

Three experiments at 4M scale, all 2-arm logistic over neural + routed-PPM, varying the mixer parameterization:

| mixer                                        | mean bpb |
|----------------------------------------------|---------:|
| fixed logistic w=0.5                         | 1.990    |
| fixed logistic w=0.6                         | 1.988    |
| fixed logistic w=0.7                         | 1.991    |
| single-global adaptive (SGD on cross-entropy)| 1.988    |
| per-class adaptive (3 scalars, one per `ByteClass`) | 1.988 |

A 2-arm log-linear mix has a single degree of freedom per context (the simplex point on the 1-simplex). The fixed-weight sweep already finds the optimum near w=0.6; per-class adaptive converges to a near-identical operating point per class. Adding parameters to the mixer side cannot move the result because the limiting factor is the predictor pair, not the weighting between them.

### 3-arm mix: Order-2 as third predictor

Wired Order-2 dense (`Order2Adaptive`) into the mix as a third arm under the `AdaptiveLogistic` mode. The mixer becomes two scalars per class — `(w_neural, w_routed)` with `w_order2 = 1 − w_neural − w_routed` — projected onto the 2-simplex by clamp-and-scale after each SGD step. CDFs combine log-linearly: `log p_mix[s] = w_n · log p_n[s] + w_r · log p_r[s] + w_o · log p_o[s] − log Z`. Gradient via standard cross-entropy:

`∂L/∂w_n = log p_o[t] − log p_n[t] + E_{p_mix}[log p_n − log p_o]`

with the symmetric form for w_r. Order-2 is advanced through every output byte (literal or match-copied) so its 2-byte context tracks the actual sequence; Upper-class positions fold both neural and Order-2 CDFs A-Z → a-z so all three inputs share the routed letter alphabet.

| ensemble                                     | mean bpb |
|----------------------------------------------|---------:|
| 4M + per-class adaptive (2-arm)              | 1.988    |
| 4M + per-class adaptive (3-arm: n+r+o2)      | **1.985** |

Per-offset, every offset improved by 0.001-0.006 (uniform). Statistically real but small. The 3-arm result confirms the predictor-diversity thesis directionally: adding a structurally different predictor moves the floor in a way that adding mixer parameters cannot. But the magnitude — 0.003 bpb — bounds the headroom of this particular third arm: Order-2 dense overlaps heavily with PPM-D order-5/8 (same alphabet, same input stream, only a shorter context). A genuinely orthogonal predictor (word-level, run-length over `\n`, number tokens) should buy more.

### Phase-2 standing post 4M + 3-arm

```
Predictor                                                       mean bpb     1 GB
order-2 adaptive (dense, 64 MiB)                                  2.771       346 MB
type-routed PPM, cross-stream context                             2.373       297 MB
LZ77 + routed (deterministic floor)                               2.056       257 MB
LZ77 + routed + 1M neural (logistic w=0.5)                        1.994       249 MB
LZ77 + routed + 4M neural (logistic w=0.6)                        1.988       249 MB
LZ77 + routed + 4M neural + Order-2 (adaptive 3-arm)              1.985       248 MB ← operating
target (Hutter 99%)                                               0.878       109 MB
```

The deterministic-floor-then-mix strategy continues to compound modestly. Total margin since the byte-stream order-2 baseline: −0.79 bpb / −98 MB. Still 139 MB over the 109 MB target — closing that needs a genuinely diverse new predictor type, not more of the same.

### Negative findings

- **4× neural parameters → 0.06 bpb at the ensemble level**, despite a 4% improvement at the standalone arm. Diminishing returns from neural scale-up against a stable deterministic ensemble are the expected pattern at this regime (the deterministic stack already captures much of what the neural would have predicted). Future scale-ups (8M, 16M) need a clear theory of why more neural should compound, not just more neural.
- **Mixer parameterization (1 → 3 → infinite-grained) does not move the floor** when the predictor pair is correlated. Per-context PAQ-style mixers don't help here; the next architectural lever is component diversity.
- **Order-2 dense is mostly redundant with PPM-D**. 0.003 bpb gain says it captures almost-nothing PPM-D doesn't. A useful third arm needs a different alphabet or a different conditioning structure.

### Open / next levers

- **Genuinely diverse third predictor**: position-in-word / capitalization-pattern / word-boundary-conditional model; RLE for `\n`-runs in markup; integer-token model for sequences of digits. Each ~50 LOC, no training cost.
- **8M neural** — same arch shape, `N_LAYERS=16`. Multi-day training, expected gain bounded by the 1M→4M trajectory (~0.05 bpb at the ensemble).
- **Optimal-parse LZ** — Storer-Szymanski DP. Classical 3-7% LZ gain. Has not been attempted yet.
- **Eval window scale-up** — 16 KiB windows are content-variance-noisy; 256 KiB would let us detect 0.001 bpb changes reliably. The 4M experiments are already running into the noise floor.

---

## 2026-05-09 → 2026-05-10 — LZ77 Pre-Pass; Cross-Stream PPM Context; 1M Neural Restart; Phase-2 Ensemble Under 2.0 bpb

CDR/Claude completed the deterministic-stack iteration through cross-stream PPM context plus an LZ77 pre-pass with bucketed offset/length encoding and lazy parsing, dropping the 5-offset bench mean to 2.056 bpb / 257 MB on 1 GB. Restarted neural training at 1M scale (the same arch as 2026-04-24) — landed train loss 1.289 nats / info-bpb 1.86 at step 13646, beating the historical 1M baseline by ~11% and giving a clean phase-2 starting point. Integrated the 1M neural into the LZ-routed codec as a logistic-mixed component at literal positions, with case-folding-aware Upper-class handling, hitting the **first sub-2.0-bpb operating point: 1.994 bpb / 249 MB on 1 GB**. The deterministic-floor-then-mix strategy compounded as designed: −0.93 bpb total margin over the byte-stream order-2 dense baseline (2.898), spread across LZ (~0.85), cross-stream PPM context (~0.07), and neural mixing (~0.06). Still 140 MB over the 109 MB Hutter target — closing that gap requires bigger neural (next training run) or a denser cmix-style predictor ensemble.

### Deterministic stack: cross-stream PPM context

Refactored `Ppm` to take caller-provided context bytes rather than maintaining its own per-arm history. `RoutedProbs` now keeps a single shared byte history (case-folded for letters) and feeds the same recent-bytes window to both letter and non-letter PPMs at every position. Letter-arm contexts now include surrounding markup (`</text>` → newline-ish priors); non-letter-arm contexts include surrounding letter context for predicting markup transitions. The bench measured the architectural change in isolation:

| Configuration                              | mean bpb | Δ |
|--------------------------------------------|---------:|---|
| type-routed PPM, letter-only context       |    2.649 | — |
| type-routed PPM, cross-stream context      |    2.373 | −0.276 |

Bigger lift than expected. Per-class predictors were leaving real signal on the table by ignoring the surrounding stream. Routed-alone now at 2.373 / 297 MB.

A subtle correctness bug surfaced after the LZ layer landed: matched bytes in the LZ flow weren't being pushed into the routed predictors' shared history (only literals were), so the next literal's PPM context conditioned on stale pre-match bytes instead of the actual recent output. Encoder and decoder made the same mistake symmetrically (so roundtrip worked), but PPM was wrong. Fixed via `RoutedProbs::note_match_bytes`, called by the LZ wrapper after each match emission. Bench delta: −0.045 bpb.

### LZ77 pre-pass: window, encoding, parse

`src/lz77.rs`, ~600 LOC. Hash-chain matcher (`hash[3-byte-prefix] → most-recent pos`, `prev[pos & WINDOW_MASK] → previous pos with same hash`), chain bound 32 — zlib's "fast" preset. 4 MiB sliding window covers the bench's 4 MiB pre-warm.

Three encoding refinements layered in sequence (each measurement isolating that lever):

| Configuration | mean bpb | Δ vs prev |
|---|---:|---:|
| LZ77 raw 33-bit match (`MIN_MATCH=16`)        | 2.303 | −0.346 vs no-LZ |
| LZ77 bucketed offset + Order-0 length (`MIN_MATCH=6`) | 2.194 | −0.109 |
| LZ77 + lazy parse (peek `pos+1`, defer if longer match starts there) | 2.152 | −0.042 |
| LZ77 + cross-stream PPM context fix          | 2.077 | −0.075 |
| LZ77 + 4 MiB pre-warm (vs 1 MiB)              | 2.056 | −0.021 |

Match cost dropped from 33 bits raw to ~16-19 bits bucketed/adaptive — `MIN_MATCH=6` becomes net positive. Lazy parse contributed the canonical 2% zlib refinement. The 1 MiB → 4 MiB pre-warm bump benefited PPM convergence more than LZ-window depth — at 1 MiB the LZ window was already near saturation for the 16 KiB measurement.

Per-offset, the XML-heavy windows (500M, 900M) bottom out near 0.82 bpb (essentially at Hutter SOTA at 16 KiB scale) — LZ captures the long-range exact repetition that PPM can't see at any practical order. Prose-heavy offsets stay in the 2.6-3.0 band where LZ contributes less.

### 1M neural restart: trajectory and kill

Restarted RWKV training at 1M (`D_MODEL=256, N_LAYERS=2`) byte-level, identity tokenizer (vocab=256, no merges), `--seed 1`, defaults otherwise (batch=32, seq=256, bptt_chunk=32, decay_horizon=1M, peak LR 3e-4). Tokenization 1 GB → 1B tokens in 7.3 s; training proceeded at ~3360 steps/h on M3 Pro Metal, RSS stable.

Six checkpoints landed before kill:

| step  | train loss (nats) | info-bpb | Δ |
|------:|------------------:|---------:|---|
|  6885 |            1.4808 |     2.14 | — |
| 13646 |        **1.2888** | **1.86** | −0.192 |
| 20488 |            1.3877 |     2.00 | +0.099 |
| 27502 |            1.3074 |     1.89 | −0.080 |
| 34512 |            1.3685 |     1.97 | +0.061 |
| 41523 |            1.3054 |     1.88 | −0.063 |

Best at step 13646 (4 h in). The next 28k steps (9 h) oscillated in the 1.29-1.39 band — same alternating-stale pattern that defeats the plateau detector (3-consecutive-stale rule). Killed at step 46150 / 13h 20min, archived as `BEST-step13646.ckpt` under `checkpoints/keep/1M-byte-2026-05-09/`.

Compared to the 2026-04-24 1M baseline (1.45 nats / 2.09 bpb at step ~14k): **this run lands ~11% lower info-bpb at the same step count**. Plausible reason: changes since 2026-04-24 to AdamW state initialization, the cross-stream-aware tokenization being a no-op identity here, and possibly seed luck. The new lower number is the spec for the ensemble.

### Phase-2 integration: neural mixed at literal positions

Refactored `RoutedProbs` into granular methods (`encode_class` / `cdf_for_class` / `encode_byte_value` and the decode mirrors) so the LZ wrapper can interpose a mixed CDF between class encoding and byte encoding without duplicating routed update logic. `LzRouted::with_neural((Weights, mix_weight, MixMode))` wires in a `ByteTransformer` that advances on every output byte (literal or match-copied) so neural state tracks the actual byte sequence rather than just literals.

Two mixing schemes implemented:

- **Linear**: `count_mix[i] = round(w · count_a[i] + (1−w) · count_b[i])`. Standard count-space arithmetic mean.
- **Logistic** (PAQ-canonical): `p_mix[s] ∝ p_a[s]^w · p_b[s]^(1−w)`. Geometric mean — sharpens where both predictors agree, downweights where they disagree.

Mix-weight sweeps:

| mix mode  | w   | mean bpb |
|-----------|-----|---------:|
| linear    | 0.5 | 2.026 |
| linear    | 0.7 | 2.028 |
| logistic  | 0.3 | 2.009 |
| logistic  | 0.5 | **1.994** |
| logistic  | 0.7 | 1.997 |

Linear was nearly flat across weights — a classical signal that the two predictors have correlated errors, so linear weighted-averaging dilutes both rather than sharpening the combined prediction. Logistic mix recovered ~0.03 bpb at the same w=0.5; the curve under logistic is shallow with a clear minimum at 0.5.

### Upper-class neural fold

The case-folding scheme stores class={Lower, Upper, NonLetter} on a 3-class type stream and encodes the *lowercased* byte through the letter PPM. The neural was trained on raw bytes, so its CDF predicts both `'A'` and `'a'` as separate symbols. To mix the two coherently at Upper positions, the neural CDF needs folding: A-Z mass moves to the corresponding a-z slot before mixing. Implemented as `fold_neural_cdf_for_upper`. The previous version skipped neural mixing entirely at Upper (4% of bytes); the fold extends mixing to those positions.

### Phase-2 standing

```
Predictor                                                         mean bpb     1 GB
order-2 adaptive (dense, 64 MiB)                                    2.771       346 MB
type-routed PPM, letter-only context                                2.649       331 MB
type-routed PPM, cross-stream context                               2.373       297 MB
LZ77 + routed (deterministic floor)                                 2.056       257 MB
LZ77 + routed + 1M neural (logistic w=0.5)                          1.994       249 MB ← operating
target (Hutter 99%)                                                 0.878       109 MB
```

Total margin from the byte-stream order-2 baseline: **−0.78 bpb / −97 MB**. Phase-2 has covered ~50% of the gap from byte-stream baseline to Hutter SOTA.

Inference rate at the 1M neural arm alone (codec test on a 16 KiB window): ~9.6 h/GB. Adding ensemble overhead probably puts the full submission at ~12-15 h/GB encode + decode, comfortably inside the 64 h Hutter budget at 1M scale.

### Open / next levers

- **Larger neural**. The 1M-vs-routed gap is small relative to the standalone-neural gap to deterministic, suggesting the neural is the limiting component. 4M or 8M restart with the same arch shape (just bumping `N_LAYERS`) is the canonical next move. Each step is a multi-day training run.
- **More predictors mixed**. Cmix uses 30+ submodels including PPM-N at multiple orders, dictionary lookups, and small specialized neurals. Adding 2-3 more components to the mix (e.g. PPM-2 dense, Markov-with-stretch contexts) and learning the mixing weights adaptively is the standard next step in PAQ/cmix evolution.
- **Optimal-parse LZ**. Storer-Szymanski or full DP optimal parse instead of greedy/lazy. Classical 3-7% gain.
- **Adaptive mix weight**. PAQ's neural-network mixer learns per-context weights instead of a fixed scalar. Bigger code change but sometimes worth 5-10%.
- **Eval window scale-up**. The 16 KiB measurement window is a content-variance-dominated estimator. For comparing ensembles at this stage, bumping to 256 KiB or 1 MiB measurement windows would tighten the per-offset spread enough to detect 0.005 bpb changes reliably.

---

## 2026-05-08 → 2026-05-09 — 8M Token Run Killed; Deterministic-First Pivot; Type-Routed PPM Beats Byte-Stream Order-2

CDR/Claude killed the 8M token-level RWKV training run after ~26 hours and ~4400 steps when the trajectory clearly tracked byte-level 8M's plateau (1.477 info-bpb) rather than meaningfully beating it as the BPE pivot expected. Pivoted to a cmix-style deterministic-first ensemble: fixed 5-offset codec eval panel, order-N adaptive ladder as baselines, type-routed codec (lowercase / uppercase / non-letter), and PPM-D in both content arms. Letter-arm at order-5 and non-letter-arm at order-8 brought routed bpb to 2.649, beating the byte-stream order-2 dense baseline (2.898) by 0.249 — the first measurement showing that routing pays. Plan-of-record continues with cross-stream PPM context, an LZ77 pre-pass, or PPM-Markov mixing as the next leverage points; the neural arm comes back only after the deterministic floor stops moving.

### 8M token-level run: trajectory and kill

Started 2026-05-08 at the configuration left in `arch.rs` from the 2026-05-07 pivot (`N_LAYERS=16, D_MODEL=256, VOCAB=4096`). Tokenization through enwik9 took 51 s, producing 407.4M tokens at 2.455 bytes/token. Training proceeded at ~310 steps/hour on M3 Pro Metal with `decay_horizon=1M` and peak LR 3e-4.

Trajectory across the 14 checkpoints landed before kill:

| step | train loss (nats/tok) | info-bpb | codec bpb (1 window) |
|-----:|----------------------:|---------:|---------------------:|
|  312 | 4.668 | 2.74 | 3.08 |
| 1259 | 3.495 | 2.05 | 1.23 |
| 1573 | 3.159 | 1.86 | 2.55 |
| 2833 | 2.993 | 1.76 | 1.49 |
| 3149 (best) | **2.618** | **1.54** | 0.87 |
| 4411 (last)| 2.771 | 1.63 | 1.83 |

Conversion: nats/tok ÷ ln(2) ÷ 2.455 bytes/tok = bits/byte. At step 3149, info-bpb 1.54 — slightly worse than the 8M byte-level plateau (1.477). The slope was still negative but visibly bending: −0.85 nats/tok over steps 1259→3149 (1890 steps), only −0.05 over steps 3149→4411 (1262 steps, with two regressions). Linear extrapolation suggested a plateau in the 2.3–2.5 nats/tok / 1.35–1.47 bpb band — matching byte-level rather than meaningfully beating it. The BPE pivot's stated goal had been to raise the loss ceiling, not just save wall-time; on this trajectory it was saving wall-time only (~40 h projected vs ~108 h byte-level).

Killed before convergence on slope-shape evidence. Wall-time projection at this trajectory: 1.4 bpb × 1 GB = 175 MB final size, ~60% over the 109 MB target, with no architectural path visible to closing that gap by depth alone. Codec single-window measurement (range 0.87–2.71 across windows on essentially the same model) confirmed the 5-offset eval panel deferred since 2026-04-23 was now blocking iteration.

### Pivot rationale: deterministic-first ensemble

CDR proposed a layered recipe: (1) generate a character type map (lowercase / uppercase / non-letter); (2) AC-encode the type map; (3) AC-encode the symbol stream; (4) neural-encode the letter stream. Initial discussion surfaced the central concern: independent per-class predictors lose cross-stream mutual information. A monolithic byte predictor sees that `>` follows `</text` and `\n` follows `\n  `; a strictly separated letter-only predictor cannot. With matched orders the cross-stream loss dominates the alphabet-shrinking win; routing only pays when the per-class predictor can run at higher order than the joint predictor would.

CDR pointed out that recommending the 8M token plateau measurement still relied on BPE — itself a context-preserving pre-pass. The right axis isn't "pre-pass yes/no" but "pre-pass that preserves cross-stream context vs destroys it." BPE preserves; LZ77 preserves; the as-stated separation destroys. Reformulated proposal: type-routed mixing where each predictor still conditions on full mixed history (or as much of it as the architecture allows), with the type map selecting the active predictor at each position. That is the cmix recipe.

Plan-of-record:

1. Iterate the deterministic stack first — minutes-to-hours per experiment vs days-to-weeks for neural runs. Establishes a hard floor before the neural arm has to deliver anything specific.
2. Reintroduce the neural arm later as a mixed component on the letter stream, restarting at 1M.
3. Multiple small specialized neurals stay on the table (e.g. letter-positions-after-whitespace, in-tag-content) but defer until the deterministic floor is squeezed dry.

The existing 8M model effort is preserved as a baseline number for what monolithic neural alone achieves and as a drop-in candidate for stage-2 mixing.

### Eval panel: fixed 5-offset codec measurement

`src/bench.rs`: offsets 100/300/500/700/900 MB into enwik9, 16 KiB measurement window each, 1 MiB pre-warm before the measured slice. Pre-warm matters: an adaptive predictor's first hundreds of bytes are paid at near-uniform cost as contexts learn, and at order-2 dense or higher, cold-start dominates the 16 KiB measurement. With pre-warm, order-1 measurement dropped from 4.806 to 3.703 bpb — cold-start was hiding ~1.1 bpb of real signal. Pre-warm is purely a bench-time concern: the real submission encodes the full 1 GB and amortizes cold-start for free.

Roundtrip verification at every offset: independent predictor instances on encode and decode, identical pre-warm. Cheap insurance against silent encode/decode desync as predictors get more elaborate.

Bench output is deterministic — two runs at the same offsets produce bit-identical bpb. Per-offset spread reflects real content heterogeneity in enwik9, not measurement noise. The `lzr bench` subcommand surfaces it as a stable iteration target, no model load required.

### Order-N adaptive ladder

Sequenced order-0 → order-1 → order-2 dense as warm-up baselines, all in `src/predict.rs`:

| Predictor | Mean bpb (5-offset) | Δ vs prev | 1 GB projection |
|-----------|--------------------:|----------:|----------------:|
| Order-0 adaptive (Laplace +1, online halving) | 5.042 | — | 630 MB |
| Order-1 adaptive (per-prev-byte counts) | 3.703 | −1.34 | 463 MB |
| Order-2 adaptive (dense byte-pair, 64 MiB state) | **2.898** | −0.81 | **362 MB** |

Order-2 dense is the byte-stream baseline for everything that follows. Order-3 dense is infeasible (16 GiB) and would also see worsening returns from the long-tail-uniform problem on a 256-byte alphabet — that's where PPM with sparse storage and escape becomes mandatory. Per-offset spread widens at order-2 (1.34 bpb range, 2.08–3.42), content heterogeneity dominating once the predictor is good enough.

Online halving threshold at TOTAL/2 (32,768 counts) keeps total bounded so the rescale to TOTAL never collapses a slot to zero mass.

### AC slice refactor

`src/ac.rs`: `Encoder::encode` and `Decoder::decode` now take `&[u32]` slices rather than `&[u32; CDF_LEN]` fixed arrays, with implied vocabulary size = `cdf.len() - 1`. The byte codec continues to pass full 257-entry slices via array-to-slice coercion (no callsite changes); PPM emits CDFs over a non-fixed alphabet of (escape ∪ seen-not-excluded) per position. Slice-based AC unblocks variable-vocab predictors generally, not just PPM. Cost: zero in the common path — the existing 257-entry callers monomorphize identically to the array version.

### Type-routed codec: routing infrastructure

`src/routed.rs`: three-class classification (`predict::ByteClass`) — `Lower` (a-z), `Upper` (A-Z), `NonLetter` (everything else). Uppercase positions are folded to lowercase before encoding via the letter arm; the case bit travels in the type map, the letter predictor only ever sees lowercase bytes.

For each input byte: encode the class via the type predictor, then encode the byte (or its lowercased form) via the per-class content predictor. Single shared AC stream, two emissions per input byte (class then byte). Decoder mirrors exactly: decode class, dispatch to per-class predictor, decode byte, optionally re-uppercase. Each sub-predictor maintains its own adaptive state independently — there is no cross-stream context at this stage. (Cross-stream conditioning is on the deferred list as the next architectural fork.)

### Routing with placeholder predictors: regression at matched orders

First measurement, order-1 type predictor + order-2 dense in both content arms:

| Predictor | Mean bpb |
|-----------|---------:|
| Order-2 adaptive (joint byte stream) | 2.898 |
| Type-routed (O1 type / O2 letter / O2 non-letter) | 3.694 |

Routing alone at matched orders is **+0.80 bpb worse** than the joint predictor. As predicted: each per-class predictor sees a strict subset of the context the joint predictor sees, and at matched depth the cross-stream context loss dominates the alphabet-shrinking win. Routing wins when the per-class predictor can run at higher order than the joint predictor would.

Order-1 type predictor (vs order-0): −0.156 bpb on the routing stack. The 3-symbol type stream has heavy run-length structure (long lowercase runs inside words); order-1 captures it cheaply.

Per-offset spread on routed (0.87 bpb) is tighter than joint order-2 dense (1.34 bpb) — per-class predictors are less affected by content heterogeneity. Useful for ensemble stability later.

### PPM-D: single-event design

`src/ppm.rs`, ~270 LOC. Order-N PPM with PPM-D escape weighting (escape ≈ n_distinct / (2T + n_distinct)) and full exclusion. Built as a single-event predictor (one AC emission per byte) rather than the multi-event "encode escape, try lower order, ..." formulation: the single 256-symbol pmf is mathematically equivalent and lets PPM drop straight into the existing `ProbSource` slot.

Algorithm: walk from the longest available context down to order 0 with a growing excluded set; at each visited context, allocate `remaining × (2c / (2T+n))` of the pmf to seen-and-not-excluded symbols and `remaining × (n / (2T+n))` to "escape further" (which becomes the new `remaining`). Order-(-1) fallback distributes leftover `remaining` uniformly over symbols still not excluded. Storage is sparse: `HashMap<Vec<u8>, ContextNode>` per order, only visited contexts cost anything. Per-node ~80–120 B; for order-5 letter PPM on the full 670 MB letter stream, distinct 5-grams are ~1–5M out of 12M possible, total working set ~80–400 MB.

Subtle pmf-conservation case found during testing: when order-0 accumulated mass on all 256 symbols (typical for the cycling-bytes test, also relevant for small effective alphabets like the non-letter sub-stream), the order-(-1) fallback has zero usable symbols but `remaining > 0`. Mathematically the last order's escape coefficient should have been 0. Fix: fold `remaining` back proportionally over already-allocated mass — equivalent and pmf-conserving. Without it the residual at CDF construction overshoots and indexes off the end.

### PPM-D in the symbol arm

PPM-D order-4 swap-in for the non-letter `Order2Adaptive`:

| Predictor | Mean bpb |
|-----------|---------:|
| Joint order-2 dense | 2.898 |
| Routed (O1 / O2 letter / **O2** non-letter) | 3.694 |
| Routed (O1 / O2 letter / **PPM-D-4** non-letter) | 3.507 |

−0.187 from PPM-4 alone on the non-letter arm. Win concentrated at XML-heavy 500M (−0.32) and 900M (−0.25); smaller (−0.08 to −0.17) at prose-heavy 100M / 300M / 700M. Routing still 0.61 bpb worse than the joint baseline because the letter arm at order-2 is the dominant cost (~67% of bytes × ~2 bits each = 1.34 of the 3.51 mean).

### PPM-D in the letter arm: routing flips to a win

PPM-D order-5 swap-in for the letter `Order2Adaptive`:

| Predictor | Mean bpb |
|-----------|---------:|
| Joint order-2 dense | 2.898 |
| Routed (O1 / O2 letter / PPM-4 non-letter) | 3.507 |
| Routed (O1 / **PPM-5** letter / PPM-4 non-letter) | **2.697** |

−0.81 bpb from the letter-arm swap. **Routing now beats joint order-2 dense by 0.20 bpb.** Per-offset wins concentrated where content is sparse-letter-friendly: 500M went 2.79 → 1.73 (−1.06), 900M 2.99 → 1.82 (−1.16). Prose-dominant offsets saw smaller (−0.55 to −0.70) but consistent wins.

The cmix-mechanism prediction held empirically: routing pays once the per-class predictor outranks the joint predictor's effective context. Letter-only PPM-5 on a 26-effective alphabet lands at the classical English-text PPM sweet spot the literature has documented for decades.

### Order sweep on the routed stack

Letter and non-letter arm predictors don't share state, so optimal orders are independent. Swept each on the 5-offset panel:

| Letter | Non-letter | Mean bpb |
|-------:|-----------:|---------:|
| 5 | 4 | 2.697 |
| 5 | 5 | 2.681 |
| 5 | 6 | 2.669 |
| 5 | 7 | 2.654 |
| 5 | 8 | **2.649** ← elbow |
| 6 | 4 | 2.705 |
| 6 | 8 | 2.656 |

Letter arm: 4→5 was a huge −0.79 jump; 5→6 regresses by ~0.008 at every non-letter order tested. On a 26-effective alphabet, order-6 contexts are sparse enough that escape cost dominates the within-context tightening. Operating point: **letter PPM order 5**.

Non-letter arm: monotonically improving with elbow at 7–8. The non-letter alphabet is small effective (~30 byte values) so very high orders still help — order-6 contexts there are well-populated. Marginal gains shrink to ~0.005 by 7→8. Operating point: **non-letter PPM order 8**.

Final: type-routed (order-1 type / PPM-D-5 letter / PPM-D-8 non-letter), **2.649 bpb mean / 331 MB on 1 GB**. Beats joint order-2 dense by 0.249 bpb.

### Standing relative to budget

| Predictor | Mean bpb | 1 GB | Gap to 109 MB |
|-----------|---------:|-----:|--------------:|
| Hutter 99% target | 0.878 | 109 MB | — |
| Phase-1 operating point | 2.649 | 331 MB | +222 MB |
| Joint order-2 dense baseline | 2.898 | 362 MB | +253 MB |
| 8M neural alone (byte-level plateau) | 1.477 | 185 MB | +76 MB |

The deterministic stack has covered the joint-order-2 → ~halfway-to-Hutter-SOTA distance, but absolute 1 GB projection is still 3.0× over budget. Further gains need structural moves (cross-stream context, LZ pre-pass, mixing), not parameter tuning — the order sweep has clearly plateaued.

### Open / next levers

- **Cross-stream PPM context**: letter arm conditions on previous letters only, missing signals like "byte after `</text>` is most likely a newline." Hash-based contexts over the full byte stream would let the letter arm see surrounding markup. ~30 LOC PPM addition, ~100–200 MB sparse memory. Top expected leverage.
- **LZ77 pre-pass**: PPM at any order can't see paragraph-level repetition (1 MB+ apart). Wikipedia boilerplate, redirect pages, infobox templates are duplicated verbatim across the corpus. Emitting (offset, length, literal) before PPM should attack this directly without violating cross-stream context preservation.
- **PPM × dense-Markov mixing**: PAQ/cmix-style logistic mixing of two predictors is straightforward. Different model classes catch different patterns; usually nets 5–10% off the PPM-alone bpb.
- **Neural arm reintroduction**: deferred until the deterministic floor stops moving (per the build-order plan). 1M baseline restart on whatever input form survives the deterministic stack — not necessarily BPE-tokenized.
- **Eval-panel window size**: 16 KiB windows with 1 MiB pre-warm gives ±0.05 bpb iteration sensitivity but 1.34 bpb spread between offsets at order-2 dense. If structural moves produce smaller incremental wins, consider widening the window to amortize content variance further.

---

## 2026-05-07 — BPE Tokenization Pivot; 1M Restart On Tokens; Embedding Ternarization

CDR/Claude pivoted away from depth-only byte-level RWKV scaling. The 8M plateau and the inference-budget tension at 12M (both documented in the prior entry, 2026-04-25 → 2026-05-03) made wall-time the binding constraint, not loss-vs-parameters. The new direction: BPE-tokenize the input to compress sequence length, restart at 1M to (re-)establish the small-model baseline on the new alphabet, scale from there. This entry covers the BPE measurement, the case-folding side question, the wholesale codec/AC/model rewrite to a 4096-token alphabet, the embedding ternarization that follows from it, and the decision to restart at 1M on tokens.

### BPE measurement on enwik9

Trained byte-level BPE on enwik9 at three target vocab sizes, measuring `bytes/token`. Pretokenization splits on ASCII whitespace boundaries — simpler than GPT-2's regex (no Unicode dependency) but sets a hard upper bound on bytes/token equal to the average pretoken length (`1.0e9 / 258.7M ≈ 3.87`). New `lzr bpe` subcommand emits both a JSON merge list and a compact binary blob (`assets/tokenizer.bin`, schema-versioned `u32` header + `u16` merge pairs, ~15 KB at vocab 4096) for runtime loading.

| vocab | merges | final tokens | bytes/token | % of ceiling | wall-time at 8M scale |
|------:|-------:|-------------:|------------:|-------------:|----------------------:|
|   256 |      0 |        1.0 B |        1.00 |          26% |             108 h     |
| 2,048 |  1,792 |       452 M  |        2.21 |          57% |              49 h     |
| 4,096 |  3,840 |       407 M  |    **2.46** |          63% |              44 h     |
| 8,192 |  7,936 |       374 M  |        2.68 |          69% |              40 h     |

Marginal returns: doubling 2K → 4K buys +11% bytes/token; doubling 4K → 8K buys +9%, clearly past the elbow. Top-30 learned tokens at 4K are sane for English-plus-XML: `"  "` (double space, 16.6M occurrences), `"th"`, `"er"`, `"in"`, `"]]"` and `"[["` (Wikipedia link delimiters at ranks 6 and 7), `"the"`, `"and"`, `"of"`, `"''"` (bold-italic), `"    "` (XML indent). 4K is the chosen knee — at the 8M-class scale the wall-time projection drops from ~108 h to ~44 h, well inside the 64 h Hutter budget; at 12M from ~175 h to ~71 h, ~1.1× over which is much closer than the 2.7× we had at byte-level.

### Side question: lowercase + uppercase-bitmap preprocessing

CDR proposed factoring case out of the input — lowercase before tokenization, AC-encode an uppercase bitmap separately, like cmix and similar context-mixing compressors do. Worked through the cost-vs-benefit math: the bitmap entropy at ~6% uppercase among the ~85% alpha bytes is ~35 MB unconditioned, ~2–5 MB after AC with context (sentence-start, after `[[`, after `\n`), against a token-stream savings of maybe 3.5–7 MB from tighter BPE merges (`The`/`the` collapse). Roughly cost ≈ savings, possibly slight net positive, not the slam-dunk it looks like. The deeper objection: a strong neural model can already learn case from context, so the bitmap is mostly redundant with what the model knows. Deferred — wire BPE first, measure, then A/B case folding as a polish step.

### Architectural reconsidering: stay with RWKV

The token-level pivot raised the natural question of whether RWKV is still the right architecture. Linear-time recurrence remains the right family: 407M tokens is still way too many for quadratic attention at inference (~256× slower per token than RWKV at seq=256). Within linear-time, Mamba/selective-SSM has been beating RWKV at matched param counts in 2024–2025 papers, and would be the right thing to try if the 8M plateau (1.024 nats) is architectural rather than capacity-limited. But switching means losing the BitNet kernel work, the training pipeline, the recently-debugged Metal autorelease path, and 4–6 weeks of re-validation. Tokenization is a much cheaper unblock for the same problem. Decision: stay with RWKV through the BPE integration; revisit only if tokenized end-to-end numbers suggest the architecture is the limiter.

### Codec/AC/model rewrite: VOCAB 256 → 4096, symbol type u8 → u16

Generalizing the byte-hardcoded codec stack to a 4096-token alphabet touched every file in the inference path:

- `arch.rs`: `VOCAB = 4096`, new `CDF_LEN = VOCAB + 1` const for AC's CDF arrays.
- `ac.rs`: encoder/decoder symbol type `u8 → u16`; `[u32; 257]` CDFs become `[u32; CDF_LEN]`. The 16-bit `TOTAL = 65,536` mass stays adequate at 4K — every symbol still gets ≥1 count, leaving 61,440 for proportional distribution; minimum representable probability is 1/65,536 ≈ 1.5e-5, fine for tail tokens.
- `probs.rs`: softmax-to-CDF over `[f32; VOCAB]` (already templated; only the return type changed).
- `codec.rs`: `encode_bytes` now tokenizes input before AC-encoding the token stream; `decode_bytes` runs AC until detokenized output reaches the header's byte length, using a pre-built `Vec<Vec<u8>>` token-bytes table. Header format unchanged (4B magic + 8B byte length).
- `model.rs`: `step(token: Token)` instead of `step(token: u8)`; logits and embedding shapes auto-adapt via `VOCAB`.
- `train.rs`: tokenizes the entire training corpus once at startup (~1 GB → ~407M `u16` tokens, ~814 MB), drops the byte buffer, samples token windows for batches.
- `tokenizer.rs` (new): loads the compact binary, applies merges greedily by lowest rank within whitespace-bounded chunks, decodes via direct vocab-table lookup.

Build green at this point: 33 tests pass on submission, 36 on dev, 33 on release. End-to-end with a real trained model is unverified — unit tests cover AC + tokenizer + codec roundtrip with random/zero weights, but actual training and codec performance need a run.

### Embedding ternarization

VOCAB=4096 makes the f32 token embedding 4 MB — by far the largest single tensor in the model. Ternarizing it (per-row absmean scale, same I2_S packing as the per-layer matrices) drops it to 272 KB on disk (256 KB packed + 16 KB scales). The embedding is weight-tied to the LM head, so the tied output projection becomes a `(VOCAB × D_MODEL)` ternary matvec — the same kernel shape the per-layer code already runs, just wider. New `bitnet::dequantize_i2s_row` function unpacks one I2_S-packed row × scale to f32 for the input embedding lookup; the LM head reuses `matvec_prequant`. Compile-time `lut_supports(VOCAB, D_MODEL)` assertion in `weights.rs` guards the kernel-shape compatibility.

| Component                    | Pre-ternarization | Post-ternarization |
|------------------------------|------------------:|-------------------:|
| Token embedding (f32)        |              4 MB |              0     |
| Token embedding (packed)     |                 0 |             272 KB |
| Layer matrices (×2)          |           ~250 KB |            ~250 KB |
| Norms + mix vectors          |             ~3 KB |              ~3 KB |
| **`PACKED_WEIGHTS_LEN`**     |       **~4.25 MB** |        **~530 KB** |

Net L(C) saving ~3.7 MB. For reference: this brings the 1M token-level model to ~530 KB on disk, comparable to the byte-level 1M model's footprint (~256 KB f32 emb + ~250 KB layers ≈ 500 KB total) but with 16× the vocabulary. The tied LM head adds ~1.0M MACs to per-step inference (matching the ~920K MACs across two layers' matvecs), but uses the same LUT-accelerated kernel — wall-time projects to roughly the byte-level 1M baseline plus the 2.45× saving from sequence shortening.

Training-side: the embedding `Var` stays f32 in candle; quantization happens at checkpoint write time (post-hoc, not QAT). Train/inference numerics will diverge slightly. If quality drops, the next move is fake-quant on the embedding inside the training forward pass — same STE pattern `BitLinear::forward` already uses for the per-layer matrices.

### Decision: 1M restart on tokens (`D_MODEL = 256, N_LAYERS = 2`)

`src/arch.rs` rolled all the way back to `N_LAYERS = 2`. Restarting at the smallest scale to re-establish the baseline curve on tokens is cheap — at the projected ~2.45× wall-time speedup from tokenization the original 7-hour 1M byte-level run (2026-04-24 entry) should land in ~3 hours — and the 1M-token-level point versus the next scales determines whether the BPE pivot actually beats byte-level on a per-parameter basis. The expectation going in: 1M token-level lands lower on train-loss-per-byte-encoded than 1M byte-level (each token now carries ~2.45 bytes), and the curve to higher scales should bend further than the byte-level diminishing-returns shape did from 1M → 8M (8× params for 17% loss reduction). If it does not, BPE was the wrong lever and the architecture or objective needs to change.

### Open / deferred

- **End-to-end validation**: training kickoff + first checkpoint + codec roundtrip on real (not random/zero) weights. Unit tests pass but the full token-level pipeline has not been driven end-to-end yet on hardware.
- **Alphabet trim**: ~56 byte values (invalid UTF-8 prefixes, most C0 control codes) never appear in enwik9 but consume base-vocab slots. Saves ~1.4% of vocab. Folded out of this diff to keep the rewrite small; cheap follow-up.
- **QAT for embedding**: post-hoc quantization is the first cut. If train/inference divergence costs measurable bpb, add fake-quant on the embedding in the training forward.
- **Eval harness noise**: the 5-offset codec panel from 2026-04-23 is still deferred. Codec bpb at single 16 KiB windows remains too noisy for plateau detection; train loss is still the only trustworthy signal.
- **Case folding**: deferred per the case-folding section above. A/B test it after tokenized end-to-end numbers exist.

---

## 2026-04-25 → 2026-05-03 — 8M RWKV Plateau; Metal Autorelease Fix; Scaling To 12M

CDR/Claude attempted a sequence of scale-up runs over ~9 days on an M3 Pro, blocked initially by a candle Metal memory leak that masqueraded as "training getting OOM-killed", then landed an 8M (`D_MODEL=256, N_LAYERS=16, CM_MULT=1`) run end-to-end through plateau on 2026-05-03. The run plateaued at train loss 1.024 nats / ~1.48 info-bpb, the best checkpoint sits at step 14319, and the next scale step is queued at 12M (`N_LAYERS=26`) rather than the originally-planned 16M (`N_LAYERS=32`) — the 2× depth bump saturates the dev box's training throughput, so 12M is the partial step that fits the M3 Pro comfortably. This entry covers the leak hunt, the plumbing it forced into place, the 4M and 8M runs themselves, and the inference-budget tension at 12M.

### Memory leak: ~6 MB/sec linear RSS climb on Metal

Symptom appeared on the first 2M training attempt: kernel-killed several times with RSS growing past 100 GB. Repro pattern was a perfectly linear ~6 MB/sec RSS climb regardless of batch/seq settings, regardless of how often `device.synchronize()` was called between steps. Initial debugging chased two wrong leads:

- **Suspected the autograd graph through WKV's 256-step recurrence.** Added truncated-BPTT (`--bptt-chunk` default 32, detaching `aa`/`bb`/`pp` at chunk boundaries inside `time_mix`). Cuts the autograd-graph chain from O(T²) to O(T·C) tensors. Real win for peak BPTT memory but did nothing for the linear leak rate.
- **Suspected candle's Metal `private_buffers` pool having no cleanup hook.** `metal_backend/device.rs::drop_unused_buffers` only iterates `self.buffers` (CPU-shared pool), not `self.private_buffers` (GPU-private pool, which is the hot path for every kernel output). Vendored candle-core 0.10.2 into `vendor/`, patched `drop_unused_buffers` to also sweep `private_buffers`, plumbed it into `wait_until_completed`. Build went green, leak persisted unchanged. The pool patch was a real bug fix (the pool *did* grow monotonically) but wasn't the dominant leak source.

Root cause turned up in candle issue [#2271](https://github.com/huggingface/candle/issues/2271), reported June 2024 with a clear repro of the same pattern (a tight `matmul` loop leaking GBs), still open as of March 2026. Maintainer LaurentMazare's second comment identifies it: the **ObjC autorelease pool is not draining**. Cocoa GUI apps drain the topmost autorelease pool every runloop tick; CLI Rust binaries have no runloop, so autoreleased Metal objects (`MTLCommandBuffer` instances, status snapshots, error objects, transient `NSData`) accumulate until process exit. His workaround is wrapping the per-iteration body in `objc2::rc::autoreleasepool(|| { ... })`. Applied to `train.rs`'s per-step body, RSS plateaus cleanly in the 1–3 GB range at 2M and 2–4 GB at 8M. Vendored candle-core was reverted; the fix is a single closure wrap on the consumer side.

Lesson: search the dependency's issue tracker before writing speculative patches against vendored source. The 1M run from 2026-04-24 didn't hit this because at 1M scale the leak rate was small enough that runs completed before RSS pressure mattered; 2M was the first scale where the slope mattered.

### Plumbing that landed alongside the leak hunt

- **`--resume <ckpt>`** on `lzr train`. Loads weights from a checkpoint, rehydrates each ternary `BitLinear` as `unpack(packed)[i, j] · scale[i]` via `bitnet::unpack_i2s_to_rowmajor` and a new `Weights::load_checkpoint_force_i2s` helper (the regular load path drops the I2_S packed bytes after building the LUT buffer; resume needs them). Step counter resumes at `checkpoint.step + 1` so cosine LR + plateau detector see contiguous history. AdamW moments and the data-sampling RNG do not survive a resume — first dozen steps after restart show small loss bounce while the moments warm back up.
- **Atomic checkpoint writes** via tempfile + rename. Eliminates the truncated-`.ckpt` failure mode if the trainer is killed mid-write.
- **`--max-steps-this-run`** as a safety hatch. Counts from 0 every invocation, exits cleanly with a final checkpoint after N steps. Was the planned mitigation if the autorelease fix hadn't worked — left in as defense-in-depth.

### 4M run (D_MODEL=256, N_LAYERS=8): aborted after ~1k steps

A 4M scale step (~3.77M params) ran briefly on 2026-04-28 — last checkpoint at step 1058, train loss 1.839 nats. CDR pivoted to 8M without producing a clean plateau curve at 4M because at this point the autorelease fix was confirmed and the 8M run looked feasible inside a few days. 4M is therefore a missing data point in the scale series; the 1M → 2M → 4M → 8M trajectory has clean numbers only at 1M and 8M. Acceptable for now; if width-vs-depth scaling becomes an open question later, a clean 4M run is cheap to add.

### 8M run (D_MODEL=256, N_LAYERS=16): 5 days, 19k steps, plateau 1.024 nats

Started 2026-04-28 17:39, stopped 2026-05-03 ~09:53 after the user observed plateau. Per-step time ~5.4 s on M3 Pro Metal. Per-checkpoint codec test on 16 KiB of enwik9 still produces single-window noise too large to drive selection (best codec bpb 0.573 at step 17649, worst 3.029 at step 8989, no relationship to underlying model quality) — train loss is the only trustworthy signal until the 5-offset-panel + held-out-1-MB eval harness from 2026-04-23 lands.

| step  | train nats | codec bpb |
|------:|-----------:|----------:|
|   323 |       1.98 |      3.96 |
|   656 |       1.77 |      2.69 |
|  2326 |       1.39 |      2.04 |
|  5659 |       1.21 |      1.66 |
|  8323 |       1.12 |      1.95 |
| 10322 |       1.14 |      2.07 |
| 13653 |       1.04 |      1.94 |
| 14319 |   **1.02** |      1.94 |
| 17649 |       1.17 |      0.57 |
| 18981 |       1.22 |      0.71 |

Best checkpoint **step 14319 at 1.024 nats / ~1.48 info-bpb**, archived under `checkpoints/keep/`. The 4,662 steps (~28 hours of training) after step 14319 produced no further improvement — train loss bounces in the 1.10–1.30 band with no descending trend. Plateau detector did not fire because per-checkpoint deltas oscillate ±0.1 around the trend, never accumulating three consecutive stale checkpoints; the plateau is real but invisible to the current monitor. Cosine LR was barely past warmup (`decay_horizon = 1M`, run reached step 19k), so the ceiling is capacity, not LR.

### Diminishing returns: 8× more parameters → 17% loss reduction

| scale | best train nats |
|------:|----------------:|
|    1M |            1.24 |
|    8M |            1.02 |

For an 8× parameter increase, train-loss improvement is 0.22 nats / ~17%. For comparison, the 1M → 11M leap (transformer, since-retired) projected ~0.7 nats from train-loss alone before the codec collapse made it moot. The 8M RWKV's leverage-per-parameter is much weaker — consistent with the reading that we're up against byte-level RWKV's representational ceiling at the current `D_MODEL=256, seq=256` regime, not the LR-decay or training-time ceiling. Width (`D_MODEL=512`) was not tried because it would commit every matvec to exactly the TL1 i16 accumulator bound with no further-width headroom and force kernel work; depth doubling stays inside the kernel envelope.

### Inference budget reality check

Per-checkpoint codec test reports `encode_hours_1gb` and `decode_hours_1gb` measured on M3 Pro CPU — the same code path the eventual x86 judging machine will run, modulo per-platform single-core throughput differences. Using these numbers as a relative scale signal:

| layers | encode h/GB | decode h/GB | total h |
|-------:|------------:|------------:|--------:|
|      8 |        ~29  |        ~29  |     ~58 |
|     16 |        ~54  |        ~54  |    ~108 |

Both numbers are with the TL1-NEON LUT path active (`bitnet::matvec_ternary_lut` dispatches to `matvec_ternary_tl1_neon` on aarch64; LUT buffer is built at `Weights::from_bytes` time and the I2_S buffer is dropped to save RSS). Wall-clock scales linearly with N_LAYERS as expected — matvec is the dominant inference cost.

The Hutter wall budget is `70,000 / Geekbench5` hours on the judging machine. For a Zen 2 single-core system around Geekbench5 ~1100 that's ~64 hours total (encode + decode, since the same binary handles both at 1× scoring). **8M is already over budget**: 108 hours > 64 hours by a factor of ~1.7×. 12M (N_LAYERS=26) projects to ~88 h/GB encode → ~175 h total, over by ~2.7×. The originally-planned 16M (N_LAYERS=32) would have been ~216 hours, over by ~3.4×.

Accepted as a known constraint for the 12M scale step. The matvec dominates so cleanly that the next round of optimization has to live in the matvec kernel itself: 2-row or 4-row output tile fusion across the TL1/TL2 inner loops to amortize the LUT load, tighter activation precision (4-bit or 5-bit rather than 7-bit), or eventually a non-LUT scheme. Outside scope of this entry.

### Decision: scale to 12M (N_LAYERS=26), not 16M

Original plan was a clean 2× depth bump to `N_LAYERS = 32` (~14.88M total, "16M-class"). After landing it in `src/arch.rs` and re-running the build, projecting per-step training time at ~10.8 s on M3 Pro Metal (2× the 8M cost, ~78 hours wall for 26k steps), CDR pulled back: a 3.5-day run that pegs the dev box's GPU and thermal envelope is too disruptive when the 8M plateau already clears the "is the architecture working" bar. Backing off to `N_LAYERS = 26` (~12.10M total: 11.93M ternary `7·D_MODEL² = 458,752` per layer × 26 + ~173K f32) keeps the depth bump at 1.625× rather than 2×, projects per-step time to ~8.8 s and ~64 hours wall — a 2.6-day run that the dev box can comfortably absorb. Breaks the strict 1M → 2M → 4M → 8M doubling cadence; treated as the cost of staying inside the dev-machine envelope.

Same axis-only-moves rationale as every prior step: width preferred-over-depth would be ~15% faster per step (WKV recurrence is sequential, so depth costs more per parameter than width does) but committing to `D_MODEL=512` puts every matvec exactly at the TL1 i16 accumulator bound with no further-width headroom. Holding 256 keeps the kernel-side spec book unchanged across the whole 1M → 12M trajectory.

The expectation going in is that train loss continues bending toward 0.95 nats give-or-take, with the open question being whether the eval harness can be fixed in time to confirm; if the 12M run plateaus near 0.97–1.00 it's a ~3–5% improvement on 8M and the diminishing-returns pattern from 1M→8M continues. If it plateaus at 8M's 1.024, depth has stopped paying off (at 1.625× the parameter count) and the next scale step needs width or a different objective.

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
