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
