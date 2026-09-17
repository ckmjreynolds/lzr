---
name: bit-history-states
description: "bit-history states (lpaq-class) replace per-context StateMap slots in OrderN+SparseModel — triple win: -0.013 bpb, -22% time, -41% RSS"
metadata: 
  node_type: memory
  type: project
  originSessionId: 45c7d453-f494-4d60-bbd8-b7fb702b2453
---

FINDING measured on v0.1.0 atop commit 6303147 (the mix2 default-on baseline: enwik8 1.5586 overall).
This is recommendation #1 from the project-status review, and the biggest structural gap vs the lpaq/zpaq
compressors this codebase models.

**What changed.** The context models keyed each context directly to a 3-byte `(p: u16, n: u8)` StateMap
slot; for sparse high-order contexts (seen a handful of times) that per-context probability barely leaves
½. Replaced with the lpaq-class design: a 1-byte **bit-history state** per context + one shared 256-entry
StateMap mapping state→probability, so all contexts reaching the same short history pool their stats.
- `src/models/state.rs` (new): nonstationary state machine. State = capped/discounted `(n0,n1)` counter
  pair, one nibble each → exactly 256 states, no enumeration. Observing a bit bumps its count (saturating
  at 15) and discounts the opposite (halve excess above 3), giving recency/nonstationarity. Plus a
  reusable `BitHistory` struct (per-context `Vec<u8>` states + shared `StateMap`) that both models share.
- `StateMap` gained `with_limit()` — the shared state map uses limit 127 (heavily visited, wants
  precision) vs the per-context default 11.
- `OrderN`: order-0 stays dense/direct; **orders 1+ use BitHistory** (`BITHIST_MIN_ORDER=1`).
- `SparseModel` (sparse2): also uses BitHistory.

**Results (20 MB enwik8 entropy-stage, baseline 1.6327):** OrderN bithist 1.6174 (−0.0153); +sparse2
1.6165 (−0.0162 cumulative). **Full enwik8: overall 1.5586→1.5457 (−0.0129), byte-exact roundtrip.**
Bonus at 20 MB: **−22% encode time** (75s vs 96s — 1 byte is far more cache-friendly than the 3-byte
slot on the huge hashed tables) and **−41% peak RSS** (0.97 GB vs 1.64 GB) — the memory drop directly
helps the enwik9 <10 GB Hutter budget.

**Full enwik9 (validated byte-exact roundtrip):** overall 1.2772→**1.2729** (−0.0043), entropy
1.2530→1.2489, size 159,648,681→159,117,756 B (−531 KB). Encode 60.8 min (was ~76), decode 58.6 min
(was ~74) — ~−20%. Peak RSS: encode **7.89 GB** (was ~12, −34%), decode 10.48 GB → **8.88 GB after the
pre-reserve fix below** (now under the 10 GB Hutter limit).

**Decode-vs-encode RSS asymmetry — diagnosed + fixed.** enwik9 decode was 10.48 GB vs encode 7.89 GB;
enwik8 showed only an 85 MB gap (3.24 vs 3.32 GB) — the gap scales with data-buffer size, NOT a fixed
cost. Cause: model tables (~6.9 GB at enwik9, bits=28) are identical both sides; the difference is
transient Vec-doubling. The pipeline is casefold→entities→entropy and inverse runs it reversed, so the
entropy stage (which allocates all model tables) runs LAST on encode, FIRST on decode. On decode the
1 GB output Vec AND the match model's 1 GB `r` both grow by doubling *while* the 7 GB of tables are
resident (reallocate-and-copy spike + retained freed spans); on encode the 1 GB side is the pre-sized
input (read once), so only `r` doubles. FIX (commit-pending, allocation-only, output bit-identical):
pre-reserve the decode output (`entropy.rs` inverse, `MAX_RESERVE_PER_PAYLOAD_BYTE=256` cap vs the 4096
bomb-limit so a corrupt count can't over-allocate) and `match.r` (`match_model.rs`, capped at
MAX_R_BYTES). `with_capacity` reserves address space only (pages fault as written) → no RSS cost for a
genuine stream, no bomb risk. Result: decode 10.48→8.88 GB. Residual ~1 GB gap is inherent (decoder
holds 1 GB output resident where encoder holds 159 MB).

The bpb delta SHRINKS monotonically with scale (20 MB −0.0162 → enwik8 −0.0129 → enwik9 −0.0043): the
per-context slots the old design used also warm up given enough data, narrowing the bithist edge on the
largest corpus. Ratio win is real but modest at enwik9; the speed/memory wins carry through in full.

**Swept params:** `BITHIST_MIN_ORDER` 1 beat 2 (−0.0128) and 3 (−0.0068) — convert as much as possible,
only order-0 (256 dense contexts) wants the direct path. `STATE_MAP_LIMIT` 127 ≈ 191 (noise) > 63; 255
neutral.

Supersedes the "no obvious further StateMap shrink" claim in [[order8-model-result]]: the 3-byte slot now
survives only on the direct-path models (order-0, match, run, xmltag, iddelta). Related: [[two-layer-mixer-win]].

**Next levers in this vein (untried):** apply BitHistory to the remaining direct-path models where their
contexts are sparse; a richer state machine (lpaq keeps exact histories up to 4 bits — the nibble-pair
loses some short-history detail); a run/last-byte APM keyed on the state. See [[anchor-memories-to-commits]].
