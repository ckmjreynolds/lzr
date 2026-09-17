---
name: order8-model-result
description: "order8/9/10 extend the model chain (kept); sparse24 re-rejected; StateMap n shrunk u16→u8"
metadata:
  node_type: memory
  type: project
  originSessionId: 4f3d58fb-017d-46f5-a15a-13ba6d127f0d
---

Model-chain + memory sweep on v0.1.0 atop commit 33a4aa1, extending [[entropy-model-sweep-u8]].
Rule: keep any model that doesn't regress whole-stream enwik8 bpb, revert (remove) any that does.
All roundtrip-verified byte-identical (full enwik8 encode→decode→cmp).

Cumulative enwik8 (baseline order0..7 = 1.6893 bpb / 21,115,639 B):
- **order8**  (`OrderN::new(8)`): 1.6865 / 21,081,200  (−0.0028, w=+0.123) — kept
- **order9**  (`OrderN::new(9)`): 1.6851 / 21,064,320  (−0.0014, w=+0.066) — kept
- **order10** (`OrderN::new(10)`): 1.6845 / 21,056,526 (−0.0006, w=+0.040) — kept
- **order11** (`OrderN::new(11)`): 1.6843 / 21,053,368 (−0.0002, w=+0.004) — kept, LAST winner
- **order12** (`OrderN::new(12)`): 21,053,465 (**+97 B**, w=+0.013) — REGRESSED, reverted. The wall.
- **sparse24** (`SparseModel::new(&[2,4])`): 1.6851 / 21,063,973 (**+0.0006, w=−0.035**) — REGRESSED,
  reverted. Re-confirms the pre-rewrite result: mixer drives it negative, redundant with order chain.

Chain byte-wins: −34439/−16880/−7794/−3158 (order8→11) then **+97** at order12 — first regression,
the wall. Stopped at 11. Final default = order0..11 + sparse2 + match, **bitmask `0x7FFFF`**
(19 features; `match` = bit 18).

**StateMap slot shrunk 6 B → 3 B** (`models/statemap.rs`, both value-identical: enwik8 stays bit-for-bit
21,053,368 B, roundtrips):
- `n: Vec<u16>`→`Vec<u8>` (commit 706303c) — count capped at LIMIT=11, fits a byte. −1 B/slot.
- `p: Vec<i32>`→`Vec<u16>` (commit e399fc6) — p only ever in 0..=65535 (inits 32768, tracks toward
  target 0/65535 by rate/65536 ≤ ½, never overshoots/goes negative; must be **u16 not i16** — i16
  caps at 32767, can't hold the 32768 init). −2 B/slot.
Net −3 B/slot = **384 MiB/table enwik8 / 768 MiB/table enwik9**. Across 12 hashed tables:
~4.5 GiB enwik8 / **~9 GiB enwik9**. enwik8 encode peak RSS fell 8.8 → 5.5 GiB.

Net enwik9 decode footprint: order8..11 add ~+6 GiB, the 6→3 B shrink returns ~−9 GiB → the 12 hashed
StateMaps drop from ~18 → ~9 GiB, so total lands ~**11–12 GiB, below the original ~14 GiB** while
gaining −0.0050 bpb (1.6893→1.6843). Tables still clamp at 2^28 (`hashed_bits`, `models/mod.rs`).

Order chain fully exhausted — order12 regressed (+97 B), don't retry. StateMap now at its natural
width (u16 prob + u8 count); no obvious further shrink without precision loss.
