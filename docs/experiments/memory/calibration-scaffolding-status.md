---
name: calibration-scaffolding-status
description: The lzr Re-Pair grammar cost-model / calibration experiments were tried and removed — do not re-add
metadata: 
  node_type: memory
  type: project
  originSessionId: cd030ade-6952-42a5-8f0f-034e3feb82c3
  commit: dd7a90b
  branch: v0.1.0
---

**Code state:** branch `v0.1.0` @ commit `dd7a90b` — the state where this scaffolding is confirmed ABSENT (`prune.rs`/`repair/mod.rs` reverted; `--auto` shipped instead). If a future tree re-adds `--calibrate`/`--entropy-cost`, this dead-end warning applies.

While investigating [[operating-point-beats-cost-calibration]] (2026-07-07), experimental grammar-prune cost models were added and then **removed** once measured worse than default. Do not re-implement these — they are confirmed dead ends:

- `CostModel::Entropy` (order-0 self-information prune cost) and `EncodeOptions::with_entropy_cost` / CLI `--entropy-cost`.
- Trial-encode calibration: `entropy::meter_byte_bits`, `RepairTokenizer` injected `meter`/`forward_calibrated`/`attribute_costs`, `prune::prune_costed_pass`, `EncodeOptions::with_calibration` / CLI `--calibrate`.

Both kept too many rules and lost ~1-4% bpb. Root cause is fundamental (local per-rule decisions can't see the global model-dilution benefit of a small vocabulary), so the whole family is a dead end — the win is operating-point search, now shipped as `--auto` (see [[operating-point-beats-cost-calibration]]). The `prune.rs`/`repair/mod.rs` files were reverted to their pre-experiment state.
