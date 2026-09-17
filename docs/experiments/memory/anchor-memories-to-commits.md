---
name: anchor-memories-to-commits
description: "For this repo, anchor every project/experiment memory to BOTH the git branch and the commit hash whose code state its findings were measured against — branch is required, not optional"
metadata: 
  node_type: memory
  type: feedback
  commit: 5b2e319
  branch: v0.1.0
  originSessionId: 3047ed92-5c41-4582-8b93-839399b25c91
---

When writing or updating a memory about lzr experiments/measurements, record **branch + commit hash** whose code state the finding was validated against — in the frontmatter `metadata` (`branch:` AND `commit:`) AND as a visible `**Code state:** branch \`<branch>\` @ commit \`<hash>\`` line near the top of the body. Always include the branch; a bare hash is not enough here.

**Why branch is required, not just the hash:** this repo has multiple long-lived, *divergent* branches — `v0.1.0` (current work), `hutter` (a whole parallel neural/CM stack; the SSM arm was ported *from* it), and `main`. The same finding can hold on one branch and be meaningless on another (e.g. byte-vs-token arm, the `Model` vs `TokenModel` trait, the richer `Context` all differ between `v0.1.0` and `hutter`). A hash without a branch is also hard to locate, and branches get rebased/rewritten while the branch name stays the stable "where does this work live" pointer. Branch tells you *which reality*; hash pins *which point in it*.

**Why anchor at all:** the user does a ton of experimentation and throwing away (said explicitly, 2026-07-09). A bpb number or "X is a dead end" only means something relative to a specific tree. Uncommitted experiment code is common here — commit it (or record the hash) so the memory has a real anchor.

**How to apply:** Prefer committing the relevant work first, then tag the memory with `branch + hash`. On a branch switch or big refactor/revert, re-check that an anchored memory's branch/tree still matches before trusting it. Existing anchored memories: [[neural-arm-replay-result]] (v0.1.0 @ 5b2e319), [[operating-point-beats-cost-calibration]] / [[mgp-sequence-reparse-result]] / [[calibration-scaffolding-status]] (v0.1.0 @ dd7a90b).
