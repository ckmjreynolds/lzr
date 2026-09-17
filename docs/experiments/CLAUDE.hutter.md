# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`lzr` is a composable byte-stream compression library and CLI (Rust, edition 2024, MSRV 1.87).
Published to crates.io. It is a context-mixing compressor built as a pipeline of reversible
byte→byte transforms; the longer-term aim is a Hutter-Prize branch, which shapes several design
choices (framing factored out of the codec, minimal mixer, `hutter:` bpb reported by the CLI).

## Commands

- **Full pre-PR check suite:** `bash build.sh` — runs `fmt`, `clippy --locked --all-targets -Dwarnings`,
  a `--no-default-features` check + test, and nightly `llvm-cov` coverage (writes `lcov.info`).
  Run this before opening a PR. Requires both `stable` and `nightly` toolchains.
- **Test:** `cargo test` — a single test: `cargo test <name>`; one file: `cargo test --test roundtrip`.
- **Lint:** `cargo clippy --all-targets -- -D warnings` (warnings are errors; lint config is strict, see below).
- **Benchmarks:** `bash bench.sh` (= `cargo bench --features bench-internals`). Benches need the
  `bench-internals` feature to reach internal modules — plain `cargo bench` will not build them.
- **End-to-end roundtrip:** `bash roundtrip.sh` (enwik8) / `bash roundtrip_enwik9.sh` — compress then
  decompress a corpus and `cmp` against the original. **Do NOT run anything against the full enwik9
  (`roundtrip_enwik9.sh`, or compressing/decompressing `corpora/hutter/enwik9`) without explicit user
  approval** — a full enwik9 pass takes many minutes and destroys iteration speed. Measure on enwik8 (or
  a prefix, e.g. `head -c 20000000 corpora/hutter/enwik9`) by default; only validate on full enwik9 when
  the user asks. The two roundtrip scripts use separate scratch paths (`/tmp/out8.lzr` / `/tmp/out9.lzr`)
  so they can run concurrently; do not point them at a shared `.lzr` path, since the CLI's end-of-run size
  report re-stats the input file and a concurrent overwrite would then show a mismatched size/bpb.
- **CLI:** `cargo run --release -- <input> <output>`. Compresses by default; a `.lzr` input decompresses
  by default. Force with `-z`/`-d`. Toggle pipeline features with `--enable <name>` / `--disable <name>`
  (repeatable); `--list-features` prints every toggleable stage/model/flag with its default.

## Architecture

The codebase is layered as **container → codec core → pipeline of transforms**. Framing is deliberately
separated from compression so the codec core can later be re-wrapped in a near-zero-header framing
(the Hutter branch) without touching the pipeline.

- **`container.rs`** — the self-describing on-disk container: `magic "LZR" | version | profile (ULEB128)
  | codec core | Adler-32 footer (BE, over the *original* bytes)`. `compress*` / `decompress` live here.
  Decoding validates length/magic/version/profile, decodes the core, then re-checks the checksum — a
  container never decodes to silently-wrong bytes. `VERSION` is frozen at `0x00` during development
  (see the doc/version policy below).
- **`codec.rs`** — the framing-agnostic driver. `Profile` is a feature bitmask; `Compressor` assembles
  the selected stages into a `Pipeline`. **The `FEATURES` registry** (bit position = index) is the
  single source of truth for pipeline features; adding/reordering a stage is one edit there. Stage
  order is **registry order** (which is also bit position): `Profile::pipeline` assembles the enabled
  `Kind::Stage` entries in registry order, so the canonical chain casefold → entities → repair →
  entropy (entropy always last) is imposed by where those entries sit in `FEATURES`. Stages are
  parameterless — every stage recovers what it needs from the stream, so the same assembly serves
  encode and decode and nothing encode-only is serialized. **`repair` is default-off** (the
  strengthened CM regresses with it on — see below); the default pipeline is casefold → entities → entropy.
- **`transform.rs`** — the `Transform` trait (`forward`/`inverse`, both **by value** so a stage can free
  its input before an expensive build) and the `Pipeline` that chains stages (forward in order, inverse
  in reverse). Every stage speaks the same `Vec<u8>` domain, so stages compose and toggle freely.
- **`preprocessors/`** — the reversible byte→byte pipeline stages that run ahead of the entropy coder.
  The text stages `casefold` (ASCII case-folding via inline shift/caps-lock control symbols) and
  `entities` (XML/HTML entity folding) are self-describing with a mode-byte header, gate on the shared
  text-detection helpers in `mod.rs`, and pass binary through unchanged. `preprocessors/repair/` is the
  **byte-BPE** Re-Pair grammar tokenizer: it claims the block's unused byte values as non-terminals and
  serializes the grammar inline as plain bytes (`[R] [(sym,left,right)]*R [sequence]`, `sym`/children
  each a byte value), so a symbol id *is* a byte and the decoder recovers the whole vocabulary from the
  rule table — no side channel, no varint ids. Terminals keep their original byte value; non-terminals
  take the free byte slots in rule-creation order; capped at a 256-symbol vocabulary (`VOCAB_CAP`), no
  per-run parameters. **Re-Pair is default-off**: the strengthened CM (order0..11 + `match` + per-context
  mixer + SSE) subsumes its handful of rules and their token boundaries now *regress* whole-stream bpb
  (~0.022 on enwik8). It stays as an opt-in (`--enable repair`) for corpora where it still pays.
  "Tokenization off" = the stage is simply absent (now the default).
- **`entropy.rs` + `coder.rs` + `mixer.rs` + `apm.rs` + `models/`** — the entropy stage: a carryless
  binary range coder (`coder.rs`) driven by a logistic mix (`mixer.rs`) of byte-level bit-prediction
  `Model`s (`models/`: the `order0..11` context chain, `sparse2`, the `match` model, the `word`
  character-run model (`run.rs`, a generic char-class run keyed on the trailing letters), the
  `xmltag` model (`xmltag.rs`, keyed on the enclosing element), and the `iddelta` model (`iddelta.rs`,
  predicting a number's digits from the previous completed number — the cross-token structure of
  near-monotonic ids/timestamps), each independently toggleable; a `null` no-op is injected if none are
  selected). All are default-on. The mixer keeps one weight set per
  **(previous-byte selector × bit position)** so the
  blend specializes by local regime; a chained APM/SSE (`apm.rs`, the `sse` flag then the order-1 `sse2`
  flag) refines the mixed probability. The `surprise` flag (`surprise.rs`) feeds the coder's own recent
  coding cost back as a regime context: its quantized fast level widens the `mix2` order-1 selector and
  keys a third APM (default-on, enwik8 1.5334→1.5248). Each byte is coded as an 8-bit MSB-first
  bit-tree. Encode and decode build identical fresh predictor state (hashed table sizes derive from the
  framed byte count) so predictions match bit-for-bit.
- **`adler32.rs`, `uleb128.rs`** — checksum and varint primitives. `uleb128` is a general `u64` codec
  (1–9 bytes, used for the container profile field and Re-Pair's rule count); encodings must be
  **canonical** (overlong encodings are rejected).

## Development-mode doc & version policy

The project is early and the wire format is **unstable**, so we deliberately do not spend effort on
format documentation or version numbers — do not recreate or maintain these:

- **`container::VERSION` is frozen at `0x00`.** Do **not** bump it for format changes; an old stream
  simply fails to decode, which is fine pre-1.0. Change it only when the user explicitly asks.
- **There is no `docs/FORMAT.md`** (deleted) — do not recreate a wire-format spec. The code and its
  doc-comments are the source of truth. Do not maintain `CHANGELOG.md` or `README.md` either.
- When you change the format, just change the code and its inline docs; skip the ceremony above.

## Conventions and invariants

- **Reversibility is the core invariant.** Every stage must exactly invert; the container's Adler-32
  over the original bytes is the backstop. Tests lean heavily on `proptest` roundtrips and on
  `decompress`/`decode` **never panicking** on arbitrary/adversarial input (they run on untrusted
  decoded bytes — return `Err`, don't panic).
- **Determinism.** Encoder and decoder derive all model/table sizes from the framed count so their
  predictions stay in lock-step. Don't introduce state that differs between the two sides.
- **Untrusted-input guards.** The entropy decoder caps decoded-bytes-per-payload-byte
  (`MAX_BYTES_PER_PAYLOAD_BYTE`) as a decompression-bomb guard; keep such bounds when editing decode paths.
- **Strict lints.** `unsafe_code` is forbidden. `unwrap`/`expect`/`panic`/`todo`/`print_*`/`dbg` are
  clippy-denied in library and CLI code (allowed in tests via `clippy.toml`). clippy `pedantic`,
  `nursery`, and `cargo` groups are on. Use `#[expect(lint, reason = "...")]` (reason required) rather
  than `#[allow]`. `unused_results` and `missing_docs` are warnings — document public items and handle
  results (`drop(...)` when deliberately discarding).
- **`bench-internals` feature** widens `pub(crate)` internals (adler32, uleb128, preprocessors, transform)
  to `pub` via `cfg` + `visibility::make` so `benches/` can reach them. It is off by default; never rely
  on it for the normal build.
- The build must stay green under `--no-default-features` (build.sh checks it) — don't let default-only
  code paths bit-rot.
