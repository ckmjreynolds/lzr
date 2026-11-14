# Contributing

Thanks for your interest in contributing to `lzr`!

## How decisions are made (technical merit)

All discussions and changes are evaluated on technical merit: correctness, tests,
clarity, performance, API design, maintainability, and documentation. Decisions
are not based on who proposes a change. Please keep threads focused on the work
at hand and support claims with evidence (tests, benchmarks, minimal repros, or
references).

## Repository layout

- `crates/lzr`: the library.
- `crates/lzr-cli`: the command-line tool (installs a binary named `lzr`).
- `docs/rust-guidelines.txt`: the coding guidelines this project follows.

Shared metadata, dependency versions, and lints live in the workspace `Cargo.toml`.

## Setup

You need [rustup](https://rustup.rs) and these toolchains and tools:

```sh
rustup toolchain install stable nightly
rustup toolchain install 1.100                   # the MSRV: `rust-version` in Cargo.toml
rustup component add llvm-tools-preview --toolchain nightly
cargo install --locked cargo-nextest cargo-llvm-cov cargo-deny
cargo install --locked cargo-outdated bacon      # optional
```

## Checking your change

Run `./build.sh` before opening a pull request. CI runs the same script, one step
per job, so a clean local run means CI will pass the same checks. You can also run
single steps:

```sh
./build.sh lint doc       # clippy and rustdoc, warnings are errors
./build.sh test           # nextest + doctests on stable
./build.sh test beta      # ... or on another toolchain
./build.sh cov            # instrumented test run, writes lcov.info
./coverage.sh             # open the HTML coverage report from the last run
```

The full list of steps is at the top of `build.sh`. For a watch loop, run `bacon`:
`t` runs nextest, `T` runs the doctests.

## What we look for

### Correctness and tests

- Bug fixes include a regression test when reasonable.
- New behavior comes with tests for its edge cases (empty input, single bytes,
  block boundaries, truncated or corrupt streams).
- Prefer property tests (`proptest`) for codec round-trips.
- Tests that only use the public API go under `tests/`, not in `mod tests`.
- Tests check behavior, not restated constants or a copy of the implementation.

### Style and lints

- Formatting is enforced by `rustfmt`, and clippy warnings are errors.
- The lint set in `Cargo.toml` follows the guidelines. Project-specific changes go
  only under the "custom opt-outs / opt-ins" markers, each with a reason.
- Override a lint locally with `#[expect(..., reason = "...")]`, not `#[allow]`.
- `unsafe` code is forbidden.

### Performance

`lzr` is a compression library, so performance matters:

- Avoid unnecessary allocations and cloning.
- Prefer iterator-friendly APIs when possible.
- If you change algorithms or invariants, include benchmarks or reasoning. For
  profiling, build with `cargo build --profile profiling` (release plus debug info).

### Public API

- Changes to the public API should be clearly motivated and documented.
- Every public item has documentation; consider adding an example.

## Commit and PR guidance

- Keep PRs focused (one logical change).
- Include a clear description:
  - What changed and why
  - How it was tested
  - Any trade-offs

## Release-related changes

If your PR is intended to ship in the next release, please also:

- Note whether it is breaking/change/add/fix/perf/docs.
- Add an entry under `[Unreleased]` in `CHANGELOG.md`.

## Security issues

Please do not open public issues for security vulnerabilities. Follow the instructions in `SECURITY.md`.
