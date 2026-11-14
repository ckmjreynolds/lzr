#!/usr/bin/env bash
# Local pre-push gate and CI entry point: every CI check job runs one step of this script, so a
# clean `./build.sh` locally means the same commands pass in CI.
#
#   ./build.sh             all local steps (default)
#   ./build.sh <step>...   just those steps, in order
#
# Steps: outdated fmt lint doc msrv test [toolchain] cov deny package bench codspeed
# (`bench` and `codspeed` are not part of `all`.)
set -euo pipefail
cd "$(dirname "$0")"

# The MSRV is declared once, as `rust-version` in [workspace.package].
MSRV="$(sed -n 's/^rust-version *= *"\(.*\)"/\1/p' Cargo.toml)"
[[ -n "$MSRV" ]] || { echo "build.sh: no rust-version found in Cargo.toml" >&2; exit 1; }

# Advisory only: report outdated direct dependencies without failing the build.
step_outdated() {
  if command -v cargo-outdated >/dev/null; then
    echo "Checking for outdated dependencies:"
    cargo outdated --workspace --root-deps-only || true
  fi
}

# Format in place locally; only verify in CI.
step_fmt() {
  if [[ -n "${CI:-}" ]]; then
    cargo +stable fmt --all -- --check
  else
    cargo +stable fmt --all
  fi
}

# Warnings are errors only on stable (lint, doc). The MSRV and beta test runs gate on "builds and
# tests pass", so lint drift between toolchains cannot break CI.
step_lint() {
  cargo +stable clippy --locked --workspace --all-targets --all-features -- -Dwarnings
}

step_doc() {
  RUSTDOCFLAGS="-Dwarnings" cargo +stable doc --locked --workspace --no-deps --all-features
}

# Fast local check that the declared rust-version builds. CI goes further and runs the full test
# suite on the MSRV (`test <msrv>` in the matrix).
step_msrv() {
  cargo "+${MSRV}" check --locked --workspace --all-targets --all-features
}

# nextest does not run doctests, so they get their own pass.
step_test() {
  local toolchain="${1:-stable}"
  cargo "+${toolchain}" nextest run --locked --all-features --workspace
  cargo "+${toolchain}" test --doc --locked --all-features --workspace
}

# Single instrumented test pass: llvm-cov builds the dev profile, so overflow-checks and
# debug_assert! are active. nextest and doctest profiles feed one report (lcov.info for Codecov).
step_cov() {
  cargo +nightly llvm-cov clean --workspace
  cargo +nightly llvm-cov nextest --no-report --locked --all-features --workspace --branch
  cargo +nightly llvm-cov --doc --no-report --locked --all-features --workspace --branch
  cargo +nightly llvm-cov report --doctests --branch --lcov --output-path lcov.info
  cargo +nightly llvm-cov report --doctests --branch
}

# Advisories (RustSec), licenses, bans and sources; policy in deny.toml.
step_deny() {
  cargo deny --locked check
}

# Verify each published crate builds from its `include` list. Locally the tree is usually dirty.
step_package() {
  if [[ -n "${CI:-}" ]]; then
    cargo +stable package --locked --workspace --quiet
  else
    cargo +stable package --locked --workspace --quiet --allow-dirty
  fi
}

# Local wall-clock benchmarks (plain Divan output). Too slow and noisy for `all`.
step_bench() {
  cargo +stable bench --locked -p lzr --features bench
}

# CI only: build the benches for CodSpeed's simulation instrument; CodSpeedHQ/action then runs them.
# The mode must match the action's `mode` input.
step_codspeed() {
  cargo +stable codspeed build --locked -p lzr --features bench -m simulation
}

# `cov` doubles as the local test run, so `all` does not also run `test`.
step_all() {
  step_outdated
  step_fmt
  step_lint
  step_doc
  step_msrv
  step_cov
  step_deny
  step_package
}

if [[ -t 1 ]]; then
  clear
fi

if [[ $# -eq 0 ]]; then
  set -- all
fi

case "$1" in
  test)
    shift
    step_test "$@"
    ;;
  *)
    for step in "$@"; do
      if ! declare -F "step_${step}" >/dev/null; then
        echo "build.sh: unknown step '${step}'" >&2
        grep '^# Steps:' "$0" >&2
        exit 2
      fi
      "step_${step}"
    done
    ;;
esac
echo
