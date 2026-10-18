#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ -t 1 ]]; then
  clear
fi

MSRV="1.85"

# Advisory only: report outdated direct dependencies without failing the build.
if command -v cargo-outdated >/dev/null; then
  echo "Checking for outdated dependencies:"
  cargo outdated --root-deps-only || true
fi

# Format in place locally; only verify in CI.
if [[ -n "${CI:-}" ]]; then
  cargo +stable fmt --all -- --check
else
  cargo +stable fmt --all
fi

cargo +stable clippy --locked --all-targets --all-features -- -Dwarnings
RUSTDOCFLAGS="-Dwarnings" cargo +stable doc --locked --no-deps --all-features

# Ensure the declared rust-version actually builds.
cargo "+${MSRV}" check --locked --all-targets

# Single test run: llvm-cov builds the dev profile, so overflow-checks and
# debug_assert! are active.
cargo +nightly llvm-cov --locked --all-features --doctests --branch --lcov --quiet --output-path lcov.info
cargo +nightly llvm-cov report --doctests --branch

# Verify the published crate builds from its `include` list.
cargo +stable package --locked --allow-dirty --quiet
echo
