#!/bin/bash
export TERM="${TERM:-dumb}"
if [[ "${TERM}" != "dumb" ]]; then
  clear
fi

cargo +stable fmt || exit

# Submission (default features): must stay clippy-clean.
cargo +stable clippy --all-targets -- -Dwarnings || exit

# Training build: the dev feature set also stays clippy-clean.
cargo +stable clippy --all-targets --features dev -- -Dwarnings || exit

# Bare-bones build (no features) compiles.
cargo +stable check --no-default-features --all-targets || exit

# CLAUDE.md invariant: the submission binary must not pull in a threading
# crate. `cargo tree` splits its output into the main dep tree and a
# `[dev-dependencies]` section; we only care about the main tree.
#
# `-e=no-dev` excludes dev-dependencies entirely, leaving only the deps that
# end up in the release binary.
if cargo +stable tree --features submission -e=no-dev 2>/dev/null | \
     grep -E '^[│ ]*[├└]── (rayon|crossbeam|gemm|tokio|async-std)' ; then
  echo "ERROR: submission build pulled in a threading dependency" >&2
  exit 1
fi

# Tests: submission (default) and dev.
cargo +stable test --release -- --nocapture 2>&1 || exit
cargo +stable test --release --features dev -- --nocapture 2>&1 || exit

# Release binaries: the `test --release` passes above don't produce the
# `lzr` binary itself, only the test executable. Build both here so
# ./target/release/lzr is ready after build.sh exits.
#
# The dev build runs second and so is the one that ends up at
# ./target/release/lzr (cargo rewrites the binary in-place per feature set).
cargo +stable build --release || exit
cargo +stable build --release --features dev || exit

# Coverage (nightly).
cargo +nightly llvm-cov --doctests --branch --lcov --quiet --output-path lcov.info || exit
cargo +nightly llvm-cov report
