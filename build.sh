#!/bin/bash
export TERM="${TERM:-dumb}"
if [[ "${TERM}" != "dumb" ]]; then
  clear
fi

cargo +stable fmt || exit
cargo +stable clippy --locked --all-targets -- -Dwarnings || exit

# Validate the safe-only build (no `unsafe` feature) so it never bit-rots.
cargo +stable check --locked --no-default-features --all-targets || exit
cargo +stable test --locked --release --no-default-features -- --nocapture 2>&1 || exit

# Use nightly to get coverage reports.
cargo +nightly llvm-cov --doctests --branch --lcov --quiet --output-path lcov.info || exit
cargo +nightly llvm-cov report
