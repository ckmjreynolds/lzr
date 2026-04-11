#!/bin/bash
export TERM="${TERM:-dumb}"
if [[ "${TERM}" != "dumb" ]]; then
  clear
fi

cargo +stable fmt || exit
cargo +stable clippy --all-targets -- -Dwarnings || exit

# Validate the safe-only build (no `unsafe` feature) so it never bit-rots.
cargo +stable check --no-default-features --all-targets || exit
cargo +stable test --release --no-default-features --quiet || exit

# Use nightly to get coverage reports.
cargo +nightly llvm-cov --doctests --branch --lcov --output-path lcov.info || exit
cargo +nightly llvm-cov report
