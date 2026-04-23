#!/bin/bash
export TERM="${TERM:-dumb}"
if [[ "${TERM}" != "dumb" ]]; then
  clear
fi

# Whole build runs on nightly: the NEON dot-product intrinsic
# (`vdotq_s32`, used by the BitNet matvec kernel on aarch64) is behind the
# unstable `stdarch_neon_dotprod` feature as of Rust 1.85. Coverage also
# wants nightly (`llvm-cov --doctests --branch`). Using one toolchain
# everywhere avoids split-build inconsistencies.
TC="+nightly"

cargo $TC fmt || exit

# Submission (default features): must stay clippy-clean.
cargo $TC clippy --all-targets -- -Dwarnings || exit

# Training build: the dev feature set also stays clippy-clean.
cargo $TC clippy --all-targets --features dev -- -Dwarnings || exit

# Bare-bones build (no features) compiles.
cargo $TC check --no-default-features --all-targets || exit

# CLAUDE.md invariant: the submission binary must not pull in a threading
# crate. `cargo tree` splits its output into the main dep tree and a
# `[dev-dependencies]` section; we only care about the main tree.
#
# `-e=no-dev` excludes dev-dependencies entirely, leaving only the deps that
# end up in the release binary.
if cargo $TC tree --features submission -e=no-dev 2>/dev/null | \
     grep -E '^[│ ]*[├└]── (rayon|crossbeam|gemm|tokio|async-std)' ; then
  echo "ERROR: submission build pulled in a threading dependency" >&2
  exit 1
fi

# Tests: submission (default) and dev.
cargo $TC test --release -- --nocapture 2>&1 || exit
cargo $TC test --release --features dev -- --nocapture 2>&1 || exit

# Release binaries: the `test --release` passes above don't produce the
# `lzr` binary itself, only the test executable. Build both here so
# ./target/release/lzr is ready after build.sh exits.
#
# The dev build runs second and so is the one that ends up at
# ./target/release/lzr (cargo rewrites the binary in-place per feature set).
cargo $TC build --release || exit
cargo $TC build --release --features dev || exit

# Coverage.
cargo $TC llvm-cov --doctests --branch --lcov --quiet --output-path lcov.info || exit
cargo $TC llvm-cov report
