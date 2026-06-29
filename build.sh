#!/bin/bash
export TERM="${TERM:-dumb}"
if [[ "${TERM}" != "dumb" ]]; then
  clear
fi

# Whole build runs on nightly: coverage wants nightly
# (`llvm-cov --doctests --branch`), and any future SIMD intrinsics will
# likely need it too. Using one toolchain everywhere avoids split-build
# inconsistencies.
TC="+nightly"

cargo $TC fmt || exit

# Submission (default features): must stay clippy-clean.
cargo $TC clippy --all-targets -- -Dwarnings || exit

# Bare-bones build (no features) also compiles clippy-clean.
cargo $TC clippy --no-default-features --all-targets -- -Dwarnings || exit

# Opt-in online-neural arm (`--features arm`): not shipped, but must stay
# clippy-clean and pass its correctness tests (gradient check + round-trip).
cargo $TC clippy --features arm --all-targets -- -Dwarnings || exit

# Opt-in article-reorder preprocessor (`--features reorder`): not shipped (off by
# default), but must stay clippy-clean and pass its pipeline round-trip test.
cargo $TC clippy --features reorder --all-targets -- -Dwarnings || exit

# CLAUDE.md invariant: the submission binary must not pull in a threading
# crate. `cargo tree` splits its output into the main dep tree and a
# `[dev-dependencies]` section; we only care about the main tree.
#
# `-e=no-dev` excludes dev-dependencies entirely, leaving only the deps
# that end up in the release binary.
if cargo $TC tree --features submission -e=no-dev 2>/dev/null | \
     grep -E '^[│ ]*[├└]── (rayon|crossbeam|gemm|tokio|async-std)' ; then
  echo "ERROR: submission build pulled in a threading dependency" >&2
  exit 1
fi

# Tests (default = fast, arm-free).
cargo $TC test --release -- --nocapture 2>&1 || exit

# Arm correctness: gradient check + byte-exact round-trip (small inputs, fast).
cargo $TC test --release --features arm -- --nocapture 2>&1 || exit

# Reorder correctness: full-pipeline byte-exact round-trip on an enwik8 slice.
cargo $TC test --release --features reorder -- --nocapture 2>&1 || exit

# Release binary.
cargo $TC build --release || exit

# Coverage.
cargo $TC llvm-cov --doctests --branch --lcov --quiet --output-path lcov.info || exit
cargo $TC llvm-cov report
