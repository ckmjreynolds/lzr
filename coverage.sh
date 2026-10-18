#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ -t 1 ]]; then
  clear
fi

# Reuses the profile data from the last build.sh run; pass --fresh to re-run the tests first.
if [[ "${1:-}" == "--fresh" ]]; then
  cargo +nightly llvm-cov clean --workspace
  cargo +nightly llvm-cov --locked --doctests --branch --no-report
fi

cargo +nightly llvm-cov report --doctests --branch --open
