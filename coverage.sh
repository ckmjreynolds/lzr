#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ -t 1 ]]; then
  clear
fi

# Reuses the profile data from the last build.sh run; pass --fresh to re-run the tests first.
if [[ "${1:-}" == "--fresh" ]]; then
  ./build.sh cov
fi

cargo +nightly llvm-cov report --doctests --branch --open
