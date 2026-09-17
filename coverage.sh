#!/bin/bash
export TERM="${TERM:-dumb}"
if [[ "${TERM}" != "dumb" ]]; then
  clear
fi

# Use nightly to get coverage reports.
cargo +nightly llvm-cov report --open
