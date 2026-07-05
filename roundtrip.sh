#!/bin/bash
export TERM="${TERM:-dumb}"
if [[ "${TERM}" != "dumb" ]]; then
  clear
fi

rm -f /tmp/out.lzr
rm -f /tmp/enwik8

cargo run --release corpora/hutter/enwik8 /tmp/out.lzr
cargo run --release -- -d /tmp/out.lzr /tmp/enwik8
cmp corpora/hutter/enwik8 /tmp/enwik8
