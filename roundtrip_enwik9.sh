#!/bin/bash
rm -f /tmp/out.lzr
rm -f /tmp/enwik8

cargo run --release -- corpora/hutter/enwik9 /tmp/out.lzr "$@"
cargo run --release -- -d /tmp/out.lzr /tmp/enwik9
cmp corpora/hutter/enwik9 /tmp/enwik9
