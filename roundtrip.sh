#!/bin/bash
rm -f /tmp/out8.lzr
rm -f /tmp/enwik8

cargo run --release -- corpora/hutter/enwik8 /tmp/out8.lzr "$@"
cargo run --release -- -d /tmp/out8.lzr /tmp/enwik8
cmp corpora/hutter/enwik8 /tmp/enwik8
