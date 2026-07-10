#!/bin/bash
rm -f /tmp/out9.lzr
rm -f /tmp/enwik9

cargo run --release -- corpora/hutter/enwik9 /tmp/out9.lzr "$@"
cargo run --release -- -d /tmp/out9.lzr /tmp/enwik9
cmp corpora/hutter/enwik9 /tmp/enwik9
