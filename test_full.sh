#!/bin/bash
rm /tmp/enwik9.lzr /tmp/enwik9
cargo run --release assets/enwik9 /tmp/enwik9.lzr
cargo run --release /tmp/enwik9.lzr /tmp/enwik9
cmp assets/enwik9 /tmp/enwik9
