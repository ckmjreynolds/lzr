#!/bin/bash
rm /tmp/enwik8.lzr /tmp/enwik8
cargo run --release assets/enwik8 /tmp/enwik8.lzr
cargo run --release /tmp/enwik8.lzr /tmp/enwik8
cmp assets/enwik8 /tmp/enwik8
