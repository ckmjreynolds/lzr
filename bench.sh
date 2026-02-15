#!/bin/bash
set -euo pipefail

DIR="${1:-corpora}"

cargo build --release 2>/dev/null
LZR="./target/release/lzr"

ORIGINAL=$(tar cf - "$DIR" | wc -c | tr -d ' ')

echo "Source: $DIR ($(numfmt --grouping "$ORIGINAL") bytes tarred)"

echo ""
echo "=== Compression Ratio ==="
printf "%-11s %11s %11s\n" "Tool" "Bytes" "Ratio"
printf "%-11s %11s %11s\n" "----" "-----" "-----"

for tool in gzip zstd lz4 "$LZR"; do
    name=$(basename "$tool")
    flags=""; [[ "$name" == "zstd" ]] && flags="--single-thread"
    compressed=$(tar cf - "$DIR" | "$tool" $flags | wc -c | tr -d ' ')
    ratio=$(echo "scale=1; (1 - $compressed / $ORIGINAL) * 100" | bc)
    printf "%-11s %11s %11s\n" "$name" "$(numfmt --grouping "$compressed")" "$ratio""%"
done

echo ""
echo "=== Compression Speed ==="
printf "%-11s %11s\n" "Tool" "Real"
printf "%-11s %11s\n" "----" "----"
for tool in gzip zstd lz4 "$LZR"; do
    name=$(basename "$tool")
    flags=""; [[ "$name" == "zstd" ]] && flags="--single-thread"
    real=$({ time tar cf - "$DIR" | "$tool" $flags > /dev/null; } 2>&1 | grep real | awk '{print $2}')
    printf "%-11s %11s\n" "$name" "$real"
done

echo ""
echo "=== Round-trip Speed ==="
printf "%-11s %11s\n" "Tool" "Real"
printf "%-11s %11s\n" "----" "----"
for tool in gzip zstd lz4 "$LZR"; do
    name=$(basename "$tool")
    flags=""; [[ "$name" == "zstd" ]] && flags="--single-thread"
    real=$({ time tar cf - "$DIR" | "$tool" $flags | "$tool" -d $flags > /dev/null; } 2>&1 | grep real | awk '{print $2}')
    printf "%-11s %11s\n" "$name" "$real"
done
