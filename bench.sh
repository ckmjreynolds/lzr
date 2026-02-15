#!/bin/bash
set -euo pipefail

DIR="${1:-corpora}"

cargo build --release 2>/dev/null
LZR="./target/release/lzr"

TOOLS=("${@:2}")
if [[ ${#TOOLS[@]} -eq 0 ]]; then
    TOOLS=("gzip" "zstd" "lz4" "$LZR -1")
fi

ORIGINAL=$(tar cf - "$DIR" | wc -c | tr -d ' ')

echo "Source: $DIR ($(numfmt --grouping "$ORIGINAL") bytes tarred)"

run_tool() {
    local entry="$1"
    read -ra parts <<< "$entry"
    local bin="${parts[0]}"
    local flags=()
    if [[ ${#parts[@]} -gt 1 ]]; then
        flags=("${parts[@]:1}")
    fi
    local name
    name=$(basename "$bin")
    if [[ ${#flags[@]} -gt 0 ]]; then
        name="$name ${flags[*]}"
    fi
    # Export for caller
    _BIN="$bin"
    _FLAGS=("${flags[@]+"${flags[@]}"}")
    _NAME="$name"
}

echo ""
echo "=== Compression Ratio ==="
printf "%-20s %11s %11s\n" "Tool" "Bytes" "Ratio"
printf "%-20s %11s %11s\n" "----" "-----" "-----"

for entry in "${TOOLS[@]}"; do
    run_tool "$entry"
    compressed=$(tar cf - "$DIR" | "$_BIN" "${_FLAGS[@]+"${_FLAGS[@]}"}" | wc -c | tr -d ' ')
    ratio=$(echo "scale=4; (1 - ($compressed / $ORIGINAL)) * 100" | bc)
    ratio="${ratio%??}"
    printf "%-20s %11s %10s%%\n" "$_NAME" "$(numfmt --grouping "$compressed")" "$ratio"
done

echo ""
echo "=== Compression Speed ==="
printf "%-20s %11s\n" "Tool" "Real"
printf "%-20s %11s\n" "----" "----"
for entry in "${TOOLS[@]}"; do
    run_tool "$entry"
    real=$({ time tar cf - "$DIR" | "$_BIN" "${_FLAGS[@]+"${_FLAGS[@]}"}" > /dev/null; } 2>&1 | grep real | awk '{print $2}')
    printf "%-20s %11s\n" "$_NAME" "$real"
done

echo ""
echo "=== Round-trip Speed ==="
printf "%-20s %11s\n" "Tool" "Real"
printf "%-20s %11s\n" "----" "----"
for entry in "${TOOLS[@]}"; do
    run_tool "$entry"
    real=$({ time tar cf - "$DIR" | "$_BIN" "${_FLAGS[@]+"${_FLAGS[@]}"}" | "$_BIN" -d > /dev/null; } 2>&1 | grep real | awk '{print $2}')
    printf "%-20s %11s\n" "$_NAME" "$real"
done
