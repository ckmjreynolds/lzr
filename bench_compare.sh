#!/usr/bin/env bash
# bench_compare.sh — Compare LZR against lz4 and gzip.
# Usage: bash bench_compare.sh <file>

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <file>" >&2
    exit 1
fi

FILE="$1"
if [ ! -f "$FILE" ]; then
    echo "Error: file not found: $FILE" >&2
    exit 1
fi

ORIG_SIZE=$(wc -c < "$FILE")
TMPDIR="${TMPDIR:-/tmp}"
BASENAME=$(basename "$FILE")

# Build LZR in release mode.
cargo build --release --quiet 2>/dev/null
LZR=./target/release/lzr

# Format an integer with comma separators.
commas() { printf "%'d" "$1"; }

printf "%-20s %14s %14s %8s %16s %16s\n" "Tool" "Original" "Compressed" "Ratio" "Compress" "Decompress"
printf "%-20s %14s %14s %8s %16s %16s\n" "----" "--------" "----------" "-----" "--------" "----------"

bench() {
    local label="$1"
    local compress_cmd="$2"
    local decompress_cmd="$3"

    local comp_file="$TMPDIR/bench_${BASENAME}.compressed"

    # Compress and time.
    local t0 t1 comp_time comp_size decomp_time
    t0=$(perl -MTime::HiRes=time -e 'printf "%.6f", time')
    eval "$compress_cmd" < "$FILE" > "$comp_file"
    t1=$(perl -MTime::HiRes=time -e 'printf "%.6f", time')
    comp_time=$(perl -e "printf '%.6f', $t1 - $t0")
    comp_size=$(wc -c < "$comp_file")

    # Decompress and time.
    t0=$(perl -MTime::HiRes=time -e 'printf "%.6f", time')
    eval "$decompress_cmd" < "$comp_file" > /dev/null
    t1=$(perl -MTime::HiRes=time -e 'printf "%.6f", time')
    decomp_time=$(perl -e "printf '%.6f', $t1 - $t0")

    local ratio comp_mbs decomp_mbs
    ratio=$(perl -e "printf '%.1f', (1 - $comp_size / $ORIG_SIZE) * 100")
    comp_mbs=$(perl -e "printf '%.1f', $ORIG_SIZE / (1024*1024) / ($comp_time > 0 ? $comp_time : 0.001)")
    decomp_mbs=$(perl -e "printf '%.1f', $ORIG_SIZE / (1024*1024) / ($decomp_time > 0 ? $decomp_time : 0.001)")

    printf "%-20s %14s %14s %7s%% %10s MiB/s %10s MiB/s\n" \
        "$label" "$(commas "$ORIG_SIZE")" "$(commas "$comp_size")" \
        "$ratio" "$comp_mbs" "$decomp_mbs"
    rm -f "$comp_file"
}

bench "lzr -1 ST"  "$LZR -T 1 -l 1"  "$LZR -T 1 -d"
bench "lzr -1 MT"  "$LZR -T 0 -l 1"  "$LZR -T 0 -d"
bench "lzr -9 ST"  "$LZR -T 1 -l 9"  "$LZR -T 1 -d"
bench "lzr -9 MT"  "$LZR -T 0 -l 9"  "$LZR -T 0 -d"

if command -v lz4 &>/dev/null; then
    bench "lz4 -1 ST"  "lz4 -1 -T1 -c"  "lz4 -d -c"
    bench "lz4 -1 MT"  "lz4 -1 -c"      "lz4 -d -c"
    bench "lz4 -9 ST"  "lz4 -9 -T1 -c"  "lz4 -d -c"
    bench "lz4 -9 MT"  "lz4 -9 -c"      "lz4 -d -c"
fi
