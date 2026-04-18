#!/bin/bash
#
# benchmark.sh — head-to-head compression benchmark.
#
# Usage: bash benchmark.sh [INPUT]
#
# Defaults to ./corpora.tar if no INPUT is given. Runs lzr levels 1-4,
# lz4 levels 1 and 9 (single-threaded), and gzip levels 1 and 9; reports
# compressed size, saved% (higher = better), and encode/decode MB/s.
#
set -euo pipefail

INPUT="${1:-corpora.tar}"
if [[ ! -f "$INPUT" ]]; then
    echo "Error: input file not found: $INPUT" >&2
    exit 1
fi

# --- Locate the lzr binary (prefer the local release build) -----------------
LZR=""
if [[ -x "target/release/lzr" ]]; then
    LZR="target/release/lzr"
elif command -v lzr >/dev/null 2>&1; then
    LZR="$(command -v lzr)"
else
    echo "Building release lzr..." >&2
    cargo +stable build --release --bin lzr >&2
    LZR="target/release/lzr"
fi

for tool in lz4 gzip gunzip cmp perl awk; do
    command -v "$tool" >/dev/null 2>&1 || { echo "Error: required tool not found: $tool" >&2; exit 1; }
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# --- Portable helpers --------------------------------------------------------
if stat -f%z "$INPUT" >/dev/null 2>&1; then
    STAT_SIZE='stat -f%z'
else
    STAT_SIZE='stat -c%s'
fi

file_size() { $STAT_SIZE "$1"; }
now()       { perl -MTime::HiRes=time -e 'printf "%.6f\n", time'; }

RAW=$(file_size "$INPUT")
RAW_MB=$(awk -v r="$RAW" 'BEGIN{printf "%.2f", r/1048576}')

# --- Benchmark one (tool, level) --------------------------------------------
# $1 label   $2 level   $3 encode cmd   $4 decode cmd   $5 compressed path
# $6 decoded path
bench() {
    local tool="$1" level="$2" enc="$3" dec="$4" out="$5" roundtrip="$6"

    local t0 t1
    t0=$(now); eval "$enc"; t1=$(now)
    local enc_time comp_size ratio enc_mbs
    enc_time=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.4f", b-a}')
    comp_size=$(file_size "$out")
    ratio=$(awk -v c="$comp_size" -v r="$RAW" 'BEGIN{printf "%.1f", (1-c/r)*100}')
    enc_mbs=$(awk -v r="$RAW" -v t="$enc_time" 'BEGIN{printf "%.1f", r/1048576/t}')

    t0=$(now); eval "$dec"; t1=$(now)
    local dec_time dec_mbs
    dec_time=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.4f", b-a}')
    dec_mbs=$(awk -v r="$RAW" -v t="$dec_time" 'BEGIN{printf "%.1f", r/1048576/t}')

    if ! cmp -s "$INPUT" "$roundtrip"; then
        echo "[MISMATCH] $tool L$level decompressed output differs from input" >&2
        exit 1
    fi

    local comp_mb
    comp_mb=$(awk -v s="$comp_size" 'BEGIN{printf "%.2f", s/1048576}')
    printf '%-5s  %5s  %13s  %8s  %14s  %14s\n' \
        "$tool" "$level" "$comp_mb" "${ratio}%" "$enc_mbs" "$dec_mbs"
}

# --- Run ---------------------------------------------------------------------
echo "Input: $INPUT ($RAW_MB MB, $RAW bytes)"
echo
printf '%-5s  %5s  %13s  %8s  %14s  %14s\n' \
    "tool" "level" "comp (MB)" "saved%" "encode (MB/s)" "decode (MB/s)"
printf '%-5s  %5s  %13s  %8s  %14s  %14s\n' \
    "-----" "-----" "-------------" "--------" "--------------" "--------------"

for L in $(seq 1 9); do
    out="$TMP/lzr.L$L.lzr"; dec="$TMP/lzr.L$L.raw"
    bench lzr "$L" \
        "\"$LZR\" compress --level $L \"$INPUT\" \"$out\"" \
        "\"$LZR\" decompress \"$out\" \"$dec\"" \
        "$out" "$dec"
done

for L in 1 9; do
    out="$TMP/lz4.L$L.lz4"; dec="$TMP/lz4.L$L.raw"
    bench lz4 "$L" \
        "lz4 -$L -T1 -q -f \"$INPUT\" \"$out\"" \
        "lz4 -d -q -f \"$out\" \"$dec\"" \
        "$out" "$dec"
done

for L in 1 9; do
    out="$TMP/gzip.L$L.gz"; dec="$TMP/gzip.L$L.raw"
    bench gzip "$L" \
        "gzip -$L -c \"$INPUT\" > \"$out\"" \
        "gunzip -c \"$out\" > \"$dec\"" \
        "$out" "$dec"
done
