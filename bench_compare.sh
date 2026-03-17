#!/usr/bin/env bash
set -euo pipefail

# Benchmark three match finder implementations + gzip + lz4
# Usage: bash bench_compare.sh

TESTFILE="corpora/large/bible.txt"
ORIGINAL_SIZE=$(wc -c < "$TESTFILE")
ORIGINAL_SHA=$(shasum < "$TESTFILE" | awk '{print $1}')
TMPOUT=$(mktemp)
trap 'rm -f "$TMPOUT"' EXIT

printf "Input: %s (%s bytes)\n\n" "$TESTFILE" "$ORIGINAL_SIZE"

# ── Helper ──────────────────────────────────────────────────────────────────
bench_lzr() {
    local label="$1" level="$2"
    # Compress: measure time and size
    local t0 t1 elapsed size speed
    t0=$(python3 -c 'import time; print(time.monotonic())')
    cargo run --release -q -- --level "$level" < "$TESTFILE" > "$TMPOUT" 2>/dev/null
    t1=$(python3 -c 'import time; print(time.monotonic())')
    elapsed=$(python3 -c "print(f'{$t1 - $t0:.3f}')")
    size=$(wc -c < "$TMPOUT")
    speed=$(python3 -c "t=$t1-$t0; print(f'{$ORIGINAL_SIZE/t/1048576:.1f}') if t>0 else print('inf')")
    ratio=$(python3 -c "print(f'{$size/$ORIGINAL_SIZE*100:.1f}')")

    # Verify roundtrip
    local rt_sha
    rt_sha=$(cargo run --release -q -- -d < "$TMPOUT" 2>/dev/null | shasum | awk '{print $1}')
    local ok="OK"
    [[ "$rt_sha" == "$ORIGINAL_SHA" ]] || ok="FAIL"

    printf "%-28s  %8d  %5s%%  %6s MiB/s  [%s]\n" "$label" "$size" "$ratio" "$speed" "$ok"
}

bench_ext() {
    local label="$1" compress_cmd="$2" decompress_cmd="$3"
    local t0 t1 elapsed size speed
    t0=$(python3 -c 'import time; print(time.monotonic())')
    eval "$compress_cmd" > "$TMPOUT"
    t1=$(python3 -c 'import time; print(time.monotonic())')
    size=$(wc -c < "$TMPOUT")
    speed=$(python3 -c "t=$t1-$t0; print(f'{$ORIGINAL_SIZE/t/1048576:.1f}') if t>0 else print('inf')")
    ratio=$(python3 -c "print(f'{$size/$ORIGINAL_SIZE*100:.1f}')")

    # Verify roundtrip
    local rt_sha
    rt_sha=$(eval "$decompress_cmd" < "$TMPOUT" | shasum | awk '{print $1}')
    local ok="OK"
    [[ "$rt_sha" == "$ORIGINAL_SHA" ]] || ok="FAIL"

    printf "%-28s  %8d  %5s%%  %6s MiB/s  [%s]\n" "$label" "$size" "$ratio" "$speed" "$ok"
}

# ── Patch helper ────────────────────────────────────────────────────────────
set_finder() {
    local finder="$1"
    case "$finder" in
        MatchFinder)
            sed -i '' 's/use crate::matchfinder::[A-Za-z]*;/use crate::matchfinder::MatchFinder;/' src/encode.rs
            sed -i '' 's/let mut mf = [A-Za-z]*::new(options);/let mut mf = MatchFinder::new(options);/' src/encode.rs
            ;;
        BTreeMatchFinder)
            sed -i '' 's/use crate::matchfinder::[A-Za-z]*;/use crate::matchfinder::BTreeMatchFinder;/' src/encode.rs
            sed -i '' 's/let mut mf = [A-Za-z]*::new(options);/let mut mf = BTreeMatchFinder::new(options);/' src/encode.rs
            ;;
        HashMapMatchFinder)
            sed -i '' 's/use crate::matchfinder::[A-Za-z]*;/use crate::matchfinder::HashMapMatchFinder;/' src/encode.rs
            sed -i '' 's/let mut mf = [A-Za-z]*::new(options);/let mut mf = HashMapMatchFinder::new(options);/' src/encode.rs
            ;;
    esac
}

# ── Build all three variants (release) ──────────────────────────────────────
printf "%-28s  %8s  %5s   %6s         %s\n" "Implementation" "Size" "Ratio" "Speed" "Check"
printf "%s\n" "--------------------------------------------------------------------------"

for finder in HashMapMatchFinder BTreeMatchFinder MatchFinder; do
    set_finder "$finder"
    cargo build --release -q 2>/dev/null
    for level in 1 5 9; do
        bench_lzr "lzr $finder L$level" "$level"
    done
    echo
done

# ── External tools ──────────────────────────────────────────────────────────
for level in 1 9; do
    bench_ext "gzip -$level" "gzip -${level}c < $TESTFILE" "gzip -dc"
done
echo

for level in 1 9; do
    bench_ext "lz4 -$level" "lz4 -${level}c --no-frame-crc < $TESTFILE" "lz4 -dc"
done

# ── Restore to HashMap ─────────────────────────────────────────────────────
set_finder HashMapMatchFinder
