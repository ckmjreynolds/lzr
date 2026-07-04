//! End-to-end CLI round-trip tests: compress a file, decompress it, and assert
//! the result is byte-identical to the original.
//!
//! Direction is the `-d` flag (compress is the default), isolated in the
//! `compress`/`decompress` helpers so the tests are robust to CLI changes.

#![expect(clippy::unwrap_used, reason = "unwrap is the idiomatic failure mode in tests")]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use assert_cmd::Command;

mod common;

use common::corpus;

/// A unique temp path for this test process (parallel-safe; no `tempfile` dep).
fn temp_path(tag: &str) -> PathBuf {
    static N: AtomicU32 = AtomicU32::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("lzr-it-{}-{n}-{tag}", std::process::id()))
}

fn compress(src: &Path, dst: &Path) {
    let _ok = Command::cargo_bin("lzr").unwrap().arg(src).arg(dst).assert().success();
}

fn decompress(src: &Path, dst: &Path) {
    let _ok = Command::cargo_bin("lzr").unwrap().arg("-d").arg(src).arg(dst).assert().success();
}

fn assert_cli_roundtrip(original: &[u8], tag: &str) {
    let raw = temp_path(&format!("{tag}.bin"));
    let packed = temp_path(&format!("{tag}.lzr"));
    let back = temp_path(&format!("{tag}.out"));

    std::fs::write(&raw, original).unwrap();
    compress(&raw, &packed);
    decompress(&packed, &back);

    let result = std::fs::read(&back).unwrap();
    assert_eq!(result.len(), original.len(), "length mismatch: {tag}");
    assert_eq!(result, original, "round-trip mismatch: {tag}");

    for file in [raw, packed, back] {
        drop(std::fs::remove_file(file));
    }
}

#[test]
fn roundtrip_edge_cases() {
    assert_cli_roundtrip(b"", "empty");
    assert_cli_roundtrip(b"A", "single_byte");
    assert_cli_roundtrip(&[0u8; 4096], "zeros");
    assert_cli_roundtrip(b"the quick brown fox jumps over the lazy dog", "short_text");
}

#[test]
fn roundtrip_repetitive() {
    let Some(data) = corpus("corpora/artificial/aaa.txt") else {
        return;
    };
    assert_cli_roundtrip(&data, "aaa");
}

#[test]
fn roundtrip_random() {
    let Some(data) = corpus("corpora/artificial/random.txt") else {
        return;
    };
    // Incompressible input: cap the slice so the (unoptimized) test binary stays fast.
    assert_cli_roundtrip(&data[..data.len().min(128 * 1024)], "random");
}

#[test]
fn roundtrip_bible() {
    let Some(data) = corpus("corpora/large/bible.txt") else {
        return;
    };
    // Integration tests build the dev binary; a slice keeps this well under a second.
    assert_cli_roundtrip(&data[..data.len().min(256 * 1024)], "bible");
}

#[test]
fn decompress_garbage_fails_cleanly() {
    let bad = temp_path("bad.lzr");
    std::fs::write(&bad, b"not a real lzr stream at all").unwrap();
    let out = temp_path("bad.out");
    let _ok = Command::cargo_bin("lzr").unwrap().arg("-d").arg(&bad).arg(&out).assert().failure();
    drop(std::fs::remove_file(bad));
    drop(std::fs::remove_file(out));
}
