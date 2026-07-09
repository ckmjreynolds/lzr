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
    // The default profile already enables the swept-in model set (order-0..11, sparse2, match); a bare
    // `lzr src dst` exercises them. The explicit enables below are redundant with the defaults but kept
    // so the roundtrip tests stay pinned to the dense order-0 plus hashed order-N path regardless of
    // any future default change.
    let _ok = Command::cargo_bin("lzr")
        .unwrap()
        .arg(src)
        .arg(dst)
        .args(["--enable", "order0", "--enable", "order1", "--enable", "order2"])
        .assert()
        .success();
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

/// A `.lzr` input is decompressed without an explicit `-d` flag: the direction is
/// inferred from the extension.
#[test]
fn lzr_extension_infers_decompression() {
    let original = b"the quick brown fox jumps over the lazy dog";
    let raw = temp_path("infer.bin");
    let packed = temp_path("infer.lzr");
    let back = temp_path("infer.out");

    std::fs::write(&raw, original).unwrap();
    compress(&raw, &packed);
    // No `-d`: the `.lzr` extension alone must drive decompression.
    let _ok = Command::cargo_bin("lzr").unwrap().arg(&packed).arg(&back).assert().success();

    assert_eq!(std::fs::read(&back).unwrap(), original, "extension-inferred round-trip mismatch");

    for file in [raw, packed, back] {
        drop(std::fs::remove_file(file));
    }
}

/// `-z`/`--compress` forces compression even when the input carries the `.lzr` extension that
/// would otherwise infer decompression.
#[test]
fn compress_flag_overrides_lzr_extension() {
    let original = b"the quick brown fox jumps over the lazy dog";
    let raw = temp_path("force.bin");
    let packed = temp_path("force.lzr");
    // Compress `raw` into a genuine stream that happens to be named `.lzr`.
    std::fs::write(&raw, original).unwrap();
    compress(&raw, &packed);

    // `-z` on the `.lzr`-named input must compress it again rather than decompress it.
    let double = temp_path("force.double.lzr");
    let _ok = Command::cargo_bin("lzr")
        .unwrap()
        .arg("-z")
        .arg(&packed)
        .arg(&double)
        .args(["--enable", "order0"])
        .assert()
        .success();

    // Decompressing once must return the (still-compressed) `.lzr` stream, proving `-z` compressed.
    let back = temp_path("force.out");
    let _ok = Command::cargo_bin("lzr").unwrap().arg("-d").arg(&double).arg(&back).assert().success();
    assert_eq!(std::fs::read(&back).unwrap(), std::fs::read(&packed).unwrap(), "-z did not force compression");

    for file in [raw, packed, double, back] {
        drop(std::fs::remove_file(file));
    }
}

/// Compression with pipeline options must still round-trip: a small `--block-size` (forcing many
/// blocks) and enabled models, both reconstruct the original (the decoder reads the pipeline per
/// block from the stream).
#[test]
fn roundtrip_with_pipeline_options() {
    let original = b"the quick brown fox jumps over the lazy dog. ".repeat(200);
    for (tag, args) in [
        ("small_blocks", vec!["--block-size", "512"]),
        ("suffix_blocks", vec!["--block-size", "1K"]),
        ("models", vec!["--enable", "order2", "--enable", "match"]),
        ("both", vec!["--block-size", "256", "--enable", "order1"]),
    ] {
        let raw = temp_path(&format!("{tag}.bin"));
        let packed = temp_path(&format!("{tag}.lzr"));
        let back = temp_path(&format!("{tag}.out"));
        std::fs::write(&raw, &original).unwrap();

        let mut cmd = Command::cargo_bin("lzr").unwrap();
        let _ = cmd.arg(&raw).arg(&packed).args(&args);
        let _ok = cmd.assert().success();

        decompress(&packed, &back);
        assert_eq!(std::fs::read(&back).unwrap(), original, "round-trip mismatch: {tag}");

        for file in [raw, packed, back] {
            drop(std::fs::remove_file(file));
        }
    }
}

/// An invalid `--block-size` must fail cleanly (validated before any work).
#[test]
fn invalid_block_size_fails() {
    let raw = temp_path("badbs.bin");
    let packed = temp_path("badbs.lzr");
    std::fs::write(&raw, b"data").unwrap();
    let _ok = Command::cargo_bin("lzr").unwrap().arg(&raw).arg(&packed).arg("--block-size").arg("0").assert().failure();
    drop(std::fs::remove_file(raw));
    drop(std::fs::remove_file(packed));
}

/// Re-Pair is optional now: `--disable repair` must succeed and still round-trip (byte-level stream).
#[test]
fn disabling_repair_round_trips() {
    let raw = temp_path("norepair.bin");
    let packed = temp_path("norepair.lzr");
    let back = temp_path("norepair.out");
    let original = b"the quick brown fox the quick brown fox the quick brown fox";
    std::fs::write(&raw, original).unwrap();
    let _ok = Command::cargo_bin("lzr")
        .unwrap()
        .arg(&raw)
        .arg(&packed)
        .args(["--disable", "repair", "--enable", "order0", "--enable", "order1"])
        .assert()
        .success();
    decompress(&packed, &back);
    assert_eq!(std::fs::read(&back).unwrap(), original);
    for file in [raw, packed, back] {
        drop(std::fs::remove_file(file));
    }
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
