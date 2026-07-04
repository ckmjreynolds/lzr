//! Public-API round-trip tests over real corpus files.
//!
//! These exercise `lzr::compress`/`lzr::decompress` in-process (the `tests/cli.rs`
//! suite covers the subprocess path). Corpora are located relative to the crate
//! root and skipped when absent, so a packaged-crate build stays green.

#![expect(clippy::unwrap_used, reason = "unwrap is the idiomatic failure mode in tests")]

mod common;

use common::corpus;

fn assert_roundtrip(data: &[u8]) {
    let packed = lzr::compress(data);
    let restored = lzr::decompress(&packed).unwrap();
    assert_eq!(restored.len(), data.len(), "length mismatch");
    assert_eq!(restored, data, "round-trip mismatch");
}

#[test]
fn roundtrip_text_slice() {
    let Some(data) = corpus("corpora/large/bible.txt") else {
        return;
    };
    assert_roundtrip(&data[..data.len().min(256 * 1024)]);
}

#[test]
fn roundtrip_prose_slice() {
    let Some(data) = corpus("corpora/calgary/book1") else {
        return;
    };
    assert_roundtrip(&data[..data.len().min(128 * 1024)]);
}

#[test]
fn roundtrip_binary_slice() {
    // A non-text corpus exercises the full 0..=255 byte range through the tokenizer.
    let Some(data) = corpus("corpora/canterbury/kennedy.xls") else {
        return;
    };
    assert_roundtrip(&data[..data.len().min(128 * 1024)]);
}

#[test]
fn compression_shrinks_text() {
    let Some(data) = corpus("corpora/large/bible.txt") else {
        return;
    };
    let slice = &data[..data.len().min(256 * 1024)];
    let packed = lzr::compress(slice);
    assert!(packed.len() < slice.len(), "expected compression, {} -> {}", slice.len(), packed.len());
}
