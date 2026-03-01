//! Integration tests for the LZR public API.

use std::io::Cursor;
use std::path::Path;

use pretty_assertions::assert_eq;
use proptest::prelude::*;

// ── Helpers ──────────────────────────────────────────────────────

fn round_trip(data: &[u8]) -> Vec<u8> {
    let mut compressed = Vec::new();
    lzr::encoder::encode(Cursor::new(data), &mut compressed).expect("encode failed");
    let mut decompressed = Vec::new();
    lzr::decoder::decode(Cursor::new(&compressed), &mut decompressed).expect("decode failed");
    decompressed
}

fn compress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    lzr::encoder::encode(Cursor::new(data), &mut out).expect("encode failed");
    out
}

// ── Proptest: Small-Input Round-Trip ─────────────────────────────

proptest! {
    #[test]
    fn small_input_round_trip(data in prop::collection::vec(any::<u8>(), 0..4096)) {
        prop_assert_eq!(round_trip(&data), data);
    }
}

// ── Corpus Round-Trip (CRC32) ────────────────────────────────────

#[test]
#[ignore = "slow"]
fn corpus_round_trip() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpora");
    let mut count = 0;

    for entry in walkdir(&base) {
        let relative = entry.strip_prefix(&base).unwrap();
        let data = std::fs::read(&entry).unwrap_or_else(|e| panic!("read {}: {e}", entry.display()));

        let mut compressed = Vec::new();
        lzr::encoder::encode(Cursor::new(&data), &mut compressed)
            .unwrap_or_else(|e| panic!("encode {}: {e}", relative.display()));
        let mut result = Vec::new();
        lzr::decoder::decode(Cursor::new(&compressed), &mut result)
            .unwrap_or_else(|e| panic!("decode {}: {e}", relative.display()));

        let expected_crc = crc32fast::hash(&data);
        let actual_crc = crc32fast::hash(&result);
        assert_eq!(
            expected_crc,
            actual_crc,
            "CRC32 mismatch for {}: expected {expected_crc:#010X}, got {actual_crc:#010X}",
            relative.display(),
        );
        count += 1;
    }

    // Ensure we actually tested files (catch submodule init issues).
    assert!(count >= 50, "expected ≥50 corpus files, found {count}");
}

fn walkdir(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            files.extend(walkdir(&path));
        } else if path.is_file()
            && !path.file_name().unwrap().to_str().unwrap().starts_with('.')
            && path.extension().is_none_or(|e| e != "md")
        {
            files.push(path);
        }
    }
    files.sort();
    files
}

// ── Concatenated Streams ─────────────────────────────────────────

proptest! {
    #[test]
    fn concat_round_trip(
        a in prop::collection::vec(any::<u8>(), 0..512),
        b in prop::collection::vec(any::<u8>(), 0..512),
    ) {
        let mut stream = compress(&a);
        stream.extend_from_slice(&compress(&b));

        let mut result = Vec::new();
        lzr::decoder::decode(Cursor::new(&stream), &mut result)
            .expect("decode failed");

        let mut expected = Vec::new();
        expected.extend_from_slice(&a);
        expected.extend_from_slice(&b);
        prop_assert_eq!(result, expected);
    }
}

// ── Error Handling ───────────────────────────────────────────────

#[test]
fn error_random_bytes() {
    let garbage = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x11, 0x22, 0x33];
    let mut out = Vec::new();
    assert!(lzr::decoder::decode(Cursor::new(&garbage), &mut out).is_err());
}

#[test]
fn error_truncated_stream() {
    let compressed = compress(b"Some data to compress for truncation test");
    let truncated = &compressed[..compressed.len() / 2];
    let mut out = Vec::new();
    assert!(lzr::decoder::decode(Cursor::new(truncated), &mut out).is_err());
}

#[test]
fn error_corrupted_checksum() {
    let mut compressed = compress(b"Test data for checksum corruption");
    let len = compressed.len();
    compressed[len - 1] ^= 0xFF;
    let mut out = Vec::new();
    assert!(lzr::decoder::decode(Cursor::new(&compressed), &mut out).is_err());
}

// ── Compression Sanity ───────────────────────────────────────────

#[test]
fn compressible_data_shrinks() {
    let data = vec![b'A'; 100_000];
    let compressed = compress(&data);
    assert!(
        compressed.len() < data.len(),
        "repetitive data should compress: {} vs {} original",
        compressed.len(),
        data.len(),
    );
}
