//! Integration tests: Writer → Reader roundtrip.

use std::fmt::Write as _;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};

use lzr::{Reader, Writer};
use pretty_assertions::assert_eq;

mod common;

/// Write data, seal, read back, verify.
fn roundtrip(data: &[u8]) -> Vec<u8> {
    let buf = Vec::new();
    let mut w = Writer::new(buf).unwrap();
    w.write_all(data).unwrap();
    let buf = w.seal().unwrap();

    let cursor = Cursor::new(buf);
    let mut r = Reader::new(cursor).unwrap();
    assert_eq!(r.len(), data.len() as u64);
    assert!(r.is_sealed());

    let mut output = Vec::new();
    r.read_to_end(&mut output).unwrap();
    assert_eq!(data, &output[..]);
    output
}

#[test]
fn empty() {
    roundtrip(b"");
}

#[test]
fn small() {
    roundtrip(b"Hello, world!\n");
}

#[test]
fn medium() {
    #[allow(clippy::cast_possible_truncation)]
    let data: Vec<u8> = (0..10_000u16).map(|i| (i % 256) as u8).collect();
    roundtrip(&data);
}

#[test]
fn lipsum_1kb() {
    let data = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
                 Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \
                 Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. ";
    let mut big = Vec::new();
    for _ in 0..5 {
        big.extend_from_slice(data);
    }
    roundtrip(&big);
}

#[test]
fn line_count() {
    let data = b"line 1\nline 2\nline 3\n";
    let buf = Vec::new();
    let mut w = Writer::new(buf).unwrap();
    w.write_all(data).unwrap();
    let buf = w.seal().unwrap();

    let r = Reader::new(Cursor::new(buf)).unwrap();
    assert_eq!(r.lines(), 3);
    assert_eq!(r.len(), data.len() as u64);
}

#[test]
fn line_count_in_matched_regions() {
    // Repeated identical lines force LZ77 to match across newlines,
    // exercising the decoder-mirror path in the writer's track_token().
    let line = b"the quick brown fox jumps over the lazy dog\n";
    let mut data = Vec::new();
    for _ in 0..100 {
        data.extend_from_slice(line);
    }

    let buf = Vec::new();
    let mut w = Writer::new(buf).unwrap();
    w.write_all(&data).unwrap();
    let buf = w.seal().unwrap();

    let r = Reader::new(Cursor::new(buf)).unwrap();
    assert_eq!(r.lines(), 100);
    assert_eq!(r.len(), data.len() as u64);
}

#[test]
fn flush_creates_frame() {
    let part1 = b"first frame\n";
    let part2 = b"second frame\n";

    let buf = Vec::new();
    let mut w = Writer::new(buf).unwrap();
    w.write_all(part1).unwrap();
    w.flush().unwrap();
    w.write_all(part2).unwrap();
    let buf = w.seal().unwrap();

    let mut r = Reader::new(Cursor::new(buf)).unwrap();
    let mut output = Vec::new();
    r.read_to_end(&mut output).unwrap();

    let mut expected = Vec::new();
    expected.extend_from_slice(part1);
    expected.extend_from_slice(part2);
    assert_eq!(expected, output);
}

#[test]
fn seek_by_byte() {
    let data = b"0123456789abcdef";
    let buf = Vec::new();
    let mut w = Writer::new(buf).unwrap();
    w.write_all(data).unwrap();
    let buf = w.seal().unwrap();

    let mut r = Reader::new(Cursor::new(buf)).unwrap();

    // Seek to middle.
    r.seek(SeekFrom::Start(8)).unwrap();
    let mut out = vec![0u8; 8];
    r.read_exact(&mut out).unwrap();
    assert_eq!(&out, b"89abcdef");

    // Seek to start.
    r.seek(SeekFrom::Start(0)).unwrap();
    let mut out = vec![0u8; 4];
    r.read_exact(&mut out).unwrap();
    assert_eq!(&out, b"0123");

    // Seek from end.
    r.seek(SeekFrom::End(-4)).unwrap();
    let mut out = vec![0u8; 4];
    r.read_exact(&mut out).unwrap();
    assert_eq!(&out, b"cdef");
}

#[test]
fn seek_to_line() {
    let data = b"line 0\nline 1\nline 2\nline 3\n";
    let buf = Vec::new();
    let mut w = Writer::new(buf).unwrap();
    w.write_all(data).unwrap();
    let buf = w.seal().unwrap();

    let mut r = Reader::new(Cursor::new(buf)).unwrap();
    assert_eq!(r.lines(), 4);

    // Seek to line 0 → offset 0.
    let offset = r.seek_to_line(0).unwrap();
    assert_eq!(offset, 0);

    // Seek to line 2 → should be after "line 0\nline 1\n".
    let offset = r.seek_to_line(2).unwrap();
    assert_eq!(offset, 14); // "line 0\n" (7) + "line 1\n" (7) = 14
    let mut line = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        let n = r.read(&mut byte).unwrap();
        if n == 0 || byte[0] == b'\n' {
            break;
        }
        line.push(byte[0]);
    }
    assert_eq!(&line, b"line 2");
}

#[test]
fn seal_roundtrip() {
    let data = b"sealed data\n";
    let buf = Vec::new();
    let mut w = Writer::new(buf).unwrap();
    w.write_all(data).unwrap();
    let buf = w.seal().unwrap();

    let r = Reader::new(Cursor::new(&buf)).unwrap();
    assert!(r.is_sealed());

    // Trying to open a sealed file for appending should fail.
    let cursor = Cursor::new(buf);
    let result = Writer::open(cursor);
    assert!(result.is_err());
}

#[test]
fn multiple_writes() {
    let buf = Vec::new();
    let mut w = Writer::new(buf).unwrap();
    for i in 0..100 {
        let line = format!("line {i}\n");
        w.write_all(line.as_bytes()).unwrap();
    }
    let buf = w.seal().unwrap();

    let mut r = Reader::new(Cursor::new(buf)).unwrap();
    assert_eq!(r.lines(), 100);

    let mut output = String::new();
    r.read_to_string(&mut output).unwrap();

    let mut expected = String::new();
    for i in 0..100 {
        writeln!(expected, "line {i}").unwrap();
    }
    assert_eq!(expected, output);
}

#[test]
#[allow(clippy::cast_possible_truncation)]
fn all_levels_roundtrip() {
    let data: Vec<u8> = (0..10_000u16).map(|i| (i % 256) as u8).collect();
    for level in 1..=4u8 {
        let buf = Vec::new();
        let mut w = Writer::with_level(buf, level).unwrap();
        w.write_all(&data).unwrap();
        let buf = w.seal().unwrap();

        let mut r = Reader::new(Cursor::new(buf)).unwrap();
        let mut output = Vec::new();
        r.read_to_end(&mut output).unwrap();
        assert_eq!(data, output, "roundtrip failed at level {level}");
    }
}

// ---------------------------------------------------------------------------
// Multi-Sonnet tests (data > 256 KiB encoded → crosses Sonnet boundaries)
// ---------------------------------------------------------------------------

fn numbered_lines(min_bytes: usize) -> Vec<u8> {
    common::numbered_lines(min_bytes)
}

#[test]
fn multi_sonnet_roundtrip() {
    // 600 KiB of low-compressibility numbered lines guarantees >256 KiB encoded.
    let data = numbered_lines(600 * 1024);
    roundtrip(&data);
}

#[test]
fn multi_sonnet_seek_by_byte() {
    let data = numbered_lines(600 * 1024);
    let buf = Vec::new();
    let mut w = Writer::new(buf).unwrap();
    w.write_all(&data).unwrap();
    let buf = w.seal().unwrap();

    let mut r = Reader::new(Cursor::new(buf)).unwrap();
    let total = r.len();

    // Seek to the midpoint (should be in second Sonnet).
    let mid = total / 2;
    r.seek(SeekFrom::Start(mid)).unwrap();
    let mut out = vec![0u8; 64];
    r.read_exact(&mut out).unwrap();
    assert_eq!(&out, &data[mid as usize..mid as usize + 64]);

    // Seek near the end.
    r.seek(SeekFrom::End(-64)).unwrap();
    let mut out = vec![0u8; 64];
    r.read_exact(&mut out).unwrap();
    assert_eq!(&out, &data[data.len() - 64..]);
}

#[test]
fn multi_sonnet_seek_to_line() {
    let data = numbered_lines(600 * 1024);

    let buf = Vec::new();
    let mut w = Writer::new(buf).unwrap();
    w.write_all(&data).unwrap();
    let buf = w.seal().unwrap();

    let mut r = Reader::new(Cursor::new(buf)).unwrap();
    let reader_lines = r.lines();
    assert!(reader_lines > 0, "expected at least some lines");

    // Seek to a line in the second half (using the reader's line count).
    let target_line = reader_lines * 3 / 4;
    let offset = r.seek_to_line(target_line).unwrap();

    // Read one line and verify against the original data at that offset.
    let mut line_buf = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        let n = r.read(&mut byte).unwrap();
        if n == 0 || byte[0] == b'\n' {
            break;
        }
        line_buf.push(byte[0]);
    }

    #[allow(clippy::cast_possible_truncation)]
    let expected_line = std::str::from_utf8(&data[offset as usize..]).unwrap().lines().next().unwrap();
    assert_eq!(std::str::from_utf8(&line_buf).unwrap(), expected_line);
}

#[test]
fn reader_edge_cases() {
    // Empty sealed file.
    let buf = Vec::new();
    let w = Writer::new(buf).unwrap();
    let buf = w.seal().unwrap();
    let mut r = Reader::new(Cursor::new(&buf)).unwrap();
    assert!(r.is_empty());
    assert_eq!(r.len(), 0);
    assert_eq!(r.lines(), 0);

    // Debug format doesn't panic.
    let _ = format!("{r:?}");

    // seek_to_line out of range.
    let err = r.seek_to_line(1);
    assert!(err.is_err());

    // SeekFrom::Current positive and negative.
    let data = b"0123456789abcdef";
    let buf = Vec::new();
    let mut w = Writer::new(buf).unwrap();
    w.write_all(data).unwrap();
    let buf = w.seal().unwrap();
    let mut r = Reader::new(Cursor::new(buf)).unwrap();

    r.seek(SeekFrom::Start(4)).unwrap();
    let pos = r.seek(SeekFrom::Current(4)).unwrap();
    assert_eq!(pos, 8);

    let pos = r.seek(SeekFrom::Current(-2)).unwrap();
    assert_eq!(pos, 6);
    let mut out = vec![0u8; 4];
    r.read_exact(&mut out).unwrap();
    assert_eq!(&out, b"6789");

    // SeekFrom::End with positive offset clamps to total_bytes.
    let pos = r.seek(SeekFrom::End(100)).unwrap();
    assert_eq!(pos, data.len() as u64);

    // SeekFrom::End with negative offset past start → error.
    let err = r.seek(SeekFrom::End(-100));
    assert!(err.is_err());

    // SeekFrom::Current with negative offset past start → error.
    r.seek(SeekFrom::Start(2)).unwrap();
    let err = r.seek(SeekFrom::Current(-10));
    assert!(err.is_err());
}

#[test]
fn writer_edge_cases() {
    // Debug format doesn't panic.
    let w = Writer::new(Vec::new()).unwrap();
    let _ = format!("{w:?}");

    // Drop without seal — exercises the Drop impl.
    {
        let mut w = Writer::new(Vec::new()).unwrap();
        w.write_all(b"data that will be dropped\n").unwrap();
        // Writer drops here, calling flush() in Drop.
    }
}

#[test]
fn multi_sonnet_with_flush() {
    let data = numbered_lines(600 * 1024);
    let chunk_size = 50 * 1024;

    let buf = Vec::new();
    let mut w = Writer::new(buf).unwrap();
    for chunk in data.chunks(chunk_size) {
        w.write_all(chunk).unwrap();
        w.flush().unwrap();
    }
    let buf = w.seal().unwrap();

    let mut r = Reader::new(Cursor::new(buf)).unwrap();
    let mut output = Vec::new();
    r.read_to_end(&mut output).unwrap();
    assert_eq!(data, output);
}
