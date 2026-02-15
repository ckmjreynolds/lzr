use std::io::{self, Cursor, Read};

use crate::Result;
use crate::format::{self, Adler32};

/// A match found during compression.
struct Match {
    distance: u32,
    length: i32,
}

impl Match {
    /// Number of input bytes this match covers.
    const fn covered(&self) -> usize {
        self.length.unsigned_abs() as usize
    }

    /// Number of bytes the encoded match frame occupies.
    fn encoded_size(&self) -> usize {
        let (_, _, len_ext) = format::encode_length(self.length);
        let (_, _, dist_ext) = format::encode_distance(self.distance);
        1 + len_ext + dist_ext // token + length ext + distance ext
    }
}

/// Streaming LZR encoder. Wraps a reader; reads return compressed bytes.
///
/// # Examples
///
/// ```
/// use std::io::Read;
///
/// let data = b"hello";
/// let mut encoder = lzr::Encoder::new(&data[..]);
/// let mut output = Vec::new();
/// encoder.read_to_end(&mut output).unwrap();
///
/// let mut decompressed = Vec::new();
/// lzr::decode(&mut &output[..], &mut decompressed).unwrap();
/// assert_eq!(decompressed, data);
/// ```
#[derive(Debug)]
pub struct Encoder<R> {
    inner: R,
    level: u8,
    output: Option<Cursor<Vec<u8>>>,
}

impl<R> Encoder<R> {
    /// Creates a new encoder wrapping the given reader with the default compression level.
    #[must_use]
    pub const fn new(reader: R) -> Self {
        Self {
            inner: reader,
            level: format::DEFAULT_LEVEL,
            output: None,
        }
    }

    /// Creates a new encoder with an explicit compression level (1-9).
    ///
    /// Level 1 is fastest (smallest window), level 9 is best (largest window).
    ///
    /// # Panics
    ///
    /// Panics if `level` is not in `1..=9`.
    #[must_use]
    pub fn with_level(reader: R, level: u8) -> Self {
        assert!((format::MIN_LEVEL..=format::MAX_LEVEL).contains(&level), "compression level must be 1-9");
        Self {
            inner: reader,
            level,
            output: None,
        }
    }

    /// Consumes the encoder, returning the wrapped reader.
    #[must_use]
    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R: Read> Encoder<R> {
    /// Reads all input and compresses it, storing the result in `self.output`.
    fn compress(&mut self) -> io::Result<()> {
        let mut input = Vec::new();
        self.inner.read_to_end(&mut input)?;

        let mut adler = Adler32::new();
        adler.update(&input);

        let window_exp = format::level_to_window_exp(self.level);
        let ws = format::window_size(window_exp);
        let mut out = Vec::new();

        format::write_header(&mut out, window_exp)?;

        // Greedy compression loop.
        let mut pos = 0;
        let mut literal_start = 0;

        while pos < input.len() {
            if let Some(m) = find_best_match(&input, pos, ws) {
                if m.encoded_size() <= m.covered() {
                    // Flush pending literals.
                    if literal_start < pos {
                        emit_literals(&input[literal_start..pos], &mut out);
                    }
                    emit_match(m.distance, m.length, &mut out);
                    pos += m.covered();
                    literal_start = pos;
                    continue;
                }
            }
            pos += 1;
        }

        // Flush remaining literals.
        if literal_start < input.len() {
            emit_literals(&input[literal_start..], &mut out);
        }

        out.push(format::EOS_TOKEN);
        format::write_footer(&mut out, input.len() as u64, adler.finish())?;

        self.output = Some(Cursor::new(out));
        Ok(())
    }
}

impl<R: Read> Read for Encoder<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.output.is_none() {
            self.compress()?;
        }
        self.output.as_mut().expect("compress must set output").read(buf)
    }
}

// ---------------------------------------------------------------------------
// Greedy match search
// ---------------------------------------------------------------------------

/// Finds the best match at `pos` by searching all distances in `[1, min(pos, window_size)]`.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn find_best_match(input: &[u8], pos: usize, window_size: usize) -> Option<Match> {
    let max_dist = pos.min(window_size);
    if max_dist == 0 || pos >= input.len() {
        return None;
    }

    let remaining = input.len() - pos;
    // Max length encodable: i16::MAX = 32767.
    let max_len = remaining.min(32767);

    let mut best: Option<Match> = None;
    let mut best_savings: i64 = 0; // savings = covered - encoded_size; must be >= 0.

    for d in 1..=max_dist {
        // Forward match.
        let fwd_len = forward_match_len(input, pos, d, max_len);
        if fwd_len > 0 {
            let m = Match {
                distance: d as u32,
                length: fwd_len as i32,
            };
            let savings = m.covered() as i64 - m.encoded_size() as i64;
            if savings >= 0 && savings > best_savings {
                best_savings = savings;
                best = Some(m);
            }
        }

        // Reverse match.
        let rev_len = reverse_match_len(input, pos, d, max_len);
        if rev_len > 0 {
            let m = Match {
                distance: d as u32,
                length: -(rev_len as i32),
            };
            let savings = m.covered() as i64 - m.encoded_size() as i64;
            if savings >= 0 && savings > best_savings {
                best_savings = savings;
                best = Some(m);
            }
        }
    }

    best
}

/// Returns the length of a forward match at `pos` referencing distance `d`.
///
/// The decoder copies one byte at a time, so when length > distance the pattern
/// wraps. We simulate this with modulo: `input[pos + i] == input[(pos - d) + (i % d)]`.
fn forward_match_len(input: &[u8], pos: usize, d: usize, max_len: usize) -> usize {
    let base = pos - d;
    let mut len = 0;
    while len < max_len && input[pos + len] == input[base + (len % d)] {
        len += 1;
    }
    len
}

/// Returns the length of a reverse match at `pos` referencing distance `d`.
///
/// The decoder reads backward: `output[pos-d], output[pos-d-1], ...`
/// Positions before output start read as 0x00.
fn reverse_match_len(input: &[u8], pos: usize, d: usize, max_len: usize) -> usize {
    let mut len = 0;
    while len < max_len {
        let src_byte = if pos >= d + len {
            input[pos - d - len]
        } else {
            0x00
        };
        if input[pos + len] != src_byte {
            break;
        }
        len += 1;
    }
    len
}

// ---------------------------------------------------------------------------
// Frame emission
// ---------------------------------------------------------------------------

/// Emits literal frames for `data`, chunking into max 32767 bytes per frame.
#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn emit_literals(data: &[u8], out: &mut Vec<u8>) {
    for chunk in data.chunks(32767) {
        let (lll, ext, ext_len) = format::encode_length(chunk.len() as i32);
        out.push(format::make_token(lll, 0));
        out.extend_from_slice(&ext[..ext_len]);
        out.extend_from_slice(chunk);
    }
}

/// Emits a single match frame.
fn emit_match(distance: u32, length: i32, out: &mut Vec<u8>) {
    let (lll, len_ext, len_ext_len) = format::encode_length(length);
    let (ddddd, dist_ext, dist_ext_len) = format::encode_distance(distance);
    out.push(format::make_token(lll, ddddd));
    out.extend_from_slice(&len_ext[..len_ext_len]);
    out.extend_from_slice(&dist_ext[..dist_ext_len]);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compresses data from `reader` and writes it to `writer`.
///
/// Returns the number of bytes written.
///
/// # Errors
///
/// Returns an error if an I/O operation fails.
pub fn encode(reader: &mut dyn Read, writer: &mut dyn io::Write) -> Result<u64> {
    encode_with_level(reader, writer, format::DEFAULT_LEVEL)
}

/// Compresses data from `reader` and writes it to `writer` at the given compression level.
///
/// Level 1 is fastest (smallest window), level 9 is best (largest window).
///
/// Returns the number of bytes written.
///
/// # Errors
///
/// Returns an error if an I/O operation fails.
///
/// # Panics
///
/// Panics if `level` is not in `1..=9`.
pub fn encode_with_level(reader: &mut dyn Read, writer: &mut dyn io::Write, level: u8) -> Result<u64> {
    let mut encoder = Encoder::with_level(reader, level);
    let bytes = io::copy(&mut encoder, writer)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        let data: &[u8] = b"";
        let mut output = Vec::new();
        let bytes = encode(&mut &data[..], &mut output).unwrap();

        // Header (5) + EOS (1) + Footer (ULEB128(0)=1 byte + Adler-32=4 bytes) = 11 bytes.
        assert_eq!(bytes, 11);
        assert_eq!(output.len(), 11);
        assert_eq!(&output[..3], b"LZR");

        // Verify Adler-32 of empty is 0x00000001 LE.
        let checksum_bytes = &output[7..11];
        assert_eq!(checksum_bytes, &1u32.to_le_bytes());
    }

    #[test]
    fn single_byte_round_trip() {
        let data = b"Z";
        let mut compressed = Vec::new();
        encode(&mut &data[..], &mut compressed).unwrap();

        let mut decompressed = Vec::new();
        crate::decode(&mut &compressed[..], &mut decompressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn repeated_bytes_compress() {
        let data = vec![b'a'; 1000];
        let mut compressed = Vec::new();
        encode(&mut &data[..], &mut compressed).unwrap();

        // Should compress significantly.
        assert!(compressed.len() < data.len(), "compressed {} >= original {}", compressed.len(), data.len());

        let mut decompressed = Vec::new();
        crate::decode(&mut &compressed[..], &mut decompressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn various_strings_round_trip() {
        let cases: &[&[u8]] = &[
            b"hello world",
            b"abcabcabcabc",
            b"the quick brown fox jumps over the lazy dog",
            b"aaabbbcccaaabbbccc",
            b"\x00\x01\x02\x03\x04\x05",
        ];
        for &data in cases {
            let mut compressed = Vec::new();
            encode(&mut &data[..], &mut compressed).unwrap();

            let mut decompressed = Vec::new();
            crate::decode(&mut &compressed[..], &mut decompressed).unwrap();
            assert_eq!(decompressed, data, "round-trip failed for {:?}", String::from_utf8_lossy(data));
        }
    }

    #[test]
    fn level_stored_correctly() {
        for level in [1u8, 5, 9] {
            let data = b"test";
            let mut compressed = Vec::new();
            encode_with_level(&mut &data[..], &mut compressed, level).unwrap();

            // Flags byte is at offset 4; window exp = level + 6.
            assert_eq!(compressed[4] & 0x0F, level + 6);
        }
    }

    #[test]
    fn with_level_round_trip() {
        for level in [1u8, 9] {
            let data = b"compression level test data with some repetition repetition repetition";
            let mut compressed = Vec::new();
            encode_with_level(&mut &data[..], &mut compressed, level).unwrap();

            let mut decompressed = Vec::new();
            crate::decode(&mut &compressed[..], &mut decompressed).unwrap();
            assert_eq!(decompressed, data);
        }
    }

    #[test]
    fn encoder_into_inner() {
        let data = b"inner test";
        let encoder = Encoder::new(&data[..]);
        let inner = encoder.into_inner();
        assert_eq!(inner, data);
    }
}
