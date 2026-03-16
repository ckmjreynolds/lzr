//! LZR compression encoder.
//!
//! # Examples
//!
//! ```
//! use lzr::encode::encode;
//! use lzr::decode::decode;
//! use lzr::options::EncodeOptions;
//!
//! let input = b"Hello, world!";
//! let mut compressed = Vec::new();
//! encode(&mut &input[..], &mut compressed, &EncodeOptions::new()).unwrap();
//!
//! let mut decompressed = Vec::new();
//! decode(&mut compressed.as_slice(), &mut decompressed).unwrap();
//! assert_eq!(decompressed, input);
//! ```

use std::io::{Read, Write};

use crate::HEADER;
use crate::adler32::Adler32;
use crate::error::Result;
use crate::frame::Frame;
use crate::nibble::NibbleWriter;
use crate::options::EncodeOptions;

/// Maximum literal chunk size per frame.
const CHUNK_SIZE: usize = 4096;

/// Reads until `buf` is full or EOF. Returns the number of bytes read.
fn read_full(reader: &mut impl Read, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut total = 0;
    while total < buf.len() {
        let n = reader.read(&mut buf[total..])?;
        if n == 0 {
            break;
        }
        total += n;
    }
    Ok(total)
}

/// Compresses data from `input` and writes it to `output`.
///
/// Currently emits literal-only frames (no match finding). The output is a
/// valid LZR stream that any conforming decoder can decompress.
///
/// # Errors
///
/// Returns an error if an I/O error occurs while reading or writing.
///
/// # Examples
///
/// ```
/// use lzr::encode::encode;
/// use lzr::decode::decode;
/// use lzr::options::EncodeOptions;
///
/// let input = b"Hello, world!";
/// let mut compressed = Vec::new();
/// encode(&mut &input[..], &mut compressed, &EncodeOptions::new()).unwrap();
///
/// let mut decompressed = Vec::new();
/// decode(&mut compressed.as_slice(), &mut decompressed).unwrap();
/// assert_eq!(decompressed, input);
/// ```
#[allow(clippy::cast_possible_truncation)]
pub fn encode(input: &mut impl Read, output: &mut impl Write, _options: &EncodeOptions) -> Result<()> {
    // Header.
    output.write_all(&HEADER)?;

    // Bitstream: literal-only frames.
    let mut writer = NibbleWriter::new(&mut *output);
    let mut adler = Adler32::new();
    let mut total_len: u64 = 0;
    let mut buf = [0u8; CHUNK_SIZE];

    loop {
        let n = read_full(input, &mut buf)?;
        if n == 0 {
            break;
        }
        let chunk = &buf[..n];
        adler.update(chunk);
        total_len += n as u64;
        Frame::encode_literal(chunk, &mut writer)?;
    }

    // Pad to byte boundary if needed, then emit EOS.
    if writer.has_pending() {
        Frame::noop().encode(&mut writer)?;
    }
    Frame::EndOfStream.encode(&mut writer)?;
    let output = writer.finish()?;

    // Footer: UBLEB8 u64 length + 4-byte LE Adler-32.
    let mut footer_writer = NibbleWriter::new(&mut *output);
    footer_writer.write_ubleb8_u64(total_len)?;
    let output = footer_writer.finish()?;
    output.write_all(&adler.checksum().to_le_bytes())?;

    Ok(())
}
