//! LZR stream decoder.

use std::io::{self, BufReader, BufWriter, ErrorKind, Read, Write};

use crate::adler32::Adler32;
use crate::bleb8::{Sleb8, Ubleb8};
use crate::nibble::ReadNibble;
use crate::{MAGIC, VERSION, WINDOW_SIZE};

/// Decodes one or more concatenated LZR streams from `input` into `output`.
///
/// # Errors
///
/// Returns an error if the input is malformed (bad magic, unsupported version,
/// checksum mismatch, length mismatch) or truncated.
pub fn decode(input: impl Read, output: impl Write) -> io::Result<()> {
    let mut dec = Decoder::new(input, output);
    dec.run()?;
    dec.writer.flush()
}

/// Internal decoder state.
struct Decoder<R: Read, W: Write> {
    reader: BufReader<R>,
    writer: BufWriter<W>,
    nibble_buf: u8,
    has_low: bool,
    window: Box<[u8; WINDOW_SIZE]>,
    window_pos: usize,
    adler: Adler32,
    output_len: u64,
}

impl<R: Read, W: Write> Decoder<R, W> {
    fn new(input: R, output: W) -> Self {
        Self {
            reader: BufReader::new(input),
            writer: BufWriter::new(output),
            nibble_buf: 0,
            has_low: false,
            window: vec![0u8; WINDOW_SIZE].try_into().expect("exact size"),
            window_pos: 0,
            adler: Adler32::new(),
            output_len: 0,
        }
    }

    fn run(&mut self) -> io::Result<()> {
        while self.try_read_header()? {
            self.decode_frames()?;
            self.verify_footer()?;
            self.reset();
        }
        Ok(())
    }

    /// Resets per-stream state for concatenated streams.
    fn reset(&mut self) {
        self.nibble_buf = 0;
        self.has_low = false;
        self.window.fill(0);
        self.window_pos = 0;
        self.adler = Adler32::new();
        self.output_len = 0;
    }

    // ── Header ──────────────────────────────────────────────────────

    /// Reads a stream header, returning `Ok(true)` if found, `Ok(false)` on
    /// clean EOF, or an error for invalid/truncated headers.
    fn try_read_header(&mut self) -> io::Result<bool> {
        let mut first = [0u8; 1];
        if self.reader.read(&mut first)? == 0 {
            return Ok(false);
        }
        let mut rest = [0u8; 3];
        self.reader.read_exact(&mut rest)?;
        if [first[0], rest[0], rest[1]] != MAGIC {
            return Err(io::Error::new(ErrorKind::InvalidData, "invalid magic number"));
        }
        if rest[2] != VERSION {
            return Err(io::Error::new(ErrorKind::InvalidData, "unsupported format version"));
        }
        Ok(true)
    }

    // ── Frames ──────────────────────────────────────────────────────

    fn decode_frames(&mut self) -> io::Result<()> {
        loop {
            let d = u16::decode_ubleb8(self)?;
            if d == 0 {
                let l = u16::decode_ubleb8(self)?;
                if l == 0 {
                    break; // EOS
                }
                self.read_literals(l)?;
            } else {
                let l = i16::decode_sleb8(self)?;
                if l != 0 {
                    self.copy_match(d, l)?;
                }
                // l == 0 → no-op
            }
        }
        self.align();
        Ok(())
    }

    fn read_literals(&mut self, count: u16) -> io::Result<()> {
        let count = usize::from(count);
        let mut buf = vec![0u8; count];
        for b in &mut buf {
            *b = self.read_byte()?;
        }
        for &byte in &buf {
            self.window[self.window_pos] = byte;
            self.window_pos = (self.window_pos + 1) & (WINDOW_SIZE - 1);
        }
        self.writer.write_all(&buf)?;
        self.adler.update(&buf);
        self.output_len += count as u64;
        Ok(())
    }

    fn copy_match(&mut self, distance: u16, length: i16) -> io::Result<()> {
        let d = usize::from(distance);
        let abs_len = usize::from(length.unsigned_abs());
        // Capture start position so overlapping forward copies read
        // newly-written bytes (enables RLE via D < L).
        let start = self.window_pos;
        if length > 0 {
            for i in 0..abs_len {
                let src = start.wrapping_sub(d).wrapping_add(i) & (WINDOW_SIZE - 1);
                self.emit(self.window[src])?;
            }
        } else {
            for i in 0..abs_len {
                let src = start.wrapping_sub(d).wrapping_sub(i) & (WINDOW_SIZE - 1);
                self.emit(self.window[src])?;
            }
        }
        Ok(())
    }

    fn emit(&mut self, byte: u8) -> io::Result<()> {
        self.writer.write_all(&[byte])?;
        self.window[self.window_pos] = byte;
        self.window_pos = (self.window_pos + 1) & (WINDOW_SIZE - 1);
        self.adler.update(&[byte]);
        self.output_len += 1;
        Ok(())
    }

    // ── Footer ──────────────────────────────────────────────────────

    fn verify_footer(&mut self) -> io::Result<()> {
        let stored_len = u64::decode_ubleb8(self)?;
        self.align();
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf)?;
        let stored_checksum = u32::from_le_bytes(buf);

        if self.output_len != stored_len {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("length mismatch: expected {stored_len}, got {}", self.output_len),
            ));
        }
        let computed = self.adler.finish();
        if computed != stored_checksum {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("checksum mismatch: expected {stored_checksum:#010X}, got {computed:#010X}"),
            ));
        }
        Ok(())
    }

    // ── Nibble Helpers ─────────────────────────────────────────────

    /// Discards any pending low nibble to reach byte alignment.
    const fn align(&mut self) {
        self.has_low = false;
    }
}

impl<R: Read, W: Write> ReadNibble for Decoder<R, W> {
    type Error = io::Error;

    fn read_nibble(&mut self) -> io::Result<u8> {
        if self.has_low {
            self.has_low = false;
            Ok(self.nibble_buf & 0xF)
        } else {
            let mut buf = [0u8; 1];
            self.reader.read_exact(&mut buf)?;
            self.nibble_buf = buf[0];
            self.has_low = true;
            Ok(buf[0] >> 4)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::ErrorKind;

    use pretty_assertions::assert_eq;

    use super::*;

    fn run_decode(input: &[u8]) -> io::Result<Vec<u8>> {
        let mut output = Vec::new();
        decode(input, &mut output)?;
        Ok(output)
    }

    // ── Spec Worked Example ─────────────────────────────────────────

    #[test]
    fn worked_example_hi() {
        // FORMAT.md: 13-byte stream encoding "Hi"
        #[rustfmt::skip]
        let stream: &[u8] = &[
            0x4C, 0x5A, 0x52, 0x00, // header
            0x02, 0x84, 0x96, 0x00, // bitstream: lit(2) "Hi", EOS
            0x20,                   // footer length: UBLEB8(2)
            0xB2, 0x00, 0xFB, 0x00, // footer checksum: Adler-32("Hi")
        ];
        assert_eq!(run_decode(stream).unwrap(), b"Hi");
    }

    // ── Empty Stream ────────────────────────────────────────────────

    #[test]
    fn empty_stream() {
        // EOS immediately, zero-length output.
        #[rustfmt::skip]
        let stream: &[u8] = &[
            0x4C, 0x5A, 0x52, 0x00, // header
            0x00,                   // bitstream: EOS (D=0, L=0)
            0x00,                   // footer length: UBLEB8(0)
            0x01, 0x00, 0x00, 0x00, // footer checksum: Adler-32("")
        ];
        assert_eq!(run_decode(stream).unwrap(), b"");
    }

    // ── Empty Input ─────────────────────────────────────────────────

    #[test]
    fn empty_input() {
        assert_eq!(run_decode(&[]).unwrap(), b"");
    }

    #[test]
    fn error_bad_magic() {
        let stream: &[u8] = &[0xFF, 0x5A, 0x52, 0x00];
        let err = run_decode(stream).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        assert!(err.to_string().contains("magic"), "{err}");
    }

    #[test]
    fn error_bad_version() {
        let stream: &[u8] = &[0x4C, 0x5A, 0x52, 0x01];
        let err = run_decode(stream).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        assert!(err.to_string().contains("version"), "{err}");
    }

    #[test]
    fn error_checksum_mismatch() {
        #[rustfmt::skip]
        let stream: &[u8] = &[
            0x4C, 0x5A, 0x52, 0x00,
            0x02, 0x84, 0x96, 0x00,
            0x20,
            0xB2, 0x00, 0xFB, 0x01, // corrupted
        ];
        let err = run_decode(stream).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        assert!(err.to_string().contains("checksum"), "{err}");
    }

    #[test]
    fn error_length_mismatch() {
        #[rustfmt::skip]
        let stream: &[u8] = &[
            0x4C, 0x5A, 0x52, 0x00,
            0x02, 0x84, 0x96, 0x00,
            0x30,                   // UBLEB8(3) — wrong!
            0xB2, 0x00, 0xFB, 0x00,
        ];
        let err = run_decode(stream).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        assert!(err.to_string().contains("length"), "{err}");
    }
}
