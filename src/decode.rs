//! LZR decompression decoder.
//!
//! Reads one or more concatenated LZR streams from an [`impl Read`](Read)
//! source, decompresses each stream, and writes the result to an
//! [`impl Write`](Write) sink. Each stream is independently validated against
//! its footer (Adler-32 checksum and uncompressed length).
//!
//! See [`FORMAT.md`](../docs/FORMAT.md) for the canonical format specification.
//!
//! # Layout
//!
//! The decoder uses a flat output buffer of `OUT_BUF_SIZE` bytes, conceptually
//! divided into:
//!
//! - `out[0..WINDOW_SIZE]`: a zero-initialized "implicit zero" prefix that
//!   satisfies the format spec's "reads before the start of decompressed
//!   output return 0x00" rule. After the first shift it holds the trailing
//!   `WINDOW_SIZE` bytes of recently-emitted output instead.
//! - `out[WINDOW_SIZE..op]`: decoded but not-yet-flushed bytes.
//! - `out[op..]`: free space and wildcopy slop.
//!
//! When `op` exceeds the shift threshold, bytes
//! `out[WINDOW_SIZE..op - WINDOW_SIZE]` are flushed to the sink, the trailing
//! `WINDOW_SIZE` bytes are `copy_within`-shifted to `out[0..WINDOW_SIZE]`, and
//! `op` is reset to `WINDOW_SIZE`. All back-references at distance ≤
//! `WINDOW_SIZE` remain valid because the live window is preserved.

use std::io::{self, Read, Write};

use crate::WINDOW_SIZE;
use crate::adler32::Adler32;
use crate::error::{Error, Result};
use crate::format;
use crate::wild;

/// Output buffer size: 4 × `WINDOW_SIZE` = 256 KiB. Sized so that one max-size
/// frame (`literal_len + |match_len|` ≤ 65534 bytes plus 32 bytes of wildcopy
/// slop) can always fit between flushes after a shift back to `WINDOW_SIZE`.
const OUT_BUF_SIZE: usize = 4 * WINDOW_SIZE;

/// Wildcopy slop budget — extra bytes a 16-byte wildcopy may write past the
/// logical end of a literal or match.
const WILDCOPY_SLOP: usize = 32;

/// The largest number of output bytes a single frame can produce
/// (`literal_len + |match_len|`, both bounded by `i16::MAX`).
const MAX_FRAME_OUT: usize = 2 * (i16::MAX as usize);

/// Trigger a flush+shift before decoding a frame whenever the write head is
/// past this point. Sized so that after a shift (which leaves `op = WINDOW_SIZE`)
/// the buffer always has room for one full max frame plus wildcopy slop.
const SHIFT_THRESHOLD: usize = OUT_BUF_SIZE - MAX_FRAME_OUT - WILDCOPY_SLOP;

/// Input buffer size: large enough to always hold one max-size compressed
/// frame plus headroom. Max compressed frame ≈ 33 KiB (token + distance +
/// extension chains + 32767 literal bytes).
const IN_BUF_SIZE: usize = 128 * 1024;

/// Refill input when fewer than this many unconsumed bytes remain.
const REFILL_TARGET: usize = 64 * 1024;

/// Footer max size: 9 bytes ULEB128 length + 4 bytes checksum.
const FOOTER_MAX: usize = 13;

/// Decompresses one or more concatenated LZR streams from `input` into
/// `output`.
///
/// Each stream is validated against its footer: the decompressed byte count
/// must match the stored length, and the Adler-32 checksum must match. If data
/// remains after a footer, it is treated as a new stream beginning with a
/// header.
///
/// # Errors
///
/// Returns [`Error::Io`] on any underlying I/O failure (including unexpected
/// EOF during header or literal reads).
///
/// Returns [`Error::InvalidMagic`] or [`Error::UnsupportedVersion`] if a
/// stream header is malformed.
///
/// Returns [`Error::LengthMismatch`] if the decompressed byte count does not
/// match the footer.
///
/// Returns [`Error::ChecksumMismatch`] if the computed Adler-32 does not
/// match the footer.
pub fn decode(input: &mut impl Read, output: &mut impl Write) -> Result<()> {
    let mut state = DecodeState::new();
    state.run(input, output)
}

/// Internal decoder state. Owns input and output buffers.
struct DecodeState {
    in_buf: Vec<u8>,
    in_end: usize,
    ip: usize,
    eof: bool,

    out_buf: Vec<u8>,
    /// Logical write position in `out_buf`. Always `>= WINDOW_SIZE` so that
    /// back-references at distance up to `WINDOW_SIZE` resolve into either the
    /// live window or the zero-prefix region.
    op: usize,
    /// First not-yet-flushed byte in `out_buf`. Always `>= WINDOW_SIZE`.
    flushed: usize,
}

impl DecodeState {
    fn new() -> Self {
        Self {
            in_buf: vec![0u8; IN_BUF_SIZE],
            in_end: 0,
            ip: 0,
            eof: false,
            out_buf: vec![0u8; OUT_BUF_SIZE],
            op: WINDOW_SIZE,
            flushed: WINDOW_SIZE,
        }
    }

    /// Resets buffer state between streams (concatenated streams).
    fn reset_for_next_stream(&mut self) {
        // Wipe the live window so that the next stream's initial back-references
        // (if any are produced by a malformed encoder) read zeros, not stale
        // data from the previous stream. Spec-conformant encoders never emit
        // a back-reference past the start of their stream, so the wipe is
        // belt-and-suspenders correctness.
        self.out_buf[..self.op].fill(0);
        self.op = WINDOW_SIZE;
        self.flushed = WINDOW_SIZE;
    }

    /// Reads more compressed bytes from `input`, shifting any unconsumed
    /// bytes to the front of `in_buf`. Sets `eof` if no bytes were read.
    fn refill(&mut self, input: &mut impl Read) -> io::Result<()> {
        if self.eof {
            return Ok(());
        }
        let unconsumed = self.in_end - self.ip;
        if self.ip > 0 {
            self.in_buf.copy_within(self.ip..self.in_end, 0);
            self.in_end = unconsumed;
            self.ip = 0;
        }
        let mut filled = self.in_end;
        while filled < IN_BUF_SIZE {
            let n = input.read(&mut self.in_buf[filled..])?;
            if n == 0 {
                self.eof = true;
                break;
            }
            filled += n;
        }
        self.in_end = filled;
        Ok(())
    }

    /// Ensures at least `need` bytes are available from `ip`. Returns
    /// `UnexpectedEof` if the source is exhausted.
    fn ensure(&mut self, input: &mut impl Read, need: usize) -> io::Result<()> {
        while self.in_end - self.ip < need {
            if self.eof {
                return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
            }
            self.refill(input)?;
        }
        Ok(())
    }

    /// Flushes decoded output to `sink` and shifts the trailing window back to
    /// `out_buf[0..WINDOW_SIZE]`. After this call `op == WINDOW_SIZE` and
    /// `flushed == 0`, because the bytes that were just shifted into
    /// `out_buf[0..WINDOW_SIZE]` are the live back-reference window AND are
    /// still pending (not yet emitted to the sink). They will be emitted on
    /// the next `flush_and_shift` / `flush_remaining` call.
    fn flush_and_shift(&mut self, sink: &mut impl Write, checksum: &mut Adler32) -> io::Result<()> {
        // Bytes [flushed..op] have been decoded but not emitted yet. We may
        // emit everything up to op - WINDOW_SIZE (keeping the last WINDOW_SIZE
        // bytes for back-references).
        let keep_start = self.op - WINDOW_SIZE;
        if keep_start > self.flushed {
            let to_emit = &self.out_buf[self.flushed..keep_start];
            checksum.update(to_emit);
            sink.write_all(to_emit)?;
        }
        // Shift the live window back to the front. These bytes are still
        // pending output — they have not been emitted yet.
        self.out_buf.copy_within(keep_start..self.op, 0);
        self.op = WINDOW_SIZE;
        self.flushed = 0;
        Ok(())
    }

    /// Final flush at end of stream: emits all bytes from `flushed..op` to
    /// `sink` and updates the checksum. Does not shift; the next stream (if
    /// any) calls `reset_for_next_stream`.
    fn flush_remaining(&mut self, sink: &mut impl Write, checksum: &mut Adler32) -> io::Result<()> {
        if self.op > self.flushed {
            let bytes = &self.out_buf[self.flushed..self.op];
            checksum.update(bytes);
            sink.write_all(bytes)?;
            self.flushed = self.op;
        }
        Ok(())
    }

    /// Top-level decode loop: handles concatenated streams.
    fn run(&mut self, input: &mut impl Read, output: &mut impl Write) -> Result<()> {
        loop {
            // Try to advance to the next stream.
            if self.in_end - self.ip == 0 {
                self.refill(input)?;
                if self.in_end == 0 && self.eof {
                    return Ok(());
                }
            }
            self.decode_one_stream(input, output)?;
            if self.eof && self.in_end - self.ip == 0 {
                return Ok(());
            }
            self.reset_for_next_stream();
        }
    }

    /// Decodes one full LZR stream (header + frames + EOS + footer).
    fn decode_one_stream(&mut self, input: &mut impl Read, output: &mut impl Write) -> Result<()> {
        let mut checksum = Adler32::new();
        let mut total_len: u64 = 0;
        let stream_op_start = self.op;

        // Header.
        self.ensure(input, 4)?;
        format::decode_header(&self.in_buf, &mut self.ip)?;

        // Frame loop.
        loop {
            if !self.eof && self.in_end - self.ip < REFILL_TARGET {
                self.refill(input)?;
            }
            if self.op > SHIFT_THRESHOLD {
                self.flush_and_shift(output, &mut checksum)?;
            }

            // Decode the frame header.
            let consumed_before = self.ip;
            let (literal_len, match_len, distance) = format::decode_frame(&self.in_buf, &mut self.ip);

            // EOS sentinel: distance == 0 && literal_len == 0.
            if distance == 0 && literal_len == 0 {
                break;
            }

            // Apply literals.
            #[allow(clippy::cast_sign_loss)]
            let lit = literal_len as usize;
            if lit > 0 {
                if self.in_end - self.ip < lit {
                    // The literal payload spans a refill. Roll back the frame
                    // header consumption, refill, and retry.
                    self.ip = consumed_before;
                    if self.eof {
                        return Err(Error::Io(io::Error::from(io::ErrorKind::UnexpectedEof)));
                    }
                    self.refill(input)?;
                    continue;
                }
                self.copy_literals(lit);
            }

            // Apply match.
            if match_len != 0 {
                #[allow(clippy::cast_sign_loss)]
                let abs = match_len.unsigned_abs() as usize;
                let dist = distance as usize;
                if match_len > 0 {
                    self.copy_forward(dist, abs);
                } else {
                    self.copy_reverse(dist, abs);
                }
                total_len += abs as u64;
            }
            total_len += lit as u64;
        }

        // Final flush before footer.
        self.flush_remaining(output, &mut checksum)?;

        // Account for any bytes still in the window prefix from BEFORE this
        // stream — they should NOT be in this stream's checksum or length.
        // We've maintained the invariant that everything in [stream_op_start,
        // self.op] is this stream's output, but flush_remaining only flushes
        // [self.flushed, self.op]. The flushed bound moves with each shift,
        // and shifts can carry over bytes from the previous stream's tail in
        // the window. The checksum computed above is correct because it only
        // counted bytes that we explicitly emitted via flush_and_shift /
        // flush_remaining; both update the checksum for exactly the bytes
        // they emit. The total length is computed independently from the
        // frame deltas, which is precise.

        // Footer is 5 to 13 bytes (1–9 byte ULEB length + 4 byte checksum).
        // Require at least 5 bytes; pull more if available without erroring
        // on EOF beyond that.
        self.ensure(input, 5)?;
        while !self.eof && self.in_end - self.ip < FOOTER_MAX {
            self.refill(input)?;
        }

        let (expected_len, expected_checksum) = format::decode_footer(&self.in_buf, &mut self.ip);

        if total_len != expected_len {
            return Err(Error::LengthMismatch {
                expected: expected_len,
                actual: total_len,
            });
        }
        if checksum.checksum() != expected_checksum {
            return Err(Error::ChecksumMismatch {
                expected: expected_checksum,
                actual: checksum.checksum(),
            });
        }

        let _ = stream_op_start;
        Ok(())
    }

    /// Copies `lit` bytes of literal payload from `in_buf[ip..]` to
    /// `out_buf[op..]`. Both pointers advance by `lit`. Caller must guarantee
    /// `in_buf` has `lit` available bytes from `ip`.
    #[inline]
    fn copy_literals(&mut self, lit: usize) {
        // Wildcopy fast path: when both source and destination have ≥ 16 bytes
        // of slop past the logical end, we can copy in 16-byte chunks and
        // overshoot the literal_len safely.
        if self.ip + lit + 16 <= self.in_end && self.op + lit + 16 <= OUT_BUF_SIZE {
            let mut copied = 0;
            while copied < lit {
                wild::wildcopy_16(&mut self.out_buf, self.op + copied, &self.in_buf, self.ip + copied);
                copied += 16;
            }
        } else {
            // Bounded copy.
            self.out_buf[self.op..self.op + lit].copy_from_slice(&self.in_buf[self.ip..self.ip + lit]);
        }
        self.op += lit;
        self.ip += lit;
    }

    /// Forward copy: `out_buf[op + i] = out_buf[op - distance + i]` for `i in 0..len`.
    /// Handles overlapping (distance < len) correctly via byte-by-byte copy.
    #[inline]
    fn copy_forward(&mut self, distance: usize, len: usize) {
        let src = self.op - distance;
        if distance >= len {
            // Non-overlapping: bulk copy.
            // Use copy_within (memmove) — fast and safe.
            self.out_buf.copy_within(src..src + len, self.op);
        } else if distance >= 16 && self.op + len + 16 <= OUT_BUF_SIZE {
            // Overlapping but distance ≥ 16: wildcopy 16-byte chunks. Each
            // chunk is non-overlapping with itself (distance ≥ 16) so the
            // pattern of `len` bytes propagates correctly.
            let mut copied = 0;
            while copied < len {
                wild::wildcopy_16_within(&mut self.out_buf, self.op + copied, src + copied);
                copied += 16;
            }
        } else {
            // Tight overlap (distance < 16): byte-by-byte to preserve RLE.
            wild::copy_forward(&mut self.out_buf, self.op, src, len);
        }
        self.op += len;
    }

    /// Reverse copy: `out_buf[op + i] = out_buf[op - distance - i]` for `i in 0..len`.
    #[inline]
    fn copy_reverse(&mut self, distance: usize, len: usize) {
        let src_start = self.op - distance;
        wild::copy_reverse(&mut self.out_buf, self.op, src_start, len);
        self.op += len;
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    /// Helper: decode a byte slice and return the decompressed output.
    fn decode_bytes(compressed: &[u8]) -> std::result::Result<Vec<u8>, Error> {
        let mut output = Vec::new();
        decode(&mut &compressed[..], &mut output)?;
        Ok(output)
    }

    /// The "Hi" worked example from FORMAT.md.
    #[rustfmt::skip]
    const HI_STREAM: [u8; 17] = [
        0x4C, 0x5A, 0x52, 0x00,             // Header: LZR v0
        0x46, 0x00, 0x00, 0x48, 0x69,       // Frame: L=2, M=6, dist=0, literals "Hi"
        0x00, 0x00, 0x00,                    // EOS
        0x02,                                // Footer: ULEB128 length = 2
        0xB2, 0x00, 0xFB, 0x00,             // Footer: Adler-32 = 0x00FB00B2
    ];

    #[test]
    fn worked_example_hi() {
        let output = decode_bytes(&HI_STREAM).unwrap();
        assert_eq!(output, b"Hi");
    }

    #[test]
    fn empty_stream() {
        #[rustfmt::skip]
        let compressed = [
            0x4C, 0x5A, 0x52, 0x00,         // Header
            0x00, 0x00, 0x00,                // EOS
            0x00,                            // Footer: length = 0
            0x01, 0x00, 0x00, 0x00,          // Footer: Adler-32 of empty = 0x00000001
        ];
        let output = decode_bytes(&compressed).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn stream_concatenation() {
        let mut compressed = Vec::new();
        compressed.extend_from_slice(&HI_STREAM);
        compressed.extend_from_slice(&HI_STREAM);

        let output = decode_bytes(&compressed).unwrap();
        assert_eq!(output, b"HiHi");
    }

    #[test]
    fn invalid_magic() {
        let compressed = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let err = decode_bytes(&compressed).unwrap_err();
        assert!(matches!(err, Error::InvalidMagic));
    }

    #[test]
    fn unsupported_version() {
        let compressed = [0x4C, 0x5A, 0x52, 0x01, 0x00, 0x00, 0x00, 0x00];
        let err = decode_bytes(&compressed).unwrap_err();
        assert!(matches!(err, Error::UnsupportedVersion(0x01)));
    }

    #[test]
    fn checksum_mismatch() {
        #[rustfmt::skip]
        let compressed = [
            0x4C, 0x5A, 0x52, 0x00,         // Header
            0x46, 0x00, 0x00, 0x48, 0x69,   // Frame: "Hi"
            0x00, 0x00, 0x00,                // EOS
            0x02,                            // Footer: length = 2 (correct)
            0xFF, 0xFF, 0xFF, 0xFF,          // Footer: wrong checksum
        ];
        let err = decode_bytes(&compressed).unwrap_err();
        assert!(matches!(err, Error::ChecksumMismatch { .. }));
    }

    #[test]
    fn length_mismatch() {
        #[rustfmt::skip]
        let compressed = [
            0x4C, 0x5A, 0x52, 0x00,         // Header
            0x46, 0x00, 0x00, 0x48, 0x69,   // Frame: "Hi"
            0x00, 0x00, 0x00,                // EOS
            0x09,                            // Footer: length = 9 (wrong, should be 2)
            0xB2, 0x00, 0xFB, 0x00,          // Footer: correct checksum for "Hi"
        ];
        let err = decode_bytes(&compressed).unwrap_err();
        assert!(matches!(err, Error::LengthMismatch { .. }));
    }

    #[test]
    fn forward_match() {
        // Encode "ABCDABCD" as: 4 literals "ABCD" + forward match (distance=4, length=4).
        let mut checksum = Adler32::new();
        checksum.update(b"ABCDABCD");
        let cs = checksum.checksum().to_le_bytes();

        #[rustfmt::skip]
        let compressed = [
            0x4C, 0x5A, 0x52, 0x00,         // Header
            // Frame 1: L=4, M=7 (match_len=4), distance=4
            // Token: 100_00111 = 0x87
            0x87, 0x04, 0x00,
            0x41, 0x42, 0x43, 0x44,          // Literals: "ABCD"
            0x00, 0x00, 0x00,                // EOS
            0x08,                            // Footer: length = 8
            cs[0], cs[1], cs[2], cs[3],
        ];
        let output = decode_bytes(&compressed).unwrap();
        assert_eq!(output, b"ABCDABCD");
    }

    #[test]
    fn reverse_match() {
        // Encode "ABCDDCBA" as: 4 literals "ABCD" + reverse match (distance=1, length=-4).
        // Reverse copy from pos-1 backwards: D, C, B, A.
        let mut checksum = Adler32::new();
        checksum.update(b"ABCDDCBA");
        let cs = checksum.checksum().to_le_bytes();

        #[rustfmt::skip]
        let compressed = [
            0x4C, 0x5A, 0x52, 0x00,         // Header
            // Frame 1: L=4, M=5 (match_len=-4), distance=1
            // Token: 100_00101 = 0x85
            0x85, 0x01, 0x00,
            0x41, 0x42, 0x43, 0x44,          // Literals: "ABCD"
            0x00, 0x00, 0x00,                // EOS
            0x08,                            // Footer: length = 8
            cs[0], cs[1], cs[2], cs[3],
        ];
        let output = decode_bytes(&compressed).unwrap();
        assert_eq!(output, b"ABCDDCBA");
    }
}
