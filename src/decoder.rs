use std::io::{self, Read};

use crate::Result;
use crate::error::Error;
use crate::format::{self, Adler32};

/// State machine phases for the decoder.
#[derive(Debug)]
enum State {
    /// Expecting the 5-byte header.
    Header,
    /// Decoding frames until EOS.
    Frames,
    /// Expecting the footer (ULEB128 length + Adler-32).
    Footer,
    /// Stream fully consumed.
    Done,
}

/// Streaming LZR decoder. Wraps a reader; reads return decompressed bytes.
///
/// # Examples
///
/// ```
/// use std::io::Read;
///
/// let data = b"hello";
/// let mut compressed = Vec::new();
/// lzr::encode(&mut &data[..], &mut compressed).unwrap();
///
/// let mut decoder = lzr::Decoder::new(&compressed[..]);
/// let mut output = Vec::new();
/// decoder.read_to_end(&mut output).unwrap();
/// assert_eq!(output, data);
/// ```
#[derive(Debug)]
pub struct Decoder<R> {
    inner: R,
    state: State,
    output: Vec<u8>,
    read_pos: usize,
    adler: Adler32,
}

impl<R> Decoder<R> {
    /// Creates a new decoder wrapping the given reader.
    #[must_use]
    pub const fn new(reader: R) -> Self {
        Self {
            inner: reader,
            state: State::Header,
            output: Vec::new(),
            read_pos: 0,
            adler: Adler32::new(),
        }
    }

    /// Consumes the decoder, returning the wrapped reader.
    #[must_use]
    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R: Read> Decoder<R> {
    /// Decodes the next frame, returning `true` if output was produced or the
    /// state advanced, `false` if we reached `Done`.
    fn advance(&mut self) -> io::Result<bool> {
        match self.state {
            State::Header => {
                let _window_exp = format::read_header(&mut self.inner).map_err(io::Error::from)?;
                self.state = State::Frames;
                Ok(true)
            }
            State::Frames => self.decode_next_frame(),
            State::Footer => {
                let (expected_len, expected_checksum) = format::read_footer(&mut self.inner)?;
                if self.output.len() as u64 != expected_len {
                    return Err(io::Error::from(Error::InvalidFormat));
                }
                if self.adler.finish() != expected_checksum {
                    return Err(io::Error::from(Error::ChecksumMismatch));
                }
                self.state = State::Done;
                Ok(true)
            }
            State::Done => Ok(false),
        }
    }

    /// Decodes a single frame from the input stream.
    #[allow(clippy::cast_sign_loss)]
    fn decode_next_frame(&mut self) -> io::Result<bool> {
        let mut token_buf = [0u8; 1];
        self.inner.read_exact(&mut token_buf)?;
        let (lll, ddddd) = format::split_token(token_buf[0]);

        if lll == 0 && ddddd == 0 {
            // EOS
            self.state = State::Footer;
            return Ok(true);
        }

        if lll == 0 && ddddd > 0 {
            // Extension frame: skip N bytes in input.
            let skip = format::decode_distance(ddddd, &mut self.inner)?;
            let mut sink = io::sink();
            io::copy(&mut (&mut self.inner).take(u64::from(skip)), &mut sink)?;
            return Ok(true);
        }

        let length = format::decode_length(lll, &mut self.inner)?;

        if ddddd == 0 {
            // Literal frame.
            let count = length.unsigned_abs() as usize;
            let start = self.output.len();
            self.output.resize(start + count, 0);
            self.inner.read_exact(&mut self.output[start..])?;
            self.adler.update(&self.output[start..]);
            return Ok(true);
        }

        // Match frame.
        let distance = format::decode_distance(ddddd, &mut self.inner)? as usize;

        if length == 0 {
            // No-op.
            return Ok(true);
        }

        if length > 0 {
            // Forward copy.
            let count = length as usize;
            let pos = self.output.len();
            self.output.reserve(count);
            for i in 0..count {
                let src_idx = pos + i;
                let byte = if src_idx >= distance {
                    self.output[src_idx - distance]
                } else {
                    0x00 // before output start → 0x00
                };
                self.output.push(byte);
            }
            self.adler.update(&self.output[pos..]);
        } else {
            // Reverse copy.
            let count = length.unsigned_abs() as usize;
            let pos = self.output.len();
            self.output.reserve(count);
            for i in 0..count {
                let byte = if pos >= distance + i {
                    self.output[pos - distance - i]
                } else {
                    0x00
                };
                self.output.push(byte);
            }
            self.adler.update(&self.output[pos..]);
        }

        Ok(true)
    }
}

impl<R: Read> Read for Decoder<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // Serve any buffered bytes first.
        while self.read_pos >= self.output.len() {
            if !self.advance()? {
                return Ok(0);
            }
        }

        let available = self.output.len() - self.read_pos;
        let to_copy = available.min(buf.len());
        buf[..to_copy].copy_from_slice(&self.output[self.read_pos..self.read_pos + to_copy]);
        self.read_pos += to_copy;
        Ok(to_copy)
    }
}

/// Decompresses LZR data from `reader` and writes it to `writer`.
///
/// Returns the number of bytes written.
///
/// # Errors
///
/// Returns an error if an I/O operation fails, the format is invalid,
/// or the checksum does not match.
pub fn decode(reader: &mut dyn Read, writer: &mut dyn io::Write) -> Result<u64> {
    let mut decoder = Decoder::new(reader);
    let bytes = io::copy(&mut decoder, writer)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a minimal valid LZR stream with no frames (empty input).
    fn empty_stream() -> Vec<u8> {
        let mut buf = Vec::new();
        format::write_header(&mut buf, format::level_to_window_exp(format::DEFAULT_LEVEL)).unwrap();
        buf.push(format::EOS_TOKEN);
        // Footer: length=0, Adler-32 of empty = 1.
        format::write_footer(&mut buf, 0, 1).unwrap();
        buf
    }

    #[test]
    fn decode_empty_stream() {
        let stream = empty_stream();
        let mut output = Vec::new();
        let bytes = decode(&mut &stream[..], &mut output).unwrap();
        assert_eq!(bytes, 0);
        assert!(output.is_empty());
    }

    #[test]
    fn decode_single_byte() {
        // Encode a single byte via the encoder, then decode.
        let data = b"X";
        let mut compressed = Vec::new();
        crate::encode(&mut &data[..], &mut compressed).unwrap();

        let mut output = Vec::new();
        let bytes = decode(&mut &compressed[..], &mut output).unwrap();
        assert_eq!(bytes, 1);
        assert_eq!(output, data);
    }

    #[test]
    fn hand_crafted_literal_stream() {
        // Build: header, literal frame with "Hi", EOS, footer.
        let mut stream = Vec::new();
        format::write_header(&mut stream, 0).unwrap();

        // Literal frame: LLL=2, DDDDD=0 → token = (2 << 5) | 0 = 0x40
        stream.push(format::make_token(2, 0));
        stream.extend_from_slice(b"Hi");

        stream.push(format::EOS_TOKEN);

        let mut adler = Adler32::new();
        adler.update(b"Hi");
        format::write_footer(&mut stream, 2, adler.finish()).unwrap();

        let mut output = Vec::new();
        let bytes = decode(&mut &stream[..], &mut output).unwrap();
        assert_eq!(bytes, 2);
        assert_eq!(output, b"Hi");
    }

    #[test]
    fn invalid_magic() {
        let stream = [0x00, 0x00, 0x00, 0x00, 0x00];
        let err = decode(&mut &stream[..], &mut Vec::new()).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat));
    }

    #[test]
    fn invalid_version() {
        let stream = [0x4C, 0x5A, 0x52, 0x01, 0x00];
        let err = decode(&mut &stream[..], &mut Vec::new()).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat));
    }

    #[test]
    fn wrong_checksum() {
        let mut stream = Vec::new();
        format::write_header(&mut stream, 0).unwrap();
        stream.push(format::EOS_TOKEN);
        // Correct length (0), wrong checksum (0xBAD).
        format::write_footer(&mut stream, 0, 0xBAD).unwrap();

        let err = decode(&mut &stream[..], &mut Vec::new()).unwrap_err();
        assert!(matches!(err, Error::ChecksumMismatch));
    }

    #[test]
    fn wrong_length() {
        let mut stream = Vec::new();
        format::write_header(&mut stream, 0).unwrap();
        stream.push(format::EOS_TOKEN);
        // Wrong length (999), correct empty checksum.
        format::write_footer(&mut stream, 999, 1).unwrap();

        let err = decode(&mut &stream[..], &mut Vec::new()).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat));
    }

    #[test]
    fn decoder_into_inner() {
        let data = b"inner test";
        let decoder = Decoder::new(&data[..]);
        let inner = decoder.into_inner();
        assert_eq!(inner, data);
    }
}
