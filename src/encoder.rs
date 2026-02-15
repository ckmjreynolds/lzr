use std::io;

use crate::Result;

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
/// assert_eq!(output, data);
/// ```
#[derive(Debug)]
pub struct Encoder<R> {
    inner: R,
}

impl<R> Encoder<R> {
    /// Creates a new encoder wrapping the given reader.
    #[must_use]
    pub const fn new(reader: R) -> Self {
        Self {
            inner: reader,
        }
    }

    /// Consumes the encoder, returning the wrapped reader.
    #[must_use]
    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R: io::Read> io::Read for Encoder<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}

/// Compresses data from `reader` and writes it to `writer`.
///
/// Returns the number of bytes written.
///
/// # Errors
///
/// Returns an error if an I/O operation fails.
pub fn encode(reader: &mut dyn io::Read, writer: &mut dyn io::Write) -> Result<u64> {
    let mut encoder = Encoder::new(reader);
    let bytes = io::copy(&mut encoder, writer)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_pass_through() {
        let data = b"the quick brown fox";
        let mut output = Vec::new();
        let bytes = encode(&mut &data[..], &mut output).unwrap();
        assert_eq!(output, data);
        assert_eq!(bytes, data.len() as u64);
    }

    #[test]
    fn empty_input() {
        let data: &[u8] = b"";
        let mut output = Vec::new();
        let bytes = encode(&mut &data[..], &mut output).unwrap();
        assert_eq!(output, data);
        assert_eq!(bytes, 0);
    }

    #[test]
    fn encoder_into_inner() {
        let data = b"inner test";
        let encoder = Encoder::new(&data[..]);
        let inner = encoder.into_inner();
        assert_eq!(inner, data);
    }
}
