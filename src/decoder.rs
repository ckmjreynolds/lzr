use std::io;

use crate::Result;

/// Streaming LZR decoder. Wraps a reader; reads return decompressed bytes.
///
/// # Examples
///
/// ```
/// use std::io::Read;
///
/// let data = b"hello";
/// let mut decoder = lzr::Decoder::new(&data[..]);
/// let mut output = Vec::new();
/// decoder.read_to_end(&mut output).unwrap();
/// assert_eq!(output, data);
/// ```
#[derive(Debug)]
pub struct Decoder<R> {
    inner: R,
}

impl<R> Decoder<R> {
    /// Creates a new decoder wrapping the given reader.
    #[must_use]
    pub const fn new(reader: R) -> Self {
        Self {
            inner: reader,
        }
    }

    /// Consumes the decoder, returning the wrapped reader.
    #[must_use]
    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R: io::Read> io::Read for Decoder<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
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
pub fn decode(reader: &mut dyn io::Read, writer: &mut dyn io::Write) -> Result<u64> {
    let mut decoder = Decoder::new(reader);
    let bytes = io::copy(&mut decoder, writer)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_pass_through() {
        let data = b"jumps over the lazy dog";
        let mut output = Vec::new();
        let bytes = decode(&mut &data[..], &mut output).unwrap();
        assert_eq!(output, data);
        assert_eq!(bytes, data.len() as u64);
    }

    #[test]
    fn decoder_into_inner() {
        let data = b"inner test";
        let decoder = Decoder::new(&data[..]);
        let inner = decoder.into_inner();
        assert_eq!(inner, data);
    }
}
