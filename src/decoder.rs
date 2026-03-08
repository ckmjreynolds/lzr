//! LZR stream decoder.

use std::io::{self, Read, Write};

use crate::adler32::Adler32;

/// Decode an LZR compressed stream back to the original data.
///
/// # Errors
///
/// Returns an I/O error if reading from `input` or writing to `output` fails,
/// or if the Adler-32 checksum does not match.
pub fn decode(mut input: impl Read, mut output: impl Write) -> io::Result<()> {
    let mut header = [0u8; 4];
    input.read_exact(&mut header)?;
    if &header != crate::HEADER {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid LZR header"));
    }

    let mut data = Vec::new();
    input.read_to_end(&mut data)?;

    if data.len() < 4 {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "missing Adler-32 checksum"));
    }

    let (payload, checksum_bytes) = data.split_at(data.len() - 4);
    let stored = u32::from_le_bytes([checksum_bytes[0], checksum_bytes[1], checksum_bytes[2], checksum_bytes[3]]);

    let mut checksum = Adler32::new();
    checksum.update(payload);

    if checksum.checksum() != stored {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Adler-32 mismatch: expected {stored:#010X}, got {:#010X}", checksum.checksum()),
        ));
    }

    output.write_all(payload)?;
    Ok(())
}
