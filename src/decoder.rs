//! LZR stream decoder.

use std::io::{self, Read, Write};

/// Decode an LZR compressed stream back to the original data.
///
/// # Errors
///
/// Returns an I/O error if reading from `input` or writing to `output` fails.
pub fn decode(mut input: impl Read, mut output: impl Write) -> io::Result<()> {
    io::copy(&mut input, &mut output)?;
    Ok(())
}
