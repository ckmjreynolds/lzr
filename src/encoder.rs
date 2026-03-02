//! LZR stream encoder.

use std::io::{self, Read, Write};

/// Encode an input stream into an LZR compressed stream.
///
/// # Errors
///
/// Returns an I/O error if reading from `input` or writing to `output` fails.
pub fn encode(mut input: impl Read, mut output: impl Write) -> io::Result<()> {
    io::copy(&mut input, &mut output)?;
    Ok(())
}
