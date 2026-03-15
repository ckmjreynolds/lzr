//! LZR decompression decoder.
//!
//! # Examples
//!
//! ```
//! use lzr::decode::decode;
//!
//! let data = b"Hello, world!";
//! let mut output = Vec::new();
//! decode(&mut &data[..], &mut output).unwrap();
//! assert_eq!(output, data);
//! ```

use std::io::{Read, Write};

use crate::error::Result;

/// Decompresses data from `input` and writes it to `output`.
///
/// Currently a passthrough (`io::copy`). Real LZR decompression will replace this.
///
/// # Errors
///
/// Returns an error if an I/O error occurs while reading or writing.
///
/// # Examples
///
/// ```
/// use lzr::decode::decode;
///
/// let data = b"Hello, world!";
/// let mut output = Vec::new();
/// decode(&mut &data[..], &mut output).unwrap();
/// assert_eq!(output, data);
/// ```
pub fn decode(input: &mut impl Read, output: &mut impl Write) -> Result<()> {
    std::io::copy(input, output)?;
    Ok(())
}
