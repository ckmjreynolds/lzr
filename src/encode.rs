//! LZR compression encoder.
//!
//! # Examples
//!
//! ```
//! use lzr::encode::encode;
//! use lzr::options::EncodeOptions;
//!
//! let input = b"Hello, world!";
//! let mut output = Vec::new();
//! encode(&mut &input[..], &mut output, &EncodeOptions::default()).unwrap();
//! ```

use std::io::{Read, Write};

use crate::error::Result;
use crate::options::EncodeOptions;

/// Compresses data from `input` and writes it to `output`.
///
/// Currently a passthrough (`io::copy`). Real LZR compression will replace this.
///
/// # Errors
///
/// Returns an error if an I/O error occurs while reading or writing.
///
/// # Examples
///
/// ```
/// use lzr::encode::encode;
/// use lzr::options::EncodeOptions;
///
/// let input = b"Hello, world!";
/// let mut output = Vec::new();
/// encode(&mut &input[..], &mut output, &EncodeOptions::new()).unwrap();
/// assert_eq!(output, input);
/// ```
pub fn encode(input: &mut impl Read, output: &mut impl Write, _options: &EncodeOptions) -> Result<()> {
    std::io::copy(input, output)?;
    Ok(())
}
