//! LZR compression encoder.

use std::io::{BufRead, Write};

use crate::error::Result;
use crate::options::EncodeOptions;

/// Compresses data from `input` and writes it to `output`.
///
/// # Errors
///
/// Returns an error if an I/O error occurs while reading or writing.
///
/// # Examples
///
/// ```no_run
/// use lzr::encode::encode;
/// use lzr::options::EncodeOptions;
///
/// let input = b"Hello, world!";
/// let mut compressed = Vec::new();
/// encode(&mut &input[..], &mut compressed, &EncodeOptions::new()).unwrap();
/// ```
pub fn encode(_input: &mut impl BufRead, _output: &mut impl Write, _options: &EncodeOptions) -> Result<()> {
    todo!()
}
