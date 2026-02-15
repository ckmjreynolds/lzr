//! LZ77-based compression library.
//!
//! LZR provides streaming compression and decompression using the LZR format.
//!
//! # Examples
//!
//! ```
//! use std::io::Read;
//!
//! let data = b"hello world";
//! let mut compressed = Vec::new();
//! lzr::encode(&mut &data[..], &mut compressed).unwrap();
//!
//! let mut output = Vec::new();
//! lzr::decode(&mut &compressed[..], &mut output).unwrap();
//! assert_eq!(output, data);
//! ```

// Enable coverage attributes for nightly builds.
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

mod decoder;
mod encoder;
mod error;

pub use decoder::{Decoder, decode};
pub use encoder::{Encoder, encode};
pub use error::{Error, Result};
