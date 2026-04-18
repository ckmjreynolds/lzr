//! LZR — append-only, random-access compression format.
//!
//! Data is compressed through an LZ77 stage followed by adaptive arithmetic
//! coding, then stored in fixed 256 KiB Sonnets that enable random access by
//! byte offset or line number.
//!
//! # Quick start
//!
//! ```
//! use std::io::{Cursor, Read, Write};
//!
//! // Compress
//! let mut w = lzr::Writer::new(Vec::new()).unwrap();
//! w.write_all(b"Hello, world!\n").unwrap();
//! let compressed = w.seal().unwrap();
//!
//! // Decompress
//! let mut r = lzr::Reader::new(Cursor::new(compressed)).unwrap();
//! let mut output = String::new();
//! r.read_to_string(&mut output).unwrap();
//! assert_eq!(output, "Hello, world!\n");
//! ```

#![allow(unused_features)]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
pub(crate) mod adler32;
pub(crate) mod codec;
pub(crate) mod entropy;
mod error;
pub(crate) mod lz77;
pub(crate) mod model;
mod reader;
pub(crate) mod sonnet;
pub(crate) mod uleb128;
mod writer;

pub use error::{Error, Result};
pub use reader::Reader;
pub use writer::Writer;

/// Counts newline bytes in `data`.
#[allow(clippy::naive_bytecount)]
pub(crate) fn count_lines(data: &[u8]) -> usize {
    data.iter().filter(|&&b| b == b'\n').count()
}

/// Default compression level.
///
/// Valid levels are `1..=9`: L1 uses a single hash-table lookup (lz4-style),
/// L2–L6 a hash chain, and L7–L9 a `wabi_tree` B+tree. Each step trades
/// speed for a higher compression ratio.
pub const DEFAULT_LEVEL: u8 = lz77::DEFAULT_LEVEL;

#[cfg(any(test, feature = "bench-internals"))]
#[cfg_attr(coverage_nightly, coverage(off))]
#[allow(
    missing_docs,
    missing_copy_implementations,
    missing_debug_implementations,
    unreachable_pub,
    clippy::redundant_pub_crate,
    clippy::new_without_default,
    clippy::missing_const_for_fn
)]
pub mod bench;
