//! LZ77-based compression library.

#![allow(unused_features)]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![allow(dead_code)]

pub(crate) mod adler32;
pub(crate) mod format;
pub(crate) mod uleb128;
pub(crate) mod wild;

/// Sliding window size in bytes (64 KiB, per format spec).
pub(crate) const WINDOW_SIZE: usize = 1 << 16;

/// Encoder batch buffer size (4 MiB). Must be a power of two for [`buffer::Buffer`].
pub(crate) const BATCH_SIZE: usize = 1 << 22;

/// Encoder block size — each parallel compression job processes this many bytes.
pub(crate) const BLOCK_SIZE: usize = WINDOW_SIZE;

/// Stream decoder — decompresses LZR-encoded data.
pub mod decode;
/// Stream encoder — compresses raw data into LZR format.
pub mod encode;
/// Error types returned by encode/decode operations.
pub mod error;
/// Encoder configuration (compression level, thread count).
pub mod options;
