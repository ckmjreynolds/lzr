//! LZ77-based compression library.

#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

mod adler32;
mod bleb8;
pub mod decoder;
pub mod encoder;
mod error;
mod nibble;

/// Magic bytes: `LZR`.
pub(crate) const MAGIC: [u8; 3] = [0x4C, 0x5A, 0x52];

/// Format version byte.
pub(crate) const VERSION: u8 = 0x00;

/// Sliding window size (64 KiB).
pub(crate) const WINDOW_SIZE: usize = 65_536;
