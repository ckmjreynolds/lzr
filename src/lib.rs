//! LZ77-based compression library.

#![allow(unused_features)]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![allow(dead_code)]

pub(crate) mod buffer;
pub(crate) mod frame;
pub(crate) mod matchfinder;
pub(crate) mod ringbuf;
pub(crate) mod uleb128;

/// LZR stream header: magic bytes (`LZR`) + format version (`0x00`).
pub(crate) const HEADER: [u8; 4] = [0x4C, 0x5A, 0x52, 0x00];

/// Sliding window size in bytes (64 KiB, per format spec).
pub(crate) const WINDOW_SIZE: usize = 65_536;

pub mod adler32;
pub mod decode;
pub mod encode;
pub mod error;
pub mod options;
