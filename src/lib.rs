//! LZ77-based compression library.

#![allow(unused_features)]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

pub(crate) const HEADER: &[u8; 4] = b"LZR\0";

mod adler32;
mod nibble;
mod ring;

pub mod decoder;
pub mod encoder;
