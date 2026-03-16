//! LZ77-based compression library.

#![allow(unused_features)]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
#![allow(dead_code)]

pub(crate) mod frame;
pub(crate) mod nibble;
pub(crate) mod ringbuf;

pub mod adler32;
pub mod decode;
pub mod encode;
pub mod error;
pub mod options;
