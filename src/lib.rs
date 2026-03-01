//! LZ77-based compression library.

#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

// #[cfg(not(target_pointer_width = "64"))]
// compile_error!("lzr requires a 64-bit target!");

mod adler32;
mod bleb8;
pub mod decoder;
mod error;
mod nibble;
