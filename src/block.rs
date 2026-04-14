//! Block-level constants and footer handling for the LZR format.
//!
//! Each block is exactly [`BLOCK_SIZE`] bytes (256 KiB). Block 0 begins with
//! a 4-byte magic header ([`MAGIC`]). Every complete block ends with a
//! variable-length footer containing cumulative counters.
//!
//! # Footer layout
//!
//! ```text
//! | bytes_count (uleb128) | lines_count (uleb128) | adler32 (4 bytes) | footer_len (1 byte) |
//! ```
//!
//! `footer_len` is the **last byte** of the 256 KiB block. A reader reads that
//! byte first, then backs up `footer_len` bytes to parse the footer fields.
//! All counts are **cumulative** across blocks.

use crate::uleb128;

/// Fixed block size: 256 KiB.
pub(crate) const BLOCK_SIZE: usize = 256 * 1024;

/// Magic bytes at the start of block 0.
pub(crate) const MAGIC: [u8; 4] = *b"LZR\0";

/// Length of the magic header in bytes.
pub(crate) const MAGIC_LEN: usize = MAGIC.len();

/// Maximum footer size: 9 (max uleb128) + 9 + 4 (adler32) + 1 (len) = 23.
pub(crate) const MAX_FOOTER_SIZE: usize = 9 + 9 + 4 + 1;

/// Parsed block footer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Footer {
    /// Cumulative uncompressed byte count through end of this block.
    pub(crate) bytes_count: u64,
    /// Cumulative newline count through end of this block.
    pub(crate) lines_count: u64,
    /// Cumulative Adler-32 checksum.
    pub(crate) adler32: u32,
}

/// Returns the encoded size of a footer with the given counts.
pub(crate) fn footer_size(bytes_count: u64, lines_count: u64) -> usize {
    uleb128::uleb128_u64_len(bytes_count) + uleb128::uleb128_u64_len(lines_count) + 4 + 1
}

/// Encodes a footer into bytes.
///
/// The returned `Vec` includes the `footer_len` byte at the end.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn encode_footer(footer: &Footer) -> Vec<u8> {
    let mut buf = Vec::with_capacity(MAX_FOOTER_SIZE);
    uleb128::encode_uleb128_u64(footer.bytes_count, &mut buf);
    uleb128::encode_uleb128_u64(footer.lines_count, &mut buf);
    buf.extend_from_slice(&footer.adler32.to_le_bytes());
    // +1 for the footer_len byte itself.
    let len = buf.len() + 1;
    buf.push(len as u8);
    buf
}

/// Decodes a footer from a complete block.
///
/// Reads `footer_len` from the last byte, then parses the footer fields
/// from the appropriate offset.
pub(crate) fn decode_footer(block: &[u8]) -> Footer {
    assert!(block.len() >= MAX_FOOTER_SIZE, "block too small for footer");
    let footer_len = block[block.len() - 1] as usize;
    let footer_start = block.len() - footer_len;
    let mut pos = footer_start;

    let bytes_count = uleb128::decode_uleb128_u64(block, &mut pos);
    let lines_count = uleb128::decode_uleb128_u64(block, &mut pos);

    let adler32 = u32::from_le_bytes([block[pos], block[pos + 1], block[pos + 2], block[pos + 3]]);

    Footer {
        bytes_count,
        lines_count,
        adler32,
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn footer_roundtrip_known() {
        let footer = Footer {
            bytes_count: 1024,
            lines_count: 42,
            adler32: 0x11E6_0398,
        };

        let encoded = encode_footer(&footer);
        let expected_size = footer_size(footer.bytes_count, footer.lines_count);
        assert_eq!(encoded.len(), expected_size);

        // Place footer at the end of a block-sized buffer.
        let mut block = vec![0u8; BLOCK_SIZE];
        let start = BLOCK_SIZE - encoded.len();
        block[start..].copy_from_slice(&encoded);

        let decoded = decode_footer(&block);
        assert_eq!(footer, decoded);
    }

    #[test]
    fn footer_size_matches_encode() {
        let footer = Footer {
            bytes_count: u64::MAX,
            lines_count: u64::MAX,
            adler32: u32::MAX,
        };
        let encoded = encode_footer(&footer);
        assert_eq!(encoded.len(), footer_size(footer.bytes_count, footer.lines_count));
        assert_eq!(encoded.len(), MAX_FOOTER_SIZE);
    }

    #[test]
    fn footer_small_values() {
        let footer = Footer {
            bytes_count: 0,
            lines_count: 0,
            adler32: 1,
        };
        let encoded = encode_footer(&footer);
        // 1 + 1 + 4 + 1 = 7 bytes
        assert_eq!(encoded.len(), 7);

        let mut block = vec![0u8; BLOCK_SIZE];
        let start = BLOCK_SIZE - encoded.len();
        block[start..].copy_from_slice(&encoded);
        let decoded = decode_footer(&block);
        assert_eq!(footer, decoded);
    }

    proptest! {
        #[test]
        fn footer_roundtrip(
            bytes_count: u64,
            lines_count: u64,
            adler32: u32,
        ) {
            let footer = Footer { bytes_count, lines_count, adler32 };
            let encoded = encode_footer(&footer);
            prop_assert_eq!(encoded.len(), footer_size(bytes_count, lines_count));

            let mut block = vec![0u8; BLOCK_SIZE];
            let start = BLOCK_SIZE - encoded.len();
            block[start..].copy_from_slice(&encoded);

            let decoded = decode_footer(&block);
            prop_assert_eq!(footer, decoded);
        }
    }
}
