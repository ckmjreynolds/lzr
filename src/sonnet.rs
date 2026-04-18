//! Sonnet-level constants and footer handling for the LZR format.
//!
//! A **Sonnet** is a fixed-size block (256 KiB) — the unit of random access.
//! The first Sonnet begins after the 4-byte Invocation header ([`INVOCATION`]).
//! Every complete Sonnet ends with a variable-length footer (Couplet or Coda)
//! containing cumulative counters.
//!
//! # Footer layout (Couplet / Coda)
//!
//! ```text
//! | cumulative_bytes (uleb128) | cumulative_lines (uleb128) | checksum (4 bytes) | footer_len (1 byte) |
//! ```
//!
//! `footer_len` is the **last byte** of the Sonnet. A reader reads that byte
//! first, then backs up `footer_len` bytes to parse the footer fields.
//! All counts are **cumulative** from the start of the Opus.

use crate::error::Result;
use crate::uleb128;

/// Fixed Sonnet size: 256 KiB (262,144 bytes).
pub(crate) const SONNET_SIZE: usize = 256 * 1024;

/// Invocation header: magic bytes + version (FORMAT.md Section 1.3).
pub(crate) const INVOCATION: [u8; 4] = *b"LZR\0";

/// Length of the Invocation header in bytes.
pub(crate) const INVOCATION_LEN: usize = INVOCATION.len();

/// Maximum footer size: 9 (max uleb128) + 9 + 4 (adler32) + 1 (len) = 23.
pub(crate) const MAX_FOOTER_SIZE: usize = 9 + 9 + 4 + 1;

/// Parsed Sonnet footer (Couplet for non-final, Coda for final).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Footer {
    /// Cumulative uncompressed byte count through end of this Sonnet.
    pub(crate) bytes_count: u64,
    /// Cumulative newline count through end of this Sonnet.
    pub(crate) lines_count: u64,
    /// Cumulative Adler-32 checksum.
    pub(crate) adler32: u32,
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

/// Decodes a footer from a complete Sonnet (or suffix containing the footer).
///
/// Reads `footer_len` from the last byte, then parses the footer fields
/// from the appropriate offset.
///
/// # Errors
///
/// Returns [`crate::Error::NonCanonicalUleb128`] if either ULEB128 field is
/// non-canonical.
pub(crate) fn decode_footer(block: &[u8]) -> Result<Footer> {
    assert!(block.len() >= MAX_FOOTER_SIZE, "block too small for footer");
    let footer_len = block[block.len() - 1] as usize;
    let footer_start = block.len() - footer_len;
    let mut pos = footer_start;

    let bytes_count = uleb128::decode_uleb128_u64(block, &mut pos)?;
    let lines_count = uleb128::decode_uleb128_u64(block, &mut pos)?;

    let adler32 = u32::from_le_bytes([block[pos], block[pos + 1], block[pos + 2], block[pos + 3]]);

    Ok(Footer {
        bytes_count,
        lines_count,
        adler32,
    })
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

        let mut block = vec![0u8; SONNET_SIZE];
        let start = SONNET_SIZE - encoded.len();
        block[start..].copy_from_slice(&encoded);

        let decoded = decode_footer(&block).unwrap();
        assert_eq!(footer, decoded);
    }

    #[test]
    fn footer_max_values_fit_in_max_size() {
        let footer = Footer {
            bytes_count: u64::MAX,
            lines_count: u64::MAX,
            adler32: u32::MAX,
        };
        let encoded = encode_footer(&footer);
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
        assert_eq!(encoded.len(), 7);

        let mut block = vec![0u8; SONNET_SIZE];
        let start = SONNET_SIZE - encoded.len();
        block[start..].copy_from_slice(&encoded);
        let decoded = decode_footer(&block).unwrap();
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

            let mut block = vec![0u8; SONNET_SIZE];
            let start = SONNET_SIZE - encoded.len();
            block[start..].copy_from_slice(&encoded);

            let decoded = decode_footer(&block).unwrap();
            prop_assert_eq!(footer, decoded);
        }
    }

    #[test]
    fn footer_rejects_non_canonical_uleb128() {
        // Craft a footer where bytes_count is encoded non-canonically
        // (`0x80 0x00` for 0, instead of the canonical `0x00`).
        let mut block = vec![0u8; SONNET_SIZE];
        let footer_bytes: &[u8] = &[
            0x80, 0x00, // bytes_count (non-canonical 0)
            0x00, // lines_count = 0
            0x01, 0x00, 0x00, 0x00, // adler32 = 1
            8,    // footer_len
        ];
        let start = SONNET_SIZE - footer_bytes.len();
        block[start..].copy_from_slice(footer_bytes);
        assert!(decode_footer(&block).is_err());
    }
}
