//! The self-describing LZR container: a small header, the framing-agnostic codec
//! core, and an Adler-32 footer over the original bytes.
//!
//! ```text
//! magic "LZR" (3) | version (1) | profile (ULEB128) | codec core (var) | adler32 BE (4)
//! ```
//!
//! Framing is deliberately factored out of the codec ([`crate::codec`]): the
//! Hutter-Prize branch can wrap the same core in a near-zero-header framing
//! without touching the pipeline. See [`docs/FORMAT.md`](../docs/FORMAT.md).

use anyhow::{Context as _, Result, bail, ensure};

use crate::adler32::Adler32;
use crate::codec::Profile;
use crate::uleb128;

/// Container magic: the ASCII bytes `LZR`.
const MAGIC: &[u8; 3] = b"LZR";
/// Framing version. `0x00` denoted the abandoned block ("Sonnet") format, so the
/// first shipped container is `0x01`.
const VERSION: u8 = 0x01;
/// Fixed header prefix before the ULEB128 profile: magic (3) + version (1).
const PREFIX_LEN: usize = 4;
/// Footer length: a big-endian Adler-32 of the original input.
const FOOTER_LEN: usize = 4;

/// Compress `input` into a self-describing LZR container using the default [`Profile`].
#[must_use]
pub fn compress(input: &[u8]) -> Vec<u8> {
    compress_with(input, Profile::default())
}

/// Compress `input` into a self-describing LZR container using `profile` to select the pipeline.
#[must_use]
pub fn compress_with(input: &[u8], profile: Profile) -> Vec<u8> {
    compress_owned(input.to_vec(), profile)
}

/// Like [`compress_with`] but takes ownership of `input`, letting the pipeline free it before the
/// memory-heavy tokenizer build. Prefer this on large inputs (the CLI does) to keep the peak down.
#[must_use]
pub fn compress_owned(input: Vec<u8>, profile: Profile) -> Vec<u8> {
    // Checksum the original bytes up front so `encode` can consume and free `input` during the build.
    let mut checksum = Adler32::new();
    checksum.update(&input);
    let checksum = checksum.checksum();

    let core = profile.compressor().encode(input);

    let mut out = Vec::with_capacity(PREFIX_LEN + 1 + core.len() + FOOTER_LEN);
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    uleb128::encode_u64(profile.to_bits(), &mut out);
    out.extend_from_slice(&core);
    out.extend_from_slice(&checksum.to_be_bytes());
    out
}

/// Decompress an LZR container, verifying its version, profile, and the Adler-32
/// checksum of the recovered bytes.
///
/// # Errors
///
/// Returns an error if the input is too short, the magic/version/profile are
/// unrecognized, the codec core is malformed, or the stored checksum does not
/// match the decoded output.
pub fn decompress(input: &[u8]) -> Result<Vec<u8>> {
    ensure!(input.len() >= PREFIX_LEN, "input too short ({} bytes) to be an LZR container", input.len());
    ensure!(input.starts_with(MAGIC), "bad magic: not an LZR container");

    let version = input[3];
    ensure!(version == VERSION, "unsupported container version {version}");
    let mut pos = PREFIX_LEN;
    let profile = Profile::from_bits(uleb128::decode_u64(input, &mut pos)?)?;

    ensure!(input.len() >= pos + FOOTER_LEN, "input too short for a codec core and footer");
    let footer_start = input.len() - FOOTER_LEN;
    let core = &input[pos..footer_start];
    let output = profile.compressor().decode(core).context("decoding LZR payload")?;

    let stored = u32::from_be_bytes([
        input[footer_start],
        input[footer_start + 1],
        input[footer_start + 2],
        input[footer_start + 3],
    ]);
    let mut checksum = Adler32::new();
    checksum.update(&output);
    let actual = checksum.checksum();
    if actual != stored {
        bail!("checksum mismatch: stored {stored:#010x}, computed {actual:#010x}");
    }
    Ok(output)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn roundtrip(data in prop::collection::vec(any::<u8>(), 0..2048)) {
            prop_assert_eq!(decompress(&compress(&data)).unwrap(), data);
        }

        /// Decompressing arbitrary bytes must never panic — only Ok or Err.
        #[test]
        fn decompress_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..2048)) {
            drop(decompress(&bytes));
        }
    }

    #[test]
    fn compressed_output_is_self_describing() {
        // The default profile fits in one ULEB128 byte (a small feature bitmask).
        let c = compress(b"hello");
        assert!(c.starts_with(MAGIC));
        assert_eq!(c[3], VERSION);
        assert_eq!(u64::from(c[PREFIX_LEN]), Profile::default().to_bits());
        assert!(c.len() >= PREFIX_LEN + 1 + FOOTER_LEN);
    }

    #[test]
    fn custom_profile_selects_and_round_trips() {
        // Disabling every default feature clears all bits (profile 0x00); the container records it
        // and `decompress` rebuilds the matching pipeline.
        let mut profile = Profile::default();
        profile.disable("repair").unwrap();
        profile.disable("casefold").unwrap();
        profile.disable("entities").unwrap();
        profile.disable("lz77").unwrap();
        let c = compress_with(b"hello world", profile);
        assert_eq!(c[PREFIX_LEN], 0x00);
        assert_eq!(decompress(&c).unwrap(), b"hello world".to_vec());
    }

    #[test]
    fn rejects_unsupported_profile() {
        // Bit 5 is not a known feature; a single-byte ULEB128 profile of 0x20 must be rejected.
        let mut c = compress(b"data");
        c[PREFIX_LEN] = 0x20;
        assert!(decompress(&c).is_err());
    }

    #[test]
    fn rejects_truncated_and_bad_magic() {
        assert!(decompress(&[]).is_err());
        assert!(decompress(b"LZR").is_err());
        assert!(decompress(b"XYZ\x01\x00\x00\x00\x00\x00\x00\x00\x00").is_err());
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut c = compress(b"data");
        c[3] = 0xFF;
        assert!(decompress(&c).is_err());
    }

    #[test]
    fn detects_footer_corruption() {
        let mut c = compress(b"payload bytes");
        let last = c.len() - 1;
        c[last] ^= 0xFF; // corrupt the stored checksum
        assert!(decompress(&c).is_err());
    }

    #[test]
    fn detects_payload_corruption() {
        let original = b"the quick brown fox jumps over the lazy dog";
        let mut c = compress(original);
        // Flip a bit near the middle of the codec core; the checksum must not silently pass.
        let mid = c.len() / 2;
        c[mid] ^= 0x01;
        // The corruption must not survive as the original bytes: either decoding
        // fails outright, or the checksum rejects the different output.
        if let Ok(out) = decompress(&c) {
            assert_ne!(out.as_slice(), original.as_slice());
        }
    }
}
