//! The self-describing LZR container: a small header, the framing-agnostic codec
//! core, and an Adler-32 footer over the original bytes.
//!
//! ```text
//! magic "LZR" (3) | version (1) | profile (ULEB128) | codec core (var) | adler32 BE (4)
//! ```
//!
//! Framing is deliberately factored out of the codec ([`crate::codec`]): the
//! Hutter-Prize branch can wrap the same core in a near-zero-header framing
//! without touching the pipeline.

use anyhow::{Context as _, Result, bail, ensure};

use crate::adler32::Adler32;
use crate::codec::{EncodeOptions, Profile};
use crate::uleb128;

/// Container magic: the ASCII bytes `LZR`.
const MAGIC: &[u8; 3] = b"LZR";
/// Framing version. Frozen at `0x00` during development: the wire format is unstable and we do not
/// bump the version for format changes, so an older stream simply fails to decode (fine pre-1.0).
const VERSION: u8 = 0x00;
/// Fixed header prefix before the ULEB128 profile: magic (3) + version (1).
const PREFIX_LEN: usize = 4;
/// Footer length: a big-endian Adler-32 of the original input.
const FOOTER_LEN: usize = 4;

/// One pipeline stage's input and output size in bytes, paired with its feature name.
///
/// Returned by [`compress_owned_with_traced`] so the CLI can report each transform's own
/// bits-per-byte (`8 · output / input`) — what it did to its input.
#[derive(Debug, Clone, Copy)]
pub struct StageSize {
    /// The stage's feature name (e.g. `"repair"`).
    pub name: &'static str,
    /// The stage's input length in bytes (the previous stage's output, or the original input).
    pub input_bytes: usize,
    /// The stage's output length in bytes after it ran.
    pub output_bytes: usize,
    /// An optional stage-specific statistic as `(value, unit label)` — e.g. the Re-Pair vocabulary
    /// `(n, "tokens")` or the LZ77 match count `(n, "matches")`; `None` for stages with nothing extra
    /// to report.
    pub detail: Option<(u64, &'static str)>,
}

/// One entropy model's encode-side scorecard, for the CLI's per-model report.
///
/// Returned by [`compress_owned_with_traced`] alongside the stage sizes.
#[derive(Debug, Clone, Copy)]
pub struct ModelScore {
    /// The model's feature name (e.g. `"order0"`, or `"null"` for the injected fallback).
    pub name: &'static str,
    /// Bits per (entropy-stage input) byte if this model coded the stream alone — its raw predictive
    /// power. Near 8 means no better than chance; lower is better.
    pub bpb_alone: f64,
    /// The mixer's average final weight on this model. Near zero means it adds little *marginally* —
    /// its signal is already covered by the other models.
    pub avg_weight: f64,
}

/// Wrap a codec `core` in the container framing: magic, version, ULEB128 profile, core, Adler-32
/// footer (big-endian, over the original input).
fn frame(profile: Profile, core: &[u8], checksum: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(PREFIX_LEN + 1 + core.len() + FOOTER_LEN);
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    uleb128::encode_u64(profile.to_bits(), &mut out);
    out.extend_from_slice(core);
    out.extend_from_slice(&checksum.to_be_bytes());
    out
}

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
    compress_owned_with(input, profile, EncodeOptions::default())
}

/// Like [`compress_owned`] but with explicit encode-side [`EncodeOptions`] (e.g. the Re-Pair
/// vocabulary cap). Options are not serialized — the decoder recovers everything from the stream.
#[must_use]
pub fn compress_owned_with(input: Vec<u8>, profile: Profile, options: EncodeOptions) -> Vec<u8> {
    // Checksum the original bytes up front so `encode` can consume and free `input` during the build.
    let mut checksum = Adler32::new();
    checksum.update(&input);
    let checksum = checksum.checksum();

    let core = profile.compressor_with(options).encode(input);
    frame(profile, &core, checksum)
}

/// Like [`compress_owned_with`] but also returns each pipeline stage's output size, in order.
///
/// For the CLI's per-stage bits-per-byte report. The stage sizes are codec-core bytes; the returned
/// container additionally carries the (small, fixed) framing.
#[must_use]
pub fn compress_owned_with_traced(
    input: Vec<u8>,
    profile: Profile,
    options: EncodeOptions,
) -> (Vec<u8>, Vec<StageSize>, Vec<ModelScore>) {
    let mut checksum = Adler32::new();
    checksum.update(&input);
    let checksum = checksum.checksum();

    let (core, stages, scores) = profile.compressor_with(options).encode_traced(input);
    let out = frame(profile, &core, checksum);
    let stages = stages
        .into_iter()
        .map(|(name, input_bytes, output_bytes, detail)| StageSize {
            name,
            input_bytes,
            output_bytes,
            detail,
        })
        .collect();
    let models = scores
        .into_iter()
        .map(|(name, bpb_alone, avg_weight)| ModelScore {
            name,
            bpb_alone,
            avg_weight,
        })
        .collect();
    (out, stages, models)
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
        // The default profile serializes as a ULEB128 feature bitmask directly after the version.
        let c = compress(b"hello");
        assert!(c.starts_with(MAGIC));
        assert_eq!(c[3], VERSION);
        let mut pos = PREFIX_LEN;
        assert_eq!(uleb128::decode_u64(&c, &mut pos).unwrap(), Profile::default().to_bits());
        assert!(c.len() >= pos + FOOTER_LEN);
    }

    #[test]
    fn custom_profile_selects_and_round_trips() {
        // Disabling every default feature except the fundamental `repair` leaves only its bit set
        // (profile 0x04); the container records it and `decompress` rebuilds the matching pipeline.
        let mut profile = Profile::default();
        for feature in
            ["casefold", "entities", "lz77", "entropy", "order0", "order1", "sparse2", "sparse24", "varint", "sse"]
        {
            profile.disable(feature).unwrap();
        }
        let c = compress_with(b"hello world", profile);
        assert_eq!(c[PREFIX_LEN], 0x04); // only the mandatory repair bit
        assert_eq!(decompress(&c).unwrap(), b"hello world".to_vec());
    }

    #[test]
    fn rejects_unsupported_profile() {
        // Bit 21 is not a known feature; a container whose ULEB128 profile encodes it must be rejected.
        let mut c = Vec::new();
        c.extend_from_slice(MAGIC);
        c.push(VERSION);
        uleb128::encode_u64(1 << 21, &mut c); // profile with an unknown feature bit
        c.extend_from_slice(&[0u8; FOOTER_LEN]); // enough trailing bytes for the length checks
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
