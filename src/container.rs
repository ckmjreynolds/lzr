//! The two-tier self-describing LZR container: an outer **file** frame wrapping one-or-more
//! independent **block** frames, read and written as a **stream**.
//!
//! ```text
//! FILE:  "LZR" (3) | version (1) | block_count (ULEB128)
//!        [ block ] * block_count
//!        total_uncompressed_len (ULEB128) | adler32 BE (4, over the whole original input)
//!
//! BLOCK: comp_len (ULEB128) | unc_len (ULEB128) | profile (ULEB128 bitmask)
//!        [ codec core : exactly comp_len bytes ]
//!        adler32 BE (4, over this block's original bytes)
//! ```
//!
//! The input is split into fixed-size blocks (default [`DEFAULT_BLOCK_SIZE`]); every block carries its
//! own profile and Adler-32 and is encoded/decoded **independently**, so all model and preprocessor
//! state resets at each boundary. That makes blocks a threading boundary. The [`compress_stream`] /
//! [`decompress_stream`] entry points process blocks in a bounded window (`threads` blocks in flight)
//! so peak memory is `≈ threads × (block_size + scratch)` — **independent of the file size**, never
//! holding the whole input or output at once. `VERSION` is frozen at `0x00` during development: an old
//! stream simply fails to decode.

use std::io::{self, Read, Write};

use anyhow::{Result, ensure};
use rayon::prelude::*;

use crate::adler32::Adler32;
use crate::codec::Profile;
use crate::transform::{ModelTrace, StageTrace};
use crate::uleb128::{decode_u64, encode_u64};

/// Container magic: the ASCII bytes `LZR`.
const MAGIC: &[u8; 3] = b"LZR";
/// Framing version. Frozen at `0x00` during development: the wire format is unstable and we do not
/// bump the version for format changes, so an older stream simply fails to decode (fine pre-1.0).
const VERSION: u8 = 0x00;
/// Fixed file-header prefix before the ULEB128 block count: magic (3) + version (1).
const PREFIX_LEN: usize = 4;
/// Footer length: a big-endian Adler-32 (the file footer's, and each block footer's).
const FOOTER_LEN: usize = 4;

/// Default block size in bytes (100 MiB).
///
/// A block is the unit of independent (de)compression. At 1 GiB a ~1 GB input (e.g. enwik9) is a
/// single block, maximizing per-block model context — the deterministic CM keeps warming over the
/// whole stream with no block-boundary resets — at the cost of cross-block parallelism. It stays
/// within the `u32` ceiling on Re-Pair position indices (1 GiB < 4 GiB). Override with `--block-size`.
pub const DEFAULT_BLOCK_SIZE: usize = 1 << 30;

/// One pipeline stage's input and output size in bytes, paired with its feature name.
///
/// Returned by [`compress_stream`] so the CLI can report each transform's own bits-per-byte
/// (`8 · output / input`). With multiple blocks these are aggregated across all blocks.
#[derive(Debug, Clone, Copy)]
pub struct StageSize {
    /// The stage's feature name (e.g. `"repair"`).
    pub name: &'static str,
    /// The stage's total input length in bytes (summed over blocks).
    pub input_bytes: usize,
    /// The stage's total output length in bytes (summed over blocks).
    pub output_bytes: usize,
    /// An optional stage-specific statistic as `(value, unit label)` — e.g. the Re-Pair rule count
    /// `(n, "rules")`; `None` for stages with nothing extra to report. Counts are summed over blocks.
    pub detail: Option<(u64, &'static str)>,
}

/// One entropy model's encode-side scorecard, for the CLI's per-model report.
///
/// Returned by [`compress_stream`] alongside the stage sizes; averaged across blocks.
#[derive(Debug, Clone, Copy)]
pub struct ModelScore {
    /// The model's feature name (e.g. `"order0"`, or `"null"` for the injected fallback).
    pub name: &'static str,
    /// Bits per (entropy-stage input) byte if this model coded the stream alone — its raw predictive
    /// power. Near 8 means no better than chance; lower is better.
    pub bpb_alone: f64,
    /// The mixer's average final weight on this model. Near zero means it adds little *marginally*.
    pub avg_weight: f64,
}

/// The Adler-32 checksum of `bytes`.
fn checksum(bytes: &[u8]) -> u32 {
    let mut adler = Adler32::new();
    adler.update(bytes);
    adler.checksum()
}

/// Resolve a `threads` request to a worker count: `0` means "all cores", anything else is taken
/// literally (clamped to at least 1).
fn resolve_threads(threads: usize) -> usize {
    if threads == 0 {
        std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
    } else {
        threads
    }
}

/// The number of blocks to process concurrently: `threads` resolved to a worker count, but never more
/// than the block count (a single-block stream stays single-worker and builds no pool).
fn worker_count(threads: usize, block_count: u64) -> usize {
    resolve_threads(threads).min(usize::try_from(block_count).unwrap_or(usize::MAX).max(1))
}

/// A dedicated rayon pool of `jobs` worker threads, or `None` when a single worker suffices (so the
/// common single-block path builds no pool). Falls back to `None` if a pool cannot be built.
fn make_pool(jobs: usize) -> Option<rayon::ThreadPool> {
    if jobs <= 1 {
        return None;
    }
    rayon::ThreadPoolBuilder::new().num_threads(jobs).build().ok()
}

/// Map `f` over one window of block items, in parallel on `pool` (order preserved) or serially when a
/// single worker suffices. The one place the pool/no-pool split lives.
fn run_window<T: Send, U: Send>(
    pool: Option<&rayon::ThreadPool>,
    items: Vec<T>,
    f: impl Fn(T) -> U + Send + Sync,
) -> Vec<U> {
    match pool {
        Some(pool) => pool.install(|| items.into_par_iter().map(f).collect()),
        None => items.into_iter().map(f).collect(),
    }
}

/// Encode a ULEB128 value straight to a writer.
fn write_uleb<W: Write>(writer: &mut W, value: u64) -> io::Result<()> {
    let mut buf = Vec::with_capacity(9);
    encode_u64(value, &mut buf);
    writer.write_all(&buf)
}

/// Read a single ULEB128 value from a reader (at most 9 bytes), rejecting non-canonical encodings.
fn read_uleb<R: Read>(reader: &mut R) -> Result<u64> {
    let mut bytes = [0u8; 9];
    let mut len = 0;
    loop {
        reader.read_exact(&mut bytes[len..=len])?;
        let done = bytes[len] & 0x80 == 0;
        len += 1;
        if done || len == bytes.len() {
            break;
        }
    }
    let mut pos = 0;
    decode_u64(&bytes[..len], &mut pos)
}

/// Read up to `n` bytes (fewer only at EOF), pre-reserving `capacity`. Callers pass `capacity = n` when
/// `n` is a trusted block size (avoids the `read_to_end` doubling); the decoder passes `0` for an
/// untrusted `comp_len`, so a bogus length grows incrementally rather than pre-allocating.
fn read_chunk<R: Read>(reader: &mut R, n: usize, capacity: usize) -> io::Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(capacity);
    let _ = reader.by_ref().take(n as u64).read_to_end(&mut buf)?;
    Ok(buf)
}

/// Frame one block: `comp_len | unc_len | profile | core | adler32`.
fn frame_block(profile: Profile, unc_len: usize, core: &[u8], block_checksum: u32) -> Vec<u8> {
    let mut block = Vec::with_capacity(3 * 9 + core.len() + FOOTER_LEN);
    encode_u64(core.len() as u64, &mut block);
    encode_u64(unc_len as u64, &mut block);
    encode_u64(profile.to_bits(), &mut block);
    block.extend_from_slice(core);
    block.extend_from_slice(&block_checksum.to_be_bytes());
    block
}

/// Compress one block's `chunk` into its framed bytes, plus its per-stage/per-model traces when
/// `traced` (the CLI report needs them; the plain library path skips the tracking overhead).
fn encode_chunk(profile: Profile, chunk: Vec<u8>, traced: bool) -> (Vec<u8>, Vec<StageTrace>, Vec<ModelTrace>) {
    let unc_len = chunk.len();
    let block_checksum = checksum(&chunk);
    let (core, stages, scores) = if traced {
        profile.compressor().encode_traced(chunk)
    } else {
        (profile.compressor().encode(chunk), Vec::new(), Vec::new())
    };
    (frame_block(profile, unc_len, &core, block_checksum), stages, scores)
}

/// A parsed-but-not-yet-decoded block: its profile, expected length, core bytes, and stored checksum.
type Frame = (Profile, u64, Vec<u8>, u32);

/// Decode and verify one block frame: run its pipeline, then check its length and Adler-32.
fn decode_frame(frame: Frame) -> Result<Vec<u8>> {
    let (profile, unc_len, core, stored) = frame;
    let bytes = profile.compressor().decode(&core)?;
    ensure!(bytes.len() as u64 == unc_len, "block length does not match its header");
    let actual = checksum(&bytes);
    ensure!(actual == stored, "block checksum mismatch: stored {stored:#010x}, computed {actual:#010x}");
    Ok(bytes)
}

/// Running accumulator for a model's per-block scores, averaged in [`finalize_models`].
struct ModelAcc {
    name: &'static str,
    bpb_sum: f64,
    weight_sum: f64,
}

/// Fold one block's stage traces into the aggregate (summed by stage index; all blocks share a profile).
fn merge_stages(agg: &mut Vec<StageSize>, stages: &[StageTrace]) {
    for (i, &(name, input_bytes, output_bytes, detail)) in stages.iter().enumerate() {
        if let Some(slot) = agg.get_mut(i) {
            slot.input_bytes += input_bytes;
            slot.output_bytes += output_bytes;
            slot.detail = match (slot.detail, detail) {
                (Some((acc, unit)), Some((add, _))) => Some((acc + add, unit)),
                (existing, _) => existing,
            };
        } else {
            agg.push(StageSize {
                name,
                input_bytes,
                output_bytes,
                detail,
            });
        }
    }
}

/// Fold one block's model traces into the running per-model sums.
fn merge_models(agg: &mut Vec<ModelAcc>, models: &[ModelTrace]) {
    for (i, &(name, bpb_alone, avg_weight)) in models.iter().enumerate() {
        if let Some(slot) = agg.get_mut(i) {
            slot.bpb_sum += bpb_alone;
            slot.weight_sum += avg_weight;
        } else {
            agg.push(ModelAcc {
                name,
                bpb_sum: bpb_alone,
                weight_sum: avg_weight,
            });
        }
    }
}

/// Average the running per-model sums over `n_blocks`.
#[expect(clippy::cast_precision_loss, reason = "Diagnostic display; block counts are tiny.")]
fn finalize_models(agg: Vec<ModelAcc>, n_blocks: u64) -> Vec<ModelScore> {
    let n = n_blocks.max(1) as f64;
    agg.into_iter()
        .map(|m| ModelScore {
            name: m.name,
            bpb_alone: m.bpb_sum / n,
            avg_weight: m.weight_sum / n,
        })
        .collect()
}

/// Compress a stream of `input_len` bytes into the LZR container, writing to `writer`.
///
/// The input is split into `block_size`-byte blocks; up to `threads` blocks (0 = all cores) are
/// compressed at once, and each is written out as soon as it and its predecessors are ready — so peak
/// memory is bounded by `threads`, not the file size. Returns the aggregated per-stage and per-model
/// traces for the caller's report.
///
/// # Errors
///
/// Propagates any read or write I/O error.
pub fn compress_stream<R: Read, W: Write>(
    reader: R,
    writer: W,
    input_len: u64,
    profile: Profile,
    block_size: usize,
    threads: usize,
) -> Result<(Vec<StageSize>, Vec<ModelScore>)> {
    stream_compress(reader, writer, input_len, profile, block_size, threads, true)
}

/// The shared streaming compress loop; `traced` selects the per-chunk report tracking.
fn stream_compress<R: Read, W: Write>(
    mut reader: R,
    mut writer: W,
    input_len: u64,
    profile: Profile,
    block_size: usize,
    threads: usize,
    traced: bool,
) -> Result<(Vec<StageSize>, Vec<ModelScore>)> {
    let block_size = block_size.max(1);
    let block_count = input_len.div_ceil(block_size as u64);
    let jobs = worker_count(threads, block_count);
    let pool = make_pool(jobs);

    writer.write_all(MAGIC)?;
    writer.write_all(&[VERSION])?;
    write_uleb(&mut writer, block_count)?;

    let mut file_adler = Adler32::new();
    let mut stages_agg: Vec<StageSize> = Vec::new();
    let mut models_agg: Vec<ModelAcc> = Vec::new();

    loop {
        // Read a window of up to `jobs` chunks, folding each into the whole-file checksum as it arrives.
        // A short/empty read is EOF: the window ends and the outer loop terminates on the next pass.
        let mut window: Vec<Vec<u8>> = Vec::new();
        for _ in 0..jobs {
            let chunk = read_chunk(&mut reader, block_size, block_size)?;
            if chunk.is_empty() {
                break;
            }
            file_adler.update(&chunk);
            window.push(chunk);
        }
        if window.is_empty() {
            break;
        }
        for (block, stages, models) in run_window(pool.as_ref(), window, |c| encode_chunk(profile, c, traced)) {
            writer.write_all(&block)?;
            merge_stages(&mut stages_agg, &stages);
            merge_models(&mut models_agg, &models);
        }
    }

    write_uleb(&mut writer, input_len)?;
    writer.write_all(&file_adler.checksum().to_be_bytes())?;
    writer.flush()?;

    Ok((stages_agg, finalize_models(models_agg, block_count)))
}

/// Decompress an LZR container from `reader`, writing the recovered bytes to `writer`.
///
/// Blocks are decoded in a bounded window of up to `threads` at a time (0 = all cores), verifying each
/// block's profile/length/Adler-32 and the whole-file length and Adler-32. Peak memory is bounded by
/// `threads`, not the output size.
///
/// # Errors
///
/// Returns an error if the magic/version is unrecognized, a block frame is malformed or truncated, a
/// profile is unknown, a codec core is corrupt, any stored length/checksum mismatches, or on I/O error.
pub fn decompress_stream<R: Read, W: Write>(mut reader: R, mut writer: W, threads: usize) -> Result<()> {
    let mut header = [0u8; PREFIX_LEN];
    reader.read_exact(&mut header)?;
    ensure!(&header[..3] == MAGIC, "bad magic: not an LZR container");
    ensure!(header[3] == VERSION, "unsupported container version {}", header[3]);

    let block_count = read_uleb(&mut reader)?;
    let jobs = worker_count(threads, block_count);
    let pool = make_pool(jobs);

    let mut file_adler = Adler32::new();
    let mut total_out: u64 = 0;
    let mut remaining = block_count;

    while remaining > 0 {
        let take = usize::try_from(remaining.min(jobs as u64)).unwrap_or(1);
        // Read `take` block frames into memory (bounded by `jobs`), each self-delimited by `comp_len`.
        let mut frames: Vec<Frame> = Vec::with_capacity(take);
        for _ in 0..take {
            let comp_len = read_uleb(&mut reader)?;
            let unc_len = read_uleb(&mut reader)?;
            let profile = Profile::from_bits(read_uleb(&mut reader)?)?;
            // `comp_len` is untrusted here, so read it with no pre-allocation (grows to what exists).
            let core = read_chunk(&mut reader, usize::try_from(comp_len).unwrap_or(usize::MAX), 0)?;
            ensure!(core.len() as u64 == comp_len, "block core is truncated");
            let mut stored = [0u8; FOOTER_LEN];
            reader.read_exact(&mut stored)?;
            frames.push((profile, unc_len, core, u32::from_be_bytes(stored)));
        }
        // Decode the window (in parallel when a pool exists), preserving order.
        for bytes in run_window(pool.as_ref(), frames, decode_frame).into_iter().collect::<Result<Vec<_>>>()? {
            file_adler.update(&bytes);
            total_out += bytes.len() as u64;
            writer.write_all(&bytes)?;
        }
        remaining -= take as u64;
    }

    let total_unc = read_uleb(&mut reader)?;
    let mut stored = [0u8; FOOTER_LEN];
    reader.read_exact(&mut stored)?;
    let file_checksum = u32::from_be_bytes(stored);
    ensure!(total_out == total_unc, "decoded length does not match the file footer");
    let actual = file_adler.checksum();
    ensure!(actual == file_checksum, "checksum mismatch: stored {file_checksum:#010x}, computed {actual:#010x}");
    // Nothing may follow the footer.
    let mut extra = [0u8; 1];
    ensure!(matches!(reader.read(&mut extra), Ok(0)), "trailing bytes after the file footer");
    writer.flush()?;
    Ok(())
}

// --- In-memory convenience wrappers (used by the library API and tests) ---

/// Compress `input` into a self-describing LZR container using the default [`Profile`].
#[must_use]
pub fn compress(input: &[u8]) -> Vec<u8> {
    compress_with(input, Profile::default())
}

/// Compress `input` into a self-describing LZR container using `profile` to select the pipeline.
#[must_use]
pub fn compress_with(input: &[u8], profile: Profile) -> Vec<u8> {
    compress_to_vec(input, profile, DEFAULT_BLOCK_SIZE, false).0
}

/// Compress `input` into an in-memory LZR container with an explicit block size (no tracing).
#[must_use]
pub fn compress_owned_with_blocks(input: &[u8], profile: Profile, block_size: usize) -> Vec<u8> {
    compress_to_vec(input, profile, block_size, false).0
}

/// Like [`compress_owned_with_blocks`] but also returns aggregated per-stage and per-model traces.
#[must_use]
pub fn compress_owned_with_traced(
    input: &[u8],
    profile: Profile,
    block_size: usize,
) -> (Vec<u8>, Vec<StageSize>, Vec<ModelScore>) {
    compress_to_vec(input, profile, block_size, true)
}

/// Shared in-memory compress: stream into a `Vec` (single-threaded, since the whole buffer is already
/// resident). Writing to a `Vec` is infallible, so the streaming `Result` is always `Ok`; the
/// `unwrap_or_default` only satisfies the type and never runs (the lints forbid `expect`/`unwrap`).
fn compress_to_vec(
    input: &[u8],
    profile: Profile,
    block_size: usize,
    traced: bool,
) -> (Vec<u8>, Vec<StageSize>, Vec<ModelScore>) {
    let mut out = Vec::new();
    let (stages, models) =
        stream_compress(input, &mut out, input.len() as u64, profile, block_size, 1, traced).unwrap_or_default();
    (out, stages, models)
}

/// Decompress an in-memory LZR container.
///
/// # Errors
///
/// Returns an error if the container is malformed, corrupt, or fails any length/checksum check.
pub fn decompress(input: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    decompress_stream(input, &mut out, 0)?;
    Ok(out)
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

        /// Round-trips across a range of small block sizes, forcing many blocks.
        #[test]
        fn roundtrip_multi_block(data in prop::collection::vec(any::<u8>(), 0..4096), bs in 1usize..=257) {
            let c = compress_owned_with_blocks(&data, Profile::default(), bs);
            prop_assert_eq!(decompress(&c).unwrap(), data);
        }

        /// Round-trips across thread counts (single-threaded through several workers).
        #[test]
        fn roundtrip_threads(data in prop::collection::vec(any::<u8>(), 0..4096), threads in 0usize..=4) {
            let mut packed = Vec::new();
            drop(compress_stream(data.as_slice(), &mut packed, data.len() as u64, Profile::default(), 128, threads).unwrap());
            let mut out = Vec::new();
            decompress_stream(packed.as_slice(), &mut out, threads).unwrap();
            prop_assert_eq!(out, data);
        }

        /// Decompressing arbitrary bytes must never panic — only Ok or Err.
        #[test]
        fn decompress_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..2048)) {
            drop(decompress(&bytes));
        }
    }

    #[test]
    fn empty_input_roundtrips() {
        let c = compress(b"");
        assert_eq!(c[..3], *MAGIC);
        assert_eq!(c[3], VERSION);
        assert_eq!(decompress(&c).unwrap(), b"");
    }

    #[test]
    fn multi_block_produces_multiple_blocks() {
        // A 1000-byte input at a 100-byte block size yields 10 blocks; the header records the count.
        let data = b"the quick brown fox jumps over the lazy dog. ".repeat(64);
        let c = compress_owned_with_blocks(&data, Profile::default(), 100);
        let mut pos = PREFIX_LEN;
        let block_count = decode_u64(&c, &mut pos).unwrap();
        assert_eq!(block_count, data.len().div_ceil(100) as u64);
        assert_eq!(decompress(&c).unwrap(), data);
    }

    #[test]
    fn traced_reports_aggregated_stages() {
        // Enable repair explicitly (it is default-off) so the trace exercises a tokenizer stage too.
        let mut profile = Profile::default();
        profile.enable("repair").unwrap();
        let data = b"The QUICK brown fox. ".repeat(200);
        let (out, stages, _models) = compress_owned_with_traced(&data, profile, 128);
        assert_eq!(decompress(&out).unwrap(), data);
        assert!(stages.iter().any(|s| s.name == "repair"));
    }

    #[test]
    fn detects_block_payload_corruption() {
        let original = b"the quick brown fox jumps over the lazy dog".repeat(4);
        let mut c = compress_owned_with_blocks(&original, Profile::default(), 64);
        let mid = c.len() / 2;
        c[mid] ^= 0x01;
        if let Ok(out) = decompress(&c) {
            assert_ne!(out, original);
        }
    }

    #[test]
    fn detects_file_footer_corruption() {
        let mut c = compress(b"payload bytes");
        let last = c.len() - 1;
        c[last] ^= 0xFF; // corrupt the whole-file checksum
        assert!(decompress(&c).is_err());
    }

    #[test]
    fn rejects_truncated_and_bad_magic() {
        assert!(decompress(&[]).is_err());
        assert!(decompress(b"LZR").is_err());
        assert!(decompress(b"XYZ\x00\x00\x00\x00\x00\x00\x00\x00\x00").is_err());
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut c = compress(b"data");
        c[3] = 0xFF;
        assert!(decompress(&c).is_err());
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut c = compress(b"hello world");
        c.push(0x00); // an extra byte after the footer
        assert!(decompress(&c).is_err());
    }

    #[test]
    fn custom_profile_round_trips() {
        let mut profile = Profile::default();
        profile.enable("order1").unwrap();
        profile.enable("match").unwrap();
        let c = compress_with(b"hello world hello world hello world", profile);
        assert_eq!(decompress(&c).unwrap(), b"hello world hello world hello world");
    }
}
