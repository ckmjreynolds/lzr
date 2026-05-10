//! Fixed 5-offset codec eval panel.
//!
//! Single-window 16 KiB codec measurements (the existing per-checkpoint
//! test in `train.rs`) are dominated by content variance — the same
//! model can land 0.87 bpb on one offset and 2.65 bpb on another. For
//! iteration-driven tuning of the deterministic stack (and later for
//! ensemble weight tuning) we need a fixed measurement panel so a 1%
//! change shows up as 1%, not as ±60% of noise. See the deferred-fix
//! note in the 2026-04-23 journal entry.
//!
//! Five windows at fixed enwik9 offsets (100/300/500/700/900 MB), each
//! 16 KiB. Reports per-offset bpb plus mean. Pure deterministic CPU
//! work; no model load.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::codec::{HEADER_LEN, ProbSource, decode_bytes, encode_bytes};
use crate::tokenizer::{SCHEMA_VERSION, Token, Tokenizer};

const SAMPLE_BYTES: usize = 16 * 1024;

/// Bytes fed through the predictor before the measured slice so its
/// adaptive state has converged. 4 MiB matches the LZ77 sliding-window
/// size, lets PPM-class predictors see ~5× more contexts than at 1
/// MiB, and projects closer to the 1 GB submission's converged regime.
/// Higher-order predictors continue to benefit from more pre-warm; this
/// knob can grow if a future architectural step demands it.
const PREWARM_BYTES: usize = 1 << 22;

pub(crate) const OFFSETS: [u64; 5] = [
    100_000_000,
    300_000_000,
    500_000_000,
    700_000_000,
    900_000_000,
];

pub(crate) struct OffsetResult {
    pub offset: u64,
    pub bpb: f64,
}

pub(crate) struct BenchResult {
    pub per_offset: Vec<OffsetResult>,
    pub mean_bpb: f64,
}

/// Run the panel against a freshly-constructed predictor per offset.
///
/// `make_predictor` is a factory rather than a single instance so each
/// window starts from the predictor's empty state — otherwise the
/// later windows would benefit from the earlier windows' adaptive
/// counts and we'd report unrepresentatively-low bpb.
pub(crate) fn run<F>(corpus: &Path, mut make_predictor: F) -> Result<BenchResult>
where
    F: FnMut() -> Box<dyn ProbSource>,
{
    let mut f =
        File::open(corpus).with_context(|| format!("opening corpus {}", corpus.display()))?;
    let tok = identity_tokenizer();

    let mut per_offset = Vec::with_capacity(OFFSETS.len());
    for &off in &OFFSETS {
        // Pre-warm bytes immediately preceding the measured offset so
        // the predictor's adaptive state is in the regime the real
        // submission would be in by this byte position.
        let warm_start = off.saturating_sub(PREWARM_BYTES as u64);
        let warm_len = usize::try_from(off - warm_start)
            .expect("PREWARM_BYTES bounds the difference to usize range");
        let warm = if warm_len > 0 {
            f.seek(SeekFrom::Start(warm_start))?;
            let mut buf = vec![0u8; warm_len];
            f.read_exact(&mut buf)?;
            buf
        } else {
            Vec::new()
        };

        f.seek(SeekFrom::Start(off))?;
        let mut buf = vec![0u8; SAMPLE_BYTES];
        f.read_exact(&mut buf)?;

        // Encode with a freshly pre-warmed predictor.
        let mut probs_e = make_predictor();
        prewarm(probs_e.as_mut(), &warm);
        let mut archive = Vec::new();
        encode_bytes(&buf, &tok, probs_e.as_mut(), &mut archive)?;

        // Roundtrip-verify with an independent predictor instance
        // pre-warmed identically. Cheap insurance against silent
        // encode/decode desync as predictors get more elaborate.
        let mut probs_d = make_predictor();
        prewarm(probs_d.as_mut(), &warm);
        let mut cur = &archive[..];
        let decoded = decode_bytes(&mut cur, &tok, probs_d.as_mut())?;
        if decoded != buf {
            bail!("roundtrip mismatch at offset {off}");
        }

        let payload_bytes = archive.len().saturating_sub(HEADER_LEN);
        #[allow(clippy::cast_precision_loss)]
        let bpb = (payload_bytes as f64 * 8.0) / SAMPLE_BYTES as f64;
        per_offset.push(OffsetResult { offset: off, bpb });
    }

    #[allow(clippy::cast_precision_loss)]
    let mean_bpb = per_offset.iter().map(|r| r.bpb).sum::<f64>() / (per_offset.len() as f64);

    Ok(BenchResult {
        per_offset,
        mean_bpb,
    })
}

/// Same panel as [`run`], but for LZ77 layered on the type-routed
/// codec. Pre-warm and measurement use a single buffer so the matcher
/// can resolve back-references into the prewarm region.
///
/// If `neural` is `Some((weights, mix_weight))`, the literal-position
/// CDF is a linear mix of the routed class predictor and the neural
/// model's per-position prediction. `mix_weight` is the neural's share
/// in `[0, 1]`.
pub(crate) fn run_lz_routed(
    corpus: &Path,
    neural: Option<&(crate::weights::Weights, f32, crate::lz77::MixMode)>,
) -> Result<BenchResult> {
    use crate::lz77::{LzRouted, decode_into, encode_range};

    let mut f =
        File::open(corpus).with_context(|| format!("opening corpus {}", corpus.display()))?;
    let mut per_offset = Vec::with_capacity(OFFSETS.len());

    for &off in &OFFSETS {
        let warm_start = off.saturating_sub(PREWARM_BYTES as u64);
        let warm_len = usize::try_from(off - warm_start)
            .expect("PREWARM_BYTES bounds the difference to usize range");
        let warm = if warm_len > 0 {
            f.seek(SeekFrom::Start(warm_start))?;
            let mut buf = vec![0u8; warm_len];
            f.read_exact(&mut buf)?;
            buf
        } else {
            Vec::new()
        };

        f.seek(SeekFrom::Start(off))?;
        let mut measurement = vec![0u8; SAMPLE_BYTES];
        f.read_exact(&mut measurement)?;

        // Encode side: build a single buffer of warm + measurement so
        // the matcher's hashes can look back into the warm window.
        let mut buf: Vec<u8> = Vec::with_capacity(warm.len() + measurement.len());
        buf.extend_from_slice(&warm);
        let warm_end = buf.len();
        buf.extend_from_slice(&measurement);

        let mut probs_e = LzRouted::with_neural(neural.cloned());
        probs_e.prewarm(&warm);
        let mut archive = Vec::new();
        encode_range(&buf, warm_end, &mut probs_e, &mut archive)?;

        // Decode side: independent state, same prewarm, prepopulate
        // output buffer with the warm prefix so back-references resolve.
        let mut probs_d = LzRouted::with_neural(neural.cloned());
        probs_d.prewarm(&warm);
        let mut decoded: Vec<u8> = warm.clone();
        let mut cur = &archive[..];
        decode_into(&mut cur, &mut probs_d, &mut decoded)?;
        if &decoded[warm_end..] != measurement.as_slice() {
            bail!("LZ-routed roundtrip mismatch at offset {off}");
        }

        let payload_bytes = archive.len().saturating_sub(HEADER_LEN);
        #[allow(clippy::cast_precision_loss)]
        let bpb = (payload_bytes as f64 * 8.0) / SAMPLE_BYTES as f64;
        per_offset.push(OffsetResult { offset: off, bpb });
    }

    #[allow(clippy::cast_precision_loss)]
    let mean_bpb = per_offset.iter().map(|r| r.bpb).sum::<f64>() / (per_offset.len() as f64);

    Ok(BenchResult {
        per_offset,
        mean_bpb,
    })
}

/// Same panel as [`run`], but for the type-routed codec — three
/// independent sub-predictors, separate encode/decode entry points.
pub(crate) fn run_routed(corpus: &Path) -> Result<BenchResult> {
    use crate::routed::{
        RoutedProbs, decode_bytes as decode_routed, encode_bytes as encode_routed,
    };

    let mut f =
        File::open(corpus).with_context(|| format!("opening corpus {}", corpus.display()))?;
    let mut per_offset = Vec::with_capacity(OFFSETS.len());

    for &off in &OFFSETS {
        let warm_start = off.saturating_sub(PREWARM_BYTES as u64);
        let warm_len = usize::try_from(off - warm_start)
            .expect("PREWARM_BYTES bounds the difference to usize range");
        let warm = if warm_len > 0 {
            f.seek(SeekFrom::Start(warm_start))?;
            let mut buf = vec![0u8; warm_len];
            f.read_exact(&mut buf)?;
            buf
        } else {
            Vec::new()
        };

        f.seek(SeekFrom::Start(off))?;
        let mut buf = vec![0u8; SAMPLE_BYTES];
        f.read_exact(&mut buf)?;

        let mut probs_e = RoutedProbs::new();
        probs_e.prewarm(&warm);
        let mut archive = Vec::new();
        encode_routed(&buf, &mut probs_e, &mut archive)?;

        let mut probs_d = RoutedProbs::new();
        probs_d.prewarm(&warm);
        let mut cur = &archive[..];
        let decoded = decode_routed(&mut cur, &mut probs_d)?;
        if decoded != buf {
            bail!("routed roundtrip mismatch at offset {off}");
        }

        let payload_bytes = archive.len().saturating_sub(HEADER_LEN);
        #[allow(clippy::cast_precision_loss)]
        let bpb = (payload_bytes as f64 * 8.0) / SAMPLE_BYTES as f64;
        per_offset.push(OffsetResult { offset: off, bpb });
    }

    #[allow(clippy::cast_precision_loss)]
    let mean_bpb = per_offset.iter().map(|r| r.bpb).sum::<f64>() / (per_offset.len() as f64);

    Ok(BenchResult {
        per_offset,
        mean_bpb,
    })
}

/// Advance the predictor through `warm` without using its CDFs — pure
/// state-priming pass.
fn prewarm(probs: &mut dyn ProbSource, warm: &[u8]) {
    if warm.is_empty() {
        return;
    }
    let _ = probs.initial_cdf();
    for &b in warm {
        let _ = probs.advance(Token::from(b));
    }
}

/// Build an identity (no-merges) byte-level tokenizer in memory so the
/// codec runs as a straight byte-by-byte pipeline.
fn identity_tokenizer() -> Tokenizer {
    let mut buf = Vec::with_capacity(12);
    buf.extend_from_slice(&SCHEMA_VERSION.to_le_bytes());
    buf.extend_from_slice(&256u32.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    Tokenizer::from_bytes(&buf).expect("identity tokenizer is well-formed")
}
