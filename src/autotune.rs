//! Automatic operating-point selection.
//!
//! The final compressed size is not monotone in the Re-Pair vocabulary size, nor is the LZ77 stage a
//! uniform win: a *small* vocabulary yields a dense token stream the higher-order context models
//! exploit well, and at that operating point the LZ77 stage's match records tend to hurt (the entropy
//! coder's own match model already captures the repeats), whereas at a large vocabulary LZ77 helps.
//! The best point is data-dependent and cannot be found by any per-rule grammar cost model — the
//! benefit of a small alphabet is a *global* effect on the model statistics. So [`compress_auto`]
//! searches the true objective directly: it trial-compresses a prefix sample across a ladder of
//! vocabulary caps with LZ77 both on and off, then does one full compress at the winner.
//!
//! Everything here is encode-side: the chosen point produces an ordinary, self-describing container
//! (the LZ77 bit lives in the serialized profile; the vocabulary cap is recovered from the grammar),
//! so a stream from `compress_auto` decodes exactly like any other.

use crate::codec::{EncodeOptions, Profile};
use crate::container::{ModelScore, StageSize, compress_owned_with, compress_owned_with_traced};
use crate::preprocessors::{DEFAULT_NUM_TOKENS, MIN_NUM_TOKENS};

/// Default prefix, in bytes, the search evaluates candidates on.
///
/// The winning operating point is a property of the data's statistics more than its length, so a
/// representative prefix is enough — and it bounds the search cost independently of the (possibly
/// enormous) full input.
pub const DEFAULT_SAMPLE_BYTES: usize = 2_000_000;

/// Upper bound of the vocabulary-cap ladder. The best vocabulary is small in practice (hundreds to a
/// few thousand tokens); the natural (uncapped) cost-stop grammar is also always evaluated, so the
/// ladder need not reach the full ceiling.
const LADDER_HI: u32 = 16_384;

/// Cap on full-input trial compresses during the refinement hill-climb. A prefix sample *undershoots*
/// the full-input optimum (the best vocabulary grows with input size), so after the cheap sample search
/// the winner is polished directly on the whole input — but each such trial is a full compress, so the
/// count is bounded. The climb also stops early at a local minimum.
const MAX_REFINE_EVALS: usize = 6;

/// The pipeline knobs [`compress_auto`] chooses.
#[derive(Debug, Clone, Copy)]
pub struct OperatingPoint {
    /// The Re-Pair vocabulary cap, or `None` for the natural cost-stop grammar (no count cap).
    pub num_tokens: Option<u32>,
    /// Whether the LZ77 stage is enabled.
    pub lz77: bool,
}

/// What [`compress_auto`] chose and how much work it did, for the caller's trace.
#[derive(Debug, Clone, Copy)]
pub struct AutoReport {
    /// The winning operating point, applied to the full input.
    pub point: OperatingPoint,
    /// Number of trial compresses run on the prefix sample during the coarse search.
    pub sample_evaluations: usize,
    /// Number of full-input trial compresses run during the refinement hill-climb (0 when the sample
    /// already covered the whole input, so no refinement was needed).
    pub refine_evaluations: usize,
    /// Bytes of the input the coarse search evaluated candidates on.
    pub sampled_bytes: usize,
}

/// Returns `profile` with the LZ77 stage forced on or off. `"lz77"` is a known feature, so the toggle
/// cannot fail; on the impossible error the profile is left unchanged.
fn with_lz(mut profile: Profile, lz77: bool) -> Profile {
    if lz77 {
        drop(profile.enable("lz77"));
    } else {
        drop(profile.disable("lz77"));
    }
    profile
}

/// Encode options for a ladder rung: `None` is the natural cost-stop prune (default options), `Some(n)`
/// a hard count cap at `n` tokens. `n` comes from [`ladder`], always in the valid range, so the
/// validation cannot fail; the fallback is defensive. The Re-Pair minimal-grammar-parse re-parse
/// (MGP) is enabled for every candidate — it is a small, always-reversible win on the token stream,
/// so the search evaluates (and the winner is compressed with) MGP on throughout.
fn options_for(num_tokens: Option<u32>) -> EncodeOptions {
    num_tokens.map_or_else(EncodeOptions::default, |n| EncodeOptions::new(n).unwrap_or_default()).with_mgp()
}

/// The geometric vocabulary-cap ladder, `MIN_NUM_TOKENS` up to [`LADDER_HI`] by ~1.5×. Each rung is a
/// count-cap candidate; the search also evaluates the uncapped natural grammar separately.
fn ladder() -> Vec<u32> {
    let hi = LADDER_HI.min(DEFAULT_NUM_TOKENS);
    let mut rungs = Vec::new();
    let mut n = MIN_NUM_TOKENS;
    while n < hi {
        rungs.push(n);
        n = (n / 2).saturating_mul(3).max(n + 1);
    }
    rungs
}

/// The ordered candidate vocabulary caps: every [`ladder`] rung ascending, then `None` (the uncapped
/// natural grammar) as the largest. The refinement hill-climb walks this order.
fn candidates() -> Vec<Option<u32>> {
    ladder().into_iter().map(Some).chain(std::iter::once(None)).collect()
}

/// Full-input container size at `candidates[idx]` (LZ77 fixed), memoized into `sizes` and counted in
/// `evals`. A cache hit costs nothing; a miss is one full compress.
fn sized(
    input: &[u8],
    profile: Profile,
    lz77: bool,
    candidates: &[Option<u32>],
    idx: usize,
    sizes: &mut [Option<usize>],
    evals: &mut usize,
) -> usize {
    if let Some(size) = sizes[idx] {
        return size;
    }
    let size = compress_owned_with(input.to_vec(), with_lz(profile, lz77), options_for(candidates[idx])).len();
    sizes[idx] = Some(size);
    *evals += 1;
    size
}

/// Refine the sample-chosen vocabulary on the *full* input: a bounded hill-climb over [`candidates`]
/// from `start` (LZ77 fixed to the sample's choice), moving to whichever neighbour compresses smaller
/// until a local minimum or [`MAX_REFINE_EVALS`] full compresses. Returns the winning index and the
/// number of full compresses spent. Climbing both directions handles the usual undershoot (optimum
/// above the sample's pick) and the rare overshoot.
fn refine_on_full(
    input: &[u8],
    profile: Profile,
    lz77: bool,
    candidates: &[Option<u32>],
    start: usize,
) -> (usize, usize) {
    let mut sizes: Vec<Option<usize>> = vec![None; candidates.len()];
    let mut evals = 0usize;
    let mut best = start;
    let mut best_size = sized(input, profile, lz77, candidates, best, &mut sizes, &mut evals);
    while evals < MAX_REFINE_EVALS {
        let mut moved = false;
        let neighbours = [best.checked_sub(1), (best + 1 < candidates.len()).then_some(best + 1)];
        for nb in neighbours.into_iter().flatten() {
            if evals >= MAX_REFINE_EVALS {
                break;
            }
            let size = sized(input, profile, lz77, candidates, nb, &mut sizes, &mut evals);
            if size < best_size {
                best_size = size;
                best = nb;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    (best, evals)
}

/// Compress `input` after searching for the best operating point (vocabulary cap × LZ77 on/off).
///
/// A coarse search on a prefix sample of at most `sample_cap` bytes picks a starting point (for LZ77
/// both off and on it evaluates the natural grammar plus every [`ladder`] cap, keeping the smallest);
/// then, if the sample did not already cover the whole input, a bounded [`refine_on_full`] hill-climb
/// polishes the vocabulary directly on the full input (the sample undershoots — the best vocabulary
/// grows with input size). It starts from `profile` — the model/flag selection is preserved, only LZ77
/// is toggled — and returns the full container, its per-stage and per-model traces (as
/// [`compress_owned_with_traced`]), and an [`AutoReport`].
#[must_use]
pub fn compress_auto(
    input: Vec<u8>,
    profile: Profile,
    sample_cap: usize,
) -> (Vec<u8>, Vec<StageSize>, Vec<ModelScore>, AutoReport) {
    let sampled_bytes = input.len().min(sample_cap.max(1));
    let sample = &input[..sampled_bytes];
    let candidates = candidates();

    let mut best: Option<(usize, OperatingPoint)> = None;
    let mut sample_evaluations = 0usize;
    for lz77 in [false, true] {
        let prof = with_lz(profile, lz77);
        for &num_tokens in &candidates {
            let size = compress_owned_with(sample.to_vec(), prof, options_for(num_tokens)).len();
            sample_evaluations += 1;
            if best.is_none_or(|(best_size, _)| size < best_size) {
                best = Some((
                    size,
                    OperatingPoint {
                        num_tokens,
                        lz77,
                    },
                ));
            }
        }
    }
    // `sample`'s borrow of `input` ends here (last use above), so `input` can move into the final
    // compress below.

    // `best` is always set — the loop runs at least the first candidate. The fallback point is the plain
    // default so a degenerate search still produces a valid container.
    let sample_point = best.map_or(
        OperatingPoint {
            num_tokens: None,
            lz77: false,
        },
        |(_, p)| p,
    );

    // Refine on the full input unless the sample already covered it (then the coarse search was exact).
    let (point, refine_evaluations) = if sampled_bytes < input.len() {
        let start = candidates.iter().position(|&c| c == sample_point.num_tokens).unwrap_or(0);
        let (best_idx, evals) = refine_on_full(&input, profile, sample_point.lz77, &candidates, start);
        (
            OperatingPoint {
                num_tokens: candidates[best_idx],
                lz77: sample_point.lz77,
            },
            evals,
        )
    } else {
        (sample_point, 0)
    };

    let (output, stages, models) =
        compress_owned_with_traced(input, with_lz(profile, point.lz77), options_for(point.num_tokens));
    (
        output,
        stages,
        models,
        AutoReport {
            point,
            sample_evaluations,
            refine_evaluations,
            sampled_bytes,
        },
    )
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::container::decompress;

    #[test]
    fn ladder_is_ascending_and_bounded() {
        let rungs = ladder();
        assert_eq!(rungs.first().copied(), Some(MIN_NUM_TOKENS));
        assert!(rungs.windows(2).all(|w| w[0] < w[1]), "ladder must strictly ascend");
        assert!(rungs.iter().all(|&n| n <= LADDER_HI.min(DEFAULT_NUM_TOKENS)));
    }

    #[test]
    fn auto_roundtrips_and_reports() {
        // A repetitive corpus so the search has real candidates to weigh; the chosen point must still
        // produce a container that decodes back to the original.
        let data = b"the quick brown fox jumps over the lazy dog. ".repeat(200);
        let (output, stages, _models, report) = compress_auto(data.clone(), Profile::default(), 4096);
        assert_eq!(decompress(&output).unwrap(), data);
        // Both LZ77 states and every candidate are evaluated on the sample, so many trials ran.
        assert!(report.sample_evaluations >= 2, "expected multiple sample trials, got {}", report.sample_evaluations);
        // The 4 KiB sample is smaller than the ~9 KiB input, so a refinement pass runs too.
        assert!(report.refine_evaluations >= 1, "expected refinement, got {}", report.refine_evaluations);
        assert!(report.sampled_bytes < data.len());
        assert!(!stages.is_empty());
    }

    #[test]
    fn auto_handles_empty_and_tiny_inputs() {
        for data in [Vec::new(), b"a".to_vec(), b"abababab".to_vec()] {
            let (output, _, _, _) = compress_auto(data.clone(), Profile::default(), 1_000_000);
            assert_eq!(decompress(&output).unwrap(), data, "auto must round-trip {data:?}");
        }
    }
}
