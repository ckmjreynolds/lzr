//! Diagnostic tool: extract the Content-literal byte stream that
//! `xml-lz-cp` would defer to the per-mode model, then run alternative
//! codecs on it for comparison. Answers the architectural question
//! "would BWT on the LZ-literal residue beat the current Order-3
//! letter / Order-2 non-letter encoding?" without committing to a
//! full `xml-lz-bwt` codec implementation.
//!
//! The extraction uses lazy-parse LZ (xml-lz-word/ord3 behavior, not
//! the cost-aware Phase-7 refinement). That's a close enough proxy
//! for "the literal stream xml-lz-cp produces" since cost-aware only
//! reallocates a small fraction of marginal-match bytes.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result};

use crate::bwt_codec::BwtCodec;
use crate::classifier::{Classifier, Mode};
use crate::codec::Codec;
use crate::lz::{MIN_MATCH, Matcher};

/// Walk `measure` through the same classifier + lazy-parse LZ logic
/// the production codec uses, and return the Content-mode literal
/// bytes (i.e., the bytes that the LZ matcher did *not* absorb into
/// a match record). The `warm` prefix primes the classifier and the
/// matcher's hash chain.
pub(crate) fn extract_content_literals(warm: &[u8], measure: &[u8]) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::with_capacity(warm.len() + measure.len());
    buf.extend_from_slice(warm);
    let warm_end = buf.len();
    buf.extend_from_slice(measure);

    let mut classifier = Classifier::new();
    for &b in warm {
        classifier.advance(b);
    }
    let mut matcher = Matcher::new();
    let mut next_insert: usize = 0;
    while next_insert + 3 <= warm_end {
        matcher.insert(&buf, next_insert);
        next_insert += 1;
    }

    let mut literals: Vec<u8> = Vec::new();
    let mut i = warm_end;
    while i < buf.len() {
        while next_insert + 3 <= i {
            matcher.insert(&buf, next_insert);
            next_insert += 1;
        }
        let mode = classifier.current_mode();
        if mode == Mode::Content {
            let m_here = matcher.find_match(&buf, i);
            let take_match = match m_here {
                Some((_, len_here)) if i + (len_here as usize) <= buf.len() => {
                    let m_next = if i + 1 + MIN_MATCH <= buf.len() {
                        matcher.find_match(&buf, i + 1)
                    } else {
                        None
                    };
                    !matches!(m_next, Some((_, len_next)) if len_next > len_here)
                }
                _ => false,
            };
            if take_match {
                let (_, length) = m_here.unwrap();
                let len_us = length as usize;
                for &b in &buf[i..i + len_us] {
                    classifier.advance(b);
                }
                i += len_us;
            } else {
                let byte = buf[i];
                literals.push(byte);
                classifier.advance(byte);
                i += 1;
            }
        } else {
            classifier.advance(buf[i]);
            i += 1;
        }
    }
    literals
}

/// CLI entry point for `lzr analyze-literals`. Reads one panel
/// window (warm + measure) and reports:
///   - The literal-stream size in bytes
///   - Raw cost (8 bpb floor)
///   - BWT codec cost on the literal stream
///   - bpb when amortized over the full measure window
///
/// `cast_precision_loss` is suppressed at the function level: all
/// values cast to f64 here are bounded by a few MiB worth of bytes
/// in a single panel window, well within f64's 53-bit mantissa.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn run_analyze_literals(
    corpus: &Path,
    offset: u64,
    measure_bytes: usize,
    warm_bytes: usize,
) -> Result<()> {
    let mut f =
        File::open(corpus).with_context(|| format!("opening corpus {}", corpus.display()))?;

    let warm_start = offset.saturating_sub(warm_bytes as u64);
    let warm_len = usize::try_from(offset - warm_start).expect("warm bytes fits usize");
    let warm = if warm_len > 0 {
        f.seek(SeekFrom::Start(warm_start))?;
        let mut b = vec![0u8; warm_len];
        f.read_exact(&mut b)?;
        b
    } else {
        Vec::new()
    };

    f.seek(SeekFrom::Start(offset))?;
    let mut measure = vec![0u8; measure_bytes];
    f.read_exact(&mut measure)?;

    println!("Corpus:  {}", corpus.display());
    println!(
        "Window:  warm = {} bytes, measure = {} bytes",
        warm.len(),
        measure.len(),
    );
    println!();

    let literals = extract_content_literals(&warm, &measure);
    let lit_n = literals.len();
    println!(
        "Literal Content bytes extracted: {lit_n} ({:.2}% of measure)",
        100.0 * lit_n as f64 / measure.len() as f64,
    );
    println!();

    let raw_bits = (lit_n as u64) * 8;
    let raw_bpb_global = (raw_bits as f64) / (measure.len() as f64 * 8.0) * 8.0;
    println!(
        "Encoding the literal stream raw (8 bpb floor): {raw_bits} bits = {:.3} bpb on the literal stream, {:.3} bpb on the full measure",
        8.0, raw_bpb_global,
    );
    println!();

    // BWT codec on the literal stream as a standalone block.
    if literals.is_empty() {
        println!("No literal bytes — nothing to BWT.");
        return Ok(());
    }
    let codec = BwtCodec;
    let (archive, _decomp) = codec.encode_window(b"", &literals)?;
    // Roundtrip sanity check.
    let decoded = codec.decode_window(b"", &archive)?;
    if decoded != literals {
        anyhow::bail!("BWT roundtrip failed on literal stream");
    }
    let bwt_bits = (archive.len() as u64) * 8;
    let bwt_bpb_lit = (bwt_bits as f64) / (lit_n as f64);
    let bwt_bpb_global = (bwt_bits as f64) / (measure.len() as f64);
    println!(
        "Encoding the literal stream via BWT (Order-1 AC on MTF): {bwt_bits} bits = {bwt_bpb_lit:.3} bpb on the literal stream, {bwt_bpb_global:.3} bpb on the full measure",
    );
    println!(
        "  Compression ratio on literals: {:.2}x  (raw 8.0 -> {bwt_bpb_lit:.3} bpb)",
        8.0 / bwt_bpb_lit,
    );
    Ok(())
}
