//! Mode-routed compression CLI — Hutter Prize attempt, v2.
//!
//! The v2 design starts from a deterministic mode classifier and
//! routes each segment to a dedicated codec. The goal is to find an
//! architecturally diverse predictor mix before reaching for neural
//! arms — the v1 4M-neural plateau (`JOURNAL.md` 2026-05-10) showed
//! that mixer parameterization saturates when predictors are
//! correlated.
//!
//! Phase 0 (current): eval infrastructure only — multi-offset panel,
//! per-component bit decomposition, null codec for sanity.

#![deny(unsafe_code)]
// Pedantic gives noisy hits on patterns we use everywhere.
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::redundant_pub_crate)]
#![allow(clippy::module_name_repetitions)]

mod ac;
mod analyze;
mod analyze_wiki;
mod bit_pred;
mod bits;
mod bpe;
mod bwt;
mod bwt_codec;
mod classifier;
mod classifier_stats;
mod codec;
mod eval;
mod lz;
mod match_model;
mod models;
mod mtf;
mod null;
mod paq_codec;
mod ppm;
mod recon;
mod survey;
mod tok_lz;
mod tokenizer;
mod wiki_classifier;
mod xml_codec;
mod xml_lz_cp;
mod xml_lz_ord3;
mod xml_lz_ppm;
mod xml_lz_ppmc;
mod xml_lz_word;
mod xml_ppm;
mod xml_tok;
mod xml_wiki_lz_cp;

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "lzr", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Extract Content-literal bytes from one panel window (using
    /// the production classifier + lazy-parse LZ matcher) and report
    /// how various codecs compress that stream. Diagnostic only — no
    /// archive is produced. Used to decide whether a deferred-BWT
    /// integration would beat the current per-mode literal codec.
    /// Scan a prefix of the corpus and report wiki sub-mode coverage
    /// within Content (links, templates, headings) plus top distinct
    /// link targets and template names. Recon for Phase 12.
    AnalyzeWiki {
        /// Corpus file path.
        #[arg(long, default_value = "assets/enwik8")]
        corpus: PathBuf,
        /// How many bytes from the start to scan.
        #[arg(long, default_value_t = 16 * 1024 * 1024)]
        bytes: usize,
    },

    /// v3 capability survey: sweep tokenization schemes (bytes,
    /// word tokens at various dict sizes, BPE at various vocab
    /// sizes) and report Order-0 / Order-1 entropy on the corpus
    /// prefix. Goal: pick the right architecture from measurement,
    /// not vibes.
    Survey {
        /// Corpus file path.
        #[arg(long, default_value = "assets/enwik8")]
        corpus: PathBuf,
        /// How many bytes from the start of the corpus to use for
        /// both training and entropy measurement.
        #[arg(long, default_value_t = 16 * 1024 * 1024)]
        bytes: usize,
    },

    /// Panel survey: measure Order-N word entropy under the same
    /// 20×(4 MiB warm + 256 KiB measure) structure the bench panel
    /// uses. Calibration-corrected version of the prefix survey —
    /// the codec's prewarm absorbs first-occurrence tokens, so the
    /// "static-dict OOV" the prefix survey reported is much higher
    /// than the panel actually sees.
    PanelSurvey {
        /// Corpus file path.
        #[arg(long, default_value = "assets/enwik8")]
        corpus: PathBuf,
    },

    /// Phase-0 recon for v3 next-phase decisions. Streams the full
    /// corpus through XML + fine-grained wiki sub-classifier and a
    /// page-id counter, building a global word-token table indexed by
    /// `(token -> total, distinct_pages, max_in_page)`. Reports byte
    /// share per sub-mode and the in-page repetition factor for the
    /// top-N tokens — the two measurements that decide between
    /// Proposal B (wiki-sub-mode word models) and Proposal C
    /// (page-local cache).
    Recon {
        /// Corpus file path.
        #[arg(long, default_value = "assets/enwik9")]
        corpus: PathBuf,
        /// Number of top tokens to list in the per-token table.
        #[arg(long, default_value_t = 100)]
        top: usize,
    },

    /// Compress a whole corpus through a codec end-to-end (no panel
    /// structure: one call to `encode_window(b"", &corpus_bytes)`).
    /// By default also runs `decode_window` and verifies the
    /// roundtrip — disable with `--skip-verify` if you want to
    /// time only the encode side.
    Compress {
        /// Corpus file path to compress.
        #[arg(long, default_value = "assets/enwik9")]
        corpus: PathBuf,
        /// Codec name (see `bench` for the list).
        #[arg(long, default_value = "xml-tok")]
        codec: String,
        /// Output archive path.
        #[arg(long, default_value = "/tmp/lzr.archive")]
        out: PathBuf,
        /// Skip the decode-and-verify round-trip step (just encode
        /// + write archive). Useful for timing-only runs.
        #[arg(long, default_value_t = false)]
        skip_verify: bool,
    },

    AnalyzeLiterals {
        /// Corpus file path.
        #[arg(long, default_value = "assets/enwik8")]
        corpus: PathBuf,
        /// Start offset of the measure window.
        #[arg(long, default_value_t = 4_194_304)]
        offset: u64,
        /// Size of the measure window in bytes.
        #[arg(long, default_value_t = 256 * 1024)]
        measure_bytes: usize,
        /// Size of the warm prefix before the measure window.
        #[arg(long, default_value_t = 4 * 1024 * 1024)]
        warm_bytes: usize,
    },

    /// Run the multi-offset codec eval panel against a chosen codec.
    /// Reports per-window bpb and the mean; optionally dumps a
    /// per-component bit decomposition to CSV.
    Bench {
        /// Corpus file path. Defaults to enwik8 for fast development;
        /// use enwik9 for the canonical Hutter measurement.
        #[arg(long, default_value = "assets/enwik8")]
        corpus: PathBuf,
        /// Codec to use. Phase 0 has only `null` (sanity baseline);
        /// more land in subsequent phases.
        #[arg(long, default_value = "null")]
        codec: String,
        /// Reduce panel to 5 windows × 64 KiB for tight iteration.
        /// Default is 20 windows × 256 KiB.
        #[arg(long, default_value_t = false)]
        quick: bool,
        /// Dump per-component bit decomposition to CSV at this path.
        #[arg(long)]
        decompose: Option<PathBuf>,
    },
}

#[allow(clippy::cast_precision_loss)]
fn run_compress(
    corpus: &PathBuf,
    codec_name: &str,
    out: &PathBuf,
    skip_verify: bool,
) -> Result<()> {
    let codec = eval::make_codec_public(codec_name)?;

    eprintln!("Reading {} ...", corpus.display());
    let bytes = fs::read(corpus).with_context(|| format!("reading {}", corpus.display()))?;
    let n = bytes.len();
    eprintln!(
        "Input:  {n} bytes ({:.2} MiB)",
        n as f64 / (1024.0 * 1024.0),
    );

    let codec_ref: &dyn codec::Codec = codec.as_ref();
    eprintln!("Encoding through codec `{}` ...", codec_ref.name());
    let start = Instant::now();
    let (archive, _decomp) = codec_ref
        .encode_window(b"", &bytes)
        .context("encode_window failed")?;
    let encode_elapsed = start.elapsed();

    eprintln!("Writing archive to {} ...", out.display());
    fs::write(out, &archive).with_context(|| format!("writing archive to {}", out.display()))?;

    let archive_bytes = archive.len();
    let bpb = 8.0 * archive_bytes as f64 / n as f64;

    println!();
    println!("Codec:           {}", codec_ref.name());
    println!("Corpus:          {}", corpus.display());
    println!("Input bytes:     {n}");
    println!(
        "Archive bytes:   {archive_bytes}  ({:.2} MiB)",
        archive_bytes as f64 / (1024.0 * 1024.0),
    );
    println!("Compressed bpb:  {bpb:.4}");
    println!("Encode time:     {encode_elapsed:?}");
    println!(
        "Encode rate:     {:.2} MiB/s",
        n as f64 / (1024.0 * 1024.0) / encode_elapsed.as_secs_f64(),
    );

    if skip_verify {
        println!("Verify:          skipped");
        return Ok(());
    }

    eprintln!("Decoding for roundtrip verification ...");
    let decode_start = Instant::now();
    let decoded = codec_ref
        .decode_window(b"", &archive)
        .context("decode_window failed")?;
    let decode_elapsed = decode_start.elapsed();

    if decoded.len() != n {
        bail!("decode produced {} bytes; expected {n}", decoded.len());
    }
    if decoded != bytes {
        // Find first mismatch position for diagnostics.
        let mismatch = bytes
            .iter()
            .zip(decoded.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        bail!(
            "decode byte mismatch at offset {mismatch}: orig {:#04x} vs decoded {:#04x}",
            bytes[mismatch],
            decoded[mismatch]
        );
    }

    println!("Decode time:     {decode_elapsed:?}");
    println!(
        "Decode rate:     {:.2} MiB/s",
        n as f64 / (1024.0 * 1024.0) / decode_elapsed.as_secs_f64(),
    );
    println!("Roundtrip:       OK");
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Bench {
            corpus,
            codec,
            quick,
            decompose,
        } => eval::run_bench(&corpus, &codec, quick, decompose.as_deref()),
        Command::AnalyzeLiterals {
            corpus,
            offset,
            measure_bytes,
            warm_bytes,
        } => analyze::run_analyze_literals(&corpus, offset, measure_bytes, warm_bytes),
        Command::AnalyzeWiki { corpus, bytes } => analyze_wiki::run_analyze_wiki(&corpus, bytes),
        Command::Survey { corpus, bytes } => survey::run_survey(&corpus, bytes),
        Command::PanelSurvey { corpus } => survey::run_panel_survey(&corpus),
        Command::Compress {
            corpus,
            codec,
            out,
            skip_verify,
        } => run_compress(&corpus, &codec, &out, skip_verify),
        Command::Recon { corpus, top } => recon::run_recon(&corpus, top),
    }
}
