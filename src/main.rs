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

use std::path::PathBuf;

use anyhow::Result;
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
    }
}
