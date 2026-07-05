//! LZR compression CLI.

#![allow(unused_features, reason = "Required for coverage_attribute feature.")]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Context as _;
use clap::Parser;
use lzr::{EncodeOptions, Profile};

/// LZR — a context-mixing compressor.
///
/// Compression is the default, except when `input` has the compressor's `.lzr`
/// extension, in which case decompression is the default. Pass `-z`/`--compress`
/// or `-d`/`--decompress` to force a direction regardless of extension. Pipeline
/// features (see `--enable`/`--disable`) default to all enabled and apply to
/// compression only — decompression reads the pipeline the stream was written
/// with.
#[derive(Debug, Parser)]
#[command(name = "lzr", version, about)]
struct Cli {
    /// Force compression, even for a `.lzr` input.
    #[arg(short = 'z', long, conflicts_with = "decompress")]
    compress: bool,

    /// Force decompression, even for a non-`.lzr` input.
    #[arg(short, long)]
    decompress: bool,

    /// Enable a pipeline feature (repeatable). Compression only.
    #[arg(long, value_name = "FEATURE")]
    enable: Vec<String>,

    /// Disable a pipeline feature (repeatable). Compression only.
    #[arg(long, value_name = "FEATURE")]
    disable: Vec<String>,

    /// Re-Pair vocabulary cap (256..=4194302). Compression only; defaults to the maximum.
    #[arg(long, value_name = "N")]
    num_tokens: Option<u32>,

    /// LZ77 minimum match length (2..=4194303); may expand the stream. Compression only; defaults
    /// to a dynamic per-match rule that only emits matches that shrink the stream.
    #[arg(long, value_name = "N")]
    min_match: Option<u32>,

    /// Input file to read.
    input: PathBuf,

    /// Output file to write.
    output: PathBuf,
}

fn main() -> ExitCode {
    if let Err(err) = run(&Cli::parse()) {
        report(&err);
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

/// Read the input, transform it in the requested direction, write the output, and
/// print run statistics.
fn run(cli: &Cli) -> anyhow::Result<()> {
    let input = std::fs::read(&cli.input).with_context(|| format!("reading {}", cli.input.display()))?;
    let input_len = input.len();
    // Compression is the default; a `.lzr` input flips the default to decompression. Either
    // explicit flag overrides the extension (and the two flags conflict, so at most one is set).
    let decompress = !cli.compress && (cli.decompress || has_lzr_extension(&cli.input));
    let (output, stages) = if decompress {
        (lzr::decompress(&input)?, Vec::new())
    } else {
        let options = encode_options(cli)?;
        // Take ownership so the pipeline can free the input buffer before the tokenizer build.
        lzr::compress_owned_with_traced(input, profile(cli)?, options)
    };
    std::fs::write(&cli.output, &output).with_context(|| format!("writing {}", cli.output.display()))?;

    // For statistics, "original" is the uncompressed side and "compressed" the `.lzr` side,
    // regardless of direction.
    let (original, compressed) = if decompress {
        (output.len(), input_len)
    } else {
        (input_len, output.len())
    };
    report_stages(&stages);
    report_stats(original, compressed);
    Ok(())
}

/// Whether `path` has an `.lzr` extension (case-insensitive), signalling that the input is a
/// compressed stream and the direction should default to decompression.
fn has_lzr_extension(path: &std::path::Path) -> bool {
    path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("lzr"))
}

/// Build the encode-only [`EncodeOptions`] from the `--num-tokens` and `--min-match` flags, each
/// defaulting when absent. Both are validated by their respective setters.
fn encode_options(cli: &Cli) -> anyhow::Result<EncodeOptions> {
    let mut options = match cli.num_tokens {
        Some(num_tokens) => EncodeOptions::new(num_tokens)?,
        None => EncodeOptions::default(),
    };
    if let Some(min_match) = cli.min_match {
        options = options.with_min_match(min_match)?;
    }
    Ok(options)
}

/// Build the compression [`Profile`] from the `--disable`/`--enable` flags, starting from the
/// default (every feature enabled). Disables are applied first, so `--enable` wins any conflict.
fn profile(cli: &Cli) -> anyhow::Result<Profile> {
    let mut profile = Profile::default();
    for feature in &cli.disable {
        profile.disable(feature)?;
    }
    for feature in &cli.enable {
        profile.enable(feature)?;
    }
    // Reject a pipeline that would run entropy coding with no model to predict with.
    profile.validate()?;
    Ok(profile)
}

/// The size of the running executable in bytes, or `None` if it cannot be determined.
fn executable_size() -> Option<u64> {
    std::env::current_exe().ok().and_then(|path| std::fs::metadata(path).ok()).map(|meta| meta.len())
}

/// Formats an integer with US-style thousands separators (e.g. `131072` → `131,072`).
fn with_commas(n: u64) -> String {
    let digits = n.to_string();
    let len = digits.len();
    let mut out = String::with_capacity(len + len / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i != 0 && (len - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// Print each pipeline stage's own bits-per-byte to stderr (compression only): `8 · output / input`,
/// i.e. what the transform did to *its* input, not a running figure against the original. A stage
/// above 8 bpb expanded its input; below 8 it shrank it. No-op when there are no stages
/// (decompression, or an empty profile).
#[expect(clippy::print_stderr, reason = "A CLI reports run statistics to the user on stderr.")]
#[expect(clippy::cast_precision_loss, reason = "Display statistics; byte counts are well under 2^53.")]
fn report_stages(stages: &[lzr::StageSize]) {
    if stages.is_empty() {
        return;
    }
    eprintln!("stages (bpb):");
    for stage in stages {
        let bpb = if stage.input_bytes == 0 {
            0.0
        } else {
            8.0 * stage.output_bytes as f64 / stage.input_bytes as f64
        };
        eprintln!("  {:<9} {bpb:.4} bpb", stage.name);
    }
}

/// Print compression statistics to stderr: size, ratio, bits-per-byte, and a bits-per-byte figure
/// under Hutter-Prize accounting (the decompressor executable plus the `.lzr` payload).
#[expect(clippy::print_stderr, reason = "A CLI reports run statistics to the user on stderr.")]
#[expect(clippy::cast_precision_loss, reason = "Display statistics; byte counts are well under 2^53.")]
fn report_stats(original: usize, compressed: usize) {
    let (ratio, bpb) = if original == 0 {
        (0.0, 0.0)
    } else {
        let (compressed_f, original_f) = (compressed as f64, original as f64);
        ((1.0 - compressed_f / original_f) * 100.0, 8.0 * compressed_f / original_f)
    };
    eprintln!("original:   {} bytes", with_commas(original as u64));
    eprintln!("compressed: {} bytes", with_commas(compressed as u64));
    eprintln!("ratio:      {ratio:.2}%  ({bpb:.4} bpb)");

    if let Some(exe) = executable_size() {
        // Hutter rules: L(C) + L(C) (same executable) + L(.lzr).
        let hutter = exe.saturating_mul(2).saturating_add(compressed as u64);
        let hutter_bpb = if original == 0 {
            0.0
        } else {
            8.0 * hutter as f64 / original as f64
        };
        eprintln!(
            "hutter:     {hutter_bpb:.4} bpb  (2 x {} exe + {} lzr)",
            with_commas(exe),
            with_commas(compressed as u64)
        );
    }
}

/// Print an error (with its full anyhow context chain) to stderr.
#[expect(clippy::print_stderr, reason = "A CLI reports errors to the user on stderr.")]
fn report(err: &anyhow::Error) {
    eprintln!("lzr: {err:#}");
}
