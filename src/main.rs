//! LZR compression CLI.

#![allow(unused_features, reason = "Required for coverage_attribute feature.")]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Context as _;
use clap::Parser;
use lzr::Profile;

/// LZR — a context-mixing compressor.
///
/// Compresses `input` into `output` by default; pass `-d`/`--decompress` to
/// reverse the direction. Pipeline features (see `--enable`/`--disable`) default
/// to all enabled and apply to compression only — decompression reads the
/// pipeline the stream was written with.
#[derive(Debug, Parser)]
#[command(name = "lzr", version, about)]
struct Cli {
    /// Decompress the input instead of compressing it.
    #[arg(short, long)]
    decompress: bool,

    /// Enable a pipeline feature (repeatable). Compression only.
    #[arg(long, value_name = "FEATURE")]
    enable: Vec<String>,

    /// Disable a pipeline feature (repeatable). Compression only.
    #[arg(long, value_name = "FEATURE")]
    disable: Vec<String>,

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
    let output = if cli.decompress {
        lzr::decompress(&input)?
    } else {
        lzr::compress_with(&input, profile(cli)?)
    };
    std::fs::write(&cli.output, &output).with_context(|| format!("writing {}", cli.output.display()))?;

    // For statistics, "original" is the uncompressed side and "compressed" the `.lzr` side,
    // regardless of direction.
    let (original, compressed) = if cli.decompress {
        (output.len(), input.len())
    } else {
        (input.len(), output.len())
    };
    report_stats(original, compressed);
    Ok(())
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
