//! LZR compression CLI.

#![allow(unused_features, reason = "Required for coverage_attribute feature.")]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use std::fs::File;
use std::io::{BufReader, BufWriter, Write as _};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::Context as _;
use clap::Parser;
use lzr::{DEFAULT_BLOCK_SIZE, Profile};

/// LZR — a context-mixing compressor.
///
/// Compression is the default, except when `input` has the compressor's `.lzr`
/// extension, in which case decompression is the default. Pass `-z`/`--compress`
/// or `-d`/`--decompress` to force a direction regardless of extension. Pipeline
/// features (see `--enable`/`--disable`) apply to compression only — decompression
/// reads the pipeline each block was written with.
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

    /// Block size in bytes; the input is split into independently-compressed blocks of this size,
    /// processed in parallel. Accepts a plain byte count or a `K`/`M`/`G` suffix (e.g. `64M`).
    /// Compression only; defaults to 100 MiB.
    #[arg(long, value_name = "SIZE", value_parser = parse_block_size)]
    block_size: Option<usize>,

    /// Number of blocks to process in parallel: `0` uses all CPU cores, `1` is single-threaded, `N`
    /// uses N workers. Peak memory scales with this (each in-flight block needs its own scratch).
    #[arg(long, value_name = "N", default_value_t = 0)]
    threads: usize,

    /// List all pipeline features (for `--enable`/`--disable`) with their defaults, then exit.
    #[arg(long)]
    list_features: bool,

    /// Input file to read.
    #[arg(required_unless_present = "list_features")]
    input: Option<PathBuf>,

    /// Output file to write.
    #[arg(required_unless_present = "list_features")]
    output: Option<PathBuf>,
}

/// Parse a `--block-size` value: a plain byte count, or a number with a `K`/`M`/`G` (1024-based)
/// suffix. Rejects zero and anything that overflows `usize`.
fn parse_block_size(raw: &str) -> Result<usize, String> {
    let raw = raw.trim();
    let (digits, mult) = match raw.chars().last() {
        Some('K' | 'k') => (&raw[..raw.len() - 1], 1usize << 10),
        Some('M' | 'm') => (&raw[..raw.len() - 1], 1usize << 20),
        Some('G' | 'g') => (&raw[..raw.len() - 1], 1usize << 30),
        _ => (raw, 1usize),
    };
    let n: usize = digits.trim().parse().map_err(|_| format!("invalid block size {raw:?}"))?;
    let bytes = n.checked_mul(mult).ok_or_else(|| format!("block size {raw:?} overflows"))?;
    if bytes == 0 {
        return Err("block size must be greater than zero".to_owned());
    }
    Ok(bytes)
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
    if cli.list_features {
        print_features();
        return Ok(());
    }
    // clap's `required_unless_present` guarantees both are set here (the `--list-features` path
    // returned above); the `else` is defensive.
    let (Some(in_path), Some(out_path)) = (cli.input.as_deref(), cli.output.as_deref()) else {
        anyhow::bail!("INPUT and OUTPUT are required (or pass --list-features)");
    };

    // Compression is the default; a `.lzr` input flips the default to decompression. Either
    // explicit flag overrides the extension (and the two flags conflict, so at most one is set).
    let decompress = !cli.compress && (cli.decompress || has_lzr_extension(in_path));

    // Stream the file through the container so peak memory scales with the block window, not the file.
    if decompress {
        let mut reader = open_reader(in_path)?;
        let mut writer = open_writer(out_path)?;
        lzr::decompress_stream(&mut reader, &mut writer, cli.threads)?;
        writer.flush().with_context(|| format!("writing {}", out_path.display()))?;
        drop(writer);
        // "original" is the uncompressed side (the output) and "compressed" the `.lzr` input.
        report_stats(file_len(out_path), file_len(in_path));
    } else {
        let profile = profile(cli)?;
        let flags = profile.active_flags();
        let block_size = cli.block_size.unwrap_or(DEFAULT_BLOCK_SIZE);
        let input_len = file_len(in_path);
        let mut reader = open_reader(in_path)?;
        let mut writer = open_writer(out_path)?;
        let (stages, models) =
            lzr::compress_stream(&mut reader, &mut writer, input_len, profile, block_size, cli.threads)?;
        writer.flush().with_context(|| format!("writing {}", out_path.display()))?;
        drop(writer);
        report_stages(&stages);
        report_models(&models);
        report_flags(&flags);
        report_stats(input_len, file_len(out_path));
    }
    Ok(())
}

/// Open a buffered reader over `path`.
fn open_reader(path: &Path) -> anyhow::Result<BufReader<File>> {
    let file = File::open(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(BufReader::new(file))
}

/// Open a buffered writer over `path`, truncating any existing file.
fn open_writer(path: &Path) -> anyhow::Result<BufWriter<File>> {
    let file = File::create(path).with_context(|| format!("writing {}", path.display()))?;
    Ok(BufWriter::new(file))
}

/// The length of `path` in bytes, or `0` if it can't be stat'd (only used for display statistics).
fn file_len(path: &Path) -> u64 {
    std::fs::metadata(path).map_or(0, |m| m.len())
}

/// Whether `path` has an `.lzr` extension (case-insensitive), signalling that the input is a
/// compressed stream and the direction should default to decompression.
fn has_lzr_extension(path: &Path) -> bool {
    path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("lzr"))
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
        // A stage may report an extra statistic with its own unit (e.g. the Re-Pair rule count in "rules").
        match stage.detail {
            Some((count, unit)) => {
                eprintln!("  {:<9} {bpb:.4} bpb  ({} {unit})", stage.name, with_commas(count));
            }
            None => eprintln!("  {:<9} {bpb:.4} bpb", stage.name),
        }
    }
}

/// Print the per-model scorecard to stderr (compression only): each active model's standalone
/// bits-per-byte (its raw predictive power if it coded alone) and the mixer's average weight on it
/// (its marginal value — near zero means another model already says the same thing). No-op when there
/// are no models to report (decompression, or the entropy stage disabled).
#[expect(clippy::print_stderr, reason = "A CLI reports run statistics to the user on stderr.")]
fn report_models(models: &[lzr::ModelScore]) {
    if models.is_empty() {
        return;
    }
    eprintln!("models (bpb-alone / mixer weight):");
    for model in models {
        eprintln!("  {:<9} {:.4} bpb   w={:+.3}", model.name, model.bpb_alone, model.avg_weight);
    }
}

/// Print the enabled config flags to stderr (compression only), e.g. `sse`. Flags are pipeline
/// configuration rather than predictors, so they list plainly. No-op when none are enabled (or on
/// decompression).
#[expect(clippy::print_stderr, reason = "A CLI reports run statistics to the user on stderr.")]
fn report_flags(flags: &[&str]) {
    if flags.is_empty() {
        return;
    }
    eprintln!("flags:      {}", flags.join(", "));
}

/// Print compression statistics to stderr: size, ratio, bits-per-byte, and a bits-per-byte figure
/// under Hutter-Prize accounting (the decompressor executable plus the `.lzr` payload).
#[expect(clippy::print_stderr, reason = "A CLI reports run statistics to the user on stderr.")]
#[expect(clippy::cast_precision_loss, reason = "Display statistics; byte counts are well under 2^53.")]
fn report_stats(original: u64, compressed: u64) {
    let (ratio, bpb) = if original == 0 {
        (0.0, 0.0)
    } else {
        let (compressed_f, original_f) = (compressed as f64, original as f64);
        ((1.0 - compressed_f / original_f) * 100.0, 8.0 * compressed_f / original_f)
    };
    eprintln!("original:   {} bytes", with_commas(original));
    eprintln!("compressed: {} bytes", with_commas(compressed));
    eprintln!("ratio:      {ratio:.2}%  ({bpb:.4} bpb)");

    if let Some(exe) = executable_size() {
        // Hutter rules: L(C) + L(C) (same executable) + L(.lzr).
        let hutter = exe.saturating_mul(2).saturating_add(compressed);
        let hutter_bpb = if original == 0 {
            0.0
        } else {
            8.0 * hutter as f64 / original as f64
        };
        eprintln!("hutter:     {hutter_bpb:.4} bpb  (2 x {} exe + {} lzr)", with_commas(exe), with_commas(compressed));
    }
}

/// Print the available pipeline features (grouped by kind, with default/mandatory tags) to stdout —
/// the `--list-features` output. The names are exactly what `--enable` / `--disable` accept.
#[expect(clippy::print_stdout, reason = "The feature listing is this invocation's primary output.")]
fn print_features() {
    println!("Available features — toggle with --enable <name> / --disable <name>:\n");
    let all = lzr::features();
    for (label, kind) in [
        ("Stages (pipeline order)", lzr::FeatureKind::Stage),
        ("Entropy models", lzr::FeatureKind::Model),
        ("Flags", lzr::FeatureKind::Flag),
    ] {
        println!("{label}:");
        for feature in all.iter().filter(|feature| feature.kind == kind) {
            let mut tags = Vec::new();
            if feature.default_on {
                tags.push("default");
            }
            if feature.mandatory {
                tags.push("mandatory");
            }
            let tags = if tags.is_empty() {
                String::new()
            } else {
                format!("  ({})", tags.join(", "))
            };
            println!("  {:<10}{tags}", feature.name);
        }
        println!();
    }
}

/// Print an error (with its full anyhow context chain) to stderr.
#[expect(clippy::print_stderr, reason = "A CLI reports errors to the user on stderr.")]
fn report(err: &anyhow::Error) {
    eprintln!("lzr: {err:#}");
}
