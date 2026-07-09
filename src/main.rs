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
#[expect(clippy::struct_excessive_bools, reason = "CLI flags are independent on/off switches, not a state machine")]
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

    /// LZ77 minimum match length in tokens (2..=4194303); may expand the stream. Compression only;
    /// defaults to a dynamic per-match rule that only emits matches that shrink the stream.
    #[arg(long, value_name = "N")]
    min_match: Option<u32>,

    /// Re-parse the Re-Pair top-level sequence into a minimum-cost cover (MGP). Compression only;
    /// encode-side only, so the stream still decodes without any flag.
    #[arg(long)]
    mgp: bool,

    /// Auto-select the best operating point (Re-Pair vocabulary size × LZ77 on/off, with MGP on) by
    /// trial-compressing a prefix sample, then compress the whole input at the winner. This is the
    /// **default** for a bare compression; passing any pipeline flag (`--num-tokens`, `--min-match`,
    /// `--enable`, `--disable`) selects the manual path instead, and `--auto` forces the search even
    /// alongside those flags (overriding `--num-tokens` and the LZ77 toggle). Compression only.
    #[arg(long, conflicts_with = "no_auto")]
    auto: bool,

    /// Opt out of the default operating-point search and compress with the plain pipeline (honoring
    /// any `--num-tokens` / `--min-match` / `--enable` / `--disable`). Compression only.
    #[arg(long)]
    no_auto: bool,

    /// Prefix bytes `--auto` searches on (default 2,000,000). Larger samples track the full-input
    /// optimum better (the best vocabulary grows with input size) at proportionally more search cost.
    #[arg(long, value_name = "BYTES")]
    auto_sample: Option<usize>,

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

    let input = std::fs::read(in_path).with_context(|| format!("reading {}", in_path.display()))?;
    let input_len = input.len();
    // Compression is the default; a `.lzr` input flips the default to decompression. Either
    // explicit flag overrides the extension (and the two flags conflict, so at most one is set).
    let decompress = !cli.compress && (cli.decompress || has_lzr_extension(in_path));
    let (output, stages, models, flags) = if decompress {
        (lzr::decompress(&input)?, Vec::new(), Vec::new(), Vec::new())
    } else {
        let profile = profile(cli)?;
        // Capture the enabled config flags before the profile is consumed by the compressor.
        let flags = profile.active_flags();
        // Auto operating-point search is the default for a bare compression; any explicit pipeline flag
        // selects the manual path, `--no-auto` forces it, and `--auto` forces the search regardless.
        let manual_flags =
            cli.num_tokens.is_some() || cli.min_match.is_some() || !cli.enable.is_empty() || !cli.disable.is_empty();
        let use_auto = cli.auto || (!cli.no_auto && !manual_flags);
        // Take ownership so the pipeline can free the input buffer before the tokenizer build.
        let (output, stages, models) = if use_auto {
            let sample = cli.auto_sample.unwrap_or(lzr::DEFAULT_SAMPLE_BYTES);
            let (output, stages, models, report) = lzr::compress_auto(input, profile, sample);
            report_auto(&report);
            (output, stages, models)
        } else {
            let options = encode_options(cli)?;
            lzr::compress_owned_with_traced(input, profile, options)
        };
        (output, stages, models, flags)
    };
    std::fs::write(out_path, &output).with_context(|| format!("writing {}", out_path.display()))?;

    // For statistics, "original" is the uncompressed side and "compressed" the `.lzr` side,
    // regardless of direction.
    let (original, compressed) = if decompress {
        (output.len(), input_len)
    } else {
        (input_len, output.len())
    };
    report_stages(&stages);
    report_models(&models);
    report_flags(&flags);
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
    if cli.mgp {
        options = options.with_mgp();
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
        // A stage may report an extra statistic with its own unit (Re-Pair vocab in "tokens", LZ77
        // back-references in "matches").
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

/// Print the auto-selected operating point to stderr (compression only, `--auto`): the chosen Re-Pair
/// vocabulary and LZ77 state, and how much search it took.
#[expect(clippy::print_stderr, reason = "A CLI reports run statistics to the user on stderr.")]
fn report_auto(report: &lzr::AutoReport) {
    let vocab = report
        .point
        .num_tokens
        .map_or_else(|| "natural (cost-stop)".to_owned(), |n| format!("{}-token cap", with_commas(u64::from(n))));
    let lz77 = if report.point.lz77 {
        "on"
    } else {
        "off"
    };
    eprintln!(
        "auto:       {vocab}, lz77 {lz77}  ({} sample trials on {} bytes + {} full-input refine)",
        report.sample_evaluations,
        with_commas(report.sampled_bytes as u64),
        report.refine_evaluations
    );
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
