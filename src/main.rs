//! LZR compression CLI.

#![allow(unused_features, reason = "Required for coverage_attribute feature.")]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use std::path::PathBuf;

use clap::Parser;

/// LZR: a general-purpose compressor with random access and tailing support.
///
/// Compression is the default, except when `input` has the `.lzr` extension, in which case
/// decompression is the default. Pass `-z`/`--compress` or `-d`/`--decompress` to force a
/// direction regardless of extension.
#[derive(Debug, Parser)]
#[command(name = "lzr", version, about)]
struct Cli {
    /// Force compression, even for a `.lzr` input.
    #[arg(short = 'z', long, conflicts_with = "decompress")]
    compress: bool,

    /// Force decompression, even for a non-`.lzr` input.
    #[arg(short, long)]
    decompress: bool,

    /// Input file to read.
    input: PathBuf,

    /// Output file to write.
    output: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let decompress = cli.decompress || (!cli.compress && cli.input.extension().is_some_and(|e| e == "lzr"));
    let direction = if decompress {
        "decompress"
    } else {
        "compress"
    };
    anyhow::bail!("{direction} {} -> {}: the codec is not implemented yet", cli.input.display(), cli.output.display())
}
