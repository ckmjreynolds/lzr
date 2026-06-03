//! LZR v7 — clean-slate context-mixing compressor (Hutter Prize attempt).
//!
//! The codec is a single online multi-order context-mixing model over the
//! shared arithmetic coder (`cmix`), plus the `null` pass-through floor used
//! to sanity-check the eval harness. Everything else from earlier branches —
//! the neural network arm, the dictionary and context arms, the tokenizer,
//! the classifier — is gone. This branch rebuilds a deterministic, fast base
//! to extend deliberately.

#![deny(unsafe_code)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::redundant_pub_crate)]
#![allow(clippy::module_name_repetitions)]

mod ac;
mod bits;
mod cmix;
mod codec;
mod eval;
mod lmix;
mod null;

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
    /// Multi-offset eval panel with per-window bpb. 20 windows × 256 KiB by
    /// default, or `--quick` for 5 × 64 KiB.
    Bench {
        #[arg(long, default_value = "assets/enwik8")]
        corpus: PathBuf,
        /// Codec name. v7 knows: `null`, `cmix`, `lmix`, `lmatch`, `lsse`, `lword`.
        #[arg(long, default_value = "cmix")]
        codec: String,
        /// Reduce panel to 5 windows × 64 KiB for tight iteration.
        #[arg(long, default_value_t = false)]
        quick: bool,
        /// Dump per-component bit decomposition to CSV.
        #[arg(long)]
        decompose: Option<PathBuf>,
    },

    /// Compress a whole corpus end-to-end. Verifies the encode → decode
    /// round-trip unless `--skip-verify` is set.
    Compress {
        #[arg(long, default_value = "assets/enwik9")]
        corpus: PathBuf,
        #[arg(long, default_value = "cmix")]
        codec: String,
        #[arg(long, default_value = "/tmp/lzr.archive")]
        out: PathBuf,
        #[arg(long, default_value_t = false)]
        skip_verify: bool,
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
        Command::Compress {
            corpus,
            codec,
            out,
            skip_verify,
        } => run_compress(&corpus, &codec, &out, skip_verify),
    }
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
        n as f64 / (1024.0 * 1024.0)
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
    if encode_elapsed.as_secs_f64() > 0.0 {
        println!(
            "Encode rate:     {:.2} MiB/s",
            n as f64 / (1024.0 * 1024.0) / encode_elapsed.as_secs_f64(),
        );
    }

    if skip_verify {
        println!("Verify:          skipped");
        return Ok(());
    }

    eprintln!("Decoding for round-trip verification ...");
    let decode_start = Instant::now();
    let decoded = codec_ref
        .decode_window(b"", &archive)
        .context("decode_window failed")?;
    let decode_elapsed = decode_start.elapsed();

    if decoded.len() != n {
        bail!("decode produced {} bytes; expected {n}", decoded.len());
    }
    if decoded != bytes {
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
    println!("Round-trip:      OK");
    Ok(())
}
