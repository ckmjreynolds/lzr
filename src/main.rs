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
mod dict;
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
        /// Codec name. v7 knows: `null`, `cmix`, `lmix`, `lmatch`, `lsse`, `lword`, `lhi`, `ldict`, `lnn`, `lnndict`, `lrnn`, `lrnndict`, `lgru`, `lgrudict`.
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

    /// Attribute a codec's coding cost by byte class, to see where the bits go.
    /// Warms on the first `--warm-mb` MiB, then measures the next `--measure-mb`.
    Analyze {
        #[arg(long, default_value = "assets/enwik8")]
        corpus: PathBuf,
        /// Codec whose model to analyze (one of the `l*` codecs).
        #[arg(long, default_value = "lhi")]
        codec: String,
        #[arg(long, default_value_t = 4)]
        warm_mb: usize,
        #[arg(long, default_value_t = 4)]
        measure_mb: usize,
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
        Command::Analyze {
            corpus,
            codec,
            warm_mb,
            measure_mb,
        } => run_analyze(&corpus, &codec, warm_mb, measure_mb),
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

#[allow(clippy::cast_precision_loss)]
fn run_analyze(corpus: &PathBuf, codec: &str, warm_mb: usize, measure_mb: usize) -> Result<()> {
    let flags = lmix::flags_for(codec).with_context(|| {
        format!("'{codec}' is not an analyzable codec (use one of the l* codecs)")
    })?;

    let warm_len = warm_mb * 1024 * 1024;
    let measure_len = measure_mb * 1024 * 1024;
    let bytes = fs::read(corpus).with_context(|| format!("reading {}", corpus.display()))?;
    if bytes.len() < warm_len + measure_len {
        bail!(
            "corpus {} has {} bytes; need {} (warm {warm_mb} MiB + measure {measure_mb} MiB)",
            corpus.display(),
            bytes.len(),
            warm_len + measure_len,
        );
    }
    let warm = &bytes[..warm_len];
    let measure = &bytes[warm_len..warm_len + measure_len];

    eprintln!(
        "Analyzing `{codec}` on {} (warm {warm_mb} MiB, measure {measure_mb} MiB) ...",
        corpus.display()
    );
    let report = lmix::residual_report(flags, warm, measure);

    let total_bits = report.total_bits();
    let total_count = report.total_count();
    let mut order: Vec<usize> = (0..lmix::CLASS_NAMES.len()).collect();
    order.sort_by(|&a, &b| report.bits[b].total_cmp(&report.bits[a]));

    println!();
    println!("Codec:  {codec}");
    println!("Corpus: {}  (measure {measure_mb} MiB)", corpus.display());
    println!();
    println!("  class      bytes    %bytes      bits    %bits     bpb");
    println!("  -------  ---------  ------  ----------  ------  ------");
    for &c in &order {
        let cnt = report.count[c];
        if cnt == 0 {
            continue;
        }
        let bits = report.bits[c];
        println!(
            "  {:<7}  {:>9}  {:>5.1}%  {:>10.0}  {:>5.1}%  {:>6.3}",
            lmix::CLASS_NAMES[c],
            cnt,
            100.0 * cnt as f64 / total_count as f64,
            bits,
            100.0 * bits / total_bits,
            bits / cnt as f64,
        );
    }
    println!("  -------  ---------  ------  ----------  ------  ------");
    println!(
        "  {:<7}  {:>9}  {:>5}   {:>10.0}  {:>5}   {:>6.3}",
        "total",
        total_count,
        "",
        total_bits,
        "",
        total_bits / total_count as f64,
    );
    println!();
    Ok(())
}
