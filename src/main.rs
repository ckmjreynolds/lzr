//! LZR v8 — neural-first compressor (Hutter Prize attempt).
//!
//! Single binary, single-threaded, no `burn`: the trained `BitNet` transformer
//! weights are baked in via `include_bytes!` and run through a hand-rolled
//! scalar forward (`v8`) driving a range coder. The same binary compresses and
//! decompresses (the `comp9a == decomp9` relaxation). The training side —
//! `burn`/GPU, behind the `neural` feature — lives in `examples/neural.rs` and
//! produces the weight blob this binary embeds.

#![deny(unsafe_code)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::redundant_pub_crate)]
#![allow(clippy::module_name_repetitions)]

mod v8;

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

/// Trained `BitNet` weights (ternary, 2-bit packed), produced offline by
/// `examples/neural.rs` and baked into the binary. This is the `2×`-counted
/// `L(D)` payload of the Hutter score.
static V8_WEIGHTS: &[u8] = include_bytes!("../assets/v8weights.bin");

/// Container BOS seed for the autoregressive context.
const BOS: i32 = 0;

#[derive(Parser, Debug)]
#[command(name = "lzr", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Compress a file to an archive (`u64` original length prefix + coded bytes).
    Compress {
        input: PathBuf,
        output: PathBuf,
        /// Skip the in-process decode round-trip check.
        #[arg(long, default_value_t = false)]
        skip_verify: bool,
    },

    /// Decompress an archive produced by `compress`.
    Decompress { input: PathBuf, output: PathBuf },

    /// Quick self-check: compress a slice of a corpus with the embedded weights
    /// and verify the round-trip, reporting bpb and per-byte timing.
    NnTest {
        #[arg(long, default_value = "assets/enwik8")]
        corpus: PathBuf,
        #[arg(long, default_value_t = 256 * 1024)]
        offset: usize,
        #[arg(long, default_value_t = 240)]
        len: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Compress {
            input,
            output,
            skip_verify,
        } => run_compress(&input, &output, skip_verify),
        Command::Decompress { input, output } => run_decompress(&input, &output),
        Command::NnTest {
            corpus,
            offset,
            len,
        } => run_nn_test(&corpus, offset, len),
    }
}

fn model() -> v8::Model {
    v8::Model::from_blob(V8_WEIGHTS)
}

#[allow(clippy::cast_precision_loss)]
fn run_compress(input: &PathBuf, output: &PathBuf, skip_verify: bool) -> Result<()> {
    let bytes = fs::read(input).with_context(|| format!("reading {}", input.display()))?;
    let n = bytes.len();
    let model = model();

    let start = Instant::now();
    let archive = model.compress(&bytes, BOS);
    let elapsed = start.elapsed();

    let mut out = (n as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&archive);
    fs::write(output, &out).with_context(|| format!("writing {}", output.display()))?;

    println!("Input:    {n} bytes");
    println!(
        "Archive:  {} bytes ({:.4} bpb)",
        out.len(),
        8.0 * archive.len() as f64 / n as f64
    );
    println!("Encode:   {elapsed:?}");

    if !skip_verify {
        let decoded = model.decompress(&archive, n, BOS);
        if decoded != bytes {
            bail!("round-trip verification failed");
        }
        println!("Verify:   round-trip OK");
    }
    Ok(())
}

#[allow(clippy::cast_possible_truncation)] // judging machine is 64-bit
fn run_decompress(input: &PathBuf, output: &PathBuf) -> Result<()> {
    let blob = fs::read(input).with_context(|| format!("reading {}", input.display()))?;
    if blob.len() < 8 {
        bail!("archive too short to hold a length prefix");
    }
    let n = u64::from_le_bytes(blob[..8].try_into().unwrap()) as usize;
    let decoded = model().decompress(&blob[8..], n, BOS);
    fs::write(output, &decoded).with_context(|| format!("writing {}", output.display()))?;
    println!("Decoded:  {n} bytes -> {}", output.display());
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn run_nn_test(corpus: &PathBuf, offset: usize, len: usize) -> Result<()> {
    let bytes = fs::read(corpus).with_context(|| format!("reading {}", corpus.display()))?;
    if offset + len > bytes.len() {
        bail!(
            "slice [{offset}, {}) exceeds corpus length {}",
            offset + len,
            bytes.len()
        );
    }
    let slice = &bytes[offset..offset + len];
    let model = model();
    eprintln!(
        "v8 weights: {} bytes embedded  (L(D) ≈ {:.4} bpb on enwik9)",
        V8_WEIGHTS.len(),
        2.0 * V8_WEIGHTS.len() as f64 / 1e9,
    );

    let start = Instant::now();
    let archive = model.compress(slice, BOS);
    let encode = start.elapsed();
    let dstart = Instant::now();
    let decoded = model.decompress(&archive, slice.len(), BOS);
    let decode = dstart.elapsed();

    let ok = decoded == slice;
    println!();
    println!("Slice:        {len} bytes  [{offset}, {})", offset + len);
    println!("Archive:      {} bytes", archive.len());
    println!(
        "L(C) bpb:     {:.4}",
        8.0 * archive.len() as f64 / slice.len() as f64
    );
    println!("Round-trip:   {}", if ok { "OK" } else { "MISMATCH" });
    println!(
        "Throughput:   {:.3} ms/byte encode, {:.3} ms/byte decode",
        encode.as_secs_f64() * 1e3 / len as f64,
        decode.as_secs_f64() * 1e3 / len as f64,
    );
    if !ok {
        bail!("v8 round-trip mismatch");
    }
    Ok(())
}
