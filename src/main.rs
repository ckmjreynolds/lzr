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

mod bpe;
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

/// `L(D)` in bpb on enwik9: the shipped binary is paid `2×` (compressor +
/// reduced-multiplier decompressor) and bpb is bits per byte, so the cost is
/// `8 × 2 × binary_bytes / 1e9 = 16 × binary_bytes / 1e9`. Uses the *actual*
/// running executable (code + embedded weights), not just the weight blob.
#[allow(clippy::cast_precision_loss)]
fn ld_bpb() -> (u64, f64) {
    let bytes = std::env::current_exe()
        .ok()
        .and_then(|p| fs::metadata(p).ok())
        .map_or(0, |m| m.len());
    (bytes, 16.0 * bytes as f64 / 1e9)
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

    let lc = 8.0 * archive.len() as f64 / n as f64;
    let (bin_bytes, ld) = ld_bpb();
    println!("Input:    {n} bytes");
    println!("Archive:  {} bytes", out.len());
    println!("L(C):     {lc:.4} bpb");
    println!("L(D):     {ld:.4} bpb  ({bin_bytes} B binary × 2 penalty / enwik9)");
    println!("Net:      {:.4} bpb", lc + ld);
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
    let (bin_bytes, ld) = ld_bpb();
    let (vocab, d, layers, heads, ffn, ctx) = model.dims();
    let params = model.num_params();

    let start = Instant::now();
    let archive = model.compress(slice, BOS);
    let encode = start.elapsed();
    let dstart = Instant::now();
    let decoded = model.decompress(&archive, slice.len(), BOS);
    let decode = dstart.elapsed();

    let ok = decoded == slice;
    let lc = 8.0 * archive.len() as f64 / slice.len() as f64;
    let enc_ms_byte = encode.as_secs_f64() * 1e3 / len as f64;
    let dec_ms_byte = decode.as_secs_f64() * 1e3 / len as f64;
    let eta_h = |ms_byte: f64| ms_byte * 1e9 / 1e3 / 3600.0;

    println!();
    println!("Architecture: ternary BitNet transformer (KV-cached, CPU/auto-vec)");
    println!(
        "              d={d} layers={layers} heads={heads} ffn={ffn} ctx={ctx}  vocab={vocab}  ~{:.2}M params",
        params as f64 / 1e6,
    );
    println!(
        "Weights:      {} B blob, {bin_bytes} B binary",
        V8_WEIGHTS.len()
    );
    println!(
        "Eval slice:   {len} bytes  [{offset}, {})  -> {} archive bytes",
        offset + len,
        archive.len()
    );
    println!("L(C):         {lc:.4} bpb  (enwik8 slice)");
    println!("L(D):         {ld:.4} bpb  (binary × 2 penalty / enwik9)");
    println!("Net:          {:.4} bpb", lc + ld);
    println!("Round-trip:   {}", if ok { "OK" } else { "MISMATCH" });
    println!(
        "CPU speed:    enc {enc_ms_byte:.4} ms/byte (enwik9 ~{:.1} h), dec {dec_ms_byte:.4} ms/byte (~{:.1} h)",
        eta_h(enc_ms_byte),
        eta_h(dec_ms_byte),
    );
    if !ok {
        bail!("v8 round-trip mismatch");
    }
    Ok(())
}
