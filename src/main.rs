//! Transformer-based compression CLI — Hutter Prize attempt.

// NEON dot-product intrinsics (`vdotq_s32` / ARMv8.2 SDOT) are still behind
// an unstable feature gate as of Rust 1.85 — they give the matvec kernel a
// 4× throughput win over the `vmull_s8` + `vpadalq_s16` fallback on Apple
// Silicon. Training already requires nightly for coverage anyway, so we
// enable the feature here and run the whole toolchain on nightly.
#![cfg_attr(all(target_arch = "aarch64"), feature(stdarch_neon_dotprod))]
#![deny(unsafe_code)]
// Require explicit `unsafe { }` blocks inside `unsafe fn` — modern
// Rust 2024 safety discipline.
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(unused)]
#![warn(clippy::all)]
#![allow(clippy::missing_docs_in_private_items)]
// Keep this list short. Lints that fire only in specific hot-path or
// bit-pattern-cast sites are allowed locally at the relevant function via
// `#[allow(clippy::name)]`, not here.
#![allow(
    // `pub(crate)` inside private modules is the convention this crate uses
    // for all internal visibility; the "redundant" warning is noise.
    clippy::redundant_pub_crate,
    // Module-name repetition in a small, flat bin crate is fine (and
    // fixing it would make identifiers less readable).
    clippy::module_name_repetitions
)]

mod ac;
mod arch;
mod bitnet;
mod codec;
mod model;
mod probs;
mod weights;

#[cfg(feature = "training")]
mod train;

use std::fs::File;
use std::io::{BufReader, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use codec::{TransformerProbs, decode_bytes, encode_bytes};
use model::ByteTransformer;
use weights::Weights;

#[derive(Parser, Debug)]
#[command(name = "lzr", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Compress a file into a `.lzr` archive using the embedded model.
    Encode {
        /// Input file to compress.
        input: PathBuf,
        /// Output archive path.
        output: PathBuf,
        /// Weights file path (default: `assets/weights.bin`).
        #[arg(long, default_value = "assets/weights.bin")]
        weights: PathBuf,
    },

    /// Decompress a `.lzr` archive back to its original bytes.
    Decode {
        /// Input archive.
        input: PathBuf,
        /// Output file.
        output: PathBuf,
        /// Weights file path (default: `assets/weights.bin`).
        #[arg(long, default_value = "assets/weights.bin")]
        weights: PathBuf,
    },

    /// Train the model on a file (training feature only).
    #[cfg(feature = "training")]
    Train(train::TrainArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Encode {
            input,
            output,
            weights,
        } => cmd_encode(&input, &output, &weights),
        Command::Decode {
            input,
            output,
            weights,
        } => cmd_decode(&input, &output, &weights),
        #[cfg(feature = "training")]
        Command::Train(args) => train::run(args),
    }
}

fn cmd_encode(input: &PathBuf, output: &PathBuf, weights_path: &PathBuf) -> Result<()> {
    let src = std::fs::read(input).with_context(|| format!("reading input {}", input.display()))?;
    let weights = load_weights(weights_path)?;
    let mut model = ByteTransformer::new(weights);
    let mut probs = TransformerProbs::new(&mut model);
    let out =
        File::create(output).with_context(|| format!("creating output {}", output.display()))?;
    encode_bytes(&src, &mut probs, out)?;
    Ok(())
}

fn cmd_decode(input: &PathBuf, output: &PathBuf, weights_path: &PathBuf) -> Result<()> {
    let f = File::open(input).with_context(|| format!("opening {}", input.display()))?;
    let mut rd = BufReader::new(f);
    let weights = load_weights(weights_path)?;
    let mut model = ByteTransformer::new(weights);
    let mut probs = TransformerProbs::new(&mut model);
    let out_bytes = decode_bytes(&mut rd, &mut probs)?;
    let mut out = File::create(output).with_context(|| format!("creating {}", output.display()))?;
    out.write_all(&out_bytes)?;
    out.flush()?;
    Ok(())
}

fn load_weights(path: &PathBuf) -> Result<Weights> {
    // Accept either a raw weights file or a checkpoint file (auto-detect by
    // length; the magic check lives in `Weights::load_checkpoint`).
    let meta = std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    let len = usize::try_from(meta.len())
        .with_context(|| format!("weights file too large for this target: {}", path.display()))?;
    if len == arch::PACKED_WEIGHTS_LEN {
        Weights::load_raw(path)
    } else if len == 16 + arch::PACKED_WEIGHTS_LEN {
        let (w, _step) = Weights::load_checkpoint(path)?;
        Ok(w)
    } else {
        anyhow::bail!(
            "weights file {} has unexpected length {} (expected {} raw or {} checkpoint)",
            path.display(),
            len,
            arch::PACKED_WEIGHTS_LEN,
            16 + arch::PACKED_WEIGHTS_LEN
        )
    }
}
