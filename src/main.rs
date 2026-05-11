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
mod bench;
mod bitnet;
mod codec;
mod lz77;
mod model;
mod ppm;
mod predict;
mod probs;
mod routed;
mod tokenizer;
mod weights;

#[cfg(feature = "training")]
mod bpe;
#[cfg(feature = "training")]
mod train;

use std::fs::File;
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use codec::{TransformerProbs, decode_bytes, encode_bytes};
use model::ByteTransformer;
use tokenizer::Tokenizer;
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
        /// Tokenizer file path (default: `assets/tokenizer.bin`).
        #[arg(long, default_value = "assets/tokenizer.bin")]
        tokenizer: PathBuf,
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
        /// Tokenizer file path (default: `assets/tokenizer.bin`).
        #[arg(long, default_value = "assets/tokenizer.bin")]
        tokenizer: PathBuf,
    },

    /// Run the fixed 5-offset codec eval panel against a deterministic
    /// predictor. Used to measure phase-1 ensemble iterations against a
    /// stable baseline. With `--weights`, additionally runs the LZ-routed
    /// codec with the neural arm linearly mixed at literal positions.
    Bench {
        /// Corpus file (default: `assets/enwik9`).
        #[arg(long, default_value = "assets/enwik9")]
        corpus: PathBuf,
        /// Optional model weights (raw or checkpoint). When provided,
        /// runs the neural-mixed LZ-routed codec alongside the plain
        /// version.
        #[arg(long)]
        weights: Option<PathBuf>,
        /// Neural mixing weight in `[0, 1]`. 0.5 = equal contribution.
        #[arg(long, default_value_t = 0.5)]
        mix_weight: f32,
        /// Mixing scheme: "linear" (arithmetic mean of probs) or
        /// "logistic" (PAQ-style geometric mean, sharpens agreement).
        #[arg(long, default_value = "logistic")]
        mix_mode: String,
    },

    /// Train the model on a file (training feature only).
    #[cfg(feature = "training")]
    Train(train::TrainArgs),

    /// Train a byte-level BPE tokenizer on a corpus and emit merges as JSON.
    #[cfg(feature = "training")]
    Bpe(bpe::BpeArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Encode {
            input,
            output,
            weights,
            tokenizer,
        } => cmd_encode(&input, &output, &weights, &tokenizer),
        Command::Decode {
            input,
            output,
            weights,
            tokenizer,
        } => cmd_decode(&input, &output, &weights, &tokenizer),
        Command::Bench {
            corpus,
            weights,
            mix_weight,
            mix_mode,
        } => {
            let mode = match mix_mode.as_str() {
                "linear" => lz77::MixMode::Linear,
                "logistic" => lz77::MixMode::Logistic,
                "adaptive" | "adaptive-logistic" => lz77::MixMode::AdaptiveLogistic,
                other => {
                    anyhow::bail!("unknown mix-mode '{other}' (use linear, logistic, or adaptive)")
                }
            };
            cmd_bench(&corpus, weights.as_deref(), mix_weight, mode)
        }
        #[cfg(feature = "training")]
        Command::Train(args) => train::run(args),
        #[cfg(feature = "training")]
        Command::Bpe(args) => bpe::run(args),
    }
}

fn cmd_bench(
    corpus: &Path,
    weights_path: Option<&Path>,
    mix_weight: f32,
    mix_mode: lz77::MixMode,
) -> Result<()> {
    println!("Corpus: {}", corpus.display());
    println!();
    run_one(
        corpus,
        "order-0 adaptive (Laplace +1, online halving)",
        || Box::new(predict::Order0Adaptive::new()),
    )?;
    run_one(
        corpus,
        "order-1 adaptive (per-context Laplace +1, online halving)",
        || Box::new(predict::Order1Adaptive::new()),
    )?;
    run_one(
        corpus,
        "order-2 adaptive (dense byte-pair contexts, 64 MiB state)",
        || Box::new(predict::Order2Adaptive::new()),
    )?;
    run_routed(corpus)?;
    run_lz_routed(corpus, None, 0.0, mix_mode)?;
    if let Some(wp) = weights_path {
        let weights = load_weights(&wp.to_path_buf())?;
        let neural = (weights, mix_weight, mix_mode);
        run_lz_routed(corpus, Some(&neural), mix_weight, mix_mode)?;
    }
    Ok(())
}

fn run_routed(corpus: &Path) -> Result<()> {
    let result = bench::run_routed(corpus)?;
    println!(
        "Predictor: type-routed, cross-stream (O1 type / PPM-D-5 letter / PPM-D-8 non-letter)"
    );
    println!("  offset (B)       bpb");
    println!("  ----------       -----");
    for r in &result.per_offset {
        println!("  {:>10}     {:.3}", r.offset, r.bpb);
    }
    println!("  ----------       -----");
    println!("  mean             {:.3}", result.mean_bpb);
    println!();
    Ok(())
}

fn run_lz_routed(
    corpus: &Path,
    neural: Option<&(Weights, f32, lz77::MixMode)>,
    mix_weight: f32,
    mix_mode: lz77::MixMode,
) -> Result<()> {
    let label = if neural.is_some() {
        let mode_str = match mix_mode {
            lz77::MixMode::Linear => "linear",
            lz77::MixMode::Logistic => "logistic",
            lz77::MixMode::AdaptiveLogistic => "adaptive-3 (n+r+o2)",
        };
        format!(
            "LZ77 (4 MiB window, MIN_MATCH=6, bucketed, lazy parse) + routed + neural ({mode_str} w={mix_weight:.2})"
        )
    } else {
        "LZ77 (4 MiB window, MIN_MATCH=6, bucketed, lazy parse) + routed".to_string()
    };
    let result = bench::run_lz_routed(corpus, neural)?;
    println!("Predictor: {label}");
    println!("  offset (B)       bpb");
    println!("  ----------       -----");
    for r in &result.per_offset {
        println!("  {:>10}     {:.3}", r.offset, r.bpb);
    }
    println!("  ----------       -----");
    println!("  mean             {:.3}", result.mean_bpb);
    println!();
    Ok(())
}

fn run_one<F>(corpus: &Path, label: &str, factory: F) -> Result<()>
where
    F: FnMut() -> Box<dyn codec::ProbSource>,
{
    let result = bench::run(corpus, factory)?;
    println!("Predictor: {label}");
    println!("  offset (B)       bpb");
    println!("  ----------       -----");
    for r in &result.per_offset {
        println!("  {:>10}     {:.3}", r.offset, r.bpb);
    }
    println!("  ----------       -----");
    println!("  mean             {:.3}", result.mean_bpb);
    println!();
    Ok(())
}

fn cmd_encode(
    input: &PathBuf,
    output: &PathBuf,
    weights_path: &PathBuf,
    tokenizer_path: &PathBuf,
) -> Result<()> {
    let src = std::fs::read(input).with_context(|| format!("reading input {}", input.display()))?;
    let tokenizer = Tokenizer::load(tokenizer_path)?;
    let weights = load_weights(weights_path)?;
    let mut model = ByteTransformer::new(weights);
    let mut probs = TransformerProbs::new(&mut model);
    let out =
        File::create(output).with_context(|| format!("creating output {}", output.display()))?;
    encode_bytes(&src, &tokenizer, &mut probs, out)?;
    Ok(())
}

fn cmd_decode(
    input: &PathBuf,
    output: &PathBuf,
    weights_path: &PathBuf,
    tokenizer_path: &PathBuf,
) -> Result<()> {
    let f = File::open(input).with_context(|| format!("opening {}", input.display()))?;
    let mut rd = BufReader::new(f);
    let tokenizer = Tokenizer::load(tokenizer_path)?;
    let weights = load_weights(weights_path)?;
    let mut model = ByteTransformer::new(weights);
    let mut probs = TransformerProbs::new(&mut model);
    let out_bytes = decode_bytes(&mut rd, &tokenizer, &mut probs)?;
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
