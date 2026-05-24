//! V4 neural-first compressor — Hutter Prize attempt.
//!
//! Pure-neural codec built around the sparse-MoE byte-level AR
//! transformer trained in `lzr-neural/`. The CLI exposes only what
//! v4 needs: a panel bench, a full-corpus encode/decode, and a
//! standalone neural-arm bpb measurement.

#![deny(unsafe_code)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::redundant_pub_crate)]
#![allow(clippy::module_name_repetitions)]

mod ac;
mod bits;
mod bpe;
mod codec;
mod eval;
#[cfg(feature = "gpu-inference")]
mod gpu;
mod int_inference;
mod moe;
mod moe_arm;
mod moe_codec;
mod moe_tok_codec;
mod ngram_arm;
mod null;
mod struct_mask;
mod transformer;

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
    /// Multi-offset eval panel: 20 windows × 256 KiB by default,
    /// with per-window bpb and per-component decomposition.
    Bench {
        #[arg(long, default_value = "assets/enwik8")]
        corpus: PathBuf,
        /// Codec name. v4 currently knows: `null`, `moe`.
        #[arg(long, default_value = "null")]
        codec: String,
        /// Reduce panel to 5 windows × 64 KiB for tight iteration.
        #[arg(long, default_value_t = false)]
        quick: bool,
        /// Dump per-component bit decomposition to CSV.
        #[arg(long)]
        decompose: Option<PathBuf>,
    },

    /// Compress a whole corpus through a codec end-to-end. Verifies
    /// the encode → decode roundtrip unless `--skip-verify` is set.
    Compress {
        #[arg(long, default_value = "assets/enwik9")]
        corpus: PathBuf,
        #[arg(long, default_value = "moe")]
        codec: String,
        #[arg(long, default_value = "/tmp/lzr.archive")]
        out: PathBuf,
        #[arg(long, default_value_t = false)]
        skip_verify: bool,
    },

    /// Standalone bpb of the `MoE` arm on a corpus prefix. The codec
    /// itself isn't exercised — this is the "what would AC alone
    /// achieve from the predictor's distribution" baseline that
    /// `bench --codec moe` should approach.
    NeuralEval {
        /// Path to the .lzrm weights.
        #[arg(long)]
        weights: PathBuf,
        #[arg(long, default_value = "assets/enwik9")]
        corpus: PathBuf,
        /// Bytes to evaluate from the start of the corpus.
        #[arg(long, default_value_t = 1_000_000)]
        bytes: usize,
    },

    /// Generate a per-byte-class allowed-token bitmask by BPE-encoding
    /// a training corpus and recording which token ids appear after
    /// bytes of each class. Output is a 32 KB file consumable via
    /// `LZR_STRUCT_MASK`.
    GenMask {
        #[arg(long, default_value = "assets/enwik9")]
        corpus: PathBuf,
        #[arg(long)]
        bpe: PathBuf,
        /// Limit corpus bytes (default: full corpus).
        #[arg(long)]
        bytes: Option<usize>,
        #[arg(long, default_value = "/tmp/struct_mask.bin")]
        out: PathBuf,
    },

    /// Phase 50C Day-4 follow-up: characterize what each of the
    /// `n_experts` MoE experts actually does. Runs the model over a
    /// corpus prefix in batched-encode mode, captures per-token
    /// routing decisions, and writes a text report covering:
    /// load distribution per layer, routing entropy, byte-class ×
    /// expert crosstab, and top tokens per expert. Informs v5
    /// architectural choices (bigger MoE vs dense vs simpler router).
    RoutingAnalyze {
        #[arg(long, default_value = "assets/enwik9")]
        corpus: PathBuf,
        /// Bytes from the start of the corpus to analyze. 1 MB is
        /// enough for stable statistics on Phase 45.
        #[arg(long, default_value_t = 1_000_000)]
        bytes: usize,
        /// Path to the MoE `.lzrm` weights. If omitted, falls back
        /// to the `LZR_MOE_WEIGHTS` env var.
        #[arg(long)]
        weights: Option<PathBuf>,
        /// Path to the BPE table. If omitted, falls back to the
        /// `LZR_BPE_TABLE` env var.
        #[arg(long)]
        bpe: Option<PathBuf>,
        /// Output report file.
        #[arg(long, default_value = "/tmp/routing_report.txt")]
        out: PathBuf,
        /// Top-K tokens to list per expert.
        #[arg(long, default_value_t = 10)]
        top_k: usize,
    },
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
        n as f64 / (1024.0 * 1024.0),
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
    if let Some((quant_name, shipped_bytes, ld_bpb_on_enwik9)) = projected_ld_on_enwik9() {
        println!(
            "L(D) projection: {shipped_bytes} weight-bytes shipped at {quant_name} × Hutter 2× rule → {ld_bpb_on_enwik9:.4} bpb on 1 GB enwik9"
        );
        println!(
            "Combined L+D bpb (this corpus's L(C) + 1 GB-amortized L(D)): {:.4}",
            bpb + ld_bpb_on_enwik9
        );
    }
    println!("Encode time:     {encode_elapsed:?}");
    println!(
        "Encode rate:     {:.2} MiB/s",
        n as f64 / (1024.0 * 1024.0) / encode_elapsed.as_secs_f64(),
    );

    if skip_verify {
        println!("Verify:          skipped");
        return Ok(());
    }

    eprintln!("Decoding for roundtrip verification ...");
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
    println!(
        "Decode rate:     {:.2} MiB/s",
        n as f64 / (1024.0 * 1024.0) / decode_elapsed.as_secs_f64(),
    );
    println!("Roundtrip:       OK");
    Ok(())
}

/// Streaming `MoE` forward over a corpus prefix; reports standalone bpb.
///
/// Resets the KV cache at every `context` boundary because trained
/// absolute position embeddings are only valid for `0..context`. The
/// first byte after each reset has no context so its prediction is
/// near-uniform; we still count it in the bpb so the number stays
/// directly comparable to other byte-level compressors.
#[allow(clippy::cast_precision_loss)]
fn run_neural_eval(weights: &PathBuf, corpus: &PathBuf, bytes_to_eval: usize) -> Result<()> {
    let weights_bytes =
        fs::read(weights).with_context(|| format!("reading weights {}", weights.display()))?;
    let model = moe::MoeByteTransformer::load_lzrm(&weights_bytes)
        .with_context(|| "parsing .lzrm weights")?;
    let cfg = model.cfg;
    eprintln!(
        "Loaded MoE: layers={} heads={} d_model={} d_ff={} context={} vocab={} n_experts={}",
        cfg.n_layer, cfg.n_head, cfg.d_model, cfg.d_ff, cfg.context, cfg.vocab_size, cfg.n_experts,
    );

    let corpus_bytes =
        fs::read(corpus).with_context(|| format!("reading corpus {}", corpus.display()))?;
    let n = bytes_to_eval.min(corpus_bytes.len());
    eprintln!(
        "Evaluating first {n} bytes of {} ({:.2} MiB)",
        corpus.display(),
        n as f64 / (1024.0 * 1024.0),
    );

    let mut cache = model.new_kv_cache();
    let mut total_nats = 0_f64;
    let mut bytes_seen: usize = 0;
    let mut chunks_processed: usize = 0;
    let start = Instant::now();

    for chunk in corpus_bytes[..n].chunks(cfg.context) {
        cache.reset();
        let mut logits = model.forward_step(&mut cache, 0_u32);
        for &byte in chunk {
            let log_p = log_softmax_at(&logits, byte as usize);
            total_nats -= f64::from(log_p);
            bytes_seen += 1;
            if cache.pos < cfg.context {
                logits = model.forward_step(&mut cache, u32::from(byte));
            }
        }
        chunks_processed += 1;
        if chunks_processed % 16 == 0 {
            let bpb_so_far = total_nats / (bytes_seen as f64) / std::f64::consts::LN_2;
            let rate = (bytes_seen as f64) / start.elapsed().as_secs_f64() / (1024.0 * 1024.0);
            eprintln!(
                "  chunk {chunks_processed}  bytes={bytes_seen}  bpb={bpb_so_far:.4}  rate={rate:.2} MiB/s"
            );
        }
    }

    let bpb = total_nats / (bytes_seen as f64) / std::f64::consts::LN_2;
    let elapsed = start.elapsed();
    println!();
    println!("Weights:        {}", weights.display());
    println!("Corpus:         {}", corpus.display());
    println!("Bytes evaluated: {bytes_seen}");
    println!("Standalone bpb: {bpb:.4}");
    println!("Inference time: {elapsed:?}");
    println!(
        "Throughput:     {:.2} MiB/s",
        bytes_seen as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0),
    );
    Ok(())
}

/// Read `LZR_MOE_WEIGHTS` + `LZR_MOE_QUANT` (+ optionally
/// `LZR_BPE_TABLE`) and compute the projected `L(D)` tax the shipped
/// binary would pay on 1 GB enwik9 — under the **`2×` Hutter rule**
/// (the same binary appears in `S` as both `comp9a` and the
/// reduced-multiplier `decomp9`, so every shipped byte counts
/// twice). Loads the model so per-tensor accounting (per-channel
/// scales, mixed-precision schemes) is exact, not estimated. If
/// `LZR_BPE_TABLE` is set, its file size is included in the shipped
/// bytes (the BPE table must ship for the token codec to decode).
/// Returns `None` if the weights env var isn't set or loading fails.
/// See `JOURNAL.md` 2026-05-16 Phase 32.
#[allow(clippy::cast_precision_loss)]
fn projected_ld_on_enwik9() -> Option<(String, u64, f64)> {
    let weights_path: PathBuf = std::env::var_os("LZR_MOE_WEIGHTS")?.into();
    let bytes = fs::read(&weights_path).ok()?;
    let model = moe::MoeByteTransformer::load_lzrm(&bytes).ok()?;
    let quant = moe::Quantization::from_env(moe_arm::QUANT_ENV);
    let mut weights_bytes = model.shipped_bytes(quant);
    let mut name = quant.name().to_string();
    if let Some(bpe_path) = std::env::var_os("LZR_BPE_TABLE") {
        if let Ok(meta) = fs::metadata(PathBuf::from(&bpe_path)) {
            weights_bytes += meta.len();
            name.push_str("+bpe");
        }
    }
    // Structural mask is embedded in the binary via include_bytes!.
    weights_bytes += struct_mask::FILE_BYTES as u64;
    name.push_str("+mask");
    // 2× factor: the binary appears twice in S.
    let shipped_bytes = 2 * weights_bytes;
    let ld_bpb = 8.0 * (shipped_bytes as f64) / 1e9;
    Some((name, shipped_bytes, ld_bpb))
}

#[allow(clippy::cast_precision_loss)]
fn log_softmax_at(logits: &[f32], idx: usize) -> f32 {
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        if v > max {
            max = v;
        }
    }
    let mut sum_exp = 0_f32;
    for &v in logits {
        sum_exp += (v - max).exp();
    }
    logits[idx] - max - sum_exp.ln()
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
        Command::NeuralEval {
            weights,
            corpus,
            bytes,
        } => run_neural_eval(&weights, &corpus, bytes),
        Command::GenMask {
            corpus,
            bpe,
            bytes,
            out,
        } => run_gen_mask(&corpus, &bpe, bytes, &out),
        Command::RoutingAnalyze {
            corpus,
            bytes,
            weights,
            bpe,
            out,
            top_k,
        } => run_routing_analyze(&corpus, bytes, weights.as_deref(), bpe.as_deref(), &out, top_k),
    }
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]
fn run_routing_analyze(
    corpus: &std::path::Path,
    bytes_to_eval: usize,
    weights_arg: Option<&std::path::Path>,
    bpe_arg: Option<&std::path::Path>,
    out: &std::path::Path,
    top_k: usize,
) -> Result<()> {
    use std::io::Write;
    let weights_path: PathBuf = weights_arg
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("LZR_MOE_WEIGHTS").map(PathBuf::from))
        .context("--weights or LZR_MOE_WEIGHTS required")?;
    let bpe_path: PathBuf = bpe_arg
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("LZR_BPE_TABLE").map(PathBuf::from))
        .context("--bpe or LZR_BPE_TABLE required")?;

    eprintln!("Loading BPE table from {} ...", bpe_path.display());
    let bpe_inst = bpe::Bpe::load(&bpe_path).context("loading BPE table")?;
    eprintln!("Loading MoE weights from {} ...", weights_path.display());
    let weights_bytes = fs::read(&weights_path)
        .with_context(|| format!("reading MoE weights {}", weights_path.display()))?;
    let mut model = moe::MoeByteTransformer::load_lzrm(&weights_bytes)
        .with_context(|| "parsing .lzrm")?;
    // Use the same quantization scheme as the codec (mixed5asym was
    // shown to be the L+D sweet spot in Phase 50).
    model.apply_quantization(moe::Quantization::Mixed5Asym);
    model.prepare_int_cache();
    let cfg = model.cfg;
    if bpe_inst.vocab_size != cfg.vocab_size {
        bail!(
            "BPE vocab {} != MoE vocab {}",
            bpe_inst.vocab_size,
            cfg.vocab_size
        );
    }

    eprintln!("Reading {} bytes from {} ...", bytes_to_eval, corpus.display());
    let raw = fs::read(corpus).with_context(|| format!("reading {}", corpus.display()))?;
    let take = bytes_to_eval.min(raw.len());
    let src = &raw[..take];
    eprintln!("BPE-encoding {take} bytes ...");
    let tokens = bpe_inst.encode(src);
    eprintln!("  {} tokens ({:.2} bytes/tok)", tokens.len(), take as f64 / tokens.len() as f64);

    // Run the model in `cfg.context`-sized chunks, accumulating routing.
    eprintln!(
        "Running model.dump_routing on {} tokens (cfg.context={}, chunks ≈ {}) ...",
        tokens.len(),
        cfg.context,
        tokens.len().div_ceil(cfg.context)
    );
    let t0 = Instant::now();
    let mut combined = moe::RoutingTrace::new(cfg.n_layer, tokens.len(), cfg.n_experts);
    let mut chunk_start = 0usize;
    while chunk_start < tokens.len() {
        let chunk_end = (chunk_start + cfg.context).min(tokens.len());
        let chunk = &tokens[chunk_start..chunk_end];
        let chunk_trace = model.dump_routing(chunk);
        let chunk_len = chunk.len();
        for l in 0..cfg.n_layer {
            for i in 0..chunk_len {
                let dst = l * tokens.len() + (chunk_start + i);
                let src_idx = l * chunk_len + i;
                combined.expert_assignment[dst] = chunk_trace.expert_assignment[src_idx];
                combined.gate_val[dst] = chunk_trace.gate_val[src_idx];
                combined.router_entropy_nats[dst] = chunk_trace.router_entropy_nats[src_idx];
            }
        }
        chunk_start += cfg.context;
    }
    eprintln!("  done in {:.2}s", t0.elapsed().as_secs_f64());

    // Per-token previous-byte class (the same input the codec uses
    // for its struct_mask bias). Index i ≥ 1 uses the last byte of
    // token i-1; index 0 falls back to '\n' as the codec does.
    let mut prev_byte_class = vec![0u8; tokens.len()];
    let mut last_byte = b'\n';
    for (i, &tok) in tokens.iter().enumerate() {
        let class = struct_mask::byte_class(last_byte);
        prev_byte_class[i] = u8::try_from(class).expect("class fits u8");
        let tb = bpe_inst.token_bytes(tok);
        if let Some(&b) = tb.last() {
            last_byte = b;
        }
        if i == 0 {
            // first-token class already captured above
            let _ = i;
        }
    }

    eprintln!("Writing report to {} ...", out.display());
    let mut report = String::new();
    use std::fmt::Write as _;
    writeln!(
        report,
        "# MoE routing analysis\n\n\
         corpus:          {}\n\
         bytes analyzed:  {}\n\
         tokens analyzed: {}\n\
         model layers:    {}\n\
         experts/layer:   {}\n",
        corpus.display(),
        take,
        tokens.len(),
        cfg.n_layer,
        cfg.n_experts,
    )?;

    let n = tokens.len();
    let ne = cfg.n_experts;

    for l in 0..cfg.n_layer {
        writeln!(report, "\n## Layer {l}\n")?;

        // 1) Per-expert load histogram.
        let mut counts = vec![0u64; ne];
        for i in 0..n {
            counts[combined.expert_assignment[l * n + i] as usize] += 1;
        }
        let total = n as f64;
        let mean = total / ne as f64;
        let mut min_c = u64::MAX;
        let mut max_c = 0u64;
        for &c in &counts {
            min_c = min_c.min(c);
            max_c = max_c.max(c);
        }
        let dead = counts.iter().filter(|&&c| (c as f64) < 0.001 * total).count();
        writeln!(
            report,
            "### Load distribution\n\
             mean expected: {:.0} ({:.2}%)\n\
             min: {} ({:.2}%)   max: {} ({:.2}%)   imbalance ratio: {:.2}x\n\
             dead experts (<0.1% load): {} / {}\n",
            mean,
            100.0 / ne as f64,
            min_c,
            100.0 * (min_c as f64) / total,
            max_c,
            100.0 * (max_c as f64) / total,
            (max_c as f64) / (min_c.max(1) as f64),
            dead,
            ne,
        )?;
        writeln!(report, "  expert |   tokens  |  share")?;
        writeln!(report, "  -------+-----------+--------")?;
        for (e, &c) in counts.iter().enumerate() {
            writeln!(
                report,
                "    {:>2}   | {:>9} | {:>5.2}%",
                e,
                c,
                100.0 * (c as f64) / total
            )?;
        }

        // 2) Routing entropy distribution.
        let mut ents: Vec<f32> = (0..n).map(|i| combined.router_entropy_nats[l * n + i]).collect();
        ents.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let max_entropy = (ne as f32).ln();
        let pct = |p: f64| ents[(((ents.len() - 1) as f64) * p) as usize];
        let mean_ent: f32 = ents.iter().sum::<f32>() / ents.len() as f32;
        let confident = ents.iter().filter(|&&e| e < 0.5_f32 * max_entropy).count();
        writeln!(
            report,
            "\n### Routing entropy (nats; uniform = ln({}) = {:.3})\n\
             min: {:.3}   p10: {:.3}   median: {:.3}   p90: {:.3}   max: {:.3}\n\
             mean: {:.3}   confident (< 0.5 × max): {} / {} = {:.2}%\n",
            ne,
            max_entropy,
            ents[0],
            pct(0.10),
            pct(0.50),
            pct(0.90),
            ents[ents.len() - 1],
            mean_ent,
            confident,
            n,
            100.0 * (confident as f64) / total,
        )?;

        // 3) Byte-class × expert crosstab.
        let n_classes = struct_mask::N_CLASSES;
        let mut crosstab = vec![0u64; n_classes * ne];
        let mut class_totals = vec![0u64; n_classes];
        for i in 0..n {
            let cls = prev_byte_class[i] as usize;
            let exp = combined.expert_assignment[l * n + i] as usize;
            crosstab[cls * ne + exp] += 1;
            class_totals[cls] += 1;
        }
        writeln!(
            report,
            "\n### Byte-class × expert crosstab (rows normalized; top expert per class bolded)\n"
        )?;
        write!(report, "  class |  tokens  | top exp (share) | second   ")?;
        writeln!(report, "(rows close to {:.1}% per expert = no specialization)", 100.0 / ne as f64)?;
        writeln!(report, "  ------+----------+-----------------+------------")?;
        for cls in 0..n_classes {
            let ct = class_totals[cls];
            if ct == 0 {
                continue;
            }
            let mut sorted: Vec<(usize, u64)> = (0..ne)
                .map(|e| (e, crosstab[cls * ne + e]))
                .collect();
            sorted.sort_by_key(|&(_, c)| std::cmp::Reverse(c));
            let (e1, c1) = sorted[0];
            let (e2, c2) = sorted[1];
            writeln!(
                report,
                "   {:>2}   | {:>8} |   {:>2} ({:>5.1}%)  |  {:>2} ({:>5.1}%)",
                cls,
                ct,
                e1,
                100.0 * (c1 as f64) / ct as f64,
                e2,
                100.0 * (c2 as f64) / ct as f64,
            )?;
        }

        // 4) Top-K tokens per expert. For each expert, count token IDs
        // routed to it and pick the K most frequent.
        writeln!(
            report,
            "\n### Top {top_k} tokens per expert (by routed count)\n"
        )?;
        for e in 0..ne {
            let mut tok_counts: std::collections::HashMap<u32, u64> =
                std::collections::HashMap::new();
            for i in 0..n {
                if combined.expert_assignment[l * n + i] as usize == e {
                    *tok_counts.entry(tokens[i]).or_insert(0) += 1;
                }
            }
            let mut pairs: Vec<(u32, u64)> = tok_counts.into_iter().collect();
            pairs.sort_by_key(|&(_, c)| std::cmp::Reverse(c));
            let total_e = counts[e] as f64;
            write!(report, "  expert {e:>2} ({} tokens):", counts[e])?;
            for (tid, c) in pairs.iter().take(top_k) {
                let bs = bpe_inst.token_bytes(*tid);
                let mut s = String::new();
                for &b in bs.iter().take(20) {
                    if (0x20..=0x7e).contains(&b) {
                        s.push(b as char);
                    } else {
                        s.push_str(&format!("\\x{b:02x}"));
                    }
                }
                if bs.len() > 20 {
                    s.push('…');
                }
                write!(
                    report,
                    " [{tid:>5} \"{s}\" {:.1}%]",
                    100.0 * (*c as f64) / total_e
                )?;
            }
            writeln!(report)?;
        }
    }

    let mut f = fs::File::create(out)
        .with_context(|| format!("creating report {}", out.display()))?;
    f.write_all(report.as_bytes())?;
    eprintln!("Report written ({} bytes).", report.len());
    println!("\n{report}");
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn run_gen_mask(
    corpus: &std::path::Path,
    bpe_path: &std::path::Path,
    bytes: Option<usize>,
    out: &std::path::Path,
) -> Result<()> {
    let bpe = bpe::Bpe::load(bpe_path).context("loading BPE table")?;
    if bpe.vocab_size != struct_mask::VOCAB {
        bail!(
            "struct mask expects vocab {}, BPE has {}",
            struct_mask::VOCAB,
            bpe.vocab_size
        );
    }
    let raw = fs::read(corpus).with_context(|| format!("reading {}", corpus.display()))?;
    let take = bytes.map_or(raw.len(), |n| n.min(raw.len()));
    let src = &raw[..take];
    eprintln!(
        "GenMask: BPE-encoding {take} bytes ({:.2} MiB) ...",
        take as f64 / (1024.0 * 1024.0)
    );
    let t0 = Instant::now();
    let tokens = bpe.encode(src);
    eprintln!(
        "  {} tokens ({:.2} bytes/tok) in {:?}",
        tokens.len(),
        take as f64 / tokens.len() as f64,
        t0.elapsed()
    );

    let mut mask = struct_mask::StructMask::zeros();
    let mut byte_pos = 0_usize;
    for &tok in &tokens {
        let prev_byte = if byte_pos == 0 {
            b'\n'
        } else {
            src[byte_pos - 1]
        };
        let class = struct_mask::byte_class(prev_byte);
        mask.set(class, tok);
        byte_pos += bpe.token_byte_len(tok);
    }

    let mut populations = [0_u32; struct_mask::N_CLASSES];
    for (class, pop) in populations.iter_mut().enumerate() {
        let mut count = 0_u32;
        for tok in 0..u32::try_from(struct_mask::VOCAB).unwrap() {
            if mask.get(class, tok) {
                count += 1;
            }
        }
        *pop = count;
    }
    eprintln!("  per-class allowed-token populations:");
    let class_labels = [
        "lower", "UPPER", "digit", "space", "\\n", "ws", "<", ">", "\"", "&", ";", "=", "/", ".",
        "punct", "nonASCII",
    ];
    for (i, label) in class_labels.iter().enumerate() {
        eprintln!(
            "    [{:>2}] {:<8} {:>6} tokens ({:.1}%)",
            i,
            label,
            populations[i],
            100.0 * f64::from(populations[i]) / struct_mask::VOCAB as f64
        );
    }

    mask.save(out)?;
    eprintln!(
        "Wrote mask: {} ({} bytes)",
        out.display(),
        struct_mask::FILE_BYTES
    );
    Ok(())
}
