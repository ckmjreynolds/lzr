//! Training mode — candle-nn forward pass for the RWKV v4 byte-level model.
//! STE ternary fake-quant on every linear, `AdamW` + cosine LR, and every
//! `checkpoint_every_secs` of wall-clock time writes a timestamped checkpoint
//! then runs an encode/decode roundtrip test on a random 16 KiB slice of the
//! codec sample file.
//!
//! The WKV recurrence is implemented as a sequential loop over the time axis
//! within the training forward. At `seq_len = 256` this costs one candle op
//! per timestep per state component (~4 ops × 256 = 1024 ops per layer per
//! forward pass). Parallel-scan formulations exist but can wait until this
//! first pipeline is verified end-to-end.

use std::fs::{File, OpenOptions, create_dir_all};
use std::io::{Read, Seek, SeekFrom, Write as _};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, Var};
use candle_nn::loss::cross_entropy;
use candle_nn::ops;
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use chrono::Local;
use clap::Args;
use rand::{Rng, SeedableRng};

use crate::arch::{D_FF, D_MODEL, N_LAYERS, RMS_EPS, VOCAB};
use crate::bitnet::{pack_ternary, unpack_i2s_to_rowmajor};

/// Quantize a row-major f32 matrix to ternary `I2_S` + per-row absmean
/// scale, matching the [`BitLinear::quantize`] convention used for the
/// per-layer ternary projections. Used for the token embedding when
/// writing checkpoints.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn quantize_rows_to_ternary(
    rows_flat: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> (Vec<u8>, Vec<f32>) {
    debug_assert_eq!(rows_flat.len(), out_dim * in_dim);
    let mut packed = vec![0u8; (out_dim * in_dim) / 4];
    let mut scale = Vec::with_capacity(out_dim);
    let mut row_buf = vec![0i8; in_dim];
    for row_idx in 0..out_dim {
        let row = &rows_flat[row_idx * in_dim..(row_idx + 1) * in_dim];
        let absmean = row.iter().map(|v| v.abs()).sum::<f32>() / in_dim as f32;
        let s = absmean.max(1e-5);
        scale.push(s);
        for (dst, src) in row_buf.iter_mut().zip(row.iter()) {
            let q = (src / s).clamp(-1.0, 1.0).round();
            *dst = q as i8;
        }
        let byte_off = row_idx * in_dim / 4;
        pack_ternary(&row_buf, &mut packed[byte_off..byte_off + in_dim / 4]);
    }
    (packed, scale)
}
use crate::codec::{TransformerProbs, decode_bytes, encode_bytes};
use crate::model::ByteTransformer;
use crate::tokenizer::{Token, Tokenizer};
use crate::weights::{
    LayerTensors, LayerView, TernaryMatrix, Weights, write_checkpoint, write_weights,
};

const CODEC_SAMPLE_BYTES: usize = 16 * 1024;
const DEFAULT_SEQ_LEN: usize = 256;

/// Default truncated-BPTT chunk size. The WKV loop detaches `aa`/`bb`/`pp`
/// from the autograd graph at every multiple of this many timesteps, so the
/// in-memory BPTT graph caps at `chunk_size` deep instead of `seq_len` deep.
/// State still propagates forward (RWKV's long-range memory is in the state,
/// not the gradient path), so the model still sees full-`seq_len` context;
/// only the gradient signal is bounded.
const DEFAULT_BPTT_CHUNK: usize = 32;

/// CLI arguments for the training subcommand.
#[derive(Args, Debug, Clone)]
pub(crate) struct TrainArgs {
    /// Training data file (e.g. `assets/enwik9`).
    pub input: PathBuf,

    /// Hard cap on optimizer steps (0 = unbounded; stop via Ctrl-C or the
    /// stale-checkpoint plateau detector). Counts globally — resuming
    /// continues from the checkpoint's step toward this cap.
    #[arg(long, default_value_t = 0)]
    pub max_steps: usize,

    /// Exit cleanly after this many optimizer steps within the **current
    /// process**, after writing a final checkpoint. Counts from 0 every
    /// invocation, so `--resume` resets it. 0 disables. Useful as a safety
    /// hatch with a shell while-loop and `--resume` to cap per-process RSS
    /// growth, e.g. if a future allocator regression resurfaces.
    #[arg(long, default_value_t = 0)]
    pub max_steps_this_run: usize,

    /// Stop training after this many *consecutive* checkpoints show no
    /// train-loss improvement vs. the previous one. Set to 0 to disable
    /// plateau stopping.
    #[arg(long, default_value_t = 3)]
    pub stop_after_stale_ckpts: usize,

    /// Batch size.
    #[arg(long, default_value_t = 32)]
    pub batch: usize,

    /// Training sequence length. RWKV has no context-length bound at
    /// inference; this only controls the BPTT window during training.
    #[arg(long, default_value_t = DEFAULT_SEQ_LEN)]
    pub seq: usize,

    /// Truncated-BPTT chunk size: detach WKV state from the autograd graph
    /// every N timesteps. Caps peak training memory at O(`chunk · seq`)
    /// tensors instead of O(`seq²`). 0 disables (full BPTT through the
    /// whole sequence — large memory hit; not recommended at `seq ≥ 128`).
    #[arg(long, default_value_t = DEFAULT_BPTT_CHUNK)]
    pub bptt_chunk: usize,

    /// Emit a per-step log line every N steps.
    #[arg(long, default_value_t = 50)]
    pub log_every: usize,

    /// Seconds between checkpoints + codec tests.
    #[arg(long, default_value_t = 7200)]
    pub checkpoint_every_secs: u64,

    /// Directory for checkpoint files.
    #[arg(long, default_value = "checkpoints")]
    pub ckpt_dir: PathBuf,

    /// Path to the file used for the post-checkpoint codec test.
    /// Defaults to the training input.
    #[arg(long)]
    pub codec_sample_file: Option<PathBuf>,

    /// Training / codec log JSONL file.
    #[arg(long, default_value = "training.log")]
    pub log_file: PathBuf,

    /// RNG seed (derived from wall clock if omitted).
    #[arg(long)]
    pub seed: Option<u64>,

    /// Resume training from a checkpoint file. Restores the candle `Var`s as
    /// `ternary[i] · scale[row]` and continues the step counter from the
    /// checkpoint's saved step. `AdamW` moments and the data-sampling RNG
    /// stream are not stored in checkpoints, so those reset on resume.
    #[arg(long)]
    pub resume: Option<PathBuf>,

    /// Tokenizer file (BPE merges in compact binary format produced by
    /// `lzr bpe`). The training pipeline tokenizes the input corpus once
    /// at startup and trains the model on the resulting token stream.
    #[arg(long, default_value = "assets/tokenizer.bin")]
    pub tokenizer: PathBuf,
}

/// Entry point for `lzr train`.
#[allow(
    clippy::needless_pass_by_value,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]
pub(crate) fn run(args: TrainArgs) -> Result<()> {
    create_dir_all(&args.ckpt_dir)?;

    let device = pick_device();
    eprintln!("[lzr train] device = {device:?}");

    let tokenizer = Tokenizer::load(&args.tokenizer)
        .with_context(|| format!("loading tokenizer {}", args.tokenizer.display()))?;
    eprintln!(
        "[lzr train] tokenizer: vocab={}, merges={} ({})",
        tokenizer.vocab_size(),
        tokenizer.num_merges(),
        args.tokenizer.display()
    );
    anyhow::ensure!(
        tokenizer.vocab_size() == VOCAB,
        "tokenizer vocab {} != arch::VOCAB {VOCAB} (rebuild tokenizer or update arch)",
        tokenizer.vocab_size()
    );

    let tok_load_start = Instant::now();
    let train_bytes = std::fs::read(&args.input)
        .with_context(|| format!("reading training input {}", args.input.display()))?;
    let train_byte_len = train_bytes.len();
    let train_tokens: Vec<Token> = tokenizer.encode(&train_bytes);
    drop(train_bytes);
    #[allow(clippy::cast_precision_loss)]
    let bpt = train_byte_len as f64 / train_tokens.len().max(1) as f64;
    eprintln!(
        "[lzr train] tokenized {} bytes → {} tokens in {:.1}s ({bpt:.3} bytes/token)",
        train_byte_len,
        train_tokens.len(),
        tok_load_start.elapsed().as_secs_f64(),
    );
    anyhow::ensure!(
        train_tokens.len() > args.seq,
        "training file {} tokenized to {} tokens, less than seq+1 = {}",
        args.input.display(),
        train_tokens.len(),
        args.seq + 1,
    );

    let codec_path = args
        .codec_sample_file
        .clone()
        .unwrap_or_else(|| args.input.clone());
    let mut codec_file = File::open(&codec_path)
        .with_context(|| format!("opening codec sample {}", codec_path.display()))?;
    let codec_len = codec_file.metadata()?.len() as usize;
    anyhow::ensure!(
        codec_len >= CODEC_SAMPLE_BYTES,
        "codec sample file {} is shorter than 16 KiB",
        codec_path.display()
    );

    let resume_step = if let Some(path) = &args.resume {
        let (weights, step) = Weights::load_checkpoint_force_i2s(path)
            .with_context(|| format!("loading resume checkpoint {}", path.display()))?;
        eprintln!(
            "[lzr train] resuming from {} @ step {step} (AdamW state, RNG, and plateau \
             detector reset; weights initialised to ternary × per-row scale)",
            path.display()
        );
        Some((weights, step))
    } else {
        None
    };

    let model = match &resume_step {
        Some((weights, _)) => RwkvForTraining::from_weights(weights, &device, args.bptt_chunk)?,
        None => RwkvForTraining::new(&device, args.bptt_chunk)?,
    };
    let lr_schedule = LrSchedule::cosine();
    let mut optimizer = AdamW::new(
        model.trainable_vars(),
        ParamsAdamW {
            lr: lr_schedule.lr_max,
            weight_decay: 0.0,
            ..Default::default()
        },
    )?;

    let mut log_writer = LogWriter::open(&args.log_file)?;

    let seed = args.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
    });
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let start = Instant::now();
    let mut last_ckpt_instant = Instant::now();
    let mut last_ckpt_loss: Option<f32> = None;
    let mut stale_ckpts: usize = 0;

    // Resume continues the step counter from the checkpoint's recorded step
    // so the cosine LR schedule and the plateau detector see a contiguous
    // training history. We start at `step + 1` to avoid re-emitting a
    // checkpoint at the same step on the very next pass.
    let mut step: usize = resume_step.as_ref().map_or(0, |(_, s)| *s as usize + 1);
    let run_start_step = step;
    loop {
        if args.max_steps != 0 && step >= args.max_steps {
            break;
        }
        if args.max_steps_this_run != 0 && step - run_start_step >= args.max_steps_this_run {
            break;
        }

        let lr = lr_schedule.lr_at(step);
        optimizer.set_learning_rate(lr);

        // The forward + backward + optimizer step runs inside an ObjC
        // autorelease pool on macOS so Metal-allocated NSObjects (command
        // buffers, status snapshots, error objects, etc.) drain when the
        // closure returns instead of accumulating until process exit.
        // Without this the process leaks ~6 MB/sec on Apple Silicon — see
        // candle issue #2271. Block scope inside the closure also ensures
        // `inputs`, `targets`, `logits`, `loss` drop before the synchronize,
        // so their Arc<Buffer> strong counts are 1 by the time the buffer
        // pool's reuse logic runs.
        let loss_val = step_body(&device, || {
            let (inputs, targets) =
                sample_batch(&train_tokens, args.batch, args.seq, &mut rng, &device)?;
            let logits = model.forward(&inputs)?;
            let loss = flattened_cross_entropy(&logits, &targets)?;
            optimizer.backward_step(&loss)?;
            let loss_val = loss.to_scalar::<f32>()?;
            drop(loss);
            drop(logits);
            drop(inputs);
            drop(targets);
            device.synchronize()?;
            Ok::<_, anyhow::Error>(loss_val)
        })?;

        if step % args.log_every == 0 {
            log_writer.log_step(step, loss_val, lr, start.elapsed().as_secs_f64())?;
        }

        // Force a checkpoint write on the last step before we exit — covers
        // both the global `--max-steps` cap and the per-process
        // `--max-steps-this-run` cap that the shell-wrapper restart pattern
        // relies on. Otherwise an exit could land between scheduled
        // checkpoints, losing all progress since the last one.
        let at_max_step = (args.max_steps != 0 && step + 1 == args.max_steps)
            || (args.max_steps_this_run != 0
                && (step + 1) - run_start_step == args.max_steps_this_run);
        if last_ckpt_instant.elapsed().as_secs() >= args.checkpoint_every_secs || at_max_step {
            let ckpt_name = checkpoint_filename(step);
            let ckpt_path = args.ckpt_dir.join(&ckpt_name);
            let tensors = model.quantize_to_packed()?;
            // Write to a sibling tempfile and rename — keeps a kill mid-write
            // from leaving a truncated, unloadable checkpoint behind.
            let tmp_path = ckpt_path.with_extension("ckpt.tmp");
            {
                let mut f = File::create(&tmp_path).with_context(|| {
                    format!("creating checkpoint tempfile {}", tmp_path.display())
                })?;
                write_checkpoint_from_parts(&mut f, step as u64, &tensors)?;
                f.flush()?;
            }
            std::fs::rename(&tmp_path, &ckpt_path).with_context(|| {
                format!("renaming {} -> {}", tmp_path.display(), ckpt_path.display())
            })?;

            let codec_result =
                run_codec_test(&tensors, &tokenizer, &mut codec_file, codec_len, &mut rng)?;
            let delta = last_ckpt_loss.map(|prev| loss_val - prev);

            log_writer.log_checkpoint(step, &ckpt_path, loss_val, delta, &codec_result)?;
            print_checkpoint_console(step, &ckpt_path, loss_val, delta, &codec_result);

            if !codec_result.roundtrip_ok {
                log_writer.log_roundtrip_fail(step, &ckpt_path, &codec_result)?;
            }

            if args.stop_after_stale_ckpts > 0 {
                if let Some(prev) = last_ckpt_loss {
                    if loss_val >= prev - 1e-4 {
                        stale_ckpts += 1;
                    } else {
                        stale_ckpts = 0;
                    }
                    if stale_ckpts >= args.stop_after_stale_ckpts {
                        eprintln!(
                            "[lzr train] plateau: {stale_ckpts} stale checkpoints — stopping"
                        );
                        break;
                    }
                }
            }
            last_ckpt_loss = Some(loss_val);
            last_ckpt_instant = Instant::now();
        }

        step += 1;
    }
    Ok(())
}

fn pick_device() -> Device {
    if let Ok(d) = Device::new_metal(0) {
        return d;
    }
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    Device::Cpu
}

/// Run one training step's body inside an `ObjC` autorelease pool on macOS.
///
/// Metal's `ObjC` objects (`MTLCommandBuffer`, status / error objects,
/// transient `NSData` buffers, etc.) follow the autorelease convention —
/// they're put into the topmost autorelease pool and only released when
/// that pool drains. Cocoa GUI apps drain the pool every runloop tick;
/// command-line Rust binaries have no runloop and would never drain
/// without explicit help, so autoreleased Metal objects accumulate in RSS
/// at ~6 MB/sec under our forward/backward workload (candle issue #2271).
/// Wrapping the per-step body in `autoreleasepool` drains everything when
/// the closure returns. On non-macOS builds this is a no-op.
fn step_body<R>(_device: &Device, f: impl FnOnce() -> Result<R>) -> Result<R> {
    #[cfg(target_os = "macos")]
    {
        objc2::rc::autoreleasepool(|_| f())
    }
    #[cfg(not(target_os = "macos"))]
    {
        f()
    }
}

/// Training-time RWKV v4 model.
struct RwkvForTraining {
    tok_emb: Var,     // (VOCAB, D_MODEL)
    ln0: RMSNormVar,  // initial LayerNorm-equivalent
    ln_f: RMSNormVar, // final LayerNorm-equivalent before LM head
    layers: Vec<RwkvBlock>,
    /// Truncated-BPTT chunk size; 0 disables truncation. See `DEFAULT_BPTT_CHUNK`.
    bptt_chunk: usize,
}

impl RwkvForTraining {
    fn new(device: &Device, bptt_chunk: usize) -> Result<Self> {
        let tok_emb = gaussian_var(&[VOCAB, D_MODEL], 0.02, device)?;
        let ln0 = RMSNormVar::new(D_MODEL, device)?;
        let ln_f = RMSNormVar::new(D_MODEL, device)?;
        let layers = (0..N_LAYERS)
            .map(|_| RwkvBlock::new(device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            tok_emb,
            ln0,
            ln_f,
            layers,
            bptt_chunk,
        })
    }

    /// Build a training-ready model from a loaded checkpoint. Each ternary
    /// matrix is rehydrated as `unpack(packed)[i, j] · scale[i]`.
    fn from_weights(w: &Weights, device: &Device, bptt_chunk: usize) -> Result<Self> {
        let emb = w.tok_emb_mat();
        let mut emb_f32 = vec![0.0_f32; VOCAB * D_MODEL];
        let mut row_i8 = vec![0i8; VOCAB * D_MODEL];
        unpack_i2s_to_rowmajor(emb.packed, &mut row_i8, VOCAB, D_MODEL);
        for r in 0..VOCAB {
            let s = emb.scale[r];
            for c in 0..D_MODEL {
                emb_f32[r * D_MODEL + c] = f32::from(row_i8[r * D_MODEL + c]) * s;
            }
        }
        let tok_emb = vec_to_var(&emb_f32, &[VOCAB, D_MODEL], device)?;
        let ln0 = RMSNormVar::from_slice(w.ln0(), device)?;
        let ln_f = RMSNormVar::from_slice(w.ln_f(), device)?;
        let layers = (0..N_LAYERS)
            .map(|i| RwkvBlock::from_layer(&w.layer(i), device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            tok_emb,
            ln0,
            ln_f,
            layers,
            bptt_chunk,
        })
    }

    fn trainable_vars(&self) -> Vec<Var> {
        let mut v = vec![self.tok_emb.clone()];
        v.push(self.ln0.weight.clone());
        v.push(self.ln_f.weight.clone());
        for layer in &self.layers {
            layer.push_vars(&mut v);
        }
        v
    }

    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let b = tokens.dim(0)?;
        let t = tokens.dim(1)?;
        let flat = tokens.flatten_all()?;
        let emb = self.tok_emb.as_tensor().index_select(&flat, 0)?;
        let x = emb.reshape((b, t, D_MODEL))?;
        let mut x = self.ln0.forward(&x)?;
        for layer in &self.layers {
            x = layer.forward(&x, self.bptt_chunk)?;
        }
        let x = self.ln_f.forward(&x)?;
        let w = self.tok_emb.as_tensor();
        let x_flat = x.flatten(0, 1)?;
        let logits_flat = x_flat.matmul(&w.t()?)?;
        logits_flat.reshape((b, t, VOCAB)).map_err(Into::into)
    }

    fn quantize_to_packed(&self) -> Result<OwnedTensors> {
        let tok_emb_rows = self.tok_emb.as_tensor().to_vec2::<f32>()?;
        let tok_emb_flat: Vec<f32> = tok_emb_rows.into_iter().flatten().collect();
        let (tok_emb_packed, tok_emb_scale) =
            quantize_rows_to_ternary(&tok_emb_flat, VOCAB, D_MODEL);
        let ln0 = self.ln0.weight_as_vec()?;
        let ln_f = self.ln_f.weight_as_vec()?;
        let layers = self
            .layers
            .iter()
            .map(RwkvBlock::quantize_to_packed)
            .collect::<Result<Vec<_>>>()?;
        Ok(OwnedTensors {
            tok_emb_packed,
            tok_emb_scale,
            ln0,
            ln_f,
            layers,
        })
    }
}

struct RwkvBlock {
    tm_norm: RMSNormVar,
    time_mix_r: Var, // (D_MODEL,)
    time_mix_k: Var,
    time_mix_v: Var,
    time_decay: Var, // (D_MODEL,)
    time_first: Var,
    tm_r: BitLinear,
    tm_k: BitLinear,
    tm_v: BitLinear,
    tm_o: BitLinear,
    cm_norm: RMSNormVar,
    channel_mix_k: Var,
    channel_mix_r: Var,
    cm_k: BitLinear,
    cm_v: BitLinear,
    cm_r: BitLinear,
}

impl RwkvBlock {
    fn new(device: &Device) -> Result<Self> {
        // Time-mix / channel-mix ratios initialize in (0, 1); small Gaussian
        // noise prevents a trivial shared-parameter start.
        let mix_init = |device: &Device| -> Result<Var> {
            let t = Tensor::randn(0.5f32, 0.02f32, D_MODEL, device)?;
            Ok(Var::from_tensor(&t)?)
        };
        // Time-decay init: small negative numbers so exp(time_decay) < 1.
        let decay_init = |device: &Device| -> Result<Var> {
            let t = (Tensor::randn(0f32, 0.1f32, D_MODEL, device)? - 5.0_f64)?;
            Ok(Var::from_tensor(&t)?)
        };
        let first_init = |device: &Device| -> Result<Var> {
            let t = Tensor::randn(0f32, 0.1f32, D_MODEL, device)?;
            Ok(Var::from_tensor(&t)?)
        };
        Ok(Self {
            tm_norm: RMSNormVar::new(D_MODEL, device)?,
            time_mix_r: mix_init(device)?,
            time_mix_k: mix_init(device)?,
            time_mix_v: mix_init(device)?,
            time_decay: decay_init(device)?,
            time_first: first_init(device)?,
            tm_r: BitLinear::new(D_MODEL, D_MODEL, device)?,
            tm_k: BitLinear::new(D_MODEL, D_MODEL, device)?,
            tm_v: BitLinear::new(D_MODEL, D_MODEL, device)?,
            tm_o: BitLinear::new(D_MODEL, D_MODEL, device)?,
            cm_norm: RMSNormVar::new(D_MODEL, device)?,
            channel_mix_k: mix_init(device)?,
            channel_mix_r: mix_init(device)?,
            cm_k: BitLinear::new(D_MODEL, D_FF, device)?,
            cm_v: BitLinear::new(D_FF, D_MODEL, device)?,
            cm_r: BitLinear::new(D_MODEL, D_MODEL, device)?,
        })
    }

    fn from_layer(lv: &LayerView<'_>, device: &Device) -> Result<Self> {
        Ok(Self {
            tm_norm: RMSNormVar::from_slice(lv.tm_norm, device)?,
            time_mix_r: vec_to_var(lv.time_mix_r, &[D_MODEL], device)?,
            time_mix_k: vec_to_var(lv.time_mix_k, &[D_MODEL], device)?,
            time_mix_v: vec_to_var(lv.time_mix_v, &[D_MODEL], device)?,
            time_decay: vec_to_var(lv.time_decay, &[D_MODEL], device)?,
            time_first: vec_to_var(lv.time_first, &[D_MODEL], device)?,
            tm_r: BitLinear::from_packed(&lv.tm_r, D_MODEL, D_MODEL, device)?,
            tm_k: BitLinear::from_packed(&lv.tm_k, D_MODEL, D_MODEL, device)?,
            tm_v: BitLinear::from_packed(&lv.tm_v, D_MODEL, D_MODEL, device)?,
            tm_o: BitLinear::from_packed(&lv.tm_o, D_MODEL, D_MODEL, device)?,
            cm_norm: RMSNormVar::from_slice(lv.cm_norm, device)?,
            channel_mix_k: vec_to_var(lv.channel_mix_k, &[D_MODEL], device)?,
            channel_mix_r: vec_to_var(lv.channel_mix_r, &[D_MODEL], device)?,
            cm_k: BitLinear::from_packed(&lv.cm_k, D_MODEL, D_FF, device)?,
            cm_v: BitLinear::from_packed(&lv.cm_v, D_FF, D_MODEL, device)?,
            cm_r: BitLinear::from_packed(&lv.cm_r, D_MODEL, D_MODEL, device)?,
        })
    }

    fn push_vars(&self, out: &mut Vec<Var>) {
        out.push(self.tm_norm.weight.clone());
        out.push(self.time_mix_r.clone());
        out.push(self.time_mix_k.clone());
        out.push(self.time_mix_v.clone());
        out.push(self.time_decay.clone());
        out.push(self.time_first.clone());
        out.push(self.tm_r.weight.clone());
        out.push(self.tm_k.weight.clone());
        out.push(self.tm_v.weight.clone());
        out.push(self.tm_o.weight.clone());
        out.push(self.cm_norm.weight.clone());
        out.push(self.channel_mix_k.clone());
        out.push(self.channel_mix_r.clone());
        out.push(self.cm_k.weight.clone());
        out.push(self.cm_v.weight.clone());
        out.push(self.cm_r.weight.clone());
    }

    fn forward(&self, x: &Tensor, bptt_chunk: usize) -> Result<Tensor> {
        let residual = x.clone();
        let normed = self.tm_norm.forward(x)?;
        let tm_out = self.time_mix(&normed, bptt_chunk)?;
        let x = (residual + tm_out)?;

        let residual = x.clone();
        let normed = self.cm_norm.forward(&x)?;
        let cm_out = self.channel_mix(&normed)?;
        (residual + cm_out).map_err(Into::into)
    }

    /// RWKV v4 time-mix with sequential WKV scan.
    ///
    /// `bptt_chunk` truncates BPTT: `aa`/`bb`/`pp` are detached from the
    /// autograd graph at every multiple of `bptt_chunk` timesteps, so each
    /// `wkv_t` only backprops through at most `bptt_chunk` previous steps.
    /// Setting it to 0 disables truncation (full BPTT through the entire
    /// sequence — large memory hit).
    // Single-letter names (`b`, `t`, `r`, `k`, `v`) follow the usual tensor
    // / attention conventions; the RWKV reference uses the same spelling.
    #[allow(clippy::many_single_char_names)]
    fn time_mix(&self, x: &Tensor, bptt_chunk: usize) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let device = x.device();

        // Token shift: x_prev[:, 0] = 0; x_prev[:, i] = x[:, i-1] for i >= 1.
        let x_prev = shift_time_right(x)?;

        let mr = self.time_mix_r.as_tensor();
        let mk = self.time_mix_k.as_tensor();
        let mv = self.time_mix_v.as_tensor();
        // xr = x * mr + x_prev * (1 - mr) etc.
        let xr = lerp_tensor(x, &x_prev, mr)?;
        let xk = lerp_tensor(x, &x_prev, mk)?;
        let xv = lerp_tensor(x, &x_prev, mv)?;

        let r = ops::sigmoid(&self.tm_r.forward(&xr)?)?;
        let k = self.tm_k.forward(&xk)?;
        let v = self.tm_v.forward(&xv)?;

        // WKV running scan with the pp log-scale stability trick.
        let mut aa = Tensor::zeros((b, D_MODEL), DType::F32, device)?;
        let mut bb = Tensor::zeros((b, D_MODEL), DType::F32, device)?;
        // Very large negative number as `-inf` proxy (candle lacks a direct
        // full-inf constructor, and exp(-huge) == 0 is what the first step
        // of the scan needs).
        let mut pp = (Tensor::zeros((b, D_MODEL), DType::F32, device)? - 1.0e30_f64)?;

        let time_decay = self.time_decay.as_tensor();
        let time_first = self.time_first.as_tensor();

        let mut wkv_steps = Vec::with_capacity(t);
        for ti in 0..t {
            // Truncated-BPTT: detach state at chunk boundaries (skipping
            // ti=0 because the initial state is already a fresh leaf). After
            // detach, `wkv_t` at this and following timesteps within the
            // chunk only backprop as far as the chunk start, so the in-graph
            // tensor count is bounded by `chunk · t / chunk = t` rather than
            // `t · (t+1) / 2`.
            if bptt_chunk > 0 && ti > 0 && ti % bptt_chunk == 0 {
                aa = aa.detach();
                bb = bb.detach();
                pp = pp.detach();
            }

            let k_t = k.i((.., ti, ..))?; // (b, d)
            let v_t = v.i((.., ti, ..))?;

            let ww = k_t.broadcast_add(time_first)?;
            let qq = pp.maximum(&ww)?;
            let e1 = (&pp - &qq)?.exp()?;
            let e2 = (&ww - &qq)?.exp()?;
            let num = ((&e1 * &aa)? + (&e2 * &v_t)?)?;
            let den = ((&e1 * &bb)? + &e2)?;
            let wkv_t = (num / den)?;
            wkv_steps.push(wkv_t);

            let ww2 = pp.broadcast_add(time_decay)?;
            let qq2 = ww2.maximum(&k_t)?;
            let e1b = (&ww2 - &qq2)?.exp()?;
            let e2b = (&k_t - &qq2)?.exp()?;
            aa = ((&e1b * &aa)? + (&e2b * &v_t)?)?;
            bb = ((&e1b * &bb)? + &e2b)?;
            pp = qq2;
        }

        let wkv = Tensor::stack(&wkv_steps, 1)?; // (b, t, d)
        let rwkv = (&r * &wkv)?;
        self.tm_o.forward(&rwkv)
    }

    fn channel_mix(&self, x: &Tensor) -> Result<Tensor> {
        let x_prev = shift_time_right(x)?;
        let mk = self.channel_mix_k.as_tensor();
        let mr = self.channel_mix_r.as_tensor();
        let xk = lerp_tensor(x, &x_prev, mk)?;
        let xr = lerp_tensor(x, &x_prev, mr)?;

        let k = self.cm_k.forward(&xk)?;
        let k = k.relu()?.sqr()?; // squared-ReLU activation
        let kv = self.cm_v.forward(&k)?;
        let r = ops::sigmoid(&self.cm_r.forward(&xr)?)?;
        (&r * &kv).map_err(Into::into)
    }

    fn quantize_to_packed(&self) -> Result<OwnedLayer> {
        Ok(OwnedLayer {
            tm_norm: self.tm_norm.weight_as_vec()?,
            time_mix_r: self.time_mix_r.as_tensor().to_vec1::<f32>()?,
            time_mix_k: self.time_mix_k.as_tensor().to_vec1::<f32>()?,
            time_mix_v: self.time_mix_v.as_tensor().to_vec1::<f32>()?,
            time_decay: self.time_decay.as_tensor().to_vec1::<f32>()?,
            time_first: self.time_first.as_tensor().to_vec1::<f32>()?,
            tm_r: self.tm_r.quantize()?,
            tm_k: self.tm_k.quantize()?,
            tm_v: self.tm_v.quantize()?,
            tm_o: self.tm_o.quantize()?,
            cm_norm: self.cm_norm.weight_as_vec()?,
            channel_mix_k: self.channel_mix_k.as_tensor().to_vec1::<f32>()?,
            channel_mix_r: self.channel_mix_r.as_tensor().to_vec1::<f32>()?,
            cm_k: self.cm_k.quantize()?,
            cm_v: self.cm_v.quantize()?,
            cm_r: self.cm_r.quantize()?,
        })
    }
}

/// Shift a `(b, t, d)` tensor one step to the right along time: the output
/// at time 0 is zero, and at time `i` for `i >= 1` is the input at time
/// `i - 1`. RWKV token-shift primitive.
fn shift_time_right(x: &Tensor) -> Result<Tensor> {
    let (b, t, d) = x.dims3()?;
    if t == 0 {
        return Ok(x.clone());
    }
    let zero_row = Tensor::zeros((b, 1, d), DType::F32, x.device())?;
    let kept = x.narrow(1, 0, t - 1)?;
    Tensor::cat(&[&zero_row, &kept], 1).map_err(Into::into)
}

/// `out = cur * mix + prev * (1 - mix)`, broadcasting a 1-D `mix` across
/// `(b, t, d)` tensors.
fn lerp_tensor(cur: &Tensor, prev: &Tensor, mix: &Tensor) -> Result<Tensor> {
    let cur_m = cur.broadcast_mul(mix)?;
    // (1 - mix) via affine: y = -1 * x + 1
    let one_minus = mix.affine(-1.0, 1.0)?;
    let prev_m = prev.broadcast_mul(&one_minus)?;
    (cur_m + prev_m).map_err(Into::into)
}

struct RMSNormVar {
    weight: Var,
}

impl RMSNormVar {
    fn new(dim: usize, device: &Device) -> Result<Self> {
        let t = Tensor::ones(dim, DType::F32, device)?;
        Ok(Self {
            weight: Var::from_tensor(&t)?,
        })
    }

    fn from_slice(src: &[f32], device: &Device) -> Result<Self> {
        let t = Tensor::from_slice(src, src.len(), device)?;
        Ok(Self {
            weight: Var::from_tensor(&t)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let squared = x.sqr()?;
        let mean = squared.mean_keepdim(candle_core::D::Minus1)?;
        let scale = (mean + f64::from(RMS_EPS))?.sqrt()?.recip()?;
        let scaled = x.broadcast_mul(&scale)?;
        scaled
            .broadcast_mul(self.weight.as_tensor())
            .map_err(Into::into)
    }

    fn weight_as_vec(&self) -> Result<Vec<f32>> {
        self.weight.as_tensor().to_vec1::<f32>().map_err(Into::into)
    }
}

struct BitLinear {
    weight: Var, // (out_dim, in_dim)
    in_dim: usize,
    out_dim: usize,
}

impl BitLinear {
    #[allow(clippy::cast_precision_loss)]
    fn new(in_dim: usize, out_dim: usize, device: &Device) -> Result<Self> {
        let std = (1.0 / in_dim as f64).sqrt();
        let weight = gaussian_var(&[out_dim, in_dim], std, device)?;
        Ok(Self {
            weight,
            in_dim,
            out_dim,
        })
    }

    /// Resume-path constructor: rehydrate the unquantized `f32` weight tensor
    /// as `ternary[i, j] · scale[i]` from the I2_S-packed checkpoint bytes.
    /// The first `quantize()` call on the resumed model will reproduce a
    /// ternary lattice with a slightly rescaled per-row absmean — within the
    /// noise of one optimizer step.
    fn from_packed(
        mat: &TernaryMatrix<'_>,
        in_dim: usize,
        out_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        anyhow::ensure!(
            !mat.packed.is_empty(),
            "from_packed: TernaryMatrix.packed is empty — load the checkpoint via \
             Weights::load_checkpoint_force_i2s",
        );
        anyhow::ensure!(
            mat.scale.len() == out_dim,
            "from_packed: scale len {} != out_dim {}",
            mat.scale.len(),
            out_dim,
        );
        let mut tern = vec![0i8; out_dim * in_dim];
        unpack_i2s_to_rowmajor(mat.packed, &mut tern, out_dim, in_dim);
        let mut flat = vec![0f32; out_dim * in_dim];
        for r in 0..out_dim {
            let s = mat.scale[r];
            let row = &mut flat[r * in_dim..(r + 1) * in_dim];
            for (dst, t) in row.iter_mut().zip(&tern[r * in_dim..(r + 1) * in_dim]) {
                *dst = f32::from(*t) * s;
            }
        }
        let t = Tensor::from_vec(flat, (out_dim, in_dim), device)?;
        Ok(Self {
            weight: Var::from_tensor(&t)?,
            in_dim,
            out_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.weight.as_tensor();

        // Per-row absmean scale.
        let abs_w = w.abs()?;
        let row_mean = abs_w.mean_keepdim(candle_core::D::Minus1)?;
        let row_scale = (row_mean + 1e-5_f64)?;
        let normalized = w.broadcast_div(&row_scale)?;
        let clipped = normalized.clamp(-1f32, 1f32)?;
        let rounded = clipped.round()?;
        let w_q = rounded.broadcast_mul(&row_scale)?;
        // STE: forward uses w_q, backward flows through w.
        let w_diff = (&w_q - w)?.detach();
        let w_ste = (w + w_diff)?;

        // Per-last-axis absmax activation quant to 8-bit.
        let x_abs = x.abs()?;
        let x_max = x_abs.max_keepdim(candle_core::D::Minus1)?;
        let x_scale = (x_max / 127.0_f64)?;
        let x_scale = (x_scale + 1e-8_f64)?;
        let xn = x.broadcast_div(&x_scale)?;
        let xc = xn.clamp(-127f32, 127f32)?;
        let xr = xc.round()?;
        let x_q = xr.broadcast_mul(&x_scale)?;
        let x_diff = (&x_q - x)?.detach();
        let x_ste = (x + x_diff)?;

        let rank = x_ste.rank();
        let last = x_ste.dim(rank - 1)?;
        let x_2d = x_ste.reshape(((), last))?;
        let out_2d = x_2d.matmul(&w_ste.t()?)?;
        let mut out_shape = x_ste.dims().to_vec();
        *out_shape.last_mut().unwrap() = self.out_dim;
        out_2d.reshape(out_shape).map_err(Into::into)
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn quantize(&self) -> Result<OwnedTernary> {
        let rows = self.weight.as_tensor().to_vec2::<f32>()?;
        debug_assert_eq!(rows.len(), self.out_dim);
        debug_assert_eq!(rows[0].len(), self.in_dim);
        let mut packed = vec![0u8; (self.out_dim * self.in_dim) / 4];
        let mut scale = Vec::with_capacity(self.out_dim);
        let mut row_buf = vec![0i8; self.in_dim];
        for (row_idx, row) in rows.iter().enumerate() {
            let absmean = row.iter().map(|v| v.abs()).sum::<f32>() / self.in_dim as f32;
            let s = absmean.max(1e-5);
            scale.push(s);
            for (dst, src) in row_buf.iter_mut().zip(row.iter()) {
                let q = (src / s).clamp(-1.0, 1.0).round();
                *dst = q as i8;
            }
            let byte_off = row_idx * self.in_dim / 4;
            pack_ternary(&row_buf, &mut packed[byte_off..byte_off + self.in_dim / 4]);
        }
        Ok(OwnedTernary { packed, scale })
    }
}

#[allow(clippy::cast_possible_truncation)]
fn gaussian_var(shape: &[usize], std: f64, device: &Device) -> Result<Var> {
    let t = Tensor::randn(0f32, std as f32, shape, device)?;
    Ok(Var::from_tensor(&t)?)
}

fn vec_to_var(src: &[f32], shape: &[usize], device: &Device) -> Result<Var> {
    let expected: usize = shape.iter().product();
    anyhow::ensure!(
        src.len() == expected,
        "vec_to_var: slice len {} != product of shape {expected}",
        src.len(),
    );
    let t = Tensor::from_slice(src, shape, device)?;
    Ok(Var::from_tensor(&t)?)
}

fn flattened_cross_entropy(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let (b, t, v) = logits.dims3()?;
    let lf = logits.reshape((b * t, v))?;
    let tf = targets.reshape(b * t)?;
    cross_entropy(&lf, &tf).map_err(Into::into)
}

fn sample_batch(
    tokens: &[Token],
    batch: usize,
    seq: usize,
    rng: &mut rand::rngs::StdRng,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let max_off = tokens.len() - (seq + 1);
    let mut inputs = vec![0i64; batch * seq];
    let mut targets = vec![0i64; batch * seq];
    for bi in 0..batch {
        let off = rng.random_range(0..=max_off);
        for i in 0..seq {
            inputs[bi * seq + i] = i64::from(tokens[off + i]);
            targets[bi * seq + i] = i64::from(tokens[off + i + 1]);
        }
    }
    let inp = Tensor::from_vec(inputs, (batch, seq), device)?;
    let tgt = Tensor::from_vec(targets, (batch, seq), device)?;
    Ok((inp, tgt))
}

struct LrSchedule {
    lr_max: f64,
    lr_min: f64,
    warmup: usize,
    decay_horizon: usize,
}

impl LrSchedule {
    const fn cosine() -> Self {
        Self {
            lr_max: 3e-4,
            lr_min: 1e-5,
            warmup: 100,
            decay_horizon: 1_000_000,
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn lr_at(&self, step: usize) -> f64 {
        if step < self.warmup {
            return self.lr_max * (step as f64 + 1.0) / self.warmup as f64;
        }
        let denom = (self.decay_horizon - self.warmup).max(1) as f64;
        let progress = ((step - self.warmup) as f64 / denom).clamp(0.0, 1.0);
        let cos = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        (self.lr_max - self.lr_min).mul_add(cos, self.lr_min)
    }
}

struct OwnedTensors {
    tok_emb_packed: Vec<u8>,
    tok_emb_scale: Vec<f32>,
    ln0: Vec<f32>,
    ln_f: Vec<f32>,
    layers: Vec<OwnedLayer>,
}

struct OwnedLayer {
    tm_norm: Vec<f32>,
    time_mix_r: Vec<f32>,
    time_mix_k: Vec<f32>,
    time_mix_v: Vec<f32>,
    time_decay: Vec<f32>,
    time_first: Vec<f32>,
    tm_r: OwnedTernary,
    tm_k: OwnedTernary,
    tm_v: OwnedTernary,
    tm_o: OwnedTernary,
    cm_norm: Vec<f32>,
    channel_mix_k: Vec<f32>,
    channel_mix_r: Vec<f32>,
    cm_k: OwnedTernary,
    cm_v: OwnedTernary,
    cm_r: OwnedTernary,
}

struct OwnedTernary {
    packed: Vec<u8>,
    scale: Vec<f32>,
}

fn layers_as_refs(t: &OwnedTensors) -> [LayerTensors<'_>; N_LAYERS] {
    core::array::from_fn(|i| LayerTensors {
        tm_norm: &t.layers[i].tm_norm,
        time_mix_r: &t.layers[i].time_mix_r,
        time_mix_k: &t.layers[i].time_mix_k,
        time_mix_v: &t.layers[i].time_mix_v,
        time_decay: &t.layers[i].time_decay,
        time_first: &t.layers[i].time_first,
        tm_r_packed: &t.layers[i].tm_r.packed,
        tm_r_scale: &t.layers[i].tm_r.scale,
        tm_k_packed: &t.layers[i].tm_k.packed,
        tm_k_scale: &t.layers[i].tm_k.scale,
        tm_v_packed: &t.layers[i].tm_v.packed,
        tm_v_scale: &t.layers[i].tm_v.scale,
        tm_o_packed: &t.layers[i].tm_o.packed,
        tm_o_scale: &t.layers[i].tm_o.scale,
        cm_norm: &t.layers[i].cm_norm,
        channel_mix_k: &t.layers[i].channel_mix_k,
        channel_mix_r: &t.layers[i].channel_mix_r,
        cm_k_packed: &t.layers[i].cm_k.packed,
        cm_k_scale: &t.layers[i].cm_k.scale,
        cm_v_packed: &t.layers[i].cm_v.packed,
        cm_v_scale: &t.layers[i].cm_v.scale,
        cm_r_packed: &t.layers[i].cm_r.packed,
        cm_r_scale: &t.layers[i].cm_r.scale,
    })
}

fn write_checkpoint_from_parts(out: &mut File, step: u64, t: &OwnedTensors) -> Result<()> {
    let layers = layers_as_refs(t);
    write_checkpoint(
        out,
        step,
        &t.tok_emb_packed,
        &t.tok_emb_scale,
        &t.ln0,
        &t.ln_f,
        &layers,
    )
    .map_err(Into::into)
}

fn checkpoint_filename(step: usize) -> String {
    let ts = Local::now().format("%Y%m%d-%H%M%S");
    format!("rwkv-d{D_MODEL}-l{N_LAYERS}-step{step}-{ts}.ckpt")
}

struct CodecResult {
    offset: usize,
    encoded_len: usize,
    bpb: f64,
    encode_ms: f64,
    decode_ms: f64,
    encode_hours_1gb: f64,
    decode_hours_1gb: f64,
    roundtrip_ok: bool,
    first_mismatch: Option<usize>,
}

#[allow(clippy::cast_precision_loss)]
fn run_codec_test(
    tensors: &OwnedTensors,
    tokenizer: &Tokenizer,
    codec_file: &mut File,
    codec_len: usize,
    rng: &mut rand::rngs::StdRng,
) -> Result<CodecResult> {
    let aligned_max = (codec_len - CODEC_SAMPLE_BYTES) / CODEC_SAMPLE_BYTES;
    let offset = rng.random_range(0..=aligned_max) * CODEC_SAMPLE_BYTES;
    codec_file.seek(SeekFrom::Start(offset as u64))?;
    let mut slice = vec![0u8; CODEC_SAMPLE_BYTES];
    codec_file.read_exact(&mut slice)?;

    let mut blob = Vec::with_capacity(crate::arch::PACKED_WEIGHTS_LEN);
    let layers = layers_as_refs(tensors);
    write_weights(
        &tensors.tok_emb_packed,
        &tensors.tok_emb_scale,
        &tensors.ln0,
        &tensors.ln_f,
        &layers,
        &mut blob,
    );
    let weights = Weights::from_bytes(&blob)?;

    let mut model = ByteTransformer::new(weights);

    let enc_start = Instant::now();
    let mut archive = Vec::new();
    {
        let mut probs = TransformerProbs::new(&mut model);
        encode_bytes(&slice, tokenizer, &mut probs, &mut archive)?;
    }
    let encode_ms = enc_start.elapsed().as_secs_f64() * 1000.0;

    let dec_start = Instant::now();
    let decoded = {
        let mut cur = &archive[..];
        let mut probs = TransformerProbs::new(&mut model);
        decode_bytes(&mut cur, tokenizer, &mut probs)?
    };
    let decode_ms = dec_start.elapsed().as_secs_f64() * 1000.0;

    let roundtrip_ok = decoded == slice;
    let first_mismatch = if roundtrip_ok {
        None
    } else {
        decoded.iter().zip(slice.iter()).position(|(a, b)| a != b)
    };

    let payload_len = archive.len().saturating_sub(crate::codec::HEADER_LEN);
    let bpb = (payload_len as f64 * 8.0) / CODEC_SAMPLE_BYTES as f64;

    let encode_bps = CODEC_SAMPLE_BYTES as f64 / (encode_ms / 1000.0);
    let decode_bps = CODEC_SAMPLE_BYTES as f64 / (decode_ms / 1000.0);
    let encode_hours_1gb = 1e9 / encode_bps / 3600.0;
    let decode_hours_1gb = 1e9 / decode_bps / 3600.0;

    Ok(CodecResult {
        offset,
        encoded_len: archive.len(),
        bpb,
        encode_ms,
        decode_ms,
        encode_hours_1gb,
        decode_hours_1gb,
        roundtrip_ok,
        first_mismatch,
    })
}

fn print_checkpoint_console(
    step: usize,
    ckpt_path: &Path,
    loss: f32,
    delta: Option<f32>,
    r: &CodecResult,
) {
    let ts = Local::now().format("%Y-%m-%d %H:%M:%S");
    println!("[{ts}] checkpoint @ step {step}");
    println!("  saved:        {}", ckpt_path.display());

    let delta_str = delta.map_or_else(
        || "(first checkpoint, no baseline)".to_string(),
        |d| {
            if d.abs() < 1e-4 {
                format!("(Δ vs last ckpt: {d:+.4} — flat —)")
            } else if d < 0.0 {
                format!("(Δ vs last ckpt: {d:+.4} — improving ✓)")
            } else {
                format!("(Δ vs last ckpt: {d:+.4} — worsening ✗)")
            }
        },
    );
    println!("  model:        train loss {loss:.4} nats  {delta_str}");

    let off_start = r.offset;
    let off_end = off_start + CODEC_SAMPLE_BYTES;
    let rt = if r.roundtrip_ok {
        "roundtrip OK"
    } else {
        "ROUNDTRIP FAIL"
    };
    println!(
        "  codec test:   enwik9 offsets [{off_start:#010x} .. {off_end:#010x})  ({CODEC_SAMPLE_BYTES} bytes)"
    );
    println!(
        "                bpb {:.2}   encoded {} B   {rt}",
        r.bpb, r.encoded_len
    );
    println!(
        "                encode: {:.2} s  →  {} for the full 1 GB",
        r.encode_ms / 1000.0,
        fmt_hours(r.encode_hours_1gb)
    );
    println!(
        "                decode: {:.2} s  →  {} for the full 1 GB",
        r.decode_ms / 1000.0,
        fmt_hours(r.decode_hours_1gb)
    );
}

#[allow(clippy::cast_possible_truncation)]
fn fmt_hours(h: f64) -> String {
    if h < 100.0 {
        format!("~{h:.1} h")
    } else {
        format!("~{} h", h.round() as i64)
    }
}

struct LogWriter {
    f: File,
}

impl LogWriter {
    fn open(path: &Path) -> Result<Self> {
        let f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("opening log {}", path.display()))?;
        Ok(Self { f })
    }

    fn write_line(&mut self, entry: &serde_json::Value) -> Result<()> {
        writeln!(self.f, "{entry}")?;
        Ok(())
    }

    fn log_step(&mut self, step: usize, loss: f32, lr: f64, wall_s: f64) -> Result<()> {
        let entry = serde_json::json!({
            "t": Local::now().to_rfc3339(),
            "event": "step",
            "step": step,
            "loss": loss,
            "lr": lr,
            "wall_s": wall_s,
        });
        self.write_line(&entry)
    }

    fn log_checkpoint(
        &mut self,
        step: usize,
        ckpt_path: &Path,
        loss: f32,
        delta: Option<f32>,
        r: &CodecResult,
    ) -> Result<()> {
        let entry = serde_json::json!({
            "t": Local::now().to_rfc3339(),
            "event": "checkpoint",
            "step": step,
            "ckpt": ckpt_path.display().to_string(),
            "train_loss_nats": loss,
            "train_loss_delta": delta,
            "codec": {
                "offset": r.offset,
                "len": CODEC_SAMPLE_BYTES,
                "encoded": r.encoded_len,
                "bpb": r.bpb,
                "encode_ms": r.encode_ms,
                "decode_ms": r.decode_ms,
                "encode_hours_1gb": r.encode_hours_1gb,
                "decode_hours_1gb": r.decode_hours_1gb,
                "roundtrip_ok": r.roundtrip_ok,
            },
        });
        self.write_line(&entry)
    }

    fn log_roundtrip_fail(&mut self, step: usize, ckpt_path: &Path, r: &CodecResult) -> Result<()> {
        let entry = serde_json::json!({
            "t": Local::now().to_rfc3339(),
            "event": "roundtrip_fail",
            "step": step,
            "ckpt": ckpt_path.display().to_string(),
            "offset": r.offset,
            "len": CODEC_SAMPLE_BYTES,
            "first_mismatch": r.first_mismatch,
        });
        self.write_line(&entry)
    }
}
