//! Training mode — candle-nn forward pass matching [`crate::arch`], STE
//! ternary fake-quant on every linear, `AdamW` + cosine schedule, and every
//! `checkpoint_every_secs` of wall-clock time writes a timestamped checkpoint
//! then runs an encode/decode roundtrip test on a random 16 KiB slice of the
//! codec sample file.

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

use crate::arch::{CONTEXT_LEN, D_FF, D_MODEL, HEAD_DIM, N_HEADS, N_LAYERS, RMS_EPS, VOCAB};
use crate::bitnet::pack_ternary;
use crate::codec::{TransformerProbs, decode_bytes, encode_bytes};
use crate::model::ByteTransformer;
use crate::weights::{LayerTensors, Weights, write_checkpoint, write_weights};

const CODEC_SAMPLE_BYTES: usize = 16 * 1024;

/// CLI arguments for the training subcommand.
#[derive(Args, Debug, Clone)]
pub(crate) struct TrainArgs {
    /// Training data file (e.g. `assets/enwik9`).
    pub input: PathBuf,

    /// Hard cap on optimizer steps (0 = unbounded; stop via Ctrl-C or the
    /// stale-checkpoint plateau detector).
    #[arg(long, default_value_t = 0)]
    pub max_steps: usize,

    /// Stop training after this many *consecutive* checkpoints show no
    /// train-loss improvement vs. the previous one. Set to 0 to disable
    /// plateau stopping.
    #[arg(long, default_value_t = 3)]
    pub stop_after_stale_ckpts: usize,

    /// Batch size.
    #[arg(long, default_value_t = 32)]
    pub batch: usize,

    /// Sequence length. Must equal `CONTEXT_LEN`.
    #[arg(long, default_value_t = CONTEXT_LEN)]
    pub seq: usize,

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
}

/// Entry point for `lzr train`.
///
/// - `args` is taken by value because `clap` hands us a fresh instance and
///   we then need to partially consume fields (`.ckpt_dir`, `.log_file`).
/// - File-length `u64 -> usize` casts: enwik9 is 1 GB; we cannot meaningfully
///   run on a 32-bit target, so truncation is not a real hazard.
/// - `as_nanos() as u64`: we only need 64 bits of jitter for the RNG seed.
// The training loop's linearity (setup → loop → checkpoint → plateau check)
// reads top-to-bottom better than several one-call functions would.
#[allow(
    clippy::needless_pass_by_value,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]
pub(crate) fn run(args: TrainArgs) -> Result<()> {
    anyhow::ensure!(
        args.seq == CONTEXT_LEN,
        "seq ({}) must equal CONTEXT_LEN ({CONTEXT_LEN}); RoPE positions diverge otherwise",
        args.seq
    );
    create_dir_all(&args.ckpt_dir)?;

    let device = pick_device();
    eprintln!("[lzr train] device = {device:?}");

    let mut train_file = File::open(&args.input)
        .with_context(|| format!("opening training input {}", args.input.display()))?;
    let train_len = train_file.metadata()?.len() as usize;
    anyhow::ensure!(
        train_len > args.seq,
        "training file {} is shorter than seq+1",
        args.input.display()
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

    let model = TransformerForTraining::new(&device)?;
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

    // Training is unbounded by default. `max_steps == 0` means "no cap";
    // otherwise treat it as a hard upper bound. Exit on plateau or Ctrl-C.
    let mut step: usize = 0;
    loop {
        if args.max_steps != 0 && step >= args.max_steps {
            break;
        }

        let lr = lr_schedule.lr_at(step);
        optimizer.set_learning_rate(lr);

        let (inputs, targets) = sample_batch(
            &mut train_file,
            train_len,
            args.batch,
            args.seq,
            &mut rng,
            &device,
        )?;
        let logits = model.forward(&inputs)?;
        let loss = flattened_cross_entropy(&logits, &targets)?;
        optimizer.backward_step(&loss)?;

        let loss_val = loss.to_scalar::<f32>()?;

        if step % args.log_every == 0 {
            log_writer.log_step(step, loss_val, lr, start.elapsed().as_secs_f64())?;
        }

        let at_max_step = args.max_steps != 0 && step + 1 == args.max_steps;
        if last_ckpt_instant.elapsed().as_secs() >= args.checkpoint_every_secs || at_max_step {
            let ckpt_name = checkpoint_filename(step);
            let ckpt_path = args.ckpt_dir.join(&ckpt_name);
            let tensors = model.quantize_to_packed()?;
            let mut f = File::create(&ckpt_path)
                .with_context(|| format!("creating checkpoint {}", ckpt_path.display()))?;
            write_checkpoint_from_parts(&mut f, step as u64, &tensors)?;
            f.flush()?;
            drop(f);

            let codec_result = run_codec_test(&tensors, &mut codec_file, codec_len, &mut rng)?;
            let delta = last_ckpt_loss.map(|prev| loss_val - prev);

            log_writer.log_checkpoint(step, &ckpt_path, loss_val, delta, &codec_result)?;
            print_checkpoint_console(step, &ckpt_path, loss_val, delta, &codec_result);

            if !codec_result.roundtrip_ok {
                log_writer.log_roundtrip_fail(step, &ckpt_path, &codec_result)?;
                eprintln!(
                    "\n[lzr train] FATAL: roundtrip failed — terminating run. See {}.",
                    args.log_file.display()
                );
                std::process::exit(2);
            }

            // Plateau detection: a checkpoint is "stale" if it did not
            // improve vs. the previous one. The first checkpoint never
            // counts as stale (no baseline).
            match delta {
                Some(d) if d >= 0.0 => stale_ckpts += 1,
                Some(_) => stale_ckpts = 0,
                None => {}
            }
            if args.stop_after_stale_ckpts != 0 && stale_ckpts >= args.stop_after_stale_ckpts {
                println!(
                    "\n[lzr train] plateau reached: {stale_ckpts} consecutive checkpoint(s) \
                     without improvement — stopping."
                );
                break;
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

/// The training-time transformer. Stores every trainable parameter as an
/// explicit [`Var`] so we can (a) hand the full list to [`AdamW`], (b) read
/// them back at quantize time, and (c) avoid the [`candle_nn::VarMap`]
/// name-lookup indirection.
struct TransformerForTraining {
    tok_emb: Var, // (VOCAB, D_MODEL)
    layers: Vec<Block>,
    rope_cos: Tensor,
    rope_sin: Tensor,
}

impl TransformerForTraining {
    fn new(device: &Device) -> Result<Self> {
        let tok_emb = gaussian_var(&[VOCAB, D_MODEL], 0.02, device)?;
        let layers = (0..N_LAYERS)
            .map(|_| Block::new(device))
            .collect::<Result<Vec<_>>>()?;
        let (rope_cos, rope_sin) = rope_tables(device)?;
        Ok(Self {
            tok_emb,
            layers,
            rope_cos,
            rope_sin,
        })
    }

    fn trainable_vars(&self) -> Vec<Var> {
        let mut v = vec![self.tok_emb.clone()];
        for layer in &self.layers {
            layer.push_vars(&mut v);
        }
        v
    }

    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let b = tokens.dim(0)?;
        let t = tokens.dim(1)?;
        anyhow::ensure!(t <= CONTEXT_LEN, "sequence length {t} exceeds CONTEXT_LEN");
        let flat = tokens.flatten_all()?;
        let emb = self.tok_emb.as_tensor().index_select(&flat, 0)?;
        let mut x = emb.reshape((b, t, D_MODEL))?;
        let mask = causal_mask(t, tokens.device())?;
        for layer in &self.layers {
            x = layer.forward(&x, &self.rope_cos, &self.rope_sin, &mask)?;
        }
        let w = self.tok_emb.as_tensor();
        let x_flat = x.flatten(0, 1)?;
        let logits_flat = x_flat.matmul(&w.t()?)?;
        logits_flat.reshape((b, t, VOCAB)).map_err(Into::into)
    }

    fn quantize_to_packed(&self) -> Result<OwnedTensors> {
        let tok_emb = self.tok_emb.as_tensor().to_vec2::<f32>()?;
        let tok_emb: Vec<f32> = tok_emb.into_iter().flatten().collect();
        let layers = self
            .layers
            .iter()
            .map(Block::quantize_to_packed)
            .collect::<Result<Vec<_>>>()?;
        Ok(OwnedTensors { tok_emb, layers })
    }
}

struct Block {
    attn_norm: RMSNorm,
    mlp_norm: RMSNorm,
    q: BitLinear,
    k: BitLinear,
    v: BitLinear,
    o: BitLinear,
    w1: BitLinear,
    w2: BitLinear,
    w3: BitLinear,
}

impl Block {
    fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            attn_norm: RMSNorm::new(D_MODEL, device)?,
            mlp_norm: RMSNorm::new(D_MODEL, device)?,
            q: BitLinear::new(D_MODEL, D_MODEL, device)?,
            k: BitLinear::new(D_MODEL, D_MODEL, device)?,
            v: BitLinear::new(D_MODEL, D_MODEL, device)?,
            o: BitLinear::new(D_MODEL, D_MODEL, device)?,
            w1: BitLinear::new(D_MODEL, D_FF, device)?,
            w2: BitLinear::new(D_FF, D_MODEL, device)?,
            w3: BitLinear::new(D_MODEL, D_FF, device)?,
        })
    }

    fn push_vars(&self, out: &mut Vec<Var>) {
        out.push(self.attn_norm.weight.clone());
        out.push(self.mlp_norm.weight.clone());
        out.push(self.q.weight.clone());
        out.push(self.k.weight.clone());
        out.push(self.v.weight.clone());
        out.push(self.o.weight.clone());
        out.push(self.w1.weight.clone());
        out.push(self.w2.weight.clone());
        out.push(self.w3.weight.clone());
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let normed = self.attn_norm.forward(x)?;
        let q = self.q.forward(&normed)?;
        let k = self.k.forward(&normed)?;
        let v = self.v.forward(&normed)?;

        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;
        let attn = scaled_dot_product(&q, &k, &v, mask)?;
        let attn_out = self.o.forward(&attn)?;
        let x = (residual + attn_out)?;

        let residual = x.clone();
        let normed = self.mlp_norm.forward(&x)?;
        let gate = self.w1.forward(&normed)?;
        let up = self.w3.forward(&normed)?;
        let hidden = ops::silu(&gate)?.mul(&up)?;
        let ff_out = self.w2.forward(&hidden)?;
        (residual + ff_out).map_err(Into::into)
    }

    fn quantize_to_packed(&self) -> Result<OwnedLayer> {
        Ok(OwnedLayer {
            attn_norm: self.attn_norm.weight_as_vec()?,
            mlp_norm: self.mlp_norm.weight_as_vec()?,
            q: self.q.quantize()?,
            k: self.k.quantize()?,
            v: self.v.quantize()?,
            o: self.o.quantize()?,
            w1: self.w1.quantize()?,
            w2: self.w2.quantize()?,
            w3: self.w3.quantize()?,
        })
    }
}

struct RMSNorm {
    weight: Var,
}

impl RMSNorm {
    fn new(dim: usize, device: &Device) -> Result<Self> {
        let t = Tensor::ones(dim, DType::F32, device)?;
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
    // `in_dim` comes from `D_MODEL` / `D_FF` — small constants well within f64.
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

        // Candle's `matmul` requires both operands to have the same rank.
        // Flatten the leading batch+time dims into one, matmul in 2D, then
        // reshape back.
        let rank = x_ste.rank();
        let last = x_ste.dim(rank - 1)?;
        let x_2d = x_ste.reshape(((), last))?;
        let out_2d = x_2d.matmul(&w_ste.t()?)?;
        let mut out_shape = x_ste.dims().to_vec();
        *out_shape.last_mut().unwrap() = self.out_dim;
        out_2d.reshape(out_shape).map_err(Into::into)
    }

    // `self.in_dim` is a compile-time-configured dimension (at most `D_FF`)
    // so `usize -> f32` is exact. The per-row quantized value is clamped to
    // `[-1, 1]` before `as i8`, so truncation is impossible.
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

// `std` is the std-dev for Gaussian init, always a small literal; the f64→f32
// cast is the obvious conversion.
#[allow(clippy::cast_possible_truncation)]
fn gaussian_var(shape: &[usize], std: f64, device: &Device) -> Result<Var> {
    let t = Tensor::randn(0f32, std as f32, shape, device)?;
    Ok(Var::from_tensor(&t)?)
}

/// Candle-tensor wrapper around [`crate::arch::rope_cos_sin_tables`].
fn rope_tables(device: &Device) -> Result<(Tensor, Tensor)> {
    let half = HEAD_DIM / 2;
    let (cos_flat, sin_flat) = crate::arch::rope_cos_sin_tables();
    let cos = Tensor::from_vec(cos_flat, (CONTEXT_LEN, half), device)?;
    let sin = Tensor::from_vec(sin_flat, (CONTEXT_LEN, half), device)?;
    Ok((cos, sin))
}

// Single-letter names (`a`, `b`, `c`, `t`) follow the usual tensor-reshape
// conventions; `a`/`c` are the first/second half of the head dimension,
// `b` and `t` are batch and time.
#[allow(clippy::many_single_char_names)]
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, t, _d) = x.dims3()?;
    let reshaped = x.reshape((b, t, N_HEADS, HEAD_DIM))?;
    let half = HEAD_DIM / 2;
    let a = reshaped.i((.., .., .., ..half))?;
    let c = reshaped.i((.., .., .., half..))?;
    let cos_t = cos.i((..t, ..))?.reshape((1, t, 1, half))?;
    let sin_t = sin.i((..t, ..))?.reshape((1, t, 1, half))?;
    let new_a = (a.broadcast_mul(&cos_t)? - c.broadcast_mul(&sin_t)?)?;
    let new_c = (a.broadcast_mul(&sin_t)? + c.broadcast_mul(&cos_t)?)?;
    Tensor::cat(&[&new_a, &new_c], 3)?
        .reshape((b, t, N_HEADS * HEAD_DIM))
        .map_err(Into::into)
}

// `HEAD_DIM` is a compile-time constant; `usize -> f64` is exact. `q/k/v/b/t`
// follow tensor-indexing conventions.
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
fn scaled_dot_product(q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (b, t, _d) = q.dims3()?;
    let shape = (b, t, N_HEADS, HEAD_DIM);
    // Candle's matmul requires contiguous operands — `.transpose` alone
    // leaves a non-contiguous view, so force a copy after each permutation.
    let q = q.reshape(shape)?.transpose(1, 2)?.contiguous()?;
    let k = k.reshape(shape)?.transpose(1, 2)?.contiguous()?;
    let v = v.reshape(shape)?.transpose(1, 2)?.contiguous()?;
    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    let kt = k.transpose(2, 3)?.contiguous()?;
    let scores = q.matmul(&kt)?;
    let scores = (scores * scale)?;
    let scores = scores.broadcast_add(mask)?;
    let weights = ops::softmax_last_dim(&scores)?;
    let attended = weights.matmul(&v)?;
    attended
        .transpose(1, 2)?
        .contiguous()?
        .reshape((b, t, N_HEADS * HEAD_DIM))
        .map_err(Into::into)
}

fn causal_mask(t: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0.0_f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            data[i * t + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (1, 1, t, t), device).map_err(Into::into)
}

fn flattened_cross_entropy(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let (b, t, v) = logits.dims3()?;
    let lf = logits.reshape((b * t, v))?;
    let tf = targets.reshape(b * t)?;
    cross_entropy(&lf, &tf).map_err(Into::into)
}

fn sample_batch(
    file: &mut File,
    file_len: usize,
    batch: usize,
    seq: usize,
    rng: &mut rand::rngs::StdRng,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let max_off = file_len - (seq + 1);
    let mut window = vec![0u8; seq + 1];
    let mut inputs = vec![0i64; batch * seq];
    let mut targets = vec![0i64; batch * seq];
    for bi in 0..batch {
        let off = rng.random_range(0..=max_off) as u64;
        file.seek(SeekFrom::Start(off))?;
        file.read_exact(&mut window)?;
        for i in 0..seq {
            inputs[bi * seq + i] = i64::from(window[i]);
            targets[bi * seq + i] = i64::from(window[i + 1]);
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
    /// Step count used as the cosine-decay horizon. Training is unbounded
    /// by default, so we decay over a fixed reference horizon and then hold
    /// at `lr_min` — a conventional "warmup → cosine → hold" pattern.
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

    // `step` values stay within the decay horizon (1M) or saturate past it;
    // `usize -> f64` is exact in that regime.
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
    tok_emb: Vec<f32>,
    layers: Vec<OwnedLayer>,
}

struct OwnedLayer {
    attn_norm: Vec<f32>,
    mlp_norm: Vec<f32>,
    q: OwnedTernary,
    k: OwnedTernary,
    v: OwnedTernary,
    o: OwnedTernary,
    w1: OwnedTernary,
    w2: OwnedTernary,
    w3: OwnedTernary,
}

struct OwnedTernary {
    packed: Vec<u8>,
    scale: Vec<f32>,
}

fn layers_as_refs(t: &OwnedTensors) -> [LayerTensors<'_>; N_LAYERS] {
    core::array::from_fn(|i| LayerTensors {
        attn_norm: &t.layers[i].attn_norm,
        q_packed: &t.layers[i].q.packed,
        q_scale: &t.layers[i].q.scale,
        k_packed: &t.layers[i].k.packed,
        k_scale: &t.layers[i].k.scale,
        v_packed: &t.layers[i].v.packed,
        v_scale: &t.layers[i].v.scale,
        o_packed: &t.layers[i].o.packed,
        o_scale: &t.layers[i].o.scale,
        mlp_norm: &t.layers[i].mlp_norm,
        w1_packed: &t.layers[i].w1.packed,
        w1_scale: &t.layers[i].w1.scale,
        w2_packed: &t.layers[i].w2.packed,
        w2_scale: &t.layers[i].w2.scale,
        w3_packed: &t.layers[i].w3.packed,
        w3_scale: &t.layers[i].w3.scale,
    })
}

fn write_checkpoint_from_parts(out: &mut File, step: u64, t: &OwnedTensors) -> Result<()> {
    let layers = layers_as_refs(t);
    write_checkpoint(out, step, &t.tok_emb, &layers).map_err(Into::into)
}

fn checkpoint_filename(step: usize) -> String {
    let ts = Local::now().format("%Y%m%d-%H%M%S");
    format!("tiny-d{D_MODEL}-l{N_LAYERS}-h{N_HEADS}-step{step}-{ts}.ckpt")
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

// `usize -> f64` on `CODEC_SAMPLE_BYTES` (16 KiB) and `archive.len()` (at
// most 16 KiB compressed) cannot lose precision.
#[allow(clippy::cast_precision_loss)]
fn run_codec_test(
    tensors: &OwnedTensors,
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
    write_weights(&tensors.tok_emb, &layers, &mut blob);
    let weights = Weights::from_bytes(&blob)?;

    let mut model = ByteTransformer::new(weights);

    let enc_start = Instant::now();
    let mut archive = Vec::new();
    {
        let mut probs = TransformerProbs::new(&mut model);
        encode_bytes(&slice, &mut probs, &mut archive)?;
    }
    let encode_ms = enc_start.elapsed().as_secs_f64() * 1000.0;

    let dec_start = Instant::now();
    let decoded = {
        let mut cur = &archive[..];
        let mut probs = TransformerProbs::new(&mut model);
        decode_bytes(&mut cur, &mut probs)?
    };
    let decode_ms = dec_start.elapsed().as_secs_f64() * 1000.0;

    let roundtrip_ok = decoded == slice;
    let first_mismatch = if roundtrip_ok {
        None
    } else {
        decoded.iter().zip(slice.iter()).position(|(a, b)| a != b)
    };

    // bpb = encoded bits / original bytes. Subtract the 12-byte archive
    // header so what's reported is the model's own compression, not the
    // framing overhead.
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

// Hours-for-1 GB values top out around 1e5 even for a very slow codec, which
// fits in i64 with huge margin; the `f64 -> i64` cast is only reached for
// `h >= 100`, well-defined.
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
