//! Bitwise logistic context-mixing byte codec — the v7 logit-domain mixer.
//!
//! Where [`crate::cmix`] blends per-order count distributions *linearly*, this
//! codec mixes in the **logit (stretch) domain**, the PAQ/lpaq design. Each
//! byte is coded as eight binary decisions, MSB-first. For each bit, every
//! order model emits a probability that the bit is 1; those are mapped through
//! `stretch(p) = ln(p / (1-p))`, combined by a learned weighted sum, and mapped
//! back with `squash`. The mixer weights are trained online by gradient descent
//! on coding loss (`w += lr * (y - p) * stretch(p_k)`).
//!
//! Why logit-domain: a linear blend of agreeing predictors cannot exceed the
//! most confident input — if two models both say 0.9, the blend stays ~0.9.
//! In the logit domain their evidence *adds*, so agreement sharpens the
//! prediction past either input. That sharpening is where the bits are.
//!
//! Per-bit context: within a byte we track a tree node — start at 1, and after
//! each coded bit `node = (node << 1) | bit`, so `node` walks `1..=255` over
//! the eight decisions. Each order model is keyed on `(context_k, node)`, so a
//! model predicts a bit given both its byte-history context and the bits
//! already seen in the current byte.
//!
//! Determinism / `L(D) = 0`: identical to [`crate::cmix`]. Every lookup is
//! keyed on the full packed `(context, node)` so the `HashMap` seed is
//! irrelevant; all float work runs the same code path on both sides; encoder
//! and decoder replay the same stream and rebuild byte-identical state. No
//! weights are shipped.
//!
//! Archive layout: a 64-bit length prefix, then the AC bitstream — each of the
//! `8 * len` bits coded against a two-symbol CDF `[0, c0, TOTAL]`.

use std::collections::HashMap;

use anyhow::Result;

use crate::ac::{AcDecoder, AcEncoder, TOTAL};
use crate::bits::{BitReader, BitWriter};
use crate::codec::{Codec, Decomposition};

/// Highest context order, in bytes. Models orders `0..=MAX_ORDER`.
const MAX_ORDER: usize = 6;
/// Number of order models (one mixer input each).
const N_ORDERS: usize = MAX_ORDER + 1;
/// Mixer gradient-descent step size on coding loss. An enwik8 quick-panel
/// sweep bottomed out near 0.002 (0.05 → 2.059, 0.02 → 1.971, 0.004 → 1.936,
/// 0.002 → 1.933); below that the curve is flat.
const MIX_LR: f64 = 0.002;
/// Initial per-order mixer weight (before any online training).
const INIT_W: f64 = 0.3;
/// Floor on a bit-predictor's adaptive learning rate, so a well-observed
/// context still tracks local drift instead of freezing.
const RATE_FLOOR: f64 = 1.0 / 256.0;
/// Cap on a bit-predictor's observation count (caps the slowest rate).
const N_CAP: u32 = 255;
/// Clamp bounds on a probability before `stretch`, so the logit stays finite
/// (and the mixer never sees an infinite input → `NaN` weight).
const P_LO: f64 = 1e-6;
const P_HI: f64 = 1.0 - 1e-6;

/// Logit of a probability: `ln(p / (1-p))`, clamped to keep it finite.
#[inline]
fn stretch(p: f64) -> f64 {
    let p = p.clamp(P_LO, P_HI);
    (p / (1.0 - p)).ln()
}

/// Inverse of [`stretch`]: the logistic squashing function.
#[inline]
fn squash(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// One adaptive bit predictor: a probability of "next bit is 1" plus an
/// observation count that schedules the learning rate (fast while young,
/// floored once mature so it keeps tracking drift).
#[derive(Clone, Copy, Debug)]
struct BitModel {
    p: f64,
    n: u32,
}

impl Default for BitModel {
    fn default() -> Self {
        Self { p: 0.5, n: 0 }
    }
}

impl BitModel {
    /// Move the probability toward the observed bit and age the counter.
    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn update(&mut self, y: f64) {
        let rate = (1.0 / (f64::from(self.n) + 1.5)).max(RATE_FLOOR);
        self.p += rate * (y - self.p);
        if self.n < N_CAP {
            self.n += 1;
        }
    }
}

/// Online multi-order bit model with a logistic (logit-domain) mixer.
#[derive(Debug)]
struct Model {
    /// Order-0: one predictor per within-byte tree node (`1..=255`).
    o0: Vec<BitModel>,
    /// Orders `1..=MAX_ORDER`, sparse. `maps[k - 1]` is keyed on the packed
    /// `(context_k, node)` — see [`Model::key`].
    maps: Vec<HashMap<u64, BitModel>>,
    /// Mixer weights, trained online by gradient descent on coding loss.
    w: [f64; N_ORDERS],
    /// Rolling context — the last up-to-8 bytes, most recent in the low byte.
    ctx: u64,
}

impl Model {
    fn new() -> Self {
        Self {
            o0: vec![BitModel::default(); 256],
            maps: (0..MAX_ORDER).map(|_| HashMap::new()).collect(),
            w: [INIT_W; N_ORDERS],
            ctx: 0,
        }
    }

    /// Low `8 * k` bits of the rolling context (order-`k` byte history).
    const fn ctx_k(&self, k: usize) -> u64 {
        if k >= 8 {
            self.ctx
        } else {
            self.ctx & ((1u64 << (8 * k)) - 1)
        }
    }

    /// Packed lookup key for order `k` at within-byte tree `node`: the order's
    /// byte history shifted up by 9 bits, with `node` (`< 512`) in the low
    /// bits. For `k <= 6` the history is `<= 48` bits, so the whole key fits a
    /// `u64` losslessly — no hashing, no collisions.
    #[inline]
    const fn key(&self, k: usize, node: usize) -> u64 {
        (self.ctx_k(k) << 9) | node as u64
    }

    /// Predict P(next bit = 1) at `node`, filling `x` with each order's logit
    /// (a novel context contributes a neutral `stretch(0.5) = 0`).
    #[allow(clippy::needless_range_loop)]
    fn predict_bit(&self, node: usize, x: &mut [f64; N_ORDERS]) -> f64 {
        x[0] = stretch(self.o0[node].p);
        for k in 1..=MAX_ORDER {
            x[k] = self.maps[k - 1]
                .get(&self.key(k, node))
                .map_or(0.0, |bm| stretch(bm.p));
        }
        let s: f64 = self.w.iter().zip(x.iter()).map(|(wk, xk)| wk * xk).sum();
        squash(s)
    }

    /// After bit `y` is coded at `node`: step the mixer weights down the
    /// coding-loss gradient, then update each order's bit predictor.
    #[allow(clippy::suboptimal_flops)]
    fn update_bit(&mut self, node: usize, x: &[f64; N_ORDERS], p1: f64, y: u8) {
        let err = f64::from(y) - p1;
        for (wk, &xk) in self.w.iter_mut().zip(x.iter()) {
            *wk += MIX_LR * err * xk;
        }
        let yf = f64::from(y);
        self.o0[node].update(yf);
        for k in 1..=MAX_ORDER {
            let key = self.key(k, node);
            self.maps[k - 1].entry(key).or_default().update(yf);
        }
    }

    /// Replay a byte through predict+update without coding it — used to prime
    /// model state from the `warm` prefix on both sides identically.
    fn learn_byte(&mut self, b: u8) {
        let mut node = 1usize;
        let mut x = [0.0; N_ORDERS];
        for i in (0..8).rev() {
            let y = (b >> i) & 1;
            let p1 = self.predict_bit(node, &mut x);
            self.update_bit(node, &x, p1, y);
            node = (node << 1) | usize::from(y);
        }
        self.ctx = (self.ctx << 8) | u64::from(b);
    }
}

/// Fill a two-symbol CDF `[0, c0, TOTAL]` where symbol 0 (bit = 0) holds mass
/// `1 - p1`. `c0` is clamped to `[1, TOTAL - 1]` to satisfy the AC's
/// strictly-increasing-CDF contract.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn fill_bit_cdf(p1: f64, cdf: &mut [u32; 3]) {
    let c0 = ((1.0 - p1) * f64::from(TOTAL)).round() as u32;
    cdf[1] = c0.clamp(1, TOTAL - 1);
}

/// The v7 logistic-mixing codec: bitwise multi-order context mixing in the
/// logit domain + AC, nothing else.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LmixCodec;

impl Codec for LmixCodec {
    fn name(&self) -> &'static str {
        "lmix"
    }

    #[allow(clippy::cast_possible_truncation)]
    fn encode_window(&self, warm: &[u8], measure: &[u8]) -> Result<(Vec<u8>, Decomposition)> {
        let mut writer = BitWriter::new();
        writer.write_bits(measure.len() as u64, 64);
        let mut model = Model::new();
        for &b in warm {
            model.learn_byte(b);
        }
        {
            let mut enc = AcEncoder::new(&mut writer);
            let mut x = [0.0; N_ORDERS];
            let mut cdf = [0u32, 0, TOTAL];
            for &b in measure {
                let mut node = 1usize;
                for i in (0..8).rev() {
                    let y = (b >> i) & 1;
                    let p1 = model.predict_bit(node, &mut x);
                    fill_bit_cdf(p1, &mut cdf);
                    enc.encode(&cdf, usize::from(y));
                    model.update_bit(node, &x, p1, y);
                    node = (node << 1) | usize::from(y);
                }
                model.ctx = (model.ctx << 8) | u64::from(b);
            }
            enc.finish();
        }
        let (buf, _pad) = writer.finish();
        let mut decomp = Decomposition::new();
        decomp.add("lmix", 8 * buf.len() as u64);
        Ok((buf, decomp))
    }

    fn decode_window(&self, warm: &[u8], archive: &[u8]) -> Result<Vec<u8>> {
        let mut reader = BitReader::new(archive);
        let n = usize::try_from(reader.read_bits(64)?).expect("length prefix fits usize");
        let mut model = Model::new();
        for &b in warm {
            model.learn_byte(b);
        }
        let mut out = Vec::with_capacity(n);
        let mut dec = AcDecoder::new(&mut reader);
        let mut x = [0.0; N_ORDERS];
        let mut cdf = [0u32, 0, TOTAL];
        for _ in 0..n {
            let mut node = 1usize;
            let mut b = 0u8;
            for _ in 0..8 {
                let p1 = model.predict_bit(node, &mut x);
                fill_bit_cdf(p1, &mut cdf);
                let y = u8::try_from(dec.decode(&cdf)?).expect("bit symbol 0/1 fits u8");
                model.update_bit(node, &x, p1, y);
                b = (b << 1) | y;
                node = (node << 1) | usize::from(y);
            }
            model.ctx = (model.ctx << 8) | u64::from(b);
            out.push(b);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lmix_roundtrips_text() {
        let measure = b"the quick brown fox jumps over the lazy dog. \
                        the quick brown fox jumps over the lazy dog again."
            .repeat(40);
        let codec = LmixCodec;
        let (archive, decomp) = codec.encode_window(b"", &measure).unwrap();
        assert_eq!(decomp.total(), 8 * archive.len() as u64);
        let decoded = codec.decode_window(b"", &archive).unwrap();
        assert_eq!(decoded, measure);
        assert!(
            archive.len() < measure.len(),
            "archive {} not smaller than input {}",
            archive.len(),
            measure.len()
        );
    }

    #[test]
    fn lmix_roundtrips_with_warm_prefix() {
        let warm = b"warm priming context that both sides share verbatim. ".repeat(8);
        let measure = b"warm priming context that both sides share, then new tail.".to_vec();
        let codec = LmixCodec;
        let (archive, _) = codec.encode_window(&warm, &measure).unwrap();
        let decoded = codec.decode_window(&warm, &archive).unwrap();
        assert_eq!(decoded, measure);
    }

    #[test]
    fn lmix_roundtrips_all_byte_values() {
        let measure: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        let codec = LmixCodec;
        let (archive, _) = codec.encode_window(b"", &measure).unwrap();
        let decoded = codec.decode_window(b"", &archive).unwrap();
        assert_eq!(decoded, measure);
    }

    #[test]
    fn lmix_roundtrips_empty() {
        let codec = LmixCodec;
        let (archive, _) = codec.encode_window(b"", b"").unwrap();
        let decoded = codec.decode_window(b"", &archive).unwrap();
        assert!(decoded.is_empty());
    }
}
