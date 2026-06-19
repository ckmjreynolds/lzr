//! The encode/decode driver.
//!
//! One per-bit loop wires the preprocessor pipeline, the models, the mixer, and
//! the arithmetic coder. Encode and decode build identical fresh state so their
//! predictions match bit-for-bit. The only framing is a LEB128 varint length
//! prefix (the decoder must know how many bytes to emit); there is no header.

use crate::coder::{Decoder, Encoder};
use crate::mixer::Mixer;
use crate::models::context::ContextModel;
use crate::models::{Context, Model};
use crate::preprocessors::Pipeline;

/// The active model set. Adding a model is one line here.
fn models() -> Vec<Box<dyn Model>> {
    vec![
        Box::new(ContextModel::new(0)),
        Box::new(ContextModel::new(1)),
        Box::new(ContextModel::new(2)),
        Box::new(ContextModel::new(3)),
        Box::new(ContextModel::new(4)),
        Box::new(ContextModel::new(5)),
        Box::new(ContextModel::new(6)),
    ]
}

/// The shared predictor state driven identically by both directions: encode and
/// decode differ only in where each bit comes from (read from the input vs.
/// decoded from the stream) and which coder consumes it.
struct CodecState {
    models: Vec<Box<dyn Model>>,
    mixer: Mixer,
    ctx: Context,
    stretched: Vec<i32>,
}

impl CodecState {
    fn new() -> Self {
        let models = models();
        let stretched = vec![0i32; models.len()];
        let mixer = Mixer::new(models.len());
        Self {
            models,
            mixer,
            ctx: Context::new(),
            stretched,
        }
    }

    /// Predict the next bit as P(bit == 1) in 12-bit probability form.
    #[allow(clippy::cast_sign_loss)]
    fn predict(&mut self) -> u32 {
        for (m, s) in self.models.iter_mut().zip(&mut self.stretched) {
            *s = m.predict(&self.ctx);
        }
        self.mixer.mix(&self.stretched, usize::from(self.ctx.bpos)) as u32
    }

    /// Commit the actual `bit`: adapt the mixer and models, advance the context.
    fn commit(&mut self, bit: u8) {
        self.mixer.update(bit);
        for m in &mut self.models {
            m.update(&self.ctx, bit);
        }
        self.ctx.push_bit(bit);
    }

    /// Finalize the current symbol (byte) once its 8 bits are in.
    fn end_symbol(&mut self) {
        self.ctx.push_byte();
    }
}

#[allow(clippy::cast_possible_truncation)]
fn write_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn read_varint(input: &[u8]) -> (u64, usize) {
    let mut v = 0u64;
    let mut shift = 0;
    let mut i = 0;
    loop {
        let byte = input[i];
        v |= u64::from(byte & 0x7f) << shift;
        i += 1;
        if byte & 0x80 == 0 {
            return (v, i);
        }
        shift += 7;
    }
}

/// Compress `input` into the lzr byte stream.
pub(crate) fn encode(input: &[u8]) -> Vec<u8> {
    let data = Pipeline::default_pipeline().forward(input);

    let mut out = Vec::new();
    write_varint(&mut out, data.len() as u64);

    let mut state = CodecState::new();
    let mut enc = Encoder::new();
    for &byte in &data {
        for k in (0..8).rev() {
            let bit = (byte >> k) & 1;
            let p = state.predict();
            enc.encode(bit, p);
            state.commit(bit);
        }
        state.end_symbol();
    }

    out.extend_from_slice(&enc.finish());
    out
}

/// Decompress an lzr byte stream back into the original bytes.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn decode(input: &[u8]) -> Vec<u8> {
    let (len, header) = read_varint(input);
    let len = len as usize;

    let mut state = CodecState::new();
    let mut dec = Decoder::new(&input[header..]);
    let mut data = Vec::with_capacity(len);
    for _ in 0..len {
        let mut byte = 0u8;
        for _ in 0..8 {
            let p = state.predict();
            let bit = dec.decode(p);
            state.commit(bit);
            byte = (byte << 1) | bit;
        }
        state.end_symbol();
        data.push(byte);
    }

    Pipeline::default_pipeline().inverse(&data)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_precision_loss)]
    use super::*;

    #[test]
    fn roundtrip_text() {
        let data = b"Hello, context mixing! The quick brown fox. ".repeat(64);
        let coded = encode(&data);
        assert_eq!(decode(&coded), data);
    }

    #[test]
    fn roundtrip_empty() {
        assert_eq!(decode(&encode(b"")), b"");
    }

    #[test]
    fn roundtrip_enwik8_slice() {
        let Ok(bytes) = std::fs::read("assets/enwik8") else {
            return; // skip when the corpus is absent
        };
        let slice = &bytes[1_000_000..1_100_000];
        let coded = encode(slice);
        assert_eq!(decode(&coded), slice);
        let bpb = coded.len() as f64 * 8.0 / slice.len() as f64;
        println!("order-0..3 mix on enwik8 100 KB slice: {bpb:.4} bpb");
    }
}
