//! Reversible byte-stream preprocessors and the pipeline that chains them.
//!
//! A preprocessor transforms the data before modeling on the encode side and
//! exactly inverts it on the decode side. Adding one is a new file under
//! `preprocessors/` plus a push into [`Pipeline::default_pipeline`].
//!
//! ## Free byte values (escape/marker codes)
//!
//! A byte-frequency census (2026-06-19) of the target corpora found these byte
//! values **never occur**, so they are safe to use as escape/marker codes a
//! preprocessor can emit without ambiguity:
//!
//! - **enwik9** (the submission target): 50 absent values.
//! - **enwik8** (dev/test slice): 51 absent — the same set plus `0xEE` (238),
//!   which is free in enwik8 but **used** in enwik9, so do not rely on it.
//!
//! The 50 values free in **both** (use these):
//!
//! ```text
//! 0x00..=0x08            C0 controls (NUL..BS)          — TAB 0x09 and LF 0x0A are USED
//! 0x0B..=0x1F            VT, FF, CR, SO..US             — all other C0 controls
//! 0x7F                   DEL
//! 0xC0 0xC1              invalid UTF-8 lead bytes
//! 0xDD 0xDF
//! 0xF1..=0xFF            high bytes (0xF5..=0xFF invalid UTF-8; 0xF1..=0xF4 unused)
//! ```
//!
//! As decimals: 0-8, 11-31, 127, 192, 193, 221, 223, 241-255.
//! (0xEE / 238 is free in enwik8 only — excluded.)

pub(crate) mod null;

/// A reversible transform applied to the byte stream.
pub(crate) trait Preprocessor {
    /// Encode-side transform.
    fn forward(&self, input: &[u8]) -> Vec<u8>;
    /// Exact inverse, applied on the decode side.
    fn inverse(&self, input: &[u8]) -> Vec<u8>;
}

/// An ordered chain of preprocessors.
#[derive(Default)]
pub(crate) struct Pipeline {
    stages: Vec<Box<dyn Preprocessor>>,
}

impl Pipeline {
    /// The default pipeline used by the codec.
    pub(crate) fn default_pipeline() -> Self {
        Self {
            stages: vec![Box::new(null::Null)],
        }
    }

    /// Apply every stage in order (encode side). The input is copied once (by
    /// the first stage), not an extra time up front.
    pub(crate) fn forward(&self, input: &[u8]) -> Vec<u8> {
        let mut stages = self.stages.iter();
        let Some(first) = stages.next() else {
            return input.to_vec();
        };
        let mut data = first.forward(input);
        for stage in stages {
            data = stage.forward(&data);
        }
        data
    }

    /// Apply every stage's inverse in reverse order (decode side). The input is
    /// copied once (by the first inverse stage), not an extra time up front.
    pub(crate) fn inverse(&self, input: &[u8]) -> Vec<u8> {
        let mut stages = self.stages.iter().rev();
        let Some(first) = stages.next() else {
            return input.to_vec();
        };
        let mut data = first.inverse(input);
        for stage in stages {
            data = stage.inverse(&data);
        }
        data
    }
}
