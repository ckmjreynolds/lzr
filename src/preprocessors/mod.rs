//! Reversible byte-stream preprocessors and the pipeline that chains them.
//!
//! A preprocessor transforms the data before modeling on the encode side and
//! exactly inverts it on the decode side. Adding one is a new file under
//! `preprocessors/` plus a push into [`Pipeline::default_pipeline`].

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
