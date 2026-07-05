//! The universal byte→byte stream transform and the pipeline that chains them.
//!
//! Every pipeline stage — the byte preprocessors ([`crate::preprocessors`]), the
//! Re-Pair tokenizer ([`crate::tokenizers`]), and the entropy coder
//! ([`crate::entropy`]) — is a [`Transform`]. A `Transform` rewrites a byte stream
//! on the encode side ([`Transform::forward`]) and exactly inverts it on the decode
//! side ([`Transform::inverse`]). Because *every* stage speaks the same `Vec<u8>`
//! domain, stages compose freely: tokenization and entropy coding can each be
//! toggled on or off, and a future LZ77 stage slots in like any other.
//!
//! Both methods take the buffer **by value** so a stage can free its input before
//! an expensive build (the tokenizer drops ~1 GB before its grammar build) and so
//! [`Pipeline`] can move the buffer stage-to-stage with no per-stage clone.

/// A reversible byte→byte stream transform.
#[cfg_attr(feature = "bench-internals", visibility::make(pub))]
pub(crate) trait Transform {
    /// Encode-side transform. Takes ownership so the stage can free its input, and
    /// so [`Pipeline`] can thread one owned buffer through the chain. Infallible:
    /// the encoder controls its own input.
    fn forward(&self, input: Vec<u8>) -> Vec<u8>;

    /// Exact inverse, applied on the decode side. Also by value, for symmetry and
    /// so an expanding inverse (the tokenizer) can drop its compact input as it
    /// grows the output.
    ///
    /// # Errors
    ///
    /// Returns an error if the stage is fed corrupt or malformed data — it runs on
    /// decoded, possibly-adversarial bytes.
    fn inverse(&self, input: Vec<u8>) -> anyhow::Result<Vec<u8>>;
}

/// An ordered chain of transforms: `forward` applies them in order, `inverse`
/// applies each stage's inverse in reverse order.
pub(crate) struct Pipeline {
    stages: Vec<Box<dyn Transform>>,
}

impl Pipeline {
    /// A pipeline over the given ordered `stages`.
    pub(crate) fn new(stages: Vec<Box<dyn Transform>>) -> Self {
        Self {
            stages,
        }
    }

    /// Apply every stage in order, moving the buffer stage-to-stage (no clones). An
    /// empty pipeline is the true identity — the input is returned untouched.
    pub(crate) fn forward(&self, mut data: Vec<u8>) -> Vec<u8> {
        for stage in &self.stages {
            data = stage.forward(data);
        }
        data
    }

    /// Apply every stage's inverse in reverse order.
    ///
    /// # Errors
    ///
    /// Propagates the first stage inverse that fails (e.g. a transform fed corrupt
    /// data).
    pub(crate) fn inverse(&self, mut data: Vec<u8>) -> anyhow::Result<Vec<u8>> {
        for stage in self.stages.iter().rev() {
            data = stage.inverse(data)?;
        }
        Ok(data)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    /// A stage that appends a fixed marker byte (and strips it on inverse) — enough
    /// to observe stage ordering.
    struct Append(u8);

    impl Transform for Append {
        fn forward(&self, mut input: Vec<u8>) -> Vec<u8> {
            input.push(self.0);
            input
        }

        fn inverse(&self, mut input: Vec<u8>) -> anyhow::Result<Vec<u8>> {
            match input.pop() {
                Some(b) if b == self.0 => Ok(input),
                _ => anyhow::bail!("Append::inverse: missing marker {:#x}", self.0),
            }
        }
    }

    /// A profile with no enabled stages yields an empty pipeline, which must be the
    /// identity.
    #[test]
    fn empty_pipeline_is_identity() {
        let p = Pipeline::new(vec![]);
        assert_eq!(p.forward(b"hello".to_vec()), b"hello");
        assert_eq!(p.inverse(b"hello".to_vec()).unwrap(), b"hello");
    }

    /// `inverse` must unwind stages in the exact reverse of `forward` — the whole
    /// pipeline's correctness rests on this.
    #[test]
    fn inverse_reverses_stage_order() {
        // forward pushes 1 then 2; inverse must pop 2 (last stage) before 1.
        let p = Pipeline::new(vec![Box::new(Append(1)), Box::new(Append(2))]);
        let encoded = p.forward(b"data".to_vec());
        assert_eq!(encoded, b"data\x01\x02");
        assert_eq!(p.inverse(encoded).unwrap(), b"data");
    }
}
