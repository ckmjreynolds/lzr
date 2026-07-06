//! The universal byte→byte stream transform and the pipeline that chains them.
//!
//! Every pipeline stage — the byte preprocessors and the Re-Pair tokenizer
//! ([`crate::preprocessors`]) and the entropy coder ([`crate::entropy`]) — is a
//! [`Transform`]. A `Transform` rewrites a byte stream
//! on the encode side ([`Transform::forward`]) and exactly inverts it on the decode
//! side ([`Transform::inverse`]). Because *every* stage speaks the same `Vec<u8>`
//! domain, stages compose freely: tokenization and entropy coding can each be
//! toggled on or off, and a future LZ77 stage slots in like any other.
//!
//! Both methods take the buffer **by value** so a stage can free its input before
//! an expensive build (the tokenizer drops ~1 GB before its grammar build) and so
//! [`Pipeline`] can move the buffer stage-to-stage with no per-stage clone.

/// One stage's encode-side trace record: `(feature name, input length, output length, optional
/// `(detail value, unit label)`)`. Returned per stage by [`Pipeline::forward_traced`] for the CLI's
/// per-stage report; the unit label lets each stage name its own statistic (`tokens`, `matches`, …).
pub(crate) type StageTrace = (&'static str, usize, usize, Option<(u64, &'static str)>);

/// One entropy model's encode-side scorecard entry: `(model name, standalone bits-per-byte, average
/// mixer weight)`. Gathered by [`Pipeline::model_scores`] for the CLI's per-model report.
pub(crate) type ModelTrace = (&'static str, f64, f64);

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

    /// An optional per-stage statistic for the CLI's trace as `(value, unit label)`, derived from the
    /// stage's own `output` bytes — e.g. `(vocab, "tokens")` for Re-Pair, `(count, "matches")` for
    /// LZ77. `None` — the default — for stages with nothing extra to report. Encode-side only (see
    /// [`Pipeline::forward_traced`]).
    fn trace_detail(&self, _output: &[u8]) -> Option<(u64, &'static str)> {
        None
    }

    /// The stage's per-model scorecard from the last [`Transform::forward`], if any. Only the entropy
    /// stage returns entries; every other stage uses this empty default. Read after `forward` via
    /// [`Pipeline::model_scores`]. Encode-side only.
    fn model_scores(&self) -> Vec<ModelTrace> {
        Vec::new()
    }
}

/// An ordered chain of transforms: `forward` applies them in order, `inverse`
/// applies each stage's inverse in reverse order.
pub(crate) struct Pipeline {
    stages: Vec<Box<dyn Transform>>,
    /// Feature name of each stage, aligned with `stages`, used only to label the CLI's per-stage
    /// size report (see [`Pipeline::forward_traced`]).
    names: Vec<&'static str>,
}

impl Pipeline {
    /// A pipeline over the given ordered `stages`, each labelled by the aligned entry in `names`.
    pub(crate) fn new(stages: Vec<Box<dyn Transform>>, names: Vec<&'static str>) -> Self {
        debug_assert_eq!(stages.len(), names.len(), "each stage needs exactly one name");
        Self {
            stages,
            names,
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

    /// Like [`Pipeline::forward`] but also returns each stage's `(name, input length, output length,
    /// optional detail)` in order, so the CLI can report each transform's own bits-per-byte — what it
    /// did to *its* input, not a running figure against the original — plus any per-stage statistic
    /// ([`Transform::trace_detail`], e.g. the Re-Pair token count).
    pub(crate) fn forward_traced(&self, mut data: Vec<u8>) -> (Vec<u8>, Vec<StageTrace>) {
        let mut sizes = Vec::with_capacity(self.stages.len());
        for (stage, &name) in self.stages.iter().zip(&self.names) {
            let input_len = data.len();
            data = stage.forward(data);
            let detail = stage.trace_detail(&data);
            sizes.push((name, input_len, data.len(), detail));
        }
        (data, sizes)
    }

    /// The per-model scorecard gathered from every stage after a [`Pipeline::forward_traced`] run —
    /// in practice only the entropy stage contributes. Call after `forward_traced`.
    pub(crate) fn model_scores(&self) -> Vec<ModelTrace> {
        self.stages.iter().flat_map(|stage| stage.model_scores()).collect()
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
        let p = Pipeline::new(vec![], vec![]);
        assert_eq!(p.forward(b"hello".to_vec()), b"hello");
        assert_eq!(p.inverse(b"hello".to_vec()).unwrap(), b"hello");
    }

    /// `inverse` must unwind stages in the exact reverse of `forward` — the whole
    /// pipeline's correctness rests on this.
    #[test]
    fn inverse_reverses_stage_order() {
        // forward pushes 1 then 2; inverse must pop 2 (last stage) before 1.
        let p = Pipeline::new(vec![Box::new(Append(1)), Box::new(Append(2))], vec!["one", "two"]);
        let encoded = p.forward(b"data".to_vec());
        assert_eq!(encoded, b"data\x01\x02");
        assert_eq!(p.inverse(encoded).unwrap(), b"data");
    }

    /// `forward_traced` must return the same bytes as `forward`, plus each stage's name and its own
    /// input and output length (the second stage's input is the first's output) in pipeline order.
    /// The `Append` stages report no detail, so each carries `None`.
    #[test]
    fn forward_traced_records_each_stage() {
        let p = Pipeline::new(vec![Box::new(Append(1)), Box::new(Append(2))], vec!["one", "two"]);
        let (encoded, sizes) = p.forward_traced(b"data".to_vec());
        assert_eq!(encoded, b"data\x01\x02");
        assert_eq!(sizes, vec![("one", 4, 5, None), ("two", 5, 6, None)]);
    }
}
