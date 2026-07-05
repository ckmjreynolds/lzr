//! Reversible stream transforms and the pipeline that chains them.
//!
//! A [`Transform`] rewrites a stream on the encode side ([`Transform::forward`])
//! and exactly inverts it on the decode side ([`Transform::inverse`]). The trait
//! is generic over the element type, so the *same* trait serves both the byte
//! preprocessors (`Transform<u8>`, before tokenization) and the token
//! preprocessors (`Transform<u15>`, after it). Adding a real stage is a new type
//! that implements `Transform` plus a push into the relevant [`Pipeline`].
//!
//! The concrete byte preprocessors live in their own modules ([`casefold`],
//! [`entities`]); this file holds only the framework — the trait, the pipeline,
//! the identity stages, and the text-detection helpers the byte stages share.

mod casefold;
mod entities;
mod lz77;

pub(crate) use casefold::CaseFolding;
pub(crate) use entities::EntityFolding;
pub(crate) use lz77::Lz77;

/// Minimum fraction of bytes that must be common-text bytes for the input to count
/// as text (below this, a byte stage skips its transform and passes through).
pub(super) const TEXT_FRACTION: f64 = 0.95;

/// Whether `byte` is a byte we expect to see in plain text: printable ASCII plus
/// tab, newline, and carriage return. Shared by the byte preprocessors' text gate.
pub(super) const fn is_text_byte(byte: u8) -> bool {
    matches!(byte, b'\t' | b'\n' | b'\r' | 0x20..=0x7E)
}

/// The `N` lowest byte values absent from `present`, or `None` if fewer than `N`
/// exist. Byte preprocessors use this to claim unused byte values as their control
/// symbols (deterministically, so the decoder derives the same set from its own
/// scan of the reconstructed stream).
pub(super) fn spare_bytes<const N: usize>(present: &[bool; 256]) -> Option<[u8; N]> {
    let mut unused = (0u8..=255).filter(|&b| !present[usize::from(b)]);
    let mut out = [0u8; N];
    for slot in &mut out {
        *slot = unused.next()?;
    }
    Some(out)
}

/// A reversible transform over a stream of `T`.
pub(crate) trait Transform<T> {
    /// Encode-side transform. Infallible: the encoder controls its own input.
    fn forward(&self, input: &[T]) -> Vec<T>;

    /// Exact inverse, applied on the decode side. Fallible: it runs on decoded,
    /// possibly-corrupt data.
    fn inverse(&self, input: &[T]) -> anyhow::Result<Vec<T>>;
}

/// An ordered chain of same-domain transforms: `forward` applies them in order,
/// `inverse` applies each stage's inverse in reverse order.
pub(crate) struct Pipeline<T> {
    stages: Vec<Box<dyn Transform<T>>>,
}

impl<T: Clone> Pipeline<T> {
    /// A pipeline over the given ordered `stages`.
    pub(crate) fn new(stages: Vec<Box<dyn Transform<T>>>) -> Self {
        Self {
            stages,
        }
    }

    /// Apply every stage in order. The input is copied once (by the first stage),
    /// not an extra time up front.
    pub(crate) fn forward(&self, input: &[T]) -> Vec<T> {
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

    /// Apply every stage's inverse in reverse order.
    ///
    /// # Errors
    ///
    /// Propagates the first stage inverse that fails (e.g. a transform fed
    /// corrupt data).
    pub(crate) fn inverse(&self, input: &[T]) -> anyhow::Result<Vec<T>> {
        let mut stages = self.stages.iter().rev();
        let Some(first) = stages.next() else {
            return Ok(input.to_vec());
        };
        let mut data = first.inverse(input)?;
        for stage in stages {
            data = stage.inverse(&data)?;
        }
        Ok(data)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    /// A profile with no enabled stages yields an empty pipeline, which must be the
    /// identity — the path `Profile::byte_preprocessors`/`token_preprocessors` take
    /// when nothing is selected, in place of an explicit identity stage.
    #[test]
    fn empty_pipeline_is_identity() {
        let p: Pipeline<u8> = Pipeline::new(vec![]);
        assert_eq!(p.forward(b"hello"), b"hello");
        assert_eq!(p.inverse(b"hello").unwrap(), b"hello");
    }
}
