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

use arbitrary_int::u15;

mod casefold;
mod entities;

pub(crate) use casefold::CaseFolding;
pub(crate) use entities::EntityFolding;

/// Minimum fraction of bytes that must be common-text bytes for the input to count
/// as text (below this, a byte stage skips its transform and passes through).
pub(super) const TEXT_FRACTION: f64 = 0.95;

/// Whether `byte` is a byte we expect to see in plain text: printable ASCII plus
/// tab, newline, and carriage return. Shared by the byte preprocessors' text gate.
pub(super) const fn is_text_byte(byte: u8) -> bool {
    matches!(byte, b'\t' | b'\n' | b'\r' | 0x20..=0x7E)
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

/// Identity byte preprocessor: the NULL stage for the `Transform<u8>` slot.
pub(crate) struct NullBytes;

impl Transform<u8> for NullBytes {
    fn forward(&self, input: &[u8]) -> Vec<u8> {
        input.to_vec()
    }

    fn inverse(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        Ok(input.to_vec())
    }
}

/// Identity token preprocessor: the NULL stage for the `Transform<u15>` slot.
pub(crate) struct NullTokens;

impl Transform<u15> for NullTokens {
    fn forward(&self, input: &[u15]) -> Vec<u15> {
        input.to_vec()
    }

    fn inverse(&self, input: &[u15]) -> anyhow::Result<Vec<u15>> {
        Ok(input.to_vec())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn null_bytes_roundtrip(data in prop::collection::vec(any::<u8>(), 0..1024)) {
            let s = NullBytes;
            prop_assert_eq!(s.inverse(&s.forward(&data)).unwrap(), data);
        }

        #[test]
        fn null_pipeline_roundtrip(data in prop::collection::vec(any::<u8>(), 0..1024)) {
            let stages: Vec<Box<dyn Transform<u8>>> = vec![Box::new(NullBytes)];
            let p = Pipeline::new(stages);
            prop_assert_eq!(p.inverse(&p.forward(&data)).unwrap(), data);
        }
    }

    #[test]
    fn null_tokens_roundtrip() {
        let data: Vec<u15> = (0..100u16).map(u15::new).collect();
        let s = NullTokens;
        assert_eq!(s.inverse(&s.forward(&data)).unwrap(), data);
    }

    #[test]
    fn empty_pipeline_is_identity() {
        let p: Pipeline<u8> = Pipeline::new(vec![]);
        assert_eq!(p.forward(b"hello"), b"hello");
        assert_eq!(p.inverse(b"hello").unwrap(), b"hello");
    }
}
