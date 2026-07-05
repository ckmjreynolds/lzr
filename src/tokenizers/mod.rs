//! Reversible byte↔token conversion.
//!
//! A [`Tokenize`] maps the byte stream to a stream of `u15` tokens on the encode
//! side and back on the decode side. It is distinct from a preprocessor
//! `Transform` because it changes the element type, and its inverse is genuinely
//! fallible — a decoded token may lie outside the range a given tokenizer can invert.

use arbitrary_int::u15;

mod repair;

#[cfg_attr(feature = "bench-internals", visibility::make(pub))]
pub(crate) use self::repair::RepairTokenizer;

/// A reversible mapping between a byte stream and a `u15` token stream.
#[cfg_attr(feature = "bench-internals", visibility::make(pub))]
pub(crate) trait Tokenize {
    /// Encode-side: bytes to tokens. Infallible. Takes the input *by value* so a tokenizer may free
    /// it before an expensive build — at gigabyte scale holding both the bytes and the derived state
    /// is the difference between fitting in memory and not.
    fn forward(&self, input: Vec<u8>) -> Vec<u15>;

    /// Decode-side: tokens back to bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if a token cannot be mapped back to bytes by this
    /// tokenizer (e.g. it is out of the representable range).
    fn inverse(&self, input: &[u15]) -> anyhow::Result<Vec<u8>>;
}

/// Identity tokenizer: each byte becomes the numerically equal token. Its inverse
/// is the one NULL stage that can genuinely fail — a token ≥ 256 has no single
/// byte, which is exactly the guard that catches a wrong 8-bit symbol mask
/// upstream (see [`crate::models::Context::push_symbol`]).
pub(crate) struct NullTokenizer;

impl Tokenize for NullTokenizer {
    fn forward(&self, input: Vec<u8>) -> Vec<u15> {
        input.iter().map(|&b| u15::new(u16::from(b))).collect()
    }

    fn inverse(&self, input: &[u15]) -> anyhow::Result<Vec<u8>> {
        input
            .iter()
            .map(|&t| {
                u8::try_from(t.value())
                    .map_err(|_| anyhow::anyhow!("null detokenize: token {} is not a byte value", t.value()))
            })
            .collect()
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn null_tokenizer_roundtrip(data in prop::collection::vec(any::<u8>(), 0..1024)) {
            let t = NullTokenizer;
            prop_assert_eq!(t.inverse(&t.forward(data.clone())).unwrap(), data);
        }
    }

    #[test]
    fn forward_maps_bytes_to_equal_tokens() {
        let t = NullTokenizer;
        assert_eq!(t.forward(vec![0, 65, 255]), vec![u15::new(0), u15::new(65), u15::new(255)]);
    }

    #[test]
    fn detokenize_rejects_non_byte_tokens() {
        let t = NullTokenizer;
        // 255 is the largest byte value and inverts fine.
        assert_eq!(t.inverse(&[u15::new(255)]).unwrap(), vec![255u8]);
        // 256 (and anything above) has no single-byte inverse.
        assert!(t.inverse(&[u15::new(256)]).is_err());
        assert!(t.inverse(&[u15::new(0x7FFF)]).is_err());
    }
}
