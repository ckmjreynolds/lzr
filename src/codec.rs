//! Bridges LZ77 sequences and arithmetic coding.
//!
//! Provides [`ModelSet`] (five adaptive models for the different parts of an
//! LZ77 sequence) and functions to encode/decode sequences through the entropy
//! coder. Used by the [`Writer`](crate::Writer) and [`Reader`](crate::Reader)
//! as well as the shadow decoder that tracks committed sequences.

use crate::entropy;
use crate::lz77::{Sequence, Token};
use crate::model::Model;

/// Five adaptive frequency models, one for each field of an LZ77 sequence.
pub(crate) struct ModelSet {
    /// Model for `literal_len` (0..=255).
    pub(crate) lit_len: Model,
    /// Model for literal bytes.
    pub(crate) literals: Model,
    /// Model for `match_len` (0 = no match, 3..=255).
    pub(crate) match_len: Model,
    /// Model for the low byte of `match_distance`.
    pub(crate) dist_lo: Model,
    /// Model for the high byte of `match_distance`.
    pub(crate) dist_hi: Model,
}

impl ModelSet {
    /// Creates a new set of uniform models.
    pub(crate) const fn new() -> Self {
        Self {
            lit_len: Model::new(),
            literals: Model::new(),
            match_len: Model::new(),
            dist_lo: Model::new(),
            dist_hi: Model::new(),
        }
    }

    /// Resets all models to uniform (for block boundaries).
    pub(crate) const fn reset(&mut self) {
        self.lit_len.reset();
        self.literals.reset();
        self.match_len.reset();
        self.dist_lo.reset();
        self.dist_hi.reset();
    }
}

/// Marker distance for SEAL (end of file).
pub(crate) const SEAL_DISTANCE: u16 = 0;

/// Marker distance for end-of-frame (frame boundary, entropy reset).
pub(crate) const EOFRAME_DISTANCE: u16 = 0xFFFF;

/// Returns `true` if this sequence is a marker (`literal_len=0`, `match_len=0`).
pub(crate) const fn is_marker(seq: Sequence) -> bool {
    seq.literal_len == 0 && seq.match_len == 0
}

/// Returns `true` if this sequence is the SEAL sentinel.
pub(crate) const fn is_seal(seq: Sequence) -> bool {
    is_marker(seq) && seq.match_distance == SEAL_DISTANCE
}

/// Returns `true` if this sequence is an end-of-frame marker.
pub(crate) const fn is_eoframe(seq: Sequence) -> bool {
    is_marker(seq) && seq.match_distance == EOFRAME_DISTANCE
}

/// Encodes one LZ77 token through the entropy encoder.
///
/// Encodes in order: `literal_len`, each literal byte, `match_len`,
/// and (if `match_len > 0`) `match_distance` as two bytes (lo, hi).
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn encode_sequence(enc: &mut entropy::Encoder, models: &mut ModelSet, token: &Token, output: &mut Vec<u8>) {
    let seq = &token.seq;
    let lits = &token.literals[..seq.literal_len as usize];

    enc.encode(&mut models.lit_len, seq.literal_len, output);
    for &b in lits {
        enc.encode(&mut models.literals, b, output);
    }
    enc.encode(&mut models.match_len, seq.match_len, output);
    // Encode distance for matches AND markers (literal_len=0, match_len=0).
    if seq.match_len > 0 || is_marker(*seq) {
        enc.encode(&mut models.dist_lo, seq.match_distance as u8, output);
        enc.encode(&mut models.dist_hi, (seq.match_distance >> 8) as u8, output);
    }
}

/// Decodes one LZ77 token from the entropy decoder.
pub(crate) fn decode_sequence(dec: &mut entropy::Decoder, models: &mut ModelSet, input: &mut &[u8]) -> Token {
    let literal_len = dec.decode(&mut models.lit_len, input);
    let mut literals = [0u8; 255];
    for b in &mut literals[..literal_len as usize] {
        *b = dec.decode(&mut models.literals, input);
    }
    let match_len = dec.decode(&mut models.match_len, input);
    // Decode distance for matches AND markers (literal_len=0, match_len=0).
    let match_distance = if match_len > 0 || (literal_len == 0 && match_len == 0) {
        let lo = dec.decode(&mut models.dist_lo, input);
        let hi = dec.decode(&mut models.dist_hi, input);
        u16::from(lo) | (u16::from(hi) << 8)
    } else {
        0
    };

    Token {
        seq: Sequence {
            literal_len,
            match_len,
            match_distance,
        },
        literals,
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::lz77;

    /// Encode a sequence then decode it, verifying roundtrip.
    fn roundtrip_sequence(token: &Token) -> Token {
        let mut enc = entropy::Encoder::new();
        let mut enc_models = ModelSet::new();
        let mut compressed = Vec::new();

        encode_sequence(&mut enc, &mut enc_models, token, &mut compressed);
        enc.flush(&mut compressed);

        let mut dec = entropy::Decoder::new();
        let mut dec_models = ModelSet::new();
        let mut input: &[u8] = &compressed;

        decode_sequence(&mut dec, &mut dec_models, &mut input)
    }

    #[test]
    fn markers() {
        let seal = Sequence {
            literal_len: 0,
            match_len: 0,
            match_distance: SEAL_DISTANCE,
        };
        assert!(is_seal(seal));
        assert!(is_marker(seal));
        assert!(!is_eoframe(seal));

        let eoframe = Sequence {
            literal_len: 0,
            match_len: 0,
            match_distance: EOFRAME_DISTANCE,
        };
        assert!(is_eoframe(eoframe));
        assert!(is_marker(eoframe));
        assert!(!is_seal(eoframe));

        let not_marker = Sequence {
            literal_len: 1,
            match_len: 0,
            match_distance: 0,
        };
        assert!(!is_marker(not_marker));
        assert!(!is_seal(not_marker));
        assert!(!is_eoframe(not_marker));
    }

    #[test]
    fn roundtrip_literal_only() {
        let token = Token {
            seq: Sequence {
                literal_len: 5,
                match_len: 0,
                match_distance: 0,
            },
            literals: {
                let mut buf = [0u8; 255];
                buf[..5].copy_from_slice(b"hello");
                buf
            },
        };

        let decoded = roundtrip_sequence(&token);
        assert_eq!(token.seq, decoded.seq);
        assert_eq!(token.literals[..5], decoded.literals[..5]);
    }

    #[test]
    fn roundtrip_with_match() {
        let token = Token {
            seq: Sequence {
                literal_len: 3,
                match_len: 10,
                match_distance: 42,
            },
            literals: {
                let mut buf = [0u8; 255];
                buf[..3].copy_from_slice(b"abc");
                buf
            },
        };

        let decoded = roundtrip_sequence(&token);
        assert_eq!(token.seq, decoded.seq);
        assert_eq!(token.literals[..3], decoded.literals[..3]);
    }

    #[test]
    fn roundtrip_seal() {
        let seal_token = Token {
            seq: Sequence {
                literal_len: 0,
                match_len: 0,
                match_distance: SEAL_DISTANCE,
            },
            literals: [0u8; 255],
        };
        let decoded = roundtrip_sequence(&seal_token);
        assert!(is_seal(decoded.seq));
    }

    #[test]
    fn roundtrip_eoframe() {
        let eof_token = Token {
            seq: Sequence {
                literal_len: 0,
                match_len: 0,
                match_distance: EOFRAME_DISTANCE,
            },
            literals: [0u8; 255],
        };
        let decoded = roundtrip_sequence(&eof_token);
        assert!(is_eoframe(decoded.seq));
    }

    #[test]
    fn multi_sequence_roundtrip() {
        // Encode multiple sequences, then decode them all.
        let data = crate::bench::lipsum_bytes(4096);
        let mut lz_enc = lz77::Encoder::new();
        lz_enc.feed(&data);
        lz_enc.finish();

        let mut tokens = Vec::new();
        while let Some(token) = lz_enc.next() {
            tokens.push(token);
        }

        // Entropy encode all sequences.
        let mut enc = entropy::Encoder::new();
        let mut models = ModelSet::new();
        let mut compressed = Vec::new();
        for token in &tokens {
            encode_sequence(&mut enc, &mut models, token, &mut compressed);
        }
        enc.flush(&mut compressed);

        // Entropy decode all sequences.
        let mut dec = entropy::Decoder::new();
        let mut dec_models = ModelSet::new();
        let mut input: &[u8] = &compressed;
        let mut decoded_tokens = Vec::new();
        for _ in 0..tokens.len() {
            decoded_tokens.push(decode_sequence(&mut dec, &mut dec_models, &mut input));
        }

        // Verify all sequences match.
        for (orig, decoded) in tokens.iter().zip(decoded_tokens.iter()) {
            assert_eq!(orig.seq, decoded.seq);
            let len = orig.seq.literal_len as usize;
            assert_eq!(&orig.literals[..len], &decoded.literals[..len]);
        }

        // LZ77 decode to verify full pipeline.
        let mut lz_dec = lz77::Decoder::new();
        let mut output = Vec::new();
        for token in &decoded_tokens {
            let lits = &token.literals[..token.seq.literal_len as usize];
            lz_dec.decode(token.seq, lits, &mut output);
        }
        assert_eq!(data, output);
    }

    proptest! {
        #[test]
        fn roundtrip_random_sequence(
            literal_len in 0u8..=255,
            match_len in prop::sample::select(
                std::iter::once(0u8).chain(3..=255).collect::<Vec<_>>()
            ),
            match_distance in 1u16..=65_535,
            literal_data in prop::collection::vec(any::<u8>(), 255),
        ) {
            let mut literals = [0u8; 255];
            literals.copy_from_slice(&literal_data);
            let token = Token {
                seq: Sequence {
                    literal_len,
                    match_len,
                    match_distance: if match_len > 0 { match_distance } else { 0 },
                },
                literals,
            };

            let decoded = roundtrip_sequence(&token);
            prop_assert_eq!(token.seq, decoded.seq);
            let len = token.seq.literal_len as usize;
            prop_assert_eq!(&token.literals[..len], &decoded.literals[..len]);
        }
    }
}
