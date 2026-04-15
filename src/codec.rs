//! Bridges LZ77 sequences and arithmetic coding via tag-based framing.
//!
//! Provides [`ModelSet`] (six adaptive models) and functions to encode/decode
//! tokens through the entropy coder. Every token begins with a **tag** symbol
//! from a 4-symbol alphabet (LINE, EOH, EOS, EOO) per FORMAT.md Section 2.

use crate::entropy;
use crate::lz77::{Sequence, Token};
use crate::model::{FreqModel, Model, TagModel};

// ---------------------------------------------------------------------------
// Tag constants (FORMAT.md Section 2.1)
// ---------------------------------------------------------------------------

/// Normal LZ77 sequence (literals and/or a match).
pub(crate) const TAG_LINE: u8 = 0;
/// End of Haiku — finalize arithmetic coder, begin new Haiku.
pub(crate) const TAG_EOH: u8 = 1;
/// End of Sonnet — finalize, validate Couplet, reset all state.
pub(crate) const TAG_EOS: u8 = 2;
/// End of Opus — finalize, validate Coda, archive is sealed.
pub(crate) const TAG_EOO: u8 = 3;

// ---------------------------------------------------------------------------
// ModelSet
// ---------------------------------------------------------------------------

/// Six adaptive frequency models, one for each field of a token.
pub(crate) struct ModelSet {
    /// Token type tag (4-symbol alphabet).
    pub(crate) tag: TagModel,
    /// Literal byte count (0..=255).
    pub(crate) lit_len: Model,
    /// Literal byte values.
    pub(crate) literals: Model,
    /// Match copy length (0 = no match; 3..=255).
    pub(crate) match_len: Model,
    /// Low byte of match distance.
    pub(crate) dist_lo: Model,
    /// High byte of match distance.
    pub(crate) dist_hi: Model,
}

impl ModelSet {
    /// Creates a new set of uniform models.
    pub(crate) const fn new() -> Self {
        Self {
            tag: TagModel::new(),
            lit_len: Model::new(),
            literals: Model::new(),
            match_len: Model::new(),
            dist_lo: Model::new(),
            dist_hi: Model::new(),
        }
    }

    /// Resets all models to uniform (for Sonnet boundaries).
    pub(crate) fn reset(&mut self) {
        self.tag.reset();
        self.lit_len.reset();
        self.literals.reset();
        self.match_len.reset();
        self.dist_lo.reset();
        self.dist_hi.reset();
    }
}

// ---------------------------------------------------------------------------
// Decoded token type
// ---------------------------------------------------------------------------

/// Result of decoding one token from the stream.
#[allow(clippy::large_enum_variant)]
pub(crate) enum DecodedToken {
    /// Normal LZ77 sequence.
    Line(Token),
    /// End of Haiku — finalize coder, models carry over.
    EndOfHaiku,
    /// End of Sonnet — finalize coder, reset all state.
    EndOfSonnet,
    /// End of Opus — finalize coder, archive sealed.
    EndOfOpus,
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/// Encodes one LINE token through the entropy encoder (FORMAT.md Section 2.2).
///
/// Encodes: tag(LINE), `literal_len`, `match_len`, [`dist_lo`, `dist_hi`], literals.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn encode_line(enc: &mut entropy::Encoder, models: &mut ModelSet, token: &Token, output: &mut Vec<u8>) {
    let seq = &token.seq;
    let lits = &token.literals[..seq.literal_len as usize];

    enc.encode(TAG_LINE, &mut models.tag, output);
    enc.encode(seq.literal_len, &mut models.lit_len, output);
    enc.encode(seq.match_len, &mut models.match_len, output);

    if seq.match_len > 0 {
        enc.encode(seq.match_distance as u8, &mut models.dist_lo, output);
        enc.encode((seq.match_distance >> 8) as u8, &mut models.dist_hi, output);
    }

    for &b in lits {
        enc.encode(b, &mut models.literals, output);
    }
}

/// Encodes a terminator tag (EOH, EOS, or EOO).
pub(crate) fn encode_terminator(enc: &mut entropy::Encoder, models: &mut ModelSet, tag: u8, output: &mut Vec<u8>) {
    debug_assert!(tag == TAG_EOH || tag == TAG_EOS || tag == TAG_EOO);
    enc.encode(tag, &mut models.tag, output);
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

/// Decodes one token from the entropy decoder (FORMAT.md Section 2).
pub(crate) fn decode_token(dec: &mut entropy::Decoder, models: &mut ModelSet, input: &mut &[u8]) -> DecodedToken {
    let tag = dec.decode(&mut models.tag, input);

    match tag {
        TAG_LINE => {
            let literal_len = dec.decode(&mut models.lit_len, input);
            let match_len = dec.decode(&mut models.match_len, input);

            let match_distance = if match_len > 0 {
                let lo = dec.decode(&mut models.dist_lo, input);
                let hi = dec.decode(&mut models.dist_hi, input);
                u16::from(lo) | (u16::from(hi) << 8)
            } else {
                0
            };

            let mut literals = [0u8; 255];
            for b in &mut literals[..literal_len as usize] {
                *b = dec.decode(&mut models.literals, input);
            }

            DecodedToken::Line(Token {
                seq: Sequence {
                    literal_len,
                    match_len,
                    match_distance,
                },
                literals,
            })
        }
        TAG_EOH => DecodedToken::EndOfHaiku,
        TAG_EOS => DecodedToken::EndOfSonnet,
        TAG_EOO => DecodedToken::EndOfOpus,
        _ => unreachable!("tag model only has 4 symbols"),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::lz77;

    /// Encode a LINE token then decode it, verifying roundtrip.
    fn roundtrip_line(token: &Token) -> Token {
        let mut enc = entropy::Encoder::new();
        let mut enc_models = ModelSet::new();
        let mut compressed = Vec::new();

        encode_line(&mut enc, &mut enc_models, token, &mut compressed);
        enc.finalize_sonnet(&mut compressed);

        let mut dec = entropy::Decoder::new();
        let mut dec_models = ModelSet::new();
        let mut input: &[u8] = &compressed;

        match decode_token(&mut dec, &mut dec_models, &mut input) {
            DecodedToken::Line(t) => t,
            _ => panic!("expected Line token"),
        }
    }

    fn roundtrip_terminator(tag: u8) {
        let mut enc = entropy::Encoder::new();
        let mut enc_models = ModelSet::new();
        let mut compressed = Vec::new();

        encode_terminator(&mut enc, &mut enc_models, tag, &mut compressed);
        enc.finalize_sonnet(&mut compressed);

        let mut dec = entropy::Decoder::new();
        let mut dec_models = ModelSet::new();
        let mut input: &[u8] = &compressed;

        let decoded = decode_token(&mut dec, &mut dec_models, &mut input);
        match (tag, decoded) {
            (TAG_EOH, DecodedToken::EndOfHaiku)
            | (TAG_EOS, DecodedToken::EndOfSonnet)
            | (TAG_EOO, DecodedToken::EndOfOpus) => {}
            _ => panic!("tag mismatch"),
        }
    }

    #[test]
    fn terminators() {
        roundtrip_terminator(TAG_EOH);
        roundtrip_terminator(TAG_EOS);
        roundtrip_terminator(TAG_EOO);
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

        let decoded = roundtrip_line(&token);
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

        let decoded = roundtrip_line(&token);
        assert_eq!(token.seq, decoded.seq);
        assert_eq!(token.literals[..3], decoded.literals[..3]);
    }

    #[test]
    fn multi_token_roundtrip() {
        let data = crate::bench::lipsum_bytes(4096);
        let mut lz_enc = lz77::Encoder::new(lz77::DEFAULT_LEVEL);
        lz_enc.feed(&data);
        lz_enc.finish();

        let mut tokens = Vec::new();
        while let Some(token) = lz_enc.next() {
            tokens.push(token);
        }

        // Entropy encode all tokens.
        let mut enc = entropy::Encoder::new();
        let mut models = ModelSet::new();
        let mut compressed = Vec::new();
        for token in &tokens {
            encode_line(&mut enc, &mut models, token, &mut compressed);
        }
        // Encode EOS terminator.
        encode_terminator(&mut enc, &mut models, TAG_EOS, &mut compressed);
        enc.finalize_sonnet(&mut compressed);

        // Decode all tokens.
        let mut dec = entropy::Decoder::new();
        let mut dec_models = ModelSet::new();
        let mut input: &[u8] = &compressed;
        let mut decoded_tokens = Vec::new();

        loop {
            match decode_token(&mut dec, &mut dec_models, &mut input) {
                DecodedToken::Line(t) => decoded_tokens.push(t),
                DecodedToken::EndOfSonnet => break,
                other => panic!(
                    "unexpected token: {}",
                    match other {
                        DecodedToken::EndOfHaiku => "EOH",
                        DecodedToken::EndOfOpus => "EOO",
                        _ => "?",
                    }
                ),
            }
        }

        // Verify sequence roundtrip.
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
        fn roundtrip_random_line(
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

            let decoded = roundtrip_line(&token);
            prop_assert_eq!(token.seq, decoded.seq);
            let len = token.seq.literal_len as usize;
            prop_assert_eq!(&token.literals[..len], &decoded.literals[..len]);
        }
    }
}
