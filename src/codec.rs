//! Encode / decode loops gluing [`crate::model::ByteTransformer`] to
//! [`crate::ac`] through a [`crate::tokenizer::Tokenizer`].
//!
//! The model predicts over a `VOCAB`-token alphabet (currently 4096 BPE
//! tokens trained on enwik9). The codec tokenizes input bytes, AC-encodes
//! the token stream, then decodes back to bytes via the tokenizer's
//! per-token vocabulary table.
//!
//! Archive format (12 bytes header + AC payload):
//! ```text
//! bytes 0..4:   magic b"LZR1"
//! bytes 4..12:  u64 LE, original byte length
//! bytes 12..:   arithmetic-coded token payload
//! ```
//!
//! The header records byte length, not token count, because byte length
//! is the user-visible truth — the decoder knows it has produced enough
//! output once detokenization yields exactly that many bytes.

use std::io::{self, BufWriter, Read, Write};

use crate::ac::{Decoder, Encoder};
use crate::arch::CDF_LEN;
use crate::model::ByteTransformer;
use crate::probs::{logits_to_cdf, uniform_cdf};
use crate::tokenizer::{Token, Tokenizer};

/// Archive-file magic.
pub(crate) const MAGIC: &[u8; 4] = b"LZR1";

/// 12-byte archive header.
pub(crate) const HEADER_LEN: usize = 12;

/// Write archive header (magic + original byte length).
pub(crate) fn write_header<W: Write>(out: &mut W, original_byte_len: u64) -> io::Result<()> {
    out.write_all(MAGIC)?;
    out.write_all(&original_byte_len.to_le_bytes())?;
    Ok(())
}

/// Read archive header; return `original_byte_len`.
pub(crate) fn read_header<R: Read>(inp: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; HEADER_LEN];
    inp.read_exact(&mut buf)?;
    if &buf[..4] != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bad archive magic",
        ));
    }
    Ok(u64::from_le_bytes(buf[4..12].try_into().unwrap()))
}

/// Abstract source of CDFs used by [`encode_payload`] and [`decode_payload`].
/// Production uses [`TransformerProbs`]; tests use a deterministic stub.
pub(crate) trait ProbSource {
    /// Return the CDF for the *first* token (the model has no history yet).
    fn initial_cdf(&mut self) -> [u32; CDF_LEN];
    /// Advance the model with the just-observed token; return the CDF for
    /// the token that follows.
    fn advance(&mut self, observed: Token) -> [u32; CDF_LEN];
}

/// Transformer-backed prob source.
pub(crate) struct TransformerProbs<'a> {
    model: &'a mut ByteTransformer,
}

impl<'a> TransformerProbs<'a> {
    pub(crate) fn new(model: &'a mut ByteTransformer) -> Self {
        model.reset();
        Self { model }
    }
}

impl ProbSource for TransformerProbs<'_> {
    fn initial_cdf(&mut self) -> [u32; CDF_LEN] {
        uniform_cdf()
    }
    fn advance(&mut self, observed: Token) -> [u32; CDF_LEN] {
        let logits = self.model.step(observed);
        logits_to_cdf(logits)
    }
}

/// Encode a token sequence into `out` (archive payload only — does not
/// write the header).
pub(crate) fn encode_payload<P, W>(tokens: &[Token], probs: &mut P, out: &mut W) -> io::Result<()>
where
    P: ProbSource + ?Sized,
    W: Write,
{
    let mut enc = Encoder::new(out);
    let mut cdf = probs.initial_cdf();
    for &t in tokens {
        enc.encode(&cdf, t)?;
        cdf = probs.advance(t);
    }
    enc.finish()?;
    Ok(())
}

/// Decode tokens from `inp` until the detokenized output reaches
/// `expected_byte_len`. Writes detokenized bytes to `out`.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn decode_payload<P, R, W>(
    inp: &mut R,
    expected_byte_len: u64,
    tokenizer: &Tokenizer,
    probs: &mut P,
    out: &mut W,
) -> io::Result<()>
where
    P: ProbSource + ?Sized,
    R: Read,
    W: Write,
{
    let mut dec = Decoder::new(inp)?;
    let mut cdf = probs.initial_cdf();
    let mut produced: u64 = 0;
    while produced < expected_byte_len {
        let t = dec.decode(&cdf)?;
        let bytes = tokenizer.token_bytes(t);
        let take = (expected_byte_len - produced).min(bytes.len() as u64) as usize;
        out.write_all(&bytes[..take])?;
        produced += take as u64;
        cdf = probs.advance(t);
    }
    Ok(())
}

/// Tokenize `src`, encode into a complete archive (header + payload).
pub(crate) fn encode_bytes<W: Write>(
    src: &[u8],
    tokenizer: &Tokenizer,
    probs: &mut (impl ProbSource + ?Sized),
    out: W,
) -> io::Result<()> {
    let tokens = tokenizer.encode(src);
    let mut w = BufWriter::new(out);
    write_header(&mut w, src.len() as u64)?;
    encode_payload(&tokens, probs, &mut w)?;
    w.flush()?;
    Ok(())
}

/// Decode a complete archive (header + payload) into a byte vec.
///
/// `expected_byte_len as usize` can theoretically truncate on 32-bit
/// platforms, but enwik9 (1 GB) fits comfortably; we'd OOM long before
/// hitting the `u32::MAX` limit.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn decode_bytes<R: Read>(
    inp: &mut R,
    tokenizer: &Tokenizer,
    probs: &mut (impl ProbSource + ?Sized),
) -> io::Result<Vec<u8>> {
    let expected_byte_len = read_header(inp)?;
    let mut out = Vec::with_capacity(expected_byte_len as usize);
    decode_payload(inp, expected_byte_len, tokenizer, probs, &mut out)?;
    Ok(out)
}

/// A [`ProbSource`] that ignores its history and cycles through a fixed list
/// of CDFs. Used for integration testing without the transformer.
#[cfg(test)]
pub(crate) struct CyclingProbs {
    cdfs: Vec<[u32; CDF_LEN]>,
    idx: usize,
}

#[cfg(test)]
impl CyclingProbs {
    pub(crate) fn new(cdfs: Vec<[u32; CDF_LEN]>) -> Self {
        assert!(!cdfs.is_empty());
        Self { cdfs, idx: 0 }
    }
}

#[cfg(test)]
impl ProbSource for CyclingProbs {
    fn initial_cdf(&mut self) -> [u32; CDF_LEN] {
        self.cdfs[0]
    }
    fn advance(&mut self, _t: Token) -> [u32; CDF_LEN] {
        self.idx = (self.idx + 1) % self.cdfs.len();
        self.cdfs[self.idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    /// Build a no-merges tokenizer (every byte is its own token, ids 0..256).
    fn identity_tokenizer() -> Tokenizer {
        use crate::tokenizer::SCHEMA_VERSION;
        let mut buf = Vec::new();
        buf.extend_from_slice(&SCHEMA_VERSION.to_le_bytes());
        buf.extend_from_slice(&256u32.to_le_bytes()); // vocab_size
        buf.extend_from_slice(&0u32.to_le_bytes()); // num_merges
        Tokenizer::from_bytes(&buf).unwrap()
    }

    #[test]
    fn payload_roundtrip_uniform() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let tokens: Vec<Token> = (0..16 * 1024).map(|_| rng.random_range(0..256)).collect();
        let mut probs_e = CyclingProbs::new(vec![uniform_cdf()]);
        let mut buf = Vec::new();
        encode_payload(&tokens, &mut probs_e, &mut buf).unwrap();

        let mut cur = &buf[..];
        let mut probs_d = CyclingProbs::new(vec![uniform_cdf()]);
        let mut dec = Decoder::new(&mut cur).unwrap();
        let mut out: Vec<Token> = Vec::with_capacity(tokens.len());
        let mut cdf = probs_d.initial_cdf();
        for _ in 0..tokens.len() {
            let t = dec.decode(&cdf).unwrap();
            out.push(t);
            cdf = probs_d.advance(t);
        }
        assert_eq!(out, tokens);
    }

    #[test]
    fn archive_roundtrip_uniform_bytes() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let src: Vec<u8> = (0..4 * 1024).map(|_| rng.random()).collect();
        let tokenizer = identity_tokenizer();

        let mut probs_e = CyclingProbs::new(vec![uniform_cdf()]);
        let mut archive = Vec::new();
        encode_bytes(&src, &tokenizer, &mut probs_e, &mut archive).unwrap();

        let mut probs_d = CyclingProbs::new(vec![uniform_cdf()]);
        let mut cur = &archive[..];
        let out = decode_bytes(&mut cur, &tokenizer, &mut probs_d).unwrap();
        assert_eq!(out, src);
    }
}
