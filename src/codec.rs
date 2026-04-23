//! Encode / decode loops gluing [`crate::model::ByteTransformer`] to
//! [`crate::ac`]. The first byte uses a uniform CDF (no context yet); every
//! subsequent byte is predicted by the model given its cache.
//!
//! Archive format:
//! ```text
//! bytes 0..4:   magic b"LZR1"
//! bytes 4..12:  u64 LE, original byte length
//! bytes 12..:   arithmetic-coded payload
//! ```

use std::io::{self, BufWriter, Read, Write};

use crate::ac::{Decoder, Encoder};
use crate::model::ByteTransformer;
use crate::probs::{logits_to_cdf, uniform_cdf};

/// Archive-file magic.
pub(crate) const MAGIC: &[u8; 4] = b"LZR1";

/// 12-byte archive header.
pub(crate) const HEADER_LEN: usize = 12;

/// Write archive header (magic + original length).
pub(crate) fn write_header<W: Write>(out: &mut W, original_len: u64) -> io::Result<()> {
    out.write_all(MAGIC)?;
    out.write_all(&original_len.to_le_bytes())?;
    Ok(())
}

/// Read archive header; return `original_len`.
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

/// Abstract source of 257-entry CDFs used by [`encode_bytes_with`] and
/// [`decode_bytes_with`]. Production uses [`TransformerProbs`]; tests use a
/// deterministic stub.
pub(crate) trait ProbSource {
    /// Return the CDF for the *next* symbol given the history observed so far
    /// (the model is updated by [`Self::advance`]).
    fn initial_cdf(&mut self) -> [u32; 257];
    /// Advance the model with the just-observed byte; return the CDF for the
    /// byte that follows.
    fn advance(&mut self, observed: u8) -> [u32; 257];
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
    fn initial_cdf(&mut self) -> [u32; 257] {
        uniform_cdf()
    }
    fn advance(&mut self, observed: u8) -> [u32; 257] {
        let logits = self.model.step(observed);
        logits_to_cdf(logits)
    }
}

/// Encode the entire `src` buffer into `out` (archive payload only — does
/// not write the header).
pub(crate) fn encode_payload<P, W>(src: &[u8], probs: &mut P, out: &mut W) -> io::Result<()>
where
    P: ProbSource,
    W: Write,
{
    let mut enc = Encoder::new(out);
    let mut cdf = probs.initial_cdf();
    for &b in src {
        enc.encode(&cdf, b)?;
        cdf = probs.advance(b);
    }
    enc.finish()?;
    Ok(())
}

/// Decode exactly `expected_len` bytes from `inp` using `probs`. Writes
/// decoded bytes to `out`.
pub(crate) fn decode_payload<P, R, W>(
    inp: &mut R,
    expected_len: u64,
    probs: &mut P,
    out: &mut W,
) -> io::Result<()>
where
    P: ProbSource,
    R: Read,
    W: Write,
{
    let mut dec = Decoder::new(inp)?;
    let mut cdf = probs.initial_cdf();
    for _ in 0..expected_len {
        let b = dec.decode(&cdf)?;
        out.write_all(&[b])?;
        cdf = probs.advance(b);
    }
    Ok(())
}

/// Encode a byte slice into a complete archive (header + payload).
pub(crate) fn encode_bytes<W: Write>(
    src: &[u8],
    probs: &mut impl ProbSource,
    out: W,
) -> io::Result<()> {
    let mut w = BufWriter::new(out);
    write_header(&mut w, src.len() as u64)?;
    encode_payload(src, probs, &mut w)?;
    w.flush()?;
    Ok(())
}

/// Decode a complete archive (header + payload) into a byte vec.
///
/// `expected_len as usize` can theoretically truncate on 32-bit platforms,
/// but enwik9 (1 GB) fits comfortably; we'd OOM long before hitting the
/// `u32::MAX` limit.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn decode_bytes<R: Read>(
    inp: &mut R,
    probs: &mut impl ProbSource,
) -> io::Result<Vec<u8>> {
    let expected_len = read_header(inp)?;
    let mut out = Vec::with_capacity(expected_len as usize);
    decode_payload(inp, expected_len, probs, &mut out)?;
    Ok(out)
}

/// A [`ProbSource`] that ignores its history and cycles through a fixed list
/// of CDFs. Used for integration testing without the transformer.
#[cfg(test)]
pub(crate) struct CyclingProbs {
    cdfs: Vec<[u32; 257]>,
    idx: usize,
}

#[cfg(test)]
impl CyclingProbs {
    pub(crate) fn new(cdfs: Vec<[u32; 257]>) -> Self {
        assert!(!cdfs.is_empty());
        Self { cdfs, idx: 0 }
    }
}

#[cfg(test)]
impl ProbSource for CyclingProbs {
    fn initial_cdf(&mut self) -> [u32; 257] {
        self.cdfs[0]
    }
    fn advance(&mut self, _b: u8) -> [u32; 257] {
        self.idx = (self.idx + 1) % self.cdfs.len();
        self.cdfs[self.idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ac::TOTAL;
    use rand::{Rng, SeedableRng};

    #[test]
    fn payload_roundtrip_uniform() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let src: Vec<u8> = (0..16 * 1024).map(|_| rng.random()).collect();
        let mut probs_e = CyclingProbs::new(vec![uniform_cdf()]);
        let mut buf = Vec::new();
        encode_payload(&src, &mut probs_e, &mut buf).unwrap();

        let mut cur = &buf[..];
        let mut probs_d = CyclingProbs::new(vec![uniform_cdf()]);
        let mut out = Vec::new();
        decode_payload(&mut cur, src.len() as u64, &mut probs_d, &mut out).unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn archive_roundtrip_uniform() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let src: Vec<u8> = (0..4 * 1024).map(|_| rng.random()).collect();

        let mut probs_e = CyclingProbs::new(vec![uniform_cdf()]);
        let mut archive = Vec::new();
        encode_bytes(&src, &mut probs_e, &mut archive).unwrap();

        let mut probs_d = CyclingProbs::new(vec![uniform_cdf()]);
        let mut cur = &archive[..];
        let out = decode_bytes(&mut cur, &mut probs_d).unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn archive_roundtrip_cycling_skewed() {
        // Build several distinct skewed CDFs and cycle through them.
        let mut cdfs = Vec::new();
        for mode in [0u8, 97, 200] {
            let mut cdf = [0u32; 257];
            let mut acc = 0u32;
            for (j, slot) in cdf.iter_mut().enumerate().take(256) {
                let freq = if j == mode as usize { TOTAL - 255 } else { 1 };
                *slot = acc;
                acc += freq;
            }
            cdf[256] = TOTAL;
            cdfs.push(cdf);
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(23);
        let src: Vec<u8> = (0..2 * 1024).map(|_| rng.random()).collect();

        let mut probs_e = CyclingProbs::new(cdfs.clone());
        let mut archive = Vec::new();
        encode_bytes(&src, &mut probs_e, &mut archive).unwrap();

        let mut probs_d = CyclingProbs::new(cdfs);
        let mut cur = &archive[..];
        let out = decode_bytes(&mut cur, &mut probs_d).unwrap();
        assert_eq!(out, src);
    }
}
