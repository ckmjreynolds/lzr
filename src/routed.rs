//! Type-routed codec.
//!
//! Each input byte is classified into one of three classes (lowercase
//! letter, uppercase letter, non-letter). The class index is encoded
//! via a type predictor; the byte itself is then encoded via the
//! per-class content predictor. Uppercase letters are folded to
//! lowercase before encoding, so the case bit lives entirely in the
//! type stream and the letter predictor only ever sees lowercase
//! bytes.
//!
//! All three predictors share a single AC stream and are invoked in
//! deterministic order (type, then byte) per input position. The
//! decoder mirrors this exactly. Each predictor maintains its own
//! adaptive state independently — there is no cross-stream context
//! at this stage. (Cross-stream conditioning is a stretch goal for
//! later iterations.)
//!
//! The placeholder configuration uses `Order0Adaptive` for the type
//! stream and `Order2Adaptive` for both content streams; the symbol
//! arm will be replaced with PPM-D next turn.

use std::collections::VecDeque;
use std::io::{self, BufWriter, Read, Write};

use crate::ac::{Decoder, Encoder};
use crate::arch::CDF_LEN;
use crate::codec::{ProbSource, read_header, write_header};
use crate::ppm::Ppm;
use crate::predict::{ByteClass, Order1Adaptive};
use crate::tokenizer::Token;

/// Max order for the symbol-arm PPM. Order sweep on the 5-offset
/// panel: 4→5: −0.016, 5→6: −0.012, 6→7: −0.015, 7→8: −0.005.
/// Elbow at 7–8; settling on 8 for the marginal win at acceptable
/// memory cost (small effective alphabet keeps sparse contexts cheap).
const NONLETTER_PPM_ORDER: usize = 8;

/// Max order for the letter-arm PPM. Sweep on the 5-offset panel:
/// 4→5: −0.79 (huge jump); 5→6: regresses by ~0.008 across non-letter
/// orders 4 and 8 (sparse-context escape cost > within-context
/// tightening). 5 is the operating point.
const LETTER_PPM_ORDER: usize = 5;

/// Bundle of predictors for type-routed encoding and decoding.
///
/// Type stream: order-1 (run-length structure within words).
/// Letter stream: PPM-D order-5 over the **mixed byte history**, so
/// letter contexts include surrounding markup (`</text>` → newline-ish
/// priors). Same for non-letter PPM-D order-8 — both arms share the
/// single byte-history buffer, advancing it on every observed byte
/// regardless of class. Letters are recorded in lowercased form so
/// PPM contexts canonicalize over case.
#[allow(clippy::struct_field_names)]
pub(crate) struct RoutedProbs {
    type_pred: Order1Adaptive,
    letter_pred: Ppm,
    nonletter_pred: Ppm,
    type_cdf: [u32; CDF_LEN],
    letter_cdf: [u32; CDF_LEN],
    nonletter_cdf: [u32; CDF_LEN],
    /// Last `MAX_PPM_ORDER` bytes seen, lowercase-folded for letters.
    /// Shared by both PPM arms.
    history: VecDeque<u8>,
    history_cap: usize,
}

impl RoutedProbs {
    pub(crate) fn new() -> Self {
        let letter_pred = Ppm::new(LETTER_PPM_ORDER);
        let nonletter_pred = Ppm::new(NONLETTER_PPM_ORDER);
        let history_cap = letter_pred.max_order().max(nonletter_pred.max_order());
        let mut s = Self {
            type_pred: Order1Adaptive::new(),
            letter_pred,
            nonletter_pred,
            type_cdf: [0; CDF_LEN],
            letter_cdf: [0; CDF_LEN],
            nonletter_cdf: [0; CDF_LEN],
            history: VecDeque::with_capacity(history_cap),
            history_cap,
        };
        s.refresh_cdfs();
        s
    }

    fn history_slice(&mut self) -> &[u8] {
        // VecDeque may be split across the ring buffer's wrap-around;
        // make_contiguous gives us a single slice for PPM lookup.
        self.history.make_contiguous()
    }

    fn push_history(&mut self, byte: u8) {
        if self.history.len() == self.history_cap {
            self.history.pop_front();
        }
        self.history.push_back(byte);
    }

    /// Snapshot the predictors' next-symbol CDFs into local state.
    pub(crate) fn refresh_cdfs(&mut self) {
        self.type_cdf = self.type_pred.initial_cdf();
        let hist = self.history_slice().to_vec();
        self.letter_cdf = self.letter_pred.build_cdf(&hist);
        self.nonletter_cdf = self.nonletter_pred.build_cdf(&hist);
    }

    /// Advance every sub-predictor through `bytes` without touching the
    /// AC. Used by the bench (and by the LZ-routed wrapper) to converge
    /// adaptive state before the measured slice.
    pub(crate) fn prewarm(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        for &b in bytes {
            self.observe(b);
        }
        self.refresh_cdfs();
    }

    /// Lowercase letters, leave non-letters unchanged. Used for
    /// canonical history entries.
    const fn fold_byte(byte: u8, class: ByteClass) -> u8 {
        match class {
            ByteClass::Upper => byte | 0x20,
            ByteClass::Lower | ByteClass::NonLetter => byte,
        }
    }

    /// Push `bytes` into the shared PPM history (case-folded) WITHOUT
    /// counting them in the PPM tables. Used by the LZ-routed wrapper
    /// after a match: the bytes appear in the output, so they should
    /// inform PPM context for the next literal — but they were already
    /// encoded via the match record, so PPM should not credit them as
    /// observations. CDFs refreshed once at the end against the new
    /// history.
    pub(crate) fn note_match_bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            let class = ByteClass::classify(b);
            let folded = Self::fold_byte(b, class);
            self.push_history(folded);
        }
        let hist = self.history_slice().to_vec();
        self.letter_cdf = self.letter_pred.build_cdf(&hist);
        self.nonletter_cdf = self.nonletter_pred.build_cdf(&hist);
    }

    /// Advance every sub-predictor through one byte without touching
    /// the AC. Records the (case-folded) byte in shared history.
    pub(crate) fn observe(&mut self, byte: u8) {
        let class = ByteClass::classify(byte);
        let _ = self.type_pred.advance(class.as_token());
        let folded = Self::fold_byte(byte, class);
        let hist = self.history_slice().to_vec();
        match class {
            ByteClass::Lower | ByteClass::Upper => {
                self.letter_pred.observe(&hist, folded);
            }
            ByteClass::NonLetter => {
                self.nonletter_pred.observe(&hist, folded);
            }
        }
        self.push_history(folded);
    }

    /// Routed CDF for the next byte under `class`. Reference into self,
    /// so callers should copy if they need to release the borrow before
    /// the next call.
    pub(crate) const fn cdf_for_class(&self, class: ByteClass) -> &[u32; CDF_LEN] {
        match class {
            ByteClass::Lower | ByteClass::Upper => &self.letter_cdf,
            ByteClass::NonLetter => &self.nonletter_cdf,
        }
    }

    /// Encode the type-class for `byte` and advance the type predictor.
    /// Returns the inferred class so the caller can dispatch the byte
    /// encoding (possibly through a mixed CDF).
    pub(crate) fn encode_class<W: Write>(
        &mut self,
        enc: &mut Encoder<W>,
        byte: u8,
    ) -> io::Result<ByteClass> {
        let class = ByteClass::classify(byte);
        enc.encode(&self.type_cdf, class.as_token())?;
        self.type_cdf = self.type_pred.advance(class.as_token());
        Ok(class)
    }

    /// Decode the next type-class symbol and advance the type predictor.
    pub(crate) fn decode_class<R: Read>(&mut self, dec: &mut Decoder<R>) -> io::Result<ByteClass> {
        let class_tok = dec.decode(&self.type_cdf)?;
        self.type_cdf = self.type_pred.advance(class_tok);
        Ok(ByteClass::from_token(class_tok))
    }

    /// Encode `byte` (case-folded for letters) under `class` using the
    /// given `byte_cdf` (which may be the routed predictor's CDF, or a
    /// mix of routed + neural). Updates the class predictor and shared
    /// history.
    pub(crate) fn encode_byte_value<W: Write>(
        &mut self,
        enc: &mut Encoder<W>,
        byte: u8,
        class: ByteClass,
        byte_cdf: &[u32; CDF_LEN],
    ) -> io::Result<()> {
        let folded = Self::fold_byte(byte, class);
        enc.encode(byte_cdf, Token::from(folded))?;
        self.update_after_byte(folded, class);
        Ok(())
    }

    /// Decode the next byte under `class` using the given `byte_cdf`.
    /// Returns the case-restored original byte.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn decode_byte_value<R: Read>(
        &mut self,
        dec: &mut Decoder<R>,
        class: ByteClass,
        byte_cdf: &[u32; CDF_LEN],
    ) -> io::Result<u8> {
        let tok = dec.decode(byte_cdf)?;
        let folded = tok as u8;
        let output = match class {
            ByteClass::Lower | ByteClass::NonLetter => folded,
            ByteClass::Upper => folded & 0x5F, // a-z → A-Z
        };
        self.update_after_byte(folded, class);
        Ok(output)
    }

    fn update_after_byte(&mut self, folded: u8, class: ByteClass) {
        let hist = self.history_slice().to_vec();
        match class {
            ByteClass::Lower | ByteClass::Upper => {
                self.letter_pred.observe(&hist, folded);
            }
            ByteClass::NonLetter => {
                self.nonletter_pred.observe(&hist, folded);
            }
        }
        self.push_history(folded);
        let hist = self.history_slice().to_vec();
        self.letter_cdf = self.letter_pred.build_cdf(&hist);
        self.nonletter_cdf = self.nonletter_pred.build_cdf(&hist);
    }

    /// Encode one byte through the routed AC pipeline (no neural mix).
    pub(crate) fn encode_byte<W: Write>(
        &mut self,
        enc: &mut Encoder<W>,
        byte: u8,
    ) -> io::Result<()> {
        let class = self.encode_class(enc, byte)?;
        let cdf = *self.cdf_for_class(class);
        self.encode_byte_value(enc, byte, class, &cdf)
    }

    /// Decode one byte from the routed AC pipeline (no neural mix).
    pub(crate) fn decode_byte<R: Read>(&mut self, dec: &mut Decoder<R>) -> io::Result<u8> {
        let class = self.decode_class(dec)?;
        let cdf = *self.cdf_for_class(class);
        self.decode_byte_value(dec, class, &cdf)
    }
}

impl Default for RoutedProbs {
    fn default() -> Self {
        Self::new()
    }
}

/// Encode `src` into a complete archive (header + payload) using
/// type-routed predictors.
pub(crate) fn encode_bytes<W: Write>(
    src: &[u8],
    routed: &mut RoutedProbs,
    out: W,
) -> io::Result<()> {
    let mut w = BufWriter::new(out);
    let len_u64 = u64::try_from(src.len()).expect("src.len fits in u64");
    write_header(&mut w, len_u64)?;
    let mut enc = Encoder::new(&mut w);
    for &byte in src {
        routed.encode_byte(&mut enc, byte)?;
    }
    enc.finish()?;
    w.flush()?;
    Ok(())
}

/// Decode a complete archive (header + payload) back to its bytes.
pub(crate) fn decode_bytes<R: Read>(inp: &mut R, routed: &mut RoutedProbs) -> io::Result<Vec<u8>> {
    let expected = read_header(inp)?;
    let cap = usize::try_from(expected).unwrap_or(usize::MAX);
    let mut out = Vec::with_capacity(cap);
    let mut dec = Decoder::new(inp)?;
    let mut produced: u64 = 0;
    while produced < expected {
        out.push(routed.decode_byte(&mut dec)?);
        produced += 1;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_short_mixed() {
        let src = b"The quick brown FOX jumps over 13 lazy dogs.\n  <tag/>";
        let mut probs_e = RoutedProbs::new();
        let mut archive = Vec::new();
        encode_bytes(src, &mut probs_e, &mut archive).unwrap();

        let mut probs_d = RoutedProbs::new();
        let mut cur = &archive[..];
        let decoded = decode_bytes(&mut cur, &mut probs_d).unwrap();
        assert_eq!(&decoded[..], src);
    }

    #[test]
    fn roundtrip_after_prewarm() {
        let warm = b"aaa bbb ccc DDD <eee> 123";
        let src = b"now MEASURE this slice.";

        let mut probs_e = RoutedProbs::new();
        probs_e.prewarm(warm);
        let mut archive = Vec::new();
        encode_bytes(src, &mut probs_e, &mut archive).unwrap();

        let mut probs_d = RoutedProbs::new();
        probs_d.prewarm(warm);
        let mut cur = &archive[..];
        let decoded = decode_bytes(&mut cur, &mut probs_d).unwrap();
        assert_eq!(&decoded[..], src);
    }
}
