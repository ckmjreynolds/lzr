//! LZR frame encoding and decoding.

use std::io::{Read, Write};

use smallvec::SmallVec;

use crate::error::{Error, Result};

/// Inline capacity for literal frames. Literals up to this size avoid heap allocation.
const LITERAL_INLINE: usize = 32;

/// Maximum magnitude for Short frames before extension (4-bit field, offset 0).
pub(crate) const SHORT_BASE_MAX: u32 = 15;

/// Maximum magnitude for Medium forward frames before extension (4-bit field, offset 2).
pub(crate) const MEDIUM_FWD_BASE_MAX: u32 = 17;

/// Maximum magnitude for Medium reverse frames before extension (3-bit effective field, offset 2).
pub(crate) const MEDIUM_REV_BASE_MAX: u32 = 9;

/// Maximum magnitude for Long frames before extension (4-bit field, offset 3).
pub(crate) const LONG_BASE_MAX: u32 = 18;

/// Returns the number of extension bytes needed for `extra` additional magnitude.
const fn ext_byte_count(extra: u32) -> usize {
    (extra / 255 + 1) as usize
}

/// Writes an extension chain if `magnitude >= base_max`.
#[allow(clippy::cast_possible_truncation)]
fn encode_extension<W: Write>(magnitude: u32, base_max: u32, w: &mut W) -> Result<()> {
    if magnitude >= base_max {
        let mut extra = magnitude - base_max;
        while extra >= 255 {
            w.write_all(&[255])?;
            extra -= 255;
        }
        w.write_all(&[extra as u8])?;
    }
    Ok(())
}

/// Reads an extension chain if `base_magnitude == base_max`, adding to the magnitude.
#[inline]
pub(crate) fn decode_extension<R: Read>(base_magnitude: u32, base_max: u32, r: &mut R) -> Result<u32> {
    if base_magnitude < base_max {
        return Ok(base_magnitude);
    }
    let mut total = base_max;
    let mut buf = [0u8];
    loop {
        r.read_exact(&mut buf)?;
        total += u32::from(buf[0]);
        if buf[0] < 255 {
            break;
        }
    }
    Ok(total)
}

/// Returns `(cost, type_index)` for the cheapest match encoding.
///
/// Type indices: 0 = RLE (Short D=1), 1 = Medium, 2 = Long.
#[inline]
const fn cheapest_match_encoding(distance: u32, abs_len: u32, is_forward: bool) -> (usize, u8) {
    let mut min_cost = usize::MAX;
    let mut best_type = 2u8;

    // RLE: 1B, D=1, forward only, L≥1.
    if is_forward && distance == 1 {
        let ext = if abs_len >= SHORT_BASE_MAX {
            ext_byte_count(abs_len - SHORT_BASE_MAX)
        } else {
            0
        };
        let cost = 1 + ext;
        if cost < min_cost {
            min_cost = cost;
            best_type = 0;
        }
    }

    // Medium: 2B, D=1–2048, |L|≥2.
    if distance <= 2048 && abs_len >= 2 {
        let base_max = if is_forward {
            MEDIUM_FWD_BASE_MAX
        } else {
            MEDIUM_REV_BASE_MAX
        };
        let ext = if abs_len >= base_max {
            ext_byte_count(abs_len - base_max)
        } else {
            0
        };
        let cost = 2 + ext;
        if cost < min_cost {
            min_cost = cost;
            best_type = 1;
        }
    }

    // Long: 3B, D=1–65536, |L|≥3.
    if abs_len >= 3 {
        let ext = if abs_len >= LONG_BASE_MAX {
            ext_byte_count(abs_len - LONG_BASE_MAX)
        } else {
            0
        };
        let cost = 3 + ext;
        if cost < min_cost {
            min_cost = cost;
            best_type = 2;
        }
    }

    (min_cost, best_type)
}

/// A single LZR frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Frame {
    /// Literal bytes.
    Literal(SmallVec<[u8; LITERAL_INLINE]>),

    /// Match copy: positive length = forward, negative length = reverse.
    Match {
        /// Distance back into the output buffer (1–65536).
        distance: u32,
        /// Signed length: positive = forward, negative = reverse.
        length: i32,
    },

    /// End of stream.
    EndOfStream,
}

impl Frame {
    /// Returns the byte cost of encoding a match with the cheapest available frame type.
    #[inline]
    pub(crate) const fn match_cost(distance: u32, abs_len: u32, is_forward: bool) -> usize {
        cheapest_match_encoding(distance, abs_len, is_forward).0
    }

    /// Returns which frame type the encoder will use: 0=rle, 1=medium, 2=long.
    #[inline]
    pub(crate) const fn match_type(distance: u32, abs_len: u32, is_forward: bool) -> u8 {
        cheapest_match_encoding(distance, abs_len, is_forward).1
    }

    /// Returns the byte savings of a match versus emitting the same bytes as literals.
    #[inline]
    #[allow(clippy::cast_possible_wrap)]
    pub(crate) const fn match_gain(distance: u32, length: i32) -> isize {
        let abs_len = length.unsigned_abs();
        let is_forward = length > 0;
        let cost = Self::match_cost(distance, abs_len, is_forward);
        if cost == usize::MAX {
            return isize::MIN / 2;
        }
        abs_len as isize - cost.cast_signed()
    }

    /// Encodes this frame into the byte stream.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn encode<W: Write>(&self, w: &mut W) -> Result<()> {
        match self {
            Self::EndOfStream => {
                w.write_all(&[0xC0])?;
            }
            Self::Literal(bytes) => {
                debug_assert!(!bytes.is_empty(), "empty literal frame");
                Self::encode_literal(bytes, w)?;
            }
            Self::Match {
                distance,
                length,
            } => {
                encode_match(w, *distance, length.unsigned_abs(), *length > 0)?;
            }
        }
        Ok(())
    }

    /// Encodes a literal frame from a borrowed slice, avoiding allocation.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn encode_literal<W: Write>(bytes: &[u8], w: &mut W) -> Result<()> {
        debug_assert!(!bytes.is_empty(), "empty literal");
        let len = bytes.len() as u32;
        let field = len.min(SHORT_BASE_MAX) as u8;
        let header = 0xC0 | (field << 1);
        w.write_all(&[header])?;
        encode_extension(len, SHORT_BASE_MAX, w)?;
        w.write_all(bytes)?;
        Ok(())
    }

    /// Decodes the next frame from the byte stream.
    #[allow(clippy::cast_possible_wrap)]
    pub(crate) fn decode<R: Read>(r: &mut R) -> Result<Self> {
        let mut header = [0u8];
        r.read_exact(&mut header)?;
        let b = header[0];

        // EOS.
        if b == 0xC0 {
            return Ok(Self::EndOfStream);
        }

        // Short: 110LLLLD (0xC1–0xDF)
        if b & 0xE0 == 0xC0 {
            let l_field = u32::from((b >> 1) & 0x0F);
            let d_flag = b & 0x01;
            let magnitude = decode_extension(l_field, SHORT_BASE_MAX, r)?;

            if d_flag == 0 {
                // Literal.
                let len = magnitude as usize;
                let mut bytes = SmallVec::with_capacity(len);
                bytes.resize(len, 0);
                r.read_exact(&mut bytes)?;
                return Ok(Self::Literal(bytes));
            }

            // RLE (forward match, distance=1).
            if magnitude == 0 {
                return Err(Error::InvalidFormat("reserved Short frame (L=0, D=1)".into()));
            }
            return Ok(Self::Match {
                distance: 1,
                length: magnitude as i32,
            });
        }

        // Long: 111SLLLL (0xE0–0xFF)
        if b & 0xE0 == 0xE0 {
            let s = (b >> 4) & 1;
            let l_field = u32::from(b & 0x0F);
            let mut dist_buf = [0u8; 2];
            r.read_exact(&mut dist_buf)?;
            let distance = u32::from(u16::from_le_bytes(dist_buf)) + 1;
            let base_magnitude = l_field + 3;
            let magnitude = decode_extension(base_magnitude, LONG_BASE_MAX, r)?;
            let length = if s == 1 {
                -(magnitude as i32)
            } else {
                magnitude as i32
            };
            return Ok(Self::Match {
                distance,
                length,
            });
        }

        // Medium: SLLLLDDD (0x00–0xBF)
        let s = (b >> 7) & 1;
        let l_field = u32::from((b >> 3) & 0x0F);
        let d_low3 = u32::from(b & 0x07);
        let mut buf = [0u8];
        r.read_exact(&mut buf)?;
        let d_high8 = u32::from(buf[0]);
        let distance = (d_high8 << 3 | d_low3) + 1;
        let base_magnitude = l_field + 2;
        let base_max = if s == 0 {
            MEDIUM_FWD_BASE_MAX
        } else {
            MEDIUM_REV_BASE_MAX
        };
        let magnitude = decode_extension(base_magnitude, base_max, r)?;
        let length = if s == 1 {
            -(magnitude as i32)
        } else {
            magnitude as i32
        };
        Ok(Self::Match {
            distance,
            length,
        })
    }
}

/// Encodes a match frame, picking the cheapest frame type.
#[allow(clippy::cast_possible_truncation)]
fn encode_match<W: Write>(w: &mut W, dist: u32, abs_len: u32, is_forward: bool) -> Result<()> {
    let (_, frame_type) = cheapest_match_encoding(dist, abs_len, is_forward);

    match frame_type {
        0 => {
            // RLE (Short with D=1): 110LLLLD, D=1
            let field = abs_len.min(SHORT_BASE_MAX) as u8;
            let header = 0xC0 | (field << 1) | 1;
            w.write_all(&[header])?;
            encode_extension(abs_len, SHORT_BASE_MAX, w)?;
        }
        1 => {
            // Medium: SLLLLDDD | DDDDDDDD
            let s = u8::from(!is_forward);
            let base_max = if is_forward {
                MEDIUM_FWD_BASE_MAX
            } else {
                MEDIUM_REV_BASE_MAX
            };
            let field = (abs_len.min(base_max) - 2) as u8;
            let d_val = dist - 1;
            let d_low3 = (d_val & 0x07) as u8;
            let d_high8 = (d_val >> 3) as u8;
            let header = (s << 7) | (field << 3) | d_low3;
            w.write_all(&[header, d_high8])?;
            encode_extension(abs_len, base_max, w)?;
        }
        _ => {
            // Long: 111SLLLL | DDDDDDDD | DDDDDDDD
            let s = u8::from(!is_forward);
            let field = (abs_len.min(LONG_BASE_MAX) - 3) as u8;
            let header = 0xE0 | (s << 4) | field;
            w.write_all(&[header])?;
            w.write_all(&((dist - 1) as u16).to_le_bytes())?;
            encode_extension(abs_len, LONG_BASE_MAX, w)?;
        }
    }

    Ok(())
}

/// Encodes a `u64` value as ULEB128 (max 9 bytes).
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn encode_uleb128_u64<W: Write>(mut value: u64, w: &mut W) -> Result<()> {
    for _ in 0..8 {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            w.write_all(&[byte])?;
            return Ok(());
        }
        w.write_all(&[byte | 0x80])?;
    }
    // 9th byte: all 8 bits are payload.
    w.write_all(&[value as u8])?;
    Ok(())
}

/// Decodes a ULEB128-encoded `u64` (max 9 bytes).
pub(crate) fn decode_uleb128_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;
    let mut buf = [0u8];

    for _ in 0..8 {
        r.read_exact(&mut buf)?;
        value |= u64::from(buf[0] & 0x7F) << shift;
        if buf[0] & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
    }
    // 9th byte: all 8 bits are payload.
    r.read_exact(&mut buf)?;
    value |= u64::from(buf[0]) << shift;
    Ok(value)
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use std::io::Cursor;

    use proptest::prelude::*;
    use smallvec::SmallVec;

    use super::*;

    // ── Short literal ───────────────────────────────────────────

    #[test]
    fn short_literal_l1() {
        let frame = Frame::Literal(SmallVec::from_slice(&[0xAA]));
        let mut buf = Vec::new();
        frame.encode(&mut buf).unwrap();
        assert_eq!(buf, [0xC2, 0xAA]); // 0xC0 | (1<<1) = 0xC2
        let decoded = Frame::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn short_literal_l14() {
        let data: SmallVec<[u8; LITERAL_INLINE]> = (0..14).collect();
        let frame = Frame::Literal(data);
        let mut buf = Vec::new();
        frame.encode(&mut buf).unwrap();
        assert_eq!(buf[0], 0xDC); // 0xC0 | (14<<1)
        assert_eq!(buf.len(), 15); // 1 header + 14 data
        let decoded = Frame::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn short_literal_l15_ext() {
        let data: SmallVec<[u8; LITERAL_INLINE]> = (0..15).collect();
        let frame = Frame::Literal(data);
        let mut buf = Vec::new();
        frame.encode(&mut buf).unwrap();
        assert_eq!(buf[0], 0xDE); // 0xC0 | (15<<1)
        assert_eq!(buf[1], 0x00); // extension byte
        assert_eq!(buf.len(), 17); // 1 header + 1 ext + 15 data
        let decoded = Frame::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn short_literal_l270_ext() {
        let data: SmallVec<[u8; LITERAL_INLINE]> = (0..270u16).map(|i| i as u8).collect();
        let frame = Frame::Literal(data);
        let mut buf = Vec::new();
        frame.encode(&mut buf).unwrap();
        assert_eq!(buf[0], 0xDE); // field=15
        assert_eq!(buf[1], 255); // ext continue
        assert_eq!(buf[2], 0); // ext terminate (255+0 = 255 extra → 15+255 = 270)
        assert_eq!(buf.len(), 273); // 1 header + 2 ext + 270 data
        let decoded = Frame::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, frame);
    }

    // ── RLE (Short match D=1) ───────────────────────────────────

    #[test]
    fn rle_l1() {
        let frame = Frame::Match {
            distance: 1,
            length: 1,
        };
        let mut buf = Vec::new();
        frame.encode(&mut buf).unwrap();
        assert_eq!(buf, [0xC3]); // 0xC0 | (1<<1) | 1
        let decoded = Frame::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn rle_l2() {
        let frame = Frame::Match {
            distance: 1,
            length: 2,
        };
        let mut buf = Vec::new();
        frame.encode(&mut buf).unwrap();
        assert_eq!(buf, [0xC5]); // 0xC0 | (2<<1) | 1
        let decoded = Frame::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn rle_l14() {
        let frame = Frame::Match {
            distance: 1,
            length: 14,
        };
        let mut buf = Vec::new();
        frame.encode(&mut buf).unwrap();
        assert_eq!(buf, [0xDD]); // 0xC0 | (14<<1) | 1
        assert_eq!(buf.len(), 1);
        let decoded = Frame::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn rle_l16_ext() {
        let frame = Frame::Match {
            distance: 1,
            length: 16,
        };
        let mut buf = Vec::new();
        frame.encode(&mut buf).unwrap();
        assert_eq!(buf, [0xDF, 0x01]); // field=15, ext=1 → 15+1=16
        let decoded = Frame::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, frame);
    }

    // ── Medium match ────────────────────────────────────────────

    #[test]
    fn medium_d100_l2() {
        let frame = Frame::Match {
            distance: 100,
            length: 2,
        };
        let mut buf = Vec::new();
        frame.encode(&mut buf).unwrap();
        // s=0, field=0, d_val=99, d_low3=3, d_high8=12
        // header = (0<<7)|(0<<3)|3 = 0x03
        assert_eq!(buf, [0x03, 0x0C]);
        let decoded = Frame::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn medium_d256_l16() {
        let frame = Frame::Match {
            distance: 256,
            length: 16,
        };
        let mut buf = Vec::new();
        frame.encode(&mut buf).unwrap();
        // s=0, field=14, d_val=255, d_low3=7, d_high8=31
        // header = (0<<7)|(14<<3)|7 = 0x77
        assert_eq!(buf, [0x77, 0x1F]);
        assert_eq!(buf.len(), 2);
        let decoded = Frame::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn medium_d256_l17_ext() {
        let frame = Frame::Match {
            distance: 256,
            length: 17,
        };
        let mut buf = Vec::new();
        frame.encode(&mut buf).unwrap();
        assert_eq!(buf, [0x7F, 0x1F, 0x00]); // field=15, ext=0
        let decoded = Frame::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn medium_d256_l18_ext() {
        let frame = Frame::Match {
            distance: 256,
            length: 18,
        };
        let mut buf = Vec::new();
        frame.encode(&mut buf).unwrap();
        assert_eq!(buf, [0x7F, 0x1F, 0x01]); // field=15, ext=1
        let decoded = Frame::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn medium_d50_l5_reverse() {
        let frame = Frame::Match {
            distance: 50,
            length: -5,
        };
        let mut buf = Vec::new();
        frame.encode(&mut buf).unwrap();
        // s=1, field=3, d_val=49, d_low3=1, d_high8=6
        // header = (1<<7)|(3<<3)|1 = 0x99
        assert_eq!(buf, [0x99, 0x06]);
        assert_eq!(buf.len(), 2);
        let decoded = Frame::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn medium_d2048_l4() {
        let frame = Frame::Match {
            distance: 2048,
            length: 4,
        };
        let mut buf = Vec::new();
        frame.encode(&mut buf).unwrap();
        // s=0, field=2, d_val=2047=0x7FF, d_low3=7, d_high8=0xFF
        // header = (0<<7)|(2<<3)|7 = 0x17
        assert_eq!(buf, [0x17, 0xFF]);
        let decoded = Frame::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn medium_d2048_l17_ext() {
        let frame = Frame::Match {
            distance: 2048,
            length: 17,
        };
        let mut buf = Vec::new();
        frame.encode(&mut buf).unwrap();
        assert_eq!(buf, [0x7F, 0xFF, 0x00]); // field=15(max), ext=0
        let decoded = Frame::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn medium_d2048_l18_ext() {
        let frame = Frame::Match {
            distance: 2048,
            length: 18,
        };
        let mut buf = Vec::new();
        frame.encode(&mut buf).unwrap();
        assert_eq!(buf, [0x7F, 0xFF, 0x01]); // field=15, ext=1
        let decoded = Frame::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn medium_d1000_l3_reverse() {
        let frame = Frame::Match {
            distance: 1000,
            length: -3,
        };
        let mut buf = Vec::new();
        frame.encode(&mut buf).unwrap();
        // s=1, field=1, d_val=999=0x3E7, d_low3=7, d_high8=124=0x7C
        // header = (1<<7)|(1<<3)|7 = 0x8F
        assert_eq!(buf, [0x8F, 0x7C]);
        let decoded = Frame::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, frame);
    }

    // ── Long match ──────────────────────────────────────────────

    #[test]
    fn long_d65535_l17() {
        let frame = Frame::Match {
            distance: 65535,
            length: 17,
        };
        let mut buf = Vec::new();
        frame.encode(&mut buf).unwrap();
        assert_eq!(buf, [0xEE, 0xFE, 0xFF]); // field=14, dist-1=0xFFFE LE
        assert_eq!(buf.len(), 3);
        let decoded = Frame::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn long_d65536_l17() {
        let frame = Frame::Match {
            distance: 65536,
            length: 17,
        };
        let mut buf = Vec::new();
        frame.encode(&mut buf).unwrap();
        assert_eq!(buf, [0xEE, 0xFF, 0xFF]); // field=14, dist-1=0xFFFF LE
        assert_eq!(buf.len(), 3);
        let decoded = Frame::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn long_d65536_l18_ext() {
        let frame = Frame::Match {
            distance: 65536,
            length: 18,
        };
        let mut buf = Vec::new();
        frame.encode(&mut buf).unwrap();
        assert_eq!(buf, [0xEF, 0xFF, 0xFF, 0x00]); // field=15, ext=0
        let decoded = Frame::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn long_d65536_l19_ext() {
        let frame = Frame::Match {
            distance: 65536,
            length: 19,
        };
        let mut buf = Vec::new();
        frame.encode(&mut buf).unwrap();
        assert_eq!(buf, [0xEF, 0xFF, 0xFF, 0x01]); // field=15, ext=1
        let decoded = Frame::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, frame);
    }

    // ── EOS ─────────────────────────────────────────────────────

    #[test]
    fn encode_decode_eos() {
        let mut buf = Vec::new();
        Frame::EndOfStream.encode(&mut buf).unwrap();
        assert_eq!(buf, [0xC0]);
        let decoded = Frame::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded, Frame::EndOfStream);
    }

    // ── match_gain ──────────────────────────────────────────────

    #[test]
    fn match_gain_rle() {
        // D=1 forward: RLE 1B base.
        assert_eq!(Frame::match_gain(1, 1), 0); // 1 − 1
        assert_eq!(Frame::match_gain(1, 2), 1); // 2 − 1
        assert_eq!(Frame::match_gain(1, 14), 13); // 14 − 1
        assert_eq!(Frame::match_gain(1, 15), 13); // 15 − 2 (ext)
    }

    #[test]
    fn match_gain_medium() {
        assert_eq!(Frame::match_gain(100, 5), 3); // 5 − 2
        assert_eq!(Frame::match_gain(256, 16), 14); // 16 − 2
        assert_eq!(Frame::match_gain(1000, 3), 1); // 3 − 2
        assert_eq!(Frame::match_gain(2048, 4), 2); // 4 − 2
    }

    #[test]
    fn match_gain_long() {
        assert_eq!(Frame::match_gain(5000, 3), 0); // 3 − 3
        assert_eq!(Frame::match_gain(65536, 17), 14); // 17 − 3
    }

    #[test]
    fn match_gain_symmetry() {
        // Same gain for D>1 medium (both use 2B).
        assert_eq!(Frame::match_gain(100, 5), Frame::match_gain(100, -5));
        // Different gain at D=1: forward uses RLE (1B), reverse uses Medium (2B).
        assert_eq!(Frame::match_gain(1, 5), 4);
        assert_eq!(Frame::match_gain(1, -5), 3);
    }

    // ── ULEB128 ─────────────────────────────────────────────────

    #[test]
    fn uleb128_round_trip_small() {
        let mut buf = Vec::new();
        encode_uleb128_u64(0, &mut buf).unwrap();
        assert_eq!(buf, [0x00]);
        assert_eq!(decode_uleb128_u64(&mut Cursor::new(&buf)).unwrap(), 0);
    }

    #[test]
    fn uleb128_round_trip_127() {
        let mut buf = Vec::new();
        encode_uleb128_u64(127, &mut buf).unwrap();
        assert_eq!(buf, [0x7F]);
        assert_eq!(decode_uleb128_u64(&mut Cursor::new(&buf)).unwrap(), 127);
    }

    #[test]
    fn uleb128_round_trip_128() {
        let mut buf = Vec::new();
        encode_uleb128_u64(128, &mut buf).unwrap();
        assert_eq!(buf, [0x80, 0x01]);
        assert_eq!(decode_uleb128_u64(&mut Cursor::new(&buf)).unwrap(), 128);
    }

    #[test]
    fn uleb128_round_trip_max() {
        let mut buf = Vec::new();
        encode_uleb128_u64(u64::MAX, &mut buf).unwrap();
        assert_eq!(buf.len(), 9);
        assert_eq!(decode_uleb128_u64(&mut Cursor::new(&buf)).unwrap(), u64::MAX);
    }

    // ── Proptest ────────────────────────────────────────────────

    fn frame_strategy() -> impl Strategy<Value = Frame> {
        prop_oneof![
            10 => prop::collection::vec(any::<u8>(), 1..=500usize)
                .prop_map(|v| Frame::Literal(SmallVec::from_vec(v))),
            // RLE (forward, distance=1).
            15 => (1..=300i32)
                .prop_map(|length| Frame::Match { distance: 1, length }),
            // Medium forward.
            10 => (1..=2048u32, 2..=300i32)
                .prop_map(|(distance, length)| Frame::Match { distance, length }),
            // Medium reverse.
            10 => (1..=2048u32, 2..=300i32)
                .prop_map(|(distance, length)| Frame::Match { distance, length: -length }),
            // Long forward.
            10 => (2049..=65536u32, 3..=300i32)
                .prop_map(|(distance, length)| Frame::Match { distance, length }),
            // Long reverse.
            10 => (2049..=65536u32, 3..=300i32)
                .prop_map(|(distance, length)| Frame::Match { distance, length: -length }),
        ]
    }

    proptest! {
        #[test]
        fn round_trip_frame_sequence(frames in prop::collection::vec(frame_strategy(), 0..=32)) {
            let mut buf = Vec::new();
            for frame in &frames {
                frame.encode(&mut buf).unwrap();
            }
            Frame::EndOfStream.encode(&mut buf).unwrap();

            let mut cursor = Cursor::new(&buf);
            let mut decoded: Vec<Frame> = Vec::new();
            loop {
                match Frame::decode(&mut cursor).unwrap() {
                    Frame::EndOfStream => break,
                    frame => decoded.push(frame),
                }
            }

            prop_assert_eq!(&decoded, &frames);
        }

        #[test]
        fn round_trip_uleb128(value: u64) {
            let mut buf = Vec::new();
            encode_uleb128_u64(value, &mut buf).unwrap();
            let decoded = decode_uleb128_u64(&mut Cursor::new(&buf)).unwrap();
            prop_assert_eq!(decoded, value);
        }
    }
}
