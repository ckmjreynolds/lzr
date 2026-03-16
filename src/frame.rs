//! LZR frame encoding and decoding.

use std::io::{Read, Write};

use smallvec::SmallVec;

use crate::error::Result;
use crate::nibble::{NibbleReader, NibbleWriter, sbleb8_i16_nibbles, ubleb8_u16_nibbles};

/// Inline capacity for literal frames. Literals up to this size avoid heap allocation.
const LITERAL_INLINE: usize = 32;

/// A single LZR frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Frame {
    /// Literal bytes (D=0, L>0).
    Literal(SmallVec<[u8; LITERAL_INLINE]>),

    /// Match copy (D>0): positive L = forward, negative L = reverse, L=0 = no-op.
    Match {
        /// Distance back into the output buffer (1 = most recent byte).
        distance: u16,
        /// Signed length: positive for forward copy, negative for reverse copy.
        length: i16,
    },

    /// End of stream (D=0, L=0).
    EndOfStream,
}

impl Frame {
    /// Creates a no-op frame that pads the nibble stream by 3 nibbles.
    ///
    /// D=8 (2 UBLEB8 nibbles) + L=0 (1 SBLEB8 nibble) = 3 nibbles total,
    /// which flips the byte-alignment parity.
    pub(crate) const fn noop() -> Self {
        Self::Match {
            distance: 8,
            length: 0,
        }
    }

    /// Returns the nibble savings of a match versus emitting the same bytes as literals.
    ///
    /// Each literal byte costs 2 nibbles in the stream. A match frame costs
    /// `UBLEB8(distance) + SBLEB8(length)` nibbles and replaces `|length|` literal bytes.
    ///
    /// - `> 0` — the match saves nibbles (net compression).
    /// - `= 0` — break-even: same nibble count as literals, but avoids literal
    ///   frame overhead, so still a win.
    /// - `< 0` — the match costs more nibbles than the literals it replaces.
    #[inline]
    #[allow(clippy::cast_possible_wrap)]
    pub(crate) fn match_gain(distance: u16, length: i16) -> isize {
        let literal_nibbles = 2 * length.unsigned_abs() as isize;
        let match_nibbles = ubleb8_u16_nibbles(distance) as isize + sbleb8_i16_nibbles(length) as isize;
        literal_nibbles - match_nibbles
    }

    /// Encodes a literal frame from a borrowed slice, avoiding allocation.
    ///
    /// This is the preferred encode path when the caller already has the bytes
    /// in a buffer (e.g. the encoder's read buffer).
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn encode_literal<W: Write>(bytes: &[u8], writer: &mut NibbleWriter<W>) -> Result<()> {
        writer.write_ubleb8_u16(0)?;
        writer.write_ubleb8_u16(bytes.len() as u16)?;
        for &b in bytes {
            writer.write_literal_byte(b)?;
        }
        Ok(())
    }

    /// Encodes this frame into the nibble stream.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn encode<W: Write>(&self, writer: &mut NibbleWriter<W>) -> Result<()> {
        match self {
            Self::EndOfStream => {
                writer.write_ubleb8_u16(0)?;
                writer.write_ubleb8_u16(0)?;
            }
            Self::Literal(bytes) => {
                Self::encode_literal(bytes, writer)?;
            }
            Self::Match {
                distance,
                length,
            } => {
                writer.write_ubleb8_u16(*distance)?;
                writer.write_sbleb8_i16(*length)?;
            }
        }
        Ok(())
    }

    /// Decodes the next frame from the nibble stream.
    pub(crate) fn decode<R: Read>(reader: &mut NibbleReader<R>) -> Result<Self> {
        let distance = reader.read_ubleb8_u16()?;
        if distance == 0 {
            let length = reader.read_ubleb8_u16()?;
            if length == 0 {
                return Ok(Self::EndOfStream);
            }
            let mut bytes = SmallVec::with_capacity(length as usize);
            for _ in 0..length {
                bytes.push(reader.read_literal_byte()?);
            }
            Ok(Self::Literal(bytes))
        } else {
            let length = reader.read_sbleb8_i16()?;
            if length == 0 {
                return Self::decode(reader);
            }
            Ok(Self::Match {
                distance,
                length,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use smallvec::SmallVec;

    use super::*;
    use crate::nibble::{NibbleReader, NibbleWriter};

    #[test]
    fn match_gain_1byte_short_distance() {
        // D=1..7 (1 nibble), L=1 (1 nibble) → 2 - 1 - 1 = 0 (break-even = win)
        assert_eq!(Frame::match_gain(1, 1), 0);
        assert_eq!(Frame::match_gain(7, 1), 0);
        // D=8 (2 nibbles), L=1 → 2 - 2 - 1 = -1 (loss)
        assert_eq!(Frame::match_gain(8, 1), -1);
    }

    #[test]
    fn match_gain_2byte_distances() {
        // L=2 (1 nibble SBLEB8: 2 is in -4..3 → wait, 2 is in range -4..3? No.
        // SBLEB8: 1 nibble = -4..=3. L=2 → 1 nibble.
        // D=1..7: 4 - 1 - 1 = 2
        assert_eq!(Frame::match_gain(1, 2), 2);
        assert_eq!(Frame::match_gain(7, 2), 2);
        // D=8..63: 4 - 2 - 1 = 1
        assert_eq!(Frame::match_gain(8, 2), 1);
        assert_eq!(Frame::match_gain(63, 2), 1);
        // D=64..511: 4 - 3 - 1 = 0 (break-even)
        assert_eq!(Frame::match_gain(64, 2), 0);
        assert_eq!(Frame::match_gain(511, 2), 0);
        // D=512+: loss
        assert_eq!(Frame::match_gain(512, 2), -1);
    }

    #[test]
    fn match_gain_3byte_full_range() {
        // L=3 (1 nibble), saves 6 literal nibbles
        // D=1..7: 6 - 1 - 1 = 4
        assert_eq!(Frame::match_gain(1, 3), 4);
        // D=4096..65535: 6 - 5 - 1 = 0 (break-even at max distance)
        assert_eq!(Frame::match_gain(4096, 3), 0);
        assert_eq!(Frame::match_gain(u16::MAX, 3), 0);
    }

    #[test]
    fn match_gain_length_boundary() {
        // L=3 costs 1 SBLEB8 nibble, L=4 costs 2 SBLEB8 nibbles
        // D=1: gain(L=3) = 6 - 1 - 1 = 4, gain(L=4) = 8 - 1 - 2 = 5
        assert_eq!(Frame::match_gain(1, 3), 4);
        assert_eq!(Frame::match_gain(1, 4), 5);
    }

    #[test]
    fn match_gain_reverse_matches() {
        // L=-1 (1 nibble SBLEB8), same gain as L=1
        assert_eq!(Frame::match_gain(1, -1), 0);
        assert_eq!(Frame::match_gain(7, -1), 0);
        assert_eq!(Frame::match_gain(8, -1), -1);

        // L=-4 (1 nibble), L=-5 (2 nibbles) — discontinuity
        // D=1, L=-4: 8 - 1 - 1 = 6
        assert_eq!(Frame::match_gain(1, -4), 6);
        // D=1, L=-5: 10 - 1 - 2 = 7
        assert_eq!(Frame::match_gain(1, -5), 7);
    }

    #[test]
    fn match_gain_symmetry() {
        // Forward and reverse of same absolute length should have same gain
        // when SBLEB8 nibble counts match.
        // L=1 and L=-1 both cost 1 nibble
        assert_eq!(Frame::match_gain(100, 1), Frame::match_gain(100, -1));
        // L=3 and L=-3 both cost 1 nibble
        assert_eq!(Frame::match_gain(100, 3), Frame::match_gain(100, -3));
        // L=4 costs 2 nibbles, L=-4 costs 1 nibble — asymmetric!
        assert!(Frame::match_gain(100, -4) > Frame::match_gain(100, 4));
    }

    fn frame_strategy() -> impl Strategy<Value = Frame> {
        prop_oneof![
            10 => prop::collection::vec(any::<u8>(), 1..=64usize)
                .prop_map(|v| Frame::Literal(SmallVec::from_vec(v))),
            90 => (1..=u16::MAX, any::<i16>())
                .prop_map(|(distance, length)| Frame::Match { distance, length }),
        ]
    }

    proptest! {
        #[test]
        fn round_trip_frame_sequence(frames in prop::collection::vec(frame_strategy(), 0..=32)) {
            let mut writer = NibbleWriter::new(Vec::new());

            // Expected: only frames the decoder will actually return
            // (excludes Match with length=0, which the decoder skips).
            let expected: Vec<&Frame> = frames.iter()
                .filter(|f| !matches!(f, Frame::Match { length: 0, .. }))
                .collect();

            for frame in &frames {
                frame.encode(&mut writer).unwrap();
            }

            // Pad to byte alignment before EOS if needed.
            if writer.has_pending() {
                Frame::noop().encode(&mut writer).unwrap();
            }

            Frame::EndOfStream.encode(&mut writer).unwrap();
            let buf = writer.finish().unwrap();

            let mut reader = NibbleReader::new(buf.as_slice());
            let mut decoded: Vec<Frame> = Vec::new();

            loop {
                match Frame::decode(&mut reader).unwrap() {
                    Frame::EndOfStream => break,
                    frame => decoded.push(frame),
                }
            }

            prop_assert_eq!(decoded.iter().collect::<Vec<_>>(), expected);
        }
    }
}
