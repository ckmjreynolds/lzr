//! LZR frame encoding and decoding.

use std::io::{Read, Write};

use smallvec::SmallVec;

use crate::error::Result;
use crate::nibble::{NibbleReader, NibbleWriter};

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
