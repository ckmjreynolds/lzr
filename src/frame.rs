//! LZR frame encoding and decoding.

use std::io::{Read, Write};

use crate::error::Result;
use crate::nibble::{NibbleReader, NibbleWriter};

/// A single LZR frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Frame {
    /// Literal bytes (D=0, L>0).
    Literal(Vec<u8>),

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
    /// Encodes this frame into the nibble stream.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn encode<W: Write>(&self, writer: &mut NibbleWriter<W>) -> Result<()> {
        match self {
            Self::EndOfStream => {
                writer.write_ubleb8_u16(0)?;
                writer.write_ubleb8_u16(0)?;
            }
            Self::Literal(bytes) => {
                writer.write_ubleb8_u16(0)?;
                writer.write_ubleb8_u16(bytes.len() as u16)?;
                for &b in bytes {
                    writer.write_literal_byte(b)?;
                }
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
            let mut bytes = Vec::with_capacity(length as usize);
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
