//! LZR frame encoding and decoding.

#![allow(unused_variables, clippy::needless_pass_by_ref_mut)]

use std::io::{Read, Write};

use crate::error::Result;
use crate::nibble::{NibbleReader, NibbleWriter};

/// A single LZR frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Frame {
    /// Literal bytes (D=0, L>0).
    Literal(Vec<u8>),

    /// Forward copy: copy `length` bytes from `distance` back (D>0, L>0).
    ForwardMatch {
        /// Distance back into the output buffer (1 = most recent byte).
        distance: u16,
        /// Number of bytes to copy.
        length: u16,
    },

    /// Reverse copy: copy `length` bytes in reverse from `distance` back (D>0, L<0).
    ReverseMatch {
        /// Distance back into the output buffer (1 = most recent byte).
        distance: u16,
        /// Number of bytes to copy (absolute value of the negative length).
        length: u16,
    },

    /// End of stream (D=0, L=0).
    EndOfStream,
}

impl Frame {
    /// Encodes this frame into the nibble stream.
    pub(crate) fn encode<W: Write>(&self, writer: &mut NibbleWriter<W>) -> Result<()> {
        todo!()
    }

    /// Decodes the next frame from the nibble stream.
    pub(crate) fn decode<R: Read>(reader: &mut NibbleReader<R>) -> Result<Self> {
        todo!()
    }
}
