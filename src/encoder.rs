//! LZR stream encoder.

use std::io::{self, Read, Write};

use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

use crate::adler32::Adler32;
use crate::nibble::NibbleWriter;
use crate::ring::RingBuffer;

/// Encode an input stream into an LZR compressed stream.
///
/// # Errors
///
/// Returns an I/O error if reading from `input` or writing to `output` fails.
pub fn encode(mut input: impl Read, mut output: impl Write) -> io::Result<()> {
    output.write_all(crate::HEADER)?;

    let mut ring = RingBuffer::new();
    let mut combined_checksum = Adler32::new();

    loop {
        let bytes_read = ring.read_batch(&mut input)?;
        if bytes_read == 0 {
            break;
        }

        let ranges = ring.block_ranges(bytes_read);

        // Dispatch blocks to rayon, collect results in order.
        let results: Vec<(Vec<u8>, Adler32)> =
            ranges.par_iter().map(|&(start, len)| compress_block(&ring, start, len)).collect();

        // Write block data and combine checksums.
        for (data, checksum) in results {
            output.write_all(&data)?;
            combined_checksum = combined_checksum.combine(&checksum);
        }

        ring.advance(bytes_read);
    }

    output.write_all(&combined_checksum.checksum().to_le_bytes())?;
    Ok(())
}

/// Compress a single block from the ring buffer.
///
/// Takes the full ring buffer and a block range within it. For now only uses
/// the block slice (copies data + computes Adler-32), but the signature accepts
/// the full ring so future LZ77 can access the lookback window.
fn compress_block(ring: &RingBuffer, start: usize, len: usize) -> (Vec<u8>, Adler32) {
    let block = &ring[start..start + len];
    let mut checksum = Adler32::new();
    checksum.update(block);
    let mut nw = NibbleWriter::with_capacity(len);
    nw.write_bytes(block);
    (nw.into_bytes(), checksum)
}
