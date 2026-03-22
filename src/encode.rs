//! LZR compression encoder.

use std::io::{BufRead, Write};
use std::ops::Range;

use rayon::prelude::*;

use crate::adler32::Adler32;
use crate::buffer::Buffer;
use crate::error::Result;
use crate::options::EncodeOptions;
use crate::uleb128::encode_uleb128_u64;
use crate::{BATCH_SIZE, BLOCK_SIZE, HEADER, WINDOW_SIZE};

/// Compresses data from `input` and writes it to `output`.
///
/// Input is read in 4 MiB batches. Each batch is split into 64 KiB blocks that
/// are compressed in parallel using Rayon. The first 64 KiB of each batch is
/// the sliding window (zeros initially, then the tail of the previous batch).
///
/// # Errors
///
/// Returns an error if an I/O error occurs while reading or writing.
///
/// # Examples
///
/// ```
/// use lzr::encode::encode;
/// use lzr::options::EncodeOptions;
///
/// let input = b"Hello, world!";
/// let mut compressed = Vec::new();
/// encode(&mut &input[..], &mut compressed, &EncodeOptions::new()).unwrap();
/// ```
pub fn encode(input: &mut impl BufRead, output: &mut impl Write, options: &EncodeOptions) -> Result<()> {
    // TODO: configure Rayon thread pool when options.get_threads() > 0.
    let _ = options;

    output.write_all(&HEADER)?;

    // Heap-allocated to avoid a 4 MiB stack frame.
    let mut buf = Buffer::<BATCH_SIZE>::new();
    let mut checksum = Adler32::new();
    let mut total_len: u64 = 0;

    loop {
        let data_len = read_batch(input, &mut buf)?;
        if data_len == 0 {
            break;
        }

        let blocks: Vec<Range<usize>> = (0..data_len)
            .step_by(BLOCK_SIZE)
            .map(|offset| {
                let start = WINDOW_SIZE + offset;
                let end = start + (data_len - offset).min(BLOCK_SIZE);
                start..end
            })
            .collect();

        let results: Vec<(Vec<u8>, Adler32)> =
            blocks.par_iter().map(|range| compress_block(&buf, range.clone(), options)).collect();

        for (compressed, block_checksum) in results {
            output.write_all(&compressed)?;
            checksum = checksum.combine(&block_checksum);
        }
        total_len += data_len as u64;

        // Slide window: copy last WINDOW_SIZE bytes to position 0 for the next batch.
        let src_start = WINDOW_SIZE + data_len - WINDOW_SIZE;
        buf.copy_within(src_start..src_start + WINDOW_SIZE, 0);
    }

    // EOS marker.
    output.write_all(&[0xC0])?;

    // Footer: uncompressed length + Adler-32 checksum.
    encode_uleb128_u64(total_len, output)?;
    output.write_all(&checksum.checksum().to_le_bytes())?;

    Ok(())
}

/// Fills the batch buffer starting at [`WINDOW_SIZE`], returning the number of bytes read.
fn read_batch(input: &mut impl BufRead, buf: &mut Buffer<BATCH_SIZE>) -> Result<usize> {
    let capacity = BATCH_SIZE - WINDOW_SIZE;
    let mut filled = 0;

    while filled < capacity {
        let (slice, _) = buf.slices_mut(WINDOW_SIZE + filled, capacity - filled);
        let n = input.read(slice)?;
        if n == 0 {
            break;
        }
        filled += n;
    }

    Ok(filled)
}

/// Compresses a single block from the batch buffer.
///
/// The block's data is `buf[range]`. The preceding [`WINDOW_SIZE`] bytes
/// (`buf[range.start - WINDOW_SIZE .. range.start]`) are the sliding window
/// context available for match references.
fn compress_block(buf: &Buffer<BATCH_SIZE>, range: Range<usize>, _options: &EncodeOptions) -> (Vec<u8>, Adler32) {
    let mut checksum = Adler32::new();
    let (s1, s2) = buf.slices(range.start, range.len());
    checksum.update(s1);
    if !s2.is_empty() {
        checksum.update(s2);
    }

    // TODO: LZ77 match finding and frame encoding.
    // Placeholder: emit all data as literal Short frames (up to 15 bytes each).
    let mut compressed = Vec::new();
    let mut pos = range.start;
    let end = range.end;

    while pos < end {
        let chunk_len = (end - pos).min(14);
        // Short literal frame: 110LLLLD, L = chunk_len, D = 0.
        #[allow(clippy::cast_possible_truncation)]
        let header = 0xC0 | ((chunk_len as u8) << 1);
        compressed.push(header);
        for i in 0..chunk_len {
            compressed.push(buf[pos + i]);
        }
        pos += chunk_len;
    }

    (compressed, checksum)
}
