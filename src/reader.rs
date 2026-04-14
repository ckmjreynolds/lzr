//! Random-access decompression reader for the LZR format.
//!
//! [`Reader`] implements [`std::io::Read`] and [`std::io::Seek`], providing
//! transparent decompression of LZR files. Block footers are cached lazily
//! during seeks, enabling efficient interpolation search for both byte offsets
//! and line numbers.

use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom};

use crate::block::{self, BLOCK_SIZE, Footer, MAGIC, MAGIC_LEN};
use crate::codec::{self, ModelSet, is_eoframe, is_seal};
use crate::entropy;
use crate::lz77;

/// Random-access decompression reader for the LZR format.
///
/// Implements [`Read`] and [`Seek`] over compressed LZR files. Additionally
/// provides [`seek_to_line`](Self::seek_to_line) for line-number-based
/// positioning.
///
/// # Examples
///
/// ```no_run
/// use std::fs::File;
/// use std::io::Read;
///
/// let file = File::open("input.lzr").unwrap();
/// let mut r = lzr::Reader::new(file).unwrap();
/// let mut contents = String::new();
/// r.read_to_string(&mut contents).unwrap();
/// ```
pub struct Reader<R: Read + Seek> {
    source: R,
    file_size: u64,
    block_count: u64,
    is_sealed: bool,
    footer_cache: HashMap<u64, Footer>,
    total_bytes: u64,
    total_lines: u64,
    cached_block: Option<CachedBlock>,
    position: u64,
}

impl<R: Read + Seek> std::fmt::Debug for Reader<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reader")
            .field("file_size", &self.file_size)
            .field("block_count", &self.block_count)
            .field("is_sealed", &self.is_sealed)
            .field("total_bytes", &self.total_bytes)
            .field("total_lines", &self.total_lines)
            .field("position", &self.position)
            .finish_non_exhaustive()
    }
}

struct CachedBlock {
    block_index: u64,
    data: Vec<u8>,
    start_bytes: u64,
    start_lines: u64,
}

impl<R: Read + Seek> Reader<R> {
    /// Opens an LZR file for reading.
    ///
    /// Validates the magic header and determines total size by reading the
    /// last block's footer (if sealed) or decoding the last partial block.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is not a valid LZR file or if any I/O
    /// operation fails.
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(mut source: R) -> io::Result<Self> {
        source.seek(SeekFrom::Start(0))?;
        let mut magic = [0u8; MAGIC_LEN];
        source.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not an LZR file: invalid magic"));
        }

        let file_size = source.seek(SeekFrom::End(0))?;
        let block_count = file_size / BLOCK_SIZE as u64;
        let partial_size = (file_size % BLOCK_SIZE as u64) as usize;

        let mut footer_cache = HashMap::new();
        let mut is_sealed = false;
        let mut total_bytes = 0u64;
        let mut total_lines = 0u64;

        if block_count > 0 {
            let last_footer = read_footer_from(&mut source, block_count - 1)?;
            total_bytes = last_footer.bytes_count;
            total_lines = last_footer.lines_count;
            footer_cache.insert(block_count - 1, last_footer);
        }

        if partial_size > 0 {
            let partial_offset = block_count * BLOCK_SIZE as u64;
            source.seek(SeekFrom::Start(partial_offset))?;
            let mut partial_data = vec![0u8; partial_size];
            source.read_exact(&mut partial_data)?;

            let data_start = if block_count == 0 {
                MAGIC_LEN
            } else {
                0
            };
            if partial_size > data_start {
                let (decoded_bytes, decoded_lines, sealed) = decode_partial_block(&partial_data[data_start..]);
                total_bytes += decoded_bytes;
                total_lines += decoded_lines;
                is_sealed = sealed;
            }
        }

        Ok(Self {
            source,
            file_size,
            block_count,
            is_sealed,
            footer_cache,
            total_bytes,
            total_lines,
            cached_block: None,
            position: 0,
        })
    }

    /// Returns the total uncompressed size in bytes.
    #[must_use]
    pub const fn len(&self) -> u64 {
        self.total_bytes
    }

    /// Returns `true` if the uncompressed content is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.total_bytes == 0
    }

    /// Returns the total number of newlines in the uncompressed content.
    #[must_use]
    pub const fn lines(&self) -> u64 {
        self.total_lines
    }

    /// Returns whether the file is sealed.
    #[must_use]
    pub const fn is_sealed(&self) -> bool {
        self.is_sealed
    }

    /// Seeks to the start of the given line number (0-indexed).
    ///
    /// After this call, subsequent `read()` calls return data starting from
    /// the first byte of the specified line.
    ///
    /// Returns the byte offset of the line start.
    ///
    /// # Errors
    ///
    /// Returns an error if the line number is out of range or if any I/O
    /// operation fails.
    ///
    /// # Panics
    ///
    /// Panics if the internal block cache is inconsistent (should not happen).
    #[allow(clippy::cast_possible_truncation)]
    pub fn seek_to_line(&mut self, line: u64) -> io::Result<u64> {
        if line > self.total_lines {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("line {line} out of range (file has {} lines)", self.total_lines),
            ));
        }

        if line == 0 {
            self.position = 0;
            return Ok(0);
        }

        let total_blocks = self.total_block_count();
        let block_idx = self.find_block_by_line(line, total_blocks)?;
        self.ensure_block_cached(block_idx)?;

        let cached = self.cached_block.as_ref().expect("block was just cached");
        let lines_before_block = cached.start_lines;
        let lines_to_skip = (line - lines_before_block) as usize;

        let mut lines_seen = 0usize;
        let mut byte_offset = 0usize;
        for (i, &b) in cached.data.iter().enumerate() {
            if b == b'\n' {
                lines_seen += 1;
                if lines_seen == lines_to_skip {
                    byte_offset = i + 1;
                    break;
                }
            }
        }

        self.position = cached.start_bytes + byte_offset as u64;
        Ok(self.position)
    }

    /// Total number of blocks including the partial last block.
    fn total_block_count(&self) -> u64 {
        let has_partial = !self.file_size.is_multiple_of(BLOCK_SIZE as u64);
        self.block_count + u64::from(has_partial)
    }

    /// Reads a block's footer from disk and caches it.
    fn read_footer(&mut self, block_idx: u64) -> io::Result<Footer> {
        if let Some(&footer) = self.footer_cache.get(&block_idx) {
            return Ok(footer);
        }
        let footer = read_footer_from(&mut self.source, block_idx)?;
        self.footer_cache.insert(block_idx, footer);
        Ok(footer)
    }

    /// Gets the cumulative byte count at the END of `block_idx`.
    fn block_end_bytes(&mut self, block_idx: u64) -> io::Result<u64> {
        if block_idx < self.block_count {
            let footer = self.read_footer(block_idx)?;
            Ok(footer.bytes_count)
        } else {
            Ok(self.total_bytes)
        }
    }

    /// Gets the cumulative byte count at the START of `block_idx`.
    fn block_start_bytes(&mut self, block_idx: u64) -> io::Result<u64> {
        if block_idx == 0 {
            Ok(0)
        } else {
            self.block_end_bytes(block_idx - 1)
        }
    }

    /// Gets the cumulative line count at the END of `block_idx`.
    fn block_end_lines(&mut self, block_idx: u64) -> io::Result<u64> {
        if block_idx < self.block_count {
            let footer = self.read_footer(block_idx)?;
            Ok(footer.lines_count)
        } else {
            Ok(self.total_lines)
        }
    }

    /// Finds the block containing byte offset `target` using interpolation search.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]
    fn find_block_by_byte(&mut self, target: u64, total_blocks: u64) -> io::Result<u64> {
        if total_blocks == 0 {
            return Ok(0);
        }

        let mut lo = 0u64;
        let mut hi = total_blocks - 1;

        if self.total_bytes > 0 {
            let estimate = ((target as f64 / self.total_bytes as f64) * total_blocks as f64) as u64;
            lo = estimate.saturating_sub(1).min(hi);
            hi = (estimate + 1).min(hi);

            while lo > 0 && self.block_start_bytes(lo)? > target {
                lo = lo.saturating_sub(lo / 2 + 1);
            }
            while hi < total_blocks - 1 && self.block_end_bytes(hi)? <= target {
                hi = (hi + hi / 2 + 1).min(total_blocks - 1);
            }
        }

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let end = self.block_end_bytes(mid)?;
            if end <= target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        Ok(lo)
    }

    /// Finds the block containing line `target_line` using interpolation search.
    fn find_block_by_line(&mut self, target_line: u64, total_blocks: u64) -> io::Result<u64> {
        if total_blocks <= 1 {
            return Ok(0);
        }

        let mut lo = 0u64;
        let mut hi = total_blocks - 1;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let end_lines = self.block_end_lines(mid)?;
            if end_lines < target_line {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        Ok(lo)
    }

    /// Ensures the block at `block_idx` is decompressed and cached.
    #[allow(clippy::cast_possible_truncation)]
    fn ensure_block_cached(&mut self, block_idx: u64) -> io::Result<()> {
        if let Some(ref cached) = self.cached_block {
            if cached.block_index == block_idx {
                return Ok(());
            }
        }

        let start_bytes = self.block_start_bytes(block_idx)?;
        let start_lines = if block_idx == 0 {
            0
        } else {
            let prev_footer = self.read_footer(block_idx - 1)?;
            prev_footer.lines_count
        };

        let block_offset = block_idx * BLOCK_SIZE as u64;
        self.source.seek(SeekFrom::Start(block_offset))?;

        let read_size = if block_idx < self.block_count {
            BLOCK_SIZE
        } else {
            (self.file_size - block_offset) as usize
        };

        let mut block_data = vec![0u8; read_size];
        self.source.read_exact(&mut block_data)?;

        let data_start = if block_idx == 0 {
            MAGIC_LEN
        } else {
            0
        };
        let data_end = if block_idx < self.block_count {
            let footer_len = block_data[BLOCK_SIZE - 1] as usize;
            BLOCK_SIZE - footer_len
        } else {
            read_size
        };

        let data = decompress_block_data(&block_data[data_start..data_end]);

        self.cached_block = Some(CachedBlock {
            block_index: block_idx,
            data,
            start_bytes,
            start_lines,
        });

        Ok(())
    }
}

/// Reads a block footer from a source without caching.
fn read_footer_from<R: Read + Seek>(source: &mut R, block_idx: u64) -> io::Result<Footer> {
    let block_offset = block_idx * BLOCK_SIZE as u64;
    source.seek(SeekFrom::Start(block_offset))?;
    let mut block_data = vec![0u8; BLOCK_SIZE];
    source.read_exact(&mut block_data)?;
    Ok(block::decode_footer(&block_data))
}

/// Decompresses a block's frame data into raw bytes.
///
/// Handles `EOFrame` markers (reset entropy decoder) and stops at SEAL or
/// when no progress is made.
fn decompress_block_data(data: &[u8]) -> Vec<u8> {
    let mut dec = entropy::Decoder::new();
    let mut models = ModelSet::new();
    let mut lz77_dec = lz77::Decoder::new();
    let mut input: &[u8] = data;
    let mut output = Vec::new();

    while !input.is_empty() {
        let remaining_before = input.len();
        let token = codec::decode_sequence(&mut dec, &mut models, &mut input);
        if input.len() == remaining_before {
            break;
        }
        if is_seal(token.seq) {
            break;
        }
        if is_eoframe(token.seq) {
            continue;
        }
        let lits = &token.literals[..token.seq.literal_len as usize];
        lz77_dec.decode(token.seq, lits, &mut output);
    }

    output
}

/// Decodes a partial block's data, returning `(bytes, lines, is_sealed)`.
#[allow(clippy::naive_bytecount)]
fn decode_partial_block(data: &[u8]) -> (u64, u64, bool) {
    let mut dec = entropy::Decoder::new();
    let mut models = ModelSet::new();
    let mut lz77_dec = lz77::Decoder::new();
    let mut input: &[u8] = data;
    let mut output = Vec::new();
    let mut sealed = false;

    while !input.is_empty() {
        let remaining_before = input.len();
        let token = codec::decode_sequence(&mut dec, &mut models, &mut input);
        if input.len() == remaining_before {
            break;
        }
        if is_seal(token.seq) {
            sealed = true;
            break;
        }
        if is_eoframe(token.seq) {
            continue;
        }
        let lits = &token.literals[..token.seq.literal_len as usize];
        lz77_dec.decode(token.seq, lits, &mut output);
    }

    let lines = output.iter().filter(|&&b| b == b'\n').count() as u64;
    (output.len() as u64, lines, sealed)
}

impl<R: Read + Seek> Read for Reader<R> {
    #[allow(clippy::cast_possible_truncation)]
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.position >= self.total_bytes || buf.is_empty() {
            return Ok(0);
        }

        let total_blocks = self.total_block_count();
        let block_idx = self.find_block_by_byte(self.position, total_blocks)?;
        self.ensure_block_cached(block_idx)?;

        let cached = self.cached_block.as_ref().expect("block was just cached");
        let offset_in_block = (self.position - cached.start_bytes) as usize;

        if offset_in_block >= cached.data.len() {
            return Ok(0);
        }

        let available = &cached.data[offset_in_block..];
        let to_copy = buf.len().min(available.len());
        buf[..to_copy].copy_from_slice(&available[..to_copy]);
        self.position += to_copy as u64;

        Ok(to_copy)
    }
}

impl<R: Read + Seek> Seek for Reader<R> {
    #[allow(clippy::cast_sign_loss)]
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(offset) => offset,
            SeekFrom::End(offset) => {
                if offset >= 0 {
                    self.total_bytes.saturating_add(offset.cast_unsigned())
                } else {
                    self.total_bytes
                        .checked_sub(offset.unsigned_abs())
                        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "seek to negative offset"))?
                }
            }
            SeekFrom::Current(offset) => {
                if offset >= 0 {
                    self.position.saturating_add(offset.cast_unsigned())
                } else {
                    self.position
                        .checked_sub(offset.unsigned_abs())
                        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "seek to negative offset"))?
                }
            }
        };

        self.position = new_pos.min(self.total_bytes);
        Ok(self.position)
    }
}
