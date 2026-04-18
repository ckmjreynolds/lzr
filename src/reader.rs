//! Random-access decompression reader for the LZR format.
//!
//! [`Reader`] implements [`std::io::Read`] and [`std::io::Seek`], providing
//! transparent decompression of LZR files. Sonnet footers are cached lazily
//! during seeks, enabling efficient interpolation search for both byte offsets
//! and line numbers.

use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom};

use crate::adler32::Adler32;
use crate::codec;
use crate::error::{Error, Result};
use crate::sonnet::{self, Footer, INVOCATION, INVOCATION_LEN, SONNET_SIZE};

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
    sonnet_count: u64,
    is_sealed: bool,
    footer_cache: HashMap<u64, Footer>,
    total_bytes: u64,
    total_lines: u64,
    cached_sonnet: Option<CachedSonnet>,
    position: u64,
}

impl<R: Read + Seek> std::fmt::Debug for Reader<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reader")
            .field("file_size", &self.file_size)
            .field("sonnet_count", &self.sonnet_count)
            .field("is_sealed", &self.is_sealed)
            .field("total_bytes", &self.total_bytes)
            .field("total_lines", &self.total_lines)
            .field("position", &self.position)
            .finish_non_exhaustive()
    }
}

struct CachedSonnet {
    sonnet_index: u64,
    data: Vec<u8>,
    start_bytes: u64,
    start_lines: u64,
}

impl<R: Read + Seek> Reader<R> {
    /// Opens an LZR file for reading.
    ///
    /// Validates the Invocation header and determines total size by reading
    /// Sonnet footers.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidMagic`] or [`Error::UnsupportedVersion`] if the
    /// file is not a valid LZR file, or [`Error::Io`] if any I/O operation fails.
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(mut source: R) -> Result<Self> {
        source.seek(SeekFrom::Start(0))?;
        let mut magic = [0u8; INVOCATION_LEN];
        source.read_exact(&mut magic)?;
        if magic[..3] != INVOCATION[..3] {
            return Err(Error::InvalidMagic);
        }
        if magic[3] != INVOCATION[3] {
            return Err(Error::UnsupportedVersion(magic[3]));
        }

        let file_size = source.seek(SeekFrom::End(0))?;
        let sonnet_count = file_size / SONNET_SIZE as u64;
        let partial_size = (file_size % SONNET_SIZE as u64) as usize;

        let mut footer_cache = HashMap::new();
        let mut is_sealed = false;
        let mut total_bytes = 0u64;
        let mut total_lines = 0u64;

        if sonnet_count > 0 {
            let last_footer = read_footer_from(&mut source, sonnet_count - 1)?;
            total_bytes = last_footer.bytes_count;
            total_lines = last_footer.lines_count;
            footer_cache.insert(sonnet_count - 1, last_footer);
        }

        if partial_size > 0 {
            let partial_offset = sonnet_count * SONNET_SIZE as u64;
            source.seek(SeekFrom::Start(partial_offset))?;
            let mut partial_data = vec![0u8; partial_size];
            source.read_exact(&mut partial_data)?;

            let data_start = if sonnet_count == 0 {
                INVOCATION_LEN
            } else {
                0
            };
            if partial_size > data_start {
                let decoded = codec::decode_sonnet_data(&partial_data[data_start..])?;
                total_bytes += decoded.output.len() as u64;
                total_lines += crate::count_lines(&decoded.output) as u64;
                is_sealed = decoded.is_sealed;
            }
        }

        Ok(Self {
            source,
            file_size,
            sonnet_count,
            is_sealed,
            footer_cache,
            total_bytes,
            total_lines,
            cached_sonnet: None,
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
    /// # Panics
    ///
    /// Panics if the internal Sonnet cache is unexpectedly empty after caching.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LineOutOfRange`] if `line` is past the end of the file,
    /// or [`Error::Io`] if any I/O operation fails.
    #[allow(clippy::cast_possible_truncation)]
    pub fn seek_to_line(&mut self, line: u64) -> Result<u64> {
        if line > self.total_lines {
            return Err(Error::LineOutOfRange {
                line,
                total: self.total_lines,
            });
        }

        if line == 0 {
            self.position = 0;
            return Ok(0);
        }

        let total_sonnets = self.total_sonnet_count();
        let sonnet_idx = self.find_sonnet_by_line(line, total_sonnets)?;
        self.ensure_sonnet_cached(sonnet_idx)?;

        let cached = self.cached_sonnet.as_ref().expect("sonnet was just cached");
        let lines_before = cached.start_lines;
        let lines_to_skip = (line - lines_before) as usize;

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

    /// Total number of Sonnets including the partial last one.
    fn total_sonnet_count(&self) -> u64 {
        let has_partial = !self.file_size.is_multiple_of(SONNET_SIZE as u64);
        self.sonnet_count + u64::from(has_partial)
    }

    /// Reads a Sonnet's footer from disk and caches it.
    fn read_footer(&mut self, sonnet_idx: u64) -> io::Result<Footer> {
        if let Some(&footer) = self.footer_cache.get(&sonnet_idx) {
            return Ok(footer);
        }
        let footer = read_footer_from(&mut self.source, sonnet_idx)?;
        self.footer_cache.insert(sonnet_idx, footer);
        Ok(footer)
    }

    /// Gets the cumulative byte count at the END of `sonnet_idx`.
    fn sonnet_end_bytes(&mut self, sonnet_idx: u64) -> io::Result<u64> {
        if sonnet_idx < self.sonnet_count {
            let footer = self.read_footer(sonnet_idx)?;
            Ok(footer.bytes_count)
        } else {
            Ok(self.total_bytes)
        }
    }

    /// Gets the cumulative byte count at the START of `sonnet_idx`.
    fn sonnet_start_bytes(&mut self, sonnet_idx: u64) -> io::Result<u64> {
        if sonnet_idx == 0 {
            Ok(0)
        } else {
            self.sonnet_end_bytes(sonnet_idx - 1)
        }
    }

    /// Gets the cumulative line count at the END of `sonnet_idx`.
    fn sonnet_end_lines(&mut self, sonnet_idx: u64) -> io::Result<u64> {
        if sonnet_idx < self.sonnet_count {
            let footer = self.read_footer(sonnet_idx)?;
            Ok(footer.lines_count)
        } else {
            Ok(self.total_lines)
        }
    }

    /// Finds the Sonnet containing byte offset `target` using interpolation search.
    ///
    /// Callers guarantee `total_sonnets > 0` (the `read()` guard at line 449
    /// ensures `position < total_bytes`, which requires at least one Sonnet).
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]
    fn find_sonnet_by_byte(&mut self, target: u64, total_sonnets: u64) -> io::Result<u64> {
        debug_assert!(total_sonnets > 0);

        let mut lo = 0u64;
        let mut hi = total_sonnets - 1;

        if self.total_bytes > 0 {
            let estimate = ((target as f64 / self.total_bytes as f64) * total_sonnets as f64) as u64;
            lo = estimate.saturating_sub(1).min(hi);
            hi = (estimate + 1).min(hi);

            while lo > 0 && self.sonnet_start_bytes(lo)? > target {
                lo = lo.saturating_sub(lo / 2 + 1);
            }
            while hi < total_sonnets - 1 && self.sonnet_end_bytes(hi)? <= target {
                hi = (hi + hi / 2 + 1).min(total_sonnets - 1);
            }
        }

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let end = self.sonnet_end_bytes(mid)?;
            if end <= target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        Ok(lo)
    }

    /// Finds the Sonnet containing line `target_line` using binary search.
    fn find_sonnet_by_line(&mut self, target_line: u64, total_sonnets: u64) -> io::Result<u64> {
        if total_sonnets <= 1 {
            return Ok(0);
        }

        let mut lo = 0u64;
        let mut hi = total_sonnets - 1;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let end_lines = self.sonnet_end_lines(mid)?;
            if end_lines < target_line {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        Ok(lo)
    }

    /// Validates a complete Sonnet's Adler-32 against its Couplet/Coda footer.
    ///
    /// The footer's checksum is cumulative from the start of the Opus; for all
    /// but the first Sonnet we prime the checksum from the previous Sonnet's
    /// footer so we only hash this Sonnet's decompressed bytes (FORMAT.md §8).
    #[allow(clippy::cast_possible_truncation)]
    fn validate_sonnet_checksum(&mut self, sonnet_idx: u64, data: &[u8]) -> Result<()> {
        let this_footer = self.read_footer(sonnet_idx)?;
        let mut checksum = if sonnet_idx == 0 {
            Adler32::new()
        } else {
            let prev = self.read_footer(sonnet_idx - 1)?;
            Adler32::from_checksum(prev.adler32)
        };
        checksum.update(data);
        let computed = checksum.checksum();
        if computed != this_footer.adler32 {
            return Err(Error::ChecksumMismatch {
                expected: this_footer.adler32,
                computed,
            });
        }
        Ok(())
    }

    /// Ensures the Sonnet at `sonnet_idx` is decompressed and cached.
    #[allow(clippy::cast_possible_truncation)]
    fn ensure_sonnet_cached(&mut self, sonnet_idx: u64) -> io::Result<()> {
        if let Some(ref cached) = self.cached_sonnet {
            if cached.sonnet_index == sonnet_idx {
                return Ok(());
            }
        }

        let start_bytes = self.sonnet_start_bytes(sonnet_idx)?;
        let start_lines = if sonnet_idx == 0 {
            0
        } else {
            let prev_footer = self.read_footer(sonnet_idx - 1)?;
            prev_footer.lines_count
        };

        let sonnet_offset = sonnet_idx * SONNET_SIZE as u64;
        self.source.seek(SeekFrom::Start(sonnet_offset))?;

        let read_size = if sonnet_idx < self.sonnet_count {
            SONNET_SIZE
        } else {
            (self.file_size - sonnet_offset) as usize
        };

        let mut sonnet_data = vec![0u8; read_size];
        self.source.read_exact(&mut sonnet_data)?;

        let data_start = if sonnet_idx == 0 {
            INVOCATION_LEN
        } else {
            0
        };
        let data_end = if sonnet_idx < self.sonnet_count {
            let footer_len = sonnet_data[SONNET_SIZE - 1] as usize;
            SONNET_SIZE - footer_len
        } else {
            read_size
        };

        let data = codec::decode_sonnet_data(&sonnet_data[data_start..data_end]).map_err(io::Error::from)?.output;

        if sonnet_idx < self.sonnet_count {
            self.validate_sonnet_checksum(sonnet_idx, &data).map_err(io::Error::from)?;
        }

        self.cached_sonnet = Some(CachedSonnet {
            sonnet_index: sonnet_idx,
            data,
            start_bytes,
            start_lines,
        });

        Ok(())
    }
}

/// Reads a Sonnet footer from a source without caching.
fn read_footer_from<R: Read + Seek>(source: &mut R, sonnet_idx: u64) -> io::Result<Footer> {
    let offset = sonnet_idx * SONNET_SIZE as u64;
    source.seek(SeekFrom::Start(offset))?;
    let mut data = vec![0u8; SONNET_SIZE];
    source.read_exact(&mut data)?;
    sonnet::decode_footer(&data).map_err(io::Error::from)
}

impl<R: Read + Seek> Read for Reader<R> {
    #[allow(clippy::cast_possible_truncation)]
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.position >= self.total_bytes || buf.is_empty() {
            return Ok(0);
        }

        let total_sonnets = self.total_sonnet_count();
        let sonnet_idx = self.find_sonnet_by_byte(self.position, total_sonnets)?;
        self.ensure_sonnet_cached(sonnet_idx)?;

        let cached = self.cached_sonnet.as_ref().expect("sonnet was just cached");
        let offset_in_sonnet = (self.position - cached.start_bytes) as usize;

        if offset_in_sonnet >= cached.data.len() {
            return Ok(0);
        }

        let available = &cached.data[offset_in_sonnet..];
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
                        .ok_or(Error::NegativeSeek)
                        .map_err(io::Error::from)?
                }
            }
            SeekFrom::Current(offset) => {
                if offset >= 0 {
                    self.position.saturating_add(offset.cast_unsigned())
                } else {
                    self.position
                        .checked_sub(offset.unsigned_abs())
                        .ok_or(Error::NegativeSeek)
                        .map_err(io::Error::from)?
                }
            }
        };

        self.position = new_pos.min(self.total_bytes);
        Ok(self.position)
    }
}
