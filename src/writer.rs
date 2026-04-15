//! Append-only compressed writer for the LZR format.
//!
//! [`Writer`] implements [`std::io::Write`]. Each `write()` call feeds raw
//! bytes through the LZ77 → entropy coding pipeline, writing encoded data
//! incrementally to the destination. Sonnets (256 KiB blocks) are automatically
//! managed with proactive boundary detection.
//!
//! Call [`Write::flush`] to create a Haiku boundary (durability point).
//! Call [`Writer::seal`] to finalize and close the Opus.

use std::io::{self, Read, Seek, Write};

use crate::adler32::Adler32;
use crate::codec::{self, ModelSet, TAG_EOH, TAG_EOO, TAG_EOS};
use crate::entropy;
use crate::lz77;
use crate::sonnet::{self, Footer, INVOCATION, INVOCATION_LEN, MAX_FOOTER_SIZE, SONNET_SIZE};

/// Upper bound: bytes per literal symbol in worst case (14 bits → 2 bytes).
const PER_LIT_UPPER: usize = 2;

/// Upper bound: bytes for tag + `lit_len` + `match_len` + `dist_lo` + `dist_hi` (5 symbols x 2).
const MATCH_UPPER: usize = 10;

/// When remaining space drops below this, switch to capped literal mode.
const THRESHOLD: usize = 600;

/// Flush `encode_buf` to dest when it exceeds this size.
const FLUSH_THRESHOLD: usize = 4096;

/// Append-only compressed writer for the LZR format.
///
/// Wraps an inner [`Write`] and compresses data written to it. Sonnets of
/// 256 KiB are automatically managed. Use [`seal`](Self::seal) to finalize.
///
/// # Examples
///
/// ```no_run
/// use std::fs::File;
/// use std::io::Write;
///
/// let file = File::create("output.lzr").unwrap();
/// let mut w = lzr::Writer::new(file).unwrap();
/// w.write_all(b"Hello, world!\n").unwrap();
/// w.seal().unwrap();
/// ```
pub struct Writer<W: Write> {
    state: Option<WriterState<W>>,
}

impl<W: Write> std::fmt::Debug for Writer<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Writer").field("sealed", &self.state.is_none()).finish()
    }
}

struct WriterState<W: Write> {
    dest: W,
    sonnet_index: u64,

    /// Bytes written to dest within the current Sonnet.
    sonnet_offset: usize,
    /// Small buffer collecting entropy encoder output between flushes.
    encode_buf: Vec<u8>,

    // Compression pipeline.
    lz77_enc: lz77::Encoder,
    entropy_enc: entropy::Encoder,
    models: ModelSet,

    // Cumulative counters (NEVER reset — cumulative across the Opus).
    total_bytes: u64,
    total_lines: u64,
    checksum: Adler32,

    // Per-Sonnet counters (reset at Sonnet boundaries).
    sonnet_bytes: u64,
    sonnet_lines: u64,
}

impl<W: Write> Writer<W> {
    /// Creates a new LZR file at the default compression level (6).
    ///
    /// # Errors
    ///
    /// Returns an error if writing the header fails.
    pub fn new(dest: W) -> io::Result<Self> {
        Self::with_level(dest, lz77::DEFAULT_LEVEL)
    }

    /// Creates a new LZR file at the given compression level (1–9).
    ///
    /// Levels 1–4 control how many candidate positions are checked for
    /// matches (1 = fastest, 4 = best compression).
    ///
    /// # Errors
    ///
    /// Returns an error if writing the header fails.
    ///
    /// # Panics
    ///
    /// Panics if `level` is not in 1..=4.
    pub fn with_level(mut dest: W, level: u8) -> io::Result<Self> {
        dest.write_all(&INVOCATION)?;
        dest.flush()?;

        Ok(Self {
            state: Some(WriterState {
                dest,
                sonnet_index: 0,
                sonnet_offset: INVOCATION_LEN,
                encode_buf: Vec::with_capacity(FLUSH_THRESHOLD * 2),
                lz77_enc: lz77::Encoder::new(level),
                entropy_enc: entropy::Encoder::new(),
                models: ModelSet::new(),
                total_bytes: 0,
                total_lines: 0,
                checksum: Adler32::new(),
                sonnet_bytes: 0,
                sonnet_lines: 0,
            }),
        })
    }

    /// Seals the Opus: encodes the EOO sentinel, writes the Coda footer,
    /// and returns the inner writer.
    ///
    /// After sealing, no further writes are possible.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to the destination fails.
    pub fn seal(mut self) -> io::Result<W> {
        let mut st = self.state.take().ok_or_else(|| io::Error::other("writer already consumed"))?;

        // Flush all remaining LZ77 data.
        for token in st.lz77_enc.flush() {
            st.track_token(&token);
            codec::encode_line(&mut st.entropy_enc, &mut st.models, &token, &mut st.encode_buf);
        }

        // Encode EOO terminator and finalize (Kireji + byte-align).
        codec::encode_terminator(&mut st.entropy_enc, &mut st.models, TAG_EOO, &mut st.encode_buf);
        st.entropy_enc.finalize_sonnet(&mut st.encode_buf);

        // Write the Coda footer.
        st.write_footer_and_pad(true)?;

        Ok(st.dest)
    }
}

impl<W: Read + Write + Seek> Writer<W> {
    /// Opens an existing LZR file for appending (cold-start resume).
    ///
    /// Reads the file to reconstruct encoder state. Refuses to open sealed files.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is not a valid LZR file, is sealed, or
    /// if any I/O operation fails.
    #[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
    pub fn open(mut dest: W) -> io::Result<Self> {
        let file_size = dest.seek(io::SeekFrom::End(0))?;

        if file_size < INVOCATION_LEN as u64 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small for LZR header"));
        }

        // Verify Invocation.
        dest.seek(io::SeekFrom::Start(0))?;
        let mut magic = [0u8; INVOCATION_LEN];
        dest.read_exact(&mut magic)?;
        if magic != INVOCATION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not an LZR file: invalid magic"));
        }

        let complete_sonnets = file_size / SONNET_SIZE as u64;
        let partial_size = (file_size % SONNET_SIZE as u64) as usize;

        // Restore cumulative state from the last complete Sonnet's footer.
        let (mut total_bytes, mut total_lines, mut checksum) = if complete_sonnets > 0 {
            let last_offset = (complete_sonnets - 1) * SONNET_SIZE as u64;
            dest.seek(io::SeekFrom::Start(last_offset))?;
            let mut block_data = vec![0u8; SONNET_SIZE];
            dest.read_exact(&mut block_data)?;
            let footer = sonnet::decode_footer(&block_data);
            (
                footer.bytes_count,
                footer.lines_count,
                Adler32::from_checksum(footer.adler32, footer.bytes_count as usize),
            )
        } else {
            (0u64, 0u64, Adler32::new())
        };

        // Read the partial Sonnet data.
        let partial_offset = complete_sonnets * SONNET_SIZE as u64;
        let data_start = if complete_sonnets == 0 {
            INVOCATION_LEN
        } else {
            0
        };

        let mut models = ModelSet::new();
        let mut sonnet_bytes = 0u64;
        let mut sonnet_lines = 0u64;

        if partial_size > data_start {
            dest.seek(io::SeekFrom::Start(partial_offset + data_start as u64))?;
            let data_len = partial_size - data_start;
            let mut partial_data = vec![0u8; data_len];
            dest.read_exact(&mut partial_data)?;

            // Decode the partial Sonnet to restore model state and count bytes/lines.
            let mut dec = entropy::Decoder::new();
            let mut lz77_dec = lz77::Decoder::new();
            let mut input: &[u8] = &partial_data;
            let mut decoded_output = Vec::new();

            let max_tokens = partial_data.len() * 8;
            for _ in 0..max_tokens {
                if input.is_empty() && !dec.has_buffered_bits() {
                    break;
                }
                let token = codec::decode_token(&mut dec, &mut models, &mut input);
                match token {
                    codec::DecodedToken::Line(t) => {
                        let lits = &t.literals[..t.seq.literal_len as usize];
                        lz77_dec.decode(t.seq, lits, &mut decoded_output);

                        let seq_bytes = u64::from(t.seq.literal_len) + u64::from(t.seq.match_len);
                        let start = decoded_output.len() - seq_bytes as usize;
                        checksum.update(&decoded_output[start..]);

                        sonnet_bytes += seq_bytes;
                        sonnet_lines += count_lines(&decoded_output[start..]) as u64;
                    }
                    codec::DecodedToken::EndOfHaiku => {
                        dec.finalize_haiku(&mut input);
                    }
                    codec::DecodedToken::EndOfSonnet => {
                        // Should not happen in partial data — it would be a complete Sonnet.
                        break;
                    }
                    codec::DecodedToken::EndOfOpus => {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "file is sealed; cannot append"));
                    }
                }
            }

            total_bytes += sonnet_bytes;
            total_lines += sonnet_lines;
        }

        // Seek to end to append.
        dest.seek(io::SeekFrom::End(0))?;

        Ok(Self {
            state: Some(WriterState {
                dest,
                sonnet_index: complete_sonnets,
                sonnet_offset: partial_size,
                encode_buf: Vec::with_capacity(FLUSH_THRESHOLD * 2),
                lz77_enc: lz77::Encoder::new(lz77::DEFAULT_LEVEL),
                entropy_enc: entropy::Encoder::new(),
                models,
                total_bytes,
                total_lines,
                checksum,
                sonnet_bytes: 0,
                sonnet_lines: 0,
            }),
        })
    }
}

/// Counts newline bytes in `data`.
#[allow(clippy::naive_bytecount)]
fn count_lines(data: &[u8]) -> usize {
    data.iter().filter(|&&b| b == b'\n').count()
}

impl<W: Write> WriterState<W> {
    /// Total encoded bytes in the current Sonnet (on disk + in buffer).
    const fn sonnet_bytes_used(&self) -> usize {
        self.sonnet_offset + self.encode_buf.len()
    }

    /// Upper-bound bytes needed to finalize (terminator + Kireji).
    #[allow(clippy::cast_possible_truncation)]
    fn compute_tail_bytes(&self) -> usize {
        let terminator_cost = 2; // 1 tag symbol, worst case 2 bytes
        let pending = self.entropy_enc.pending_bits();
        let buffered = u64::from(self.entropy_enc.bits_in_buffer());
        // +14 for terminator symbol's potential E3 accumulation
        let kireji_bits = pending + 14 + 2 + 7 + buffered;
        let kireji_bytes = kireji_bits.div_ceil(8) as usize;
        terminator_cost + kireji_bytes
    }

    /// Tracks a token's contribution to counters and checksum.
    fn track_token(&mut self, token: &lz77::Token) {
        let lits = &token.literals[..token.seq.literal_len as usize];
        let match_bytes = u64::from(token.seq.match_len);
        let lit_bytes = u64::from(token.seq.literal_len);

        self.checksum.update(lits);
        self.sonnet_bytes += lit_bytes + match_bytes;
        self.sonnet_lines += count_lines(lits) as u64;
    }

    /// Flushes the encode buffer to dest (does NOT flush dest itself).
    fn flush_encode_buf(&mut self) -> io::Result<()> {
        if !self.encode_buf.is_empty() {
            self.dest.write_all(&self.encode_buf)?;
            self.sonnet_offset += self.encode_buf.len();
            self.encode_buf.clear();
        }
        Ok(())
    }

    /// Main encoding loop: pulls LZ77 tokens and entropy-encodes them,
    /// respecting the Sonnet boundary.
    #[allow(clippy::cast_possible_truncation)]
    fn encode_tokens(&mut self) -> io::Result<()> {
        loop {
            let tail = self.compute_tail_bytes();
            let remaining = SONNET_SIZE
                .saturating_sub(self.sonnet_bytes_used())
                .saturating_sub(MAX_FOOTER_SIZE)
                .saturating_sub(tail);

            if remaining == 0 {
                break;
            }

            let token = if remaining < THRESHOLD {
                let max_lits = remaining.saturating_sub(MATCH_UPPER) / PER_LIT_UPPER;
                let max_lits = max_lits.min(255) as u8;
                if max_lits == 0 {
                    break;
                }
                self.lz77_enc.next_capped(max_lits)
            } else {
                self.lz77_enc.next()
            };

            match token {
                Some(t) => {
                    self.track_token(&t);
                    codec::encode_line(&mut self.entropy_enc, &mut self.models, &t, &mut self.encode_buf);

                    if self.encode_buf.len() >= FLUSH_THRESHOLD {
                        self.flush_encode_buf()?;
                    }
                }
                None => break,
            }
        }
        Ok(())
    }

    /// Checks if the Sonnet is full and finalizes it if so.
    fn maybe_finalize_sonnet(&mut self) -> io::Result<()> {
        let tail = self.compute_tail_bytes();
        let remaining =
            SONNET_SIZE.saturating_sub(self.sonnet_bytes_used()).saturating_sub(MAX_FOOTER_SIZE).saturating_sub(tail);

        if remaining == 0 {
            self.finalize_sonnet()?;
        }
        Ok(())
    }

    /// Finalizes the current Sonnet: writes EOS, Kireji, padding, Couplet.
    #[allow(clippy::cast_possible_truncation)]
    fn finalize_sonnet(&mut self) -> io::Result<()> {
        // Drain remaining LZ77 literals that fit.
        loop {
            let tail = self.compute_tail_bytes();
            let rem = SONNET_SIZE
                .saturating_sub(self.sonnet_bytes_used())
                .saturating_sub(MAX_FOOTER_SIZE)
                .saturating_sub(tail);
            if rem == 0 {
                break;
            }
            let max_lits = (rem.saturating_sub(MATCH_UPPER) / PER_LIT_UPPER).min(255) as u8;
            if max_lits == 0 {
                break;
            }
            match self.lz77_enc.next_capped(max_lits) {
                Some(t) => {
                    self.track_token(&t);
                    codec::encode_line(&mut self.entropy_enc, &mut self.models, &t, &mut self.encode_buf);
                }
                None => break,
            }
        }

        // Drain any remaining pending literals.
        if let Some(t) = self.lz77_enc.drain() {
            // Check if it fits.
            let tail = self.compute_tail_bytes();
            let rem = SONNET_SIZE
                .saturating_sub(self.sonnet_bytes_used())
                .saturating_sub(MAX_FOOTER_SIZE)
                .saturating_sub(tail);
            let needed = t.seq.literal_len as usize * PER_LIT_UPPER + 6; // lit-only overhead
            if rem >= needed {
                self.track_token(&t);
                codec::encode_line(&mut self.entropy_enc, &mut self.models, &t, &mut self.encode_buf);
            }
            // If it doesn't fit, those literals will be in the unconsumed input.
        }

        // EOS tag + Kireji (byte-aligned for Sonnet boundary).
        codec::encode_terminator(&mut self.entropy_enc, &mut self.models, TAG_EOS, &mut self.encode_buf);
        self.entropy_enc.finalize_sonnet(&mut self.encode_buf);

        // Write footer and pad.
        self.write_footer_and_pad(false)?;

        // Carry unconsumed input to next Sonnet.
        let carry = self.lz77_enc.unconsumed_input();
        self.lz77_enc.reset();
        self.entropy_enc.reset();
        self.models.reset();
        self.sonnet_index += 1;
        self.sonnet_offset = 0;
        self.sonnet_bytes = 0;
        self.sonnet_lines = 0;

        if !carry.is_empty() {
            self.lz77_enc.feed(&carry);
        }

        Ok(())
    }

    /// Writes the encode buffer, padding, and footer to dest.
    fn write_footer_and_pad(&mut self, is_final: bool) -> io::Result<()> {
        // Flush any remaining encoded data.
        self.flush_encode_buf()?;

        let footer = Footer {
            bytes_count: self.total_bytes + self.sonnet_bytes,
            lines_count: self.total_lines + self.sonnet_lines,
            adler32: self.checksum.checksum(),
        };
        let footer_bytes = sonnet::encode_footer(&footer);

        if !is_final {
            // Non-final Sonnet (Couplet): pad to exactly SONNET_SIZE.
            let padding_len = SONNET_SIZE - self.sonnet_offset - footer_bytes.len();
            let padding = vec![0u8; padding_len];
            self.dest.write_all(&padding)?;
        }
        self.dest.write_all(&footer_bytes)?;

        self.dest.flush()?;

        // Update cumulative counters.
        self.total_bytes += self.sonnet_bytes;
        self.total_lines += self.sonnet_lines;

        Ok(())
    }
}

impl<W: Write> Write for Writer<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let st = self.state.as_mut().ok_or_else(|| io::Error::other("writer is sealed"))?;

        if buf.is_empty() {
            return Ok(0);
        }

        st.lz77_enc.feed(buf);
        st.encode_tokens()?;
        st.maybe_finalize_sonnet()?;

        // If the Sonnet was finalized and there's carry-over data, keep encoding.
        if st.lz77_enc.has_pending_literals() || st.sonnet_offset == 0 {
            st.encode_tokens()?;
        }

        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        let st = self.state.as_mut().ok_or_else(|| io::Error::other("writer is sealed"))?;

        // Flush all remaining LZ77 data.
        for token in st.lz77_enc.flush() {
            st.track_token(&token);
            codec::encode_line(&mut st.entropy_enc, &mut st.models, &token, &mut st.encode_buf);
        }

        // EOH tag + Kireji + trailing resync bytes.
        codec::encode_terminator(&mut st.entropy_enc, &mut st.models, TAG_EOH, &mut st.encode_buf);
        st.entropy_enc.finalize_haiku(&mut st.encode_buf);

        // Write to dest and flush to disk (durability point).
        st.flush_encode_buf()?;
        st.dest.flush()?;

        // Check if Sonnet is now full after the Haiku.
        st.maybe_finalize_sonnet()?;

        Ok(())
    }
}

impl<W: Write> Drop for Writer<W> {
    fn drop(&mut self) {
        if self.state.is_some() {
            let _ = self.flush();
        }
    }
}
