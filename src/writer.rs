//! Append-only compressed writer for the LZR format.
//!
//! [`Writer`] implements [`std::io::Write`]. Each `write()` call feeds raw
//! bytes through the LZ77 → entropy coding pipeline, writing encoded data
//! incrementally to the destination. Sonnets (256 KiB blocks) are automatically
//! managed with proactive boundary detection.
//!
//! Call [`Write::flush`] to create a Haiku boundary (durability point).
//! Call [`Writer::seal`] to finalize and close the Opus.

use std::collections::VecDeque;
use std::io::{self, Read, Seek, Write};

use crate::adler32::Adler32;
use crate::codec::{self, ModelSet, TAG_EOH, TAG_EOO, TAG_EOS};
use crate::entropy;
use crate::error::{Error, Result};
use crate::lz77;
use crate::sonnet::{self, Footer, INVOCATION_LEN, MAX_FOOTER_SIZE, SONNET_SIZE, flags};

/// Upper bound: bytes per literal symbol in the entropy stream (14 bits → 2 bytes).
const ENTROPY_PER_LIT_UPPER: usize = 2;

/// Upper bound: bytes for tag + `lit_len` + `match_len` + `dist_lo` + `dist_hi` in the
/// entropy stream (5 symbols × 2 bytes).
const ENTROPY_MATCH_UPPER: usize = 10;

/// Bytes per literal in raw mode (exactly 1).
const RAW_PER_LIT_UPPER: usize = 1;

/// Bytes for tag + `lit_len` + `match_len` + `dist_lo` + `dist_hi` in raw mode.
const RAW_MATCH_UPPER: usize = 5;

/// When remaining capacity drops below this, switch to capped literal mode.
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

    /// Whether this Opus uses adaptive arithmetic coding. Set from the level's
    /// [`lz77::LevelConfig::entropy`] at construction; fixed for the Opus.
    entropy_on: bool,

    /// Bytes written to dest within the current Sonnet.
    sonnet_offset: usize,
    /// Small buffer collecting token bytes between flushes to `dest`.
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

    /// Input bytes fed but not yet reflected in `checksum` / `sonnet_bytes` /
    /// `sonnet_lines`. Each emitted token drains `literal_len + match_len`
    /// bytes from the front; remaining bytes naturally carry over when the
    /// encoder is reset at a Sonnet boundary, mirroring `lz77_enc`'s own
    /// `unconsumed_input`.
    pending: VecDeque<u8>,
    /// Reusable contiguous buffer for the bytes drained from `pending` per
    /// token; kept as a field so `Adler32::update` and `count_lines` can run
    /// over a flat slice without allocating per token.
    consume_buf: Vec<u8>,
}

impl<W: Write> Writer<W> {
    /// Creates a new LZR file at the default compression level
    /// ([`crate::DEFAULT_LEVEL`]).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if writing the header fails.
    pub fn new(dest: W) -> Result<Self> {
        Self::with_level(dest, lz77::DEFAULT_LEVEL)
    }

    /// Creates a new LZR file at the given compression level.
    ///
    /// Levels are in `1..=9`. Levels 1–3 write a raw (byte-aligned) token
    /// stream (FORMAT.md §2.4); levels 4–9 enable adaptive arithmetic coding
    /// (FORMAT.md §3). See [`crate::DEFAULT_LEVEL`] for the default.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if writing the header fails.
    ///
    /// # Panics
    ///
    /// Panics if `level` is not in `1..=9`.
    pub fn with_level(mut dest: W, level: u8) -> Result<Self> {
        let cfg = lz77::level_config(level);
        let flags_byte = if cfg.entropy {
            flags::ENTROPY
        } else {
            0
        };
        let invocation = sonnet::invocation(flags_byte);
        dest.write_all(&invocation)?;
        dest.flush()?;

        Ok(Self {
            state: Some(WriterState {
                dest,
                sonnet_index: 0,
                entropy_on: cfg.entropy,
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
                pending: VecDeque::new(),
                consume_buf: Vec::with_capacity(512),
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
    /// Returns [`Error::WriterClosed`] if the writer has already been sealed,
    /// or [`Error::Io`] if writing to the destination fails.
    pub fn seal(mut self) -> Result<W> {
        let mut st = self.state.take().ok_or(Error::WriterClosed)?;

        // Drain all remaining LZ77 data, finalizing Sonnets as needed.
        st.drain_lz77()?;

        // Encode EOO terminator. In entropy mode, also flush the Kireji.
        if st.entropy_on {
            codec::encode_terminator(&mut st.entropy_enc, &mut st.models, TAG_EOO, &mut st.encode_buf);
            st.entropy_enc.finalize_sonnet(&mut st.encode_buf);
        } else {
            codec::encode_terminator_raw(TAG_EOO, &mut st.encode_buf);
        }

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
    /// Returns [`Error::TooSmall`], [`Error::InvalidMagic`],
    /// [`Error::UnsupportedVersion`], [`Error::Sealed`], or [`Error::Io`] as
    /// appropriate.
    #[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
    pub fn open(mut dest: W) -> Result<Self> {
        let file_size = dest.seek(io::SeekFrom::End(0))?;

        if file_size < INVOCATION_LEN as u64 {
            return Err(Error::TooSmall);
        }

        // Verify Invocation.
        dest.seek(io::SeekFrom::Start(0))?;
        let mut header = [0u8; INVOCATION_LEN];
        dest.read_exact(&mut header)?;
        let entropy_on = sonnet::parse_invocation(header)?;

        let complete_sonnets = file_size / SONNET_SIZE as u64;
        let partial_size = (file_size % SONNET_SIZE as u64) as usize;

        // Restore cumulative state from the last complete Sonnet's footer.
        let (mut total_bytes, mut total_lines, mut checksum) = if complete_sonnets > 0 {
            let last_offset = (complete_sonnets - 1) * SONNET_SIZE as u64;
            dest.seek(io::SeekFrom::Start(last_offset))?;
            let mut block_data = vec![0u8; SONNET_SIZE];
            dest.read_exact(&mut block_data)?;
            let footer = sonnet::decode_footer(&block_data)?;
            (footer.bytes_count, footer.lines_count, Adler32::from_checksum(footer.adler32))
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

        if partial_size > data_start {
            dest.seek(io::SeekFrom::Start(partial_offset + data_start as u64))?;
            let data_len = partial_size - data_start;
            let mut partial_data = vec![0u8; data_len];
            dest.read_exact(&mut partial_data)?;

            let decoded = codec::decode_sonnet_data(&partial_data, entropy_on)?;
            if decoded.is_sealed {
                return Err(Error::Sealed);
            }
            models = decoded.models;
            checksum.update(&decoded.output);
            total_bytes += decoded.output.len() as u64;
            total_lines += crate::count_lines(&decoded.output) as u64;
        }

        // Seek to end to append.
        dest.seek(io::SeekFrom::End(0))?;

        // Pick a compression level in the same mode (entropy on/off) as the
        // file we're resuming. Levels are 1:1 with LevelConfig, so we pick the
        // default within the right mode.
        let resume_level = if entropy_on {
            lz77::DEFAULT_LEVEL
        } else {
            // L3: raw, lazy, widest-finder raw tier.
            3
        };

        Ok(Self {
            state: Some(WriterState {
                dest,
                sonnet_index: complete_sonnets,
                entropy_on,
                sonnet_offset: partial_size,
                encode_buf: Vec::with_capacity(FLUSH_THRESHOLD * 2),
                lz77_enc: lz77::Encoder::new(resume_level),
                entropy_enc: entropy::Encoder::new(),
                models,
                total_bytes,
                total_lines,
                checksum,
                sonnet_bytes: 0,
                sonnet_lines: 0,
                pending: VecDeque::new(),
                consume_buf: Vec::with_capacity(512),
            }),
        })
    }
}

impl<W: Write> WriterState<W> {
    /// Total encoded bytes in the current Sonnet (on disk + in buffer).
    const fn sonnet_bytes_used(&self) -> usize {
        self.sonnet_offset + self.encode_buf.len()
    }

    /// Per-literal upper bound in the current encoding mode.
    const fn per_lit_upper(&self) -> usize {
        if self.entropy_on {
            ENTROPY_PER_LIT_UPPER
        } else {
            RAW_PER_LIT_UPPER
        }
    }

    /// Per-match-header upper bound in the current encoding mode.
    const fn match_upper(&self) -> usize {
        if self.entropy_on {
            ENTROPY_MATCH_UPPER
        } else {
            RAW_MATCH_UPPER
        }
    }

    /// Upper-bound bytes needed to finalize (terminator + any trailing coder
    /// state). In raw mode this is just the 1-byte terminator. In entropy
    /// mode, add the Kireji plus whatever bits are buffered.
    #[allow(clippy::cast_possible_truncation)]
    fn compute_tail_bytes(&self) -> usize {
        if !self.entropy_on {
            return 1;
        }
        let terminator_cost = 2; // 1 tag symbol, worst case 2 bytes
        let pending = self.entropy_enc.pending_bits();
        let buffered = u64::from(self.entropy_enc.bits_in_buffer());
        // +14 for terminator symbol's potential E3 accumulation
        let kireji_bits = pending + 14 + 2 + 7 + buffered;
        let kireji_bytes = kireji_bits.div_ceil(8) as usize;
        terminator_cost + kireji_bytes
    }

    /// Bytes remaining in the current Sonnet before the footer and tail.
    fn remaining_capacity(&self) -> usize {
        let tail = self.compute_tail_bytes();
        SONNET_SIZE.saturating_sub(self.sonnet_bytes_used()).saturating_sub(MAX_FOOTER_SIZE).saturating_sub(tail)
    }

    /// Drains all pending LZ77 data, finalizing Sonnets as needed.
    fn drain_lz77(&mut self) -> io::Result<()> {
        self.lz77_enc.finish();
        loop {
            self.encode_tokens()?;
            if self.remaining_capacity() < self.match_upper() + self.per_lit_upper() {
                self.finalize_sonnet()?;
                // finalize_sonnet resets the encoder; re-finish so it can drain carry data.
                self.lz77_enc.finish();
            } else {
                break;
            }
        }
        Ok(())
    }

    /// Tracks a token's contribution to counters and checksum by draining the
    /// exact bytes it emits from the `pending` input buffer. Each LINE token
    /// consumes `literal_len + match_len` uncompressed bytes, in the same
    /// order they were fed via `Write::write`, so no decoder mirror is needed.
    fn track_token(&mut self, token: &lz77::Token) {
        let n = usize::from(token.seq.literal_len) + usize::from(token.seq.match_len);
        self.consume_buf.clear();
        self.consume_buf.extend(self.pending.drain(..n));
        self.checksum.update(&self.consume_buf);
        self.sonnet_bytes += n as u64;
        self.sonnet_lines += crate::count_lines(&self.consume_buf) as u64;
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

    /// Encodes a single LINE token through the mode-appropriate codec path.
    fn encode_line_buf(&mut self, token: &lz77::Token) {
        if self.entropy_on {
            codec::encode_line(&mut self.entropy_enc, &mut self.models, token, &mut self.encode_buf);
        } else {
            codec::encode_line_raw(token, &mut self.encode_buf);
        }
    }

    /// Encodes a terminator tag through the mode-appropriate codec path.
    fn encode_terminator_buf(&mut self, tag: u8) {
        if self.entropy_on {
            codec::encode_terminator(&mut self.entropy_enc, &mut self.models, tag, &mut self.encode_buf);
        } else {
            codec::encode_terminator_raw(tag, &mut self.encode_buf);
        }
    }

    /// Main encoding loop: pulls LZ77 tokens and entropy-encodes them,
    /// respecting the Sonnet boundary.
    #[allow(clippy::cast_possible_truncation)]
    fn encode_tokens(&mut self) -> io::Result<()> {
        loop {
            let remaining = self.remaining_capacity();

            if remaining == 0 {
                break;
            }

            let token = if remaining < THRESHOLD {
                let max_lits = remaining.saturating_sub(self.match_upper()) / self.per_lit_upper();
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
                    self.encode_line_buf(&t);

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
    ///
    /// Triggers when the remaining capacity is too small for even the smallest
    /// token (one literal, no match). Without this threshold, the encoder would
    /// stall with a few unusable bytes remaining.
    fn maybe_finalize_sonnet(&mut self) -> io::Result<()> {
        if self.remaining_capacity() < self.match_upper() + self.per_lit_upper() {
            self.finalize_sonnet()?;
        }
        Ok(())
    }

    /// Finalizes the current Sonnet: writes EOS (+ Kireji if entropy is on),
    /// writes padding + Couplet, and resets per-Sonnet state.
    fn finalize_sonnet(&mut self) -> io::Result<()> {
        self.encode_terminator_buf(TAG_EOS);
        if self.entropy_on {
            self.entropy_enc.finalize_sonnet(&mut self.encode_buf);
        }

        // Write footer and pad.
        self.write_footer_and_pad(false)?;

        // Carry unconsumed input to next Sonnet. The `pending` buffer already
        // holds exactly these bytes (they haven't been drained since no token
        // consumed them), so we only need to reset the LZ77 encoder and refeed.
        let carry = self.lz77_enc.unconsumed_input();
        debug_assert_eq!(self.pending.len(), carry.len());
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
        let st = self.state.as_mut().ok_or(Error::WriterClosed).map_err(io::Error::from)?;

        if buf.is_empty() {
            return Ok(0);
        }

        st.pending.extend(buf.iter().copied());
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
        let st = self.state.as_mut().ok_or(Error::WriterClosed).map_err(io::Error::from)?;

        // Flush all remaining LZ77 data.
        for token in st.lz77_enc.flush() {
            st.track_token(&token);
            st.encode_line_buf(&token);
        }

        // EOH tag; in entropy mode also emit Kireji + resync padding so the
        // decoder can align to the next Haiku.
        st.encode_terminator_buf(TAG_EOH);
        if st.entropy_on {
            st.entropy_enc.finalize_haiku(&mut st.encode_buf);
        }

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
