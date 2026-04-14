//! Append-only compressed writer for the LZR format.
//!
//! [`Writer`] implements [`std::io::Write`]. Each `write()` call feeds raw
//! bytes through the LZ77 → entropy coding pipeline into a 256 KiB block
//! buffer. Blocks are automatically finalized and flushed when full.
//!
//! Call [`Write::flush`] to create a durability point (frame boundary).
//! Call [`Writer::seal`] to finalize and close the file.

use std::collections::VecDeque;
use std::io::{self, Read, Seek, Write};

use crate::adler32::Adler32;
use crate::block::{self, BLOCK_SIZE, Footer, MAGIC, MAGIC_LEN};
use crate::codec::{self, ModelSet, encode_sequence, is_eoframe, is_seal};
use crate::entropy;
use crate::lz77;

/// Append-only compressed writer for the LZR format.
///
/// Wraps an inner [`Write`] and compresses data written to it. Blocks of
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
    block_index: u64,

    /// Block buffer: compressed output accumulates here until the block is full.
    block_buf: Vec<u8>,

    // Compression pipeline.
    lz77_enc: lz77::Encoder,
    entropy_enc: entropy::Encoder,
    models: ModelSet,

    // Cumulative counters (NEVER reset — all are cumulative across the file).
    total_bytes: u64,
    total_lines: u64,
    checksum: Adler32,

    // Shadow decoder for tracking committed sequences.
    shadow_dec: entropy::Decoder,
    shadow_models: ModelSet,
    shadow_read_pos: usize,

    // Rewind tracking.
    pending_input: VecDeque<u8>,
    /// `(input_byte_count, line_count)` per pending sequence.
    pending_seq_info: VecDeque<(usize, usize)>,
    committed_bytes_in_block: u64,
    committed_lines_in_block: u64,
}

impl<W: Write> Writer<W> {
    /// Creates a new LZR file, writing the magic header.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing the destination fails.
    pub fn new(mut dest: W) -> io::Result<Self> {
        let mut block_buf = Vec::with_capacity(BLOCK_SIZE);
        block_buf.extend_from_slice(&MAGIC);

        dest.flush()?;

        Ok(Self {
            state: Some(WriterState {
                dest,
                block_index: 0,
                block_buf,
                lz77_enc: lz77::Encoder::new(),
                entropy_enc: entropy::Encoder::new(),
                models: ModelSet::new(),
                total_bytes: 0,
                total_lines: 0,
                checksum: Adler32::new(),
                shadow_dec: entropy::Decoder::new(),
                shadow_models: ModelSet::new(),
                shadow_read_pos: MAGIC_LEN,
                pending_input: VecDeque::new(),
                pending_seq_info: VecDeque::new(),
                committed_bytes_in_block: 0,
                committed_lines_in_block: 0,
            }),
        })
    }

    /// Seals the file: encodes the SEAL sentinel, flushes the entropy coder,
    /// writes a footer, and returns the inner writer.
    ///
    /// After sealing, no further writes are possible. `Writer::open` refuses
    /// to open a sealed file.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to the destination fails.
    pub fn seal(mut self) -> io::Result<W> {
        let mut st = self.state.take().ok_or_else(|| io::Error::other("writer already consumed"))?;

        // Flush all remaining LZ77 data.
        for token in st.lz77_enc.flush() {
            st.encode_token(&token)?;
        }

        // Encode the SEAL sentinel.
        let seal_token = lz77::Token {
            seq: lz77::Sequence {
                literal_len: 0,
                match_len: 0,
                match_distance: codec::SEAL_DISTANCE,
            },
            literals: [0u8; 255],
        };
        encode_sequence(&mut st.entropy_enc, &mut st.models, &seal_token, &mut st.block_buf);

        // Flush entropy coder.
        st.entropy_enc.flush(&mut st.block_buf);

        // All pending sequences are now committed.
        st.commit_all_pending();

        // Write footer at end of current block data.
        let footer = Footer {
            bytes_count: st.total_bytes + st.committed_bytes_in_block,
            lines_count: st.total_lines + st.committed_lines_in_block,
            adler32: st.checksum.checksum(),
        };
        let footer_bytes = block::encode_footer(&footer);
        st.block_buf.extend_from_slice(&footer_bytes);

        // Flush everything to dest.
        st.dest.write_all(&st.block_buf)?;
        st.dest.flush()?;

        Ok(st.dest)
    }
}

impl<W: Read + Write + Seek> Writer<W> {
    /// Opens an existing LZR file for appending (cold-start resume).
    ///
    /// Reads the last incomplete block to reconstruct encoder state. Refuses
    /// to open sealed files.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is not a valid LZR file, is sealed, or
    /// if any I/O operation fails.
    #[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
    pub fn open(mut dest: W) -> io::Result<Self> {
        let file_size = dest.seek(io::SeekFrom::End(0))?;

        if file_size < MAGIC_LEN as u64 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small for LZR header"));
        }

        // Verify magic.
        dest.seek(io::SeekFrom::Start(0))?;
        let mut magic = [0u8; MAGIC_LEN];
        dest.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not an LZR file: invalid magic"));
        }

        let complete_blocks = file_size / BLOCK_SIZE as u64;
        let partial_size = (file_size % BLOCK_SIZE as u64) as usize;

        // Restore cumulative state from the last complete block's footer.
        let (mut total_bytes, mut total_lines, mut checksum) = if complete_blocks > 0 {
            let last_complete_offset = (complete_blocks - 1) * BLOCK_SIZE as u64;
            dest.seek(io::SeekFrom::Start(last_complete_offset))?;
            let mut block_data = vec![0u8; BLOCK_SIZE];
            dest.read_exact(&mut block_data)?;
            let footer = block::decode_footer(&block_data);
            (
                footer.bytes_count,
                footer.lines_count,
                Adler32::from_checksum(footer.adler32, footer.bytes_count as usize),
            )
        } else {
            (0u64, 0u64, Adler32::new())
        };

        // Read the last partial block (if any).
        let partial_offset = complete_blocks * BLOCK_SIZE as u64;
        let block_buf = if partial_size > 0 {
            dest.seek(io::SeekFrom::Start(partial_offset))?;
            let mut buf = vec![0u8; partial_size];
            dest.read_exact(&mut buf)?;
            buf
        } else if complete_blocks == 0 {
            let mut buf = Vec::with_capacity(BLOCK_SIZE);
            buf.extend_from_slice(&MAGIC);
            buf
        } else {
            Vec::with_capacity(BLOCK_SIZE)
        };

        // Check for seal in partial block.
        let data_start = if complete_blocks == 0 && partial_size > 0 {
            MAGIC_LEN
        } else {
            0
        };

        if partial_size > data_start {
            let data_region = &block_buf[data_start..];
            if check_sealed(data_region) {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "file is sealed; cannot append"));
            }
        }

        // Decode the partial block to reconstruct model state.
        let mut lz77_dec = lz77::Decoder::new();
        let mut entropy_dec = entropy::Decoder::new();
        let mut models = ModelSet::new();

        if partial_size > data_start {
            let data_region = &block_buf[data_start..];
            let mut input: &[u8] = data_region;
            let mut decoded_output = Vec::new();

            // Decode sequences until we run out of data.
            while !input.is_empty() {
                let remaining_before = input.len();
                let token = codec::decode_sequence(&mut entropy_dec, &mut models, &mut input);
                if input.len() == remaining_before {
                    break; // no progress — corrupted/torn data
                }
                if is_seal(token.seq) {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "file is sealed; cannot append"));
                }
                if is_eoframe(token.seq) {
                    continue;
                }
                let lits = &token.literals[..token.seq.literal_len as usize];
                lz77_dec.decode(token.seq, lits, &mut decoded_output);

                let seq_bytes = token.seq.literal_len as usize + token.seq.match_len as usize;
                let start = decoded_output.len() - seq_bytes;
                checksum.update(&decoded_output[start..]);

                total_bytes += seq_bytes as u64;
                total_lines += count_lines(&decoded_output[start..]) as u64;
            }
        }

        // Seek back to overwrite the partial block.
        dest.seek(io::SeekFrom::Start(partial_offset))?;

        let shadow_read_pos = block_buf.len();

        Ok(Self {
            state: Some(WriterState {
                dest,
                block_index: complete_blocks,
                block_buf,
                lz77_enc: lz77::Encoder::new(),
                entropy_enc: entropy::Encoder::new(),
                models,
                total_bytes,
                total_lines,
                checksum,
                shadow_dec: entropy::Decoder::new(),
                shadow_models: ModelSet::new(),
                shadow_read_pos,
                pending_input: VecDeque::new(),
                pending_seq_info: VecDeque::new(),
                committed_bytes_in_block: 0,
                committed_lines_in_block: 0,
            }),
        })
    }
}

/// Counts newline bytes in `data`.
#[allow(clippy::naive_bytecount)]
fn count_lines(data: &[u8]) -> usize {
    data.iter().filter(|&&b| b == b'\n').count()
}

/// Checks if a data region contains the SEAL sentinel by decoding sequences.
fn check_sealed(data: &[u8]) -> bool {
    if data.len() < 8 {
        return false;
    }

    let mut dec = entropy::Decoder::new();
    let mut models = ModelSet::new();
    let mut input: &[u8] = data;

    while !input.is_empty() {
        let remaining_before = input.len();
        let token = codec::decode_sequence(&mut dec, &mut models, &mut input);
        if input.len() == remaining_before {
            break;
        }
        if is_seal(token.seq) {
            return true;
        }
    }

    false
}

impl<W: Write> WriterState<W> {
    /// Processes a single LZ77 token through the entropy encoder and shadow
    /// decoder.
    fn encode_token(&mut self, token: &lz77::Token) -> io::Result<()> {
        let seq = &token.seq;
        let lits = &token.literals[..seq.literal_len as usize];

        let seq_lines = count_lines(lits);
        let seq_bytes = seq.literal_len as usize + seq.match_len as usize;

        self.checksum.update(lits);
        self.pending_seq_info.push_back((seq_bytes, seq_lines));

        encode_sequence(&mut self.entropy_enc, &mut self.models, token, &mut self.block_buf);

        self.advance_shadow_decoder();

        // Check if block is full.
        let ftr_size = block::footer_size(
            self.total_bytes + self.committed_bytes_in_block,
            self.total_lines + self.committed_lines_in_block,
        );
        if self.block_buf.len() + ftr_size >= BLOCK_SIZE {
            self.finalize_block()?;
        }

        Ok(())
    }

    /// Runs the shadow decoder on any new bytes in `block_buf` to retire
    /// committed sequences.
    fn advance_shadow_decoder(&mut self) {
        while self.shadow_read_pos < self.block_buf.len() && !self.pending_seq_info.is_empty() {
            let available = &self.block_buf[self.shadow_read_pos..];
            let mut input: &[u8] = available;
            let before_len = input.len();

            let _token = codec::decode_sequence(&mut self.shadow_dec, &mut self.shadow_models, &mut input);
            let consumed = before_len - input.len();

            if consumed == 0 {
                break;
            }

            self.shadow_read_pos += consumed;

            if let Some((bytes, lines)) = self.pending_seq_info.pop_front() {
                self.committed_bytes_in_block += bytes as u64;
                self.committed_lines_in_block += lines as u64;

                let drain_count = bytes.min(self.pending_input.len());
                self.pending_input.drain(..drain_count);
            }
        }
    }

    /// Commits all remaining pending sequences (called after entropy flush).
    fn commit_all_pending(&mut self) {
        while let Some((bytes, lines)) = self.pending_seq_info.pop_front() {
            self.committed_bytes_in_block += bytes as u64;
            self.committed_lines_in_block += lines as u64;
        }
        self.pending_input.clear();
    }

    /// Finalizes the current block: writes footer, flushes to dest, resets
    /// block-scoped state, and re-encodes uncommitted data.
    fn finalize_block(&mut self) -> io::Result<()> {
        let footer = Footer {
            bytes_count: self.total_bytes + self.committed_bytes_in_block,
            lines_count: self.total_lines + self.committed_lines_in_block,
            adler32: self.checksum.checksum(),
        };
        let footer_bytes = block::encode_footer(&footer);

        // Pad block to BLOCK_SIZE with footer at the end.
        self.block_buf.resize(BLOCK_SIZE - footer_bytes.len(), 0);
        self.block_buf.extend_from_slice(&footer_bytes);
        debug_assert_eq!(self.block_buf.len(), BLOCK_SIZE);

        self.dest.write_all(&self.block_buf)?;

        // Update cumulative counters.
        self.total_bytes += self.committed_bytes_in_block;
        self.total_lines += self.committed_lines_in_block;

        // Reset block-scoped state.
        self.block_index += 1;
        self.block_buf.clear();
        self.lz77_enc.reset();
        self.entropy_enc.reset();
        self.models.reset();
        self.shadow_dec.reset();
        self.shadow_models.reset();
        self.shadow_read_pos = 0;
        self.committed_bytes_in_block = 0;
        self.committed_lines_in_block = 0;

        // Re-encode uncommitted input bytes into the new block.
        let rewind_data: Vec<u8> = self.pending_input.drain(..).collect();
        self.pending_seq_info.clear();

        if !rewind_data.is_empty() {
            self.lz77_enc.feed(&rewind_data);
            while let Some(token) = self.lz77_enc.next() {
                self.encode_token_no_block_check(&token);
            }
        }

        Ok(())
    }

    /// Encodes a token without checking for block overflow.
    /// Used during rewind re-encoding to avoid infinite recursion.
    fn encode_token_no_block_check(&mut self, token: &lz77::Token) {
        let seq = &token.seq;
        let lits = &token.literals[..seq.literal_len as usize];
        let seq_lines = count_lines(lits);
        let seq_bytes = seq.literal_len as usize + seq.match_len as usize;

        self.checksum.update(lits);
        self.pending_seq_info.push_back((seq_bytes, seq_lines));
        encode_sequence(&mut self.entropy_enc, &mut self.models, token, &mut self.block_buf);
        self.advance_shadow_decoder();
    }
}

impl<W: Write> Write for Writer<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let st = self.state.as_mut().ok_or_else(|| io::Error::other("writer is sealed"))?;

        if buf.is_empty() {
            return Ok(0);
        }

        st.pending_input.extend(buf);
        st.lz77_enc.feed(buf);

        while let Some(token) = st.lz77_enc.next() {
            st.encode_token(&token)?;
        }

        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        let st = self.state.as_mut().ok_or_else(|| io::Error::other("writer is sealed"))?;

        // Flush all remaining LZ77 data (process buffered lookahead).
        for token in st.lz77_enc.flush() {
            st.encode_token(&token)?;
        }

        // Encode EOFrame marker so the decoder knows to reset entropy state.
        let eoframe = lz77::Token {
            seq: lz77::Sequence {
                literal_len: 0,
                match_len: 0,
                match_distance: codec::EOFRAME_DISTANCE,
            },
            literals: [0u8; 255],
        };
        encode_sequence(&mut st.entropy_enc, &mut st.models, &eoframe, &mut st.block_buf);

        // Entropy coder continues uninterrupted within the block — EOFrame is
        // just a marker in the stream. Only block boundaries and SEAL reset entropy.
        st.advance_shadow_decoder();
        st.commit_all_pending();

        // Advance the shadow decoder past the EOFrame marker so it stays in
        // sync with the encoder's entropy state.
        if st.shadow_read_pos < st.block_buf.len() {
            let available = &st.block_buf[st.shadow_read_pos..];
            let mut input: &[u8] = available;
            let before_len = input.len();
            let token = codec::decode_sequence(&mut st.shadow_dec, &mut st.shadow_models, &mut input);
            let consumed = before_len - input.len();
            if consumed > 0 && is_eoframe(token.seq) {
                st.shadow_read_pos += consumed;
            }
        }

        // Check if the flush pushed us past the block boundary.
        let ftr_size = block::footer_size(
            st.total_bytes + st.committed_bytes_in_block,
            st.total_lines + st.committed_lines_in_block,
        );
        if st.block_buf.len() + ftr_size >= BLOCK_SIZE {
            st.finalize_block()?;
        }

        st.dest.flush()
    }
}

impl<W: Write> Drop for Writer<W> {
    fn drop(&mut self) {
        if self.state.is_some() {
            let _ = self.flush();
        }
    }
}
