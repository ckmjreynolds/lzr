//! 16-bit adaptive arithmetic coding (encoder and decoder).
//!
//! Implements the Witten/Neal/Cleary algorithm with E1/E2/E3 normalization as
//! specified in FORMAT.md Section 3. Bits are packed MSB-first into bytes.
//!
//! The caller provides a [`FreqModel`] for each symbol. The coder is generic
//! over model type, so it monomorphizes separately for [`TagModel`] (4 symbols)
//! and [`Model`] (256 symbols) with no dynamic dispatch.
//!
//! # Haiku lifecycle
//!
//! ```text
//! Encoder::new() → encode()* → finalize()   → encode()* → finalize() → ...
//!                  ╰── Haiku 1 ──╯             ╰── Haiku 2 ──╯
//! ```
//!
//! [`finalize`](Encoder::finalize) writes the Kireji (finalization bits), pads
//! to a byte boundary, and resets the arithmetic interval for the next Haiku.
//! Models are **not** reset — the caller manages model lifetime.

use crate::model::FreqModel;

// ---------------------------------------------------------------------------
// Constants (FORMAT.md Section 3.1)
// ---------------------------------------------------------------------------

const HALF: u16 = 0x8000;
const QUARTER: u16 = 0x4000;
const THREE_QUARTER: u16 = 0xC000;

// ---------------------------------------------------------------------------
// BitWriter — MSB-first bit packer
// ---------------------------------------------------------------------------

/// Packs individual bits into bytes, MSB-first.
struct BitWriter {
    /// Partially filled byte being assembled.
    byte_buffer: u8,
    /// Number of valid bits in `byte_buffer` (0..=7).
    bits_in_buffer: u8,
    /// Total bits written since last counter reset.
    total_bits: u64,
}

impl BitWriter {
    const fn new() -> Self {
        Self {
            byte_buffer: 0,
            bits_in_buffer: 0,
            total_bits: 0,
        }
    }

    /// Appends one bit (0 or 1) to the output, emitting a byte when full.
    fn output_bit(&mut self, bit: u8, output: &mut Vec<u8>) {
        self.total_bits += 1;
        self.byte_buffer = (self.byte_buffer << 1) | (bit & 1);
        self.bits_in_buffer += 1;
        if self.bits_in_buffer == 8 {
            output.push(self.byte_buffer);
            self.byte_buffer = 0;
            self.bits_in_buffer = 0;
        }
    }

    /// Pads the current byte with zeros and emits it (if any bits pending).
    fn flush_to_byte_boundary(&mut self, output: &mut Vec<u8>) {
        if self.bits_in_buffer > 0 {
            self.byte_buffer <<= 8 - self.bits_in_buffer;
            output.push(self.byte_buffer);
            self.byte_buffer = 0;
            self.bits_in_buffer = 0;
        }
    }

    const fn bits_in_buffer(&self) -> u8 {
        self.bits_in_buffer
    }

    const fn reset(&mut self) {
        self.byte_buffer = 0;
        self.bits_in_buffer = 0;
        self.total_bits = 0;
    }
}

// ---------------------------------------------------------------------------
// BitReader — MSB-first bit reader
// ---------------------------------------------------------------------------

/// Reads individual bits from a byte stream, MSB-first.
struct BitReader {
    /// Current byte being consumed.
    byte_buffer: u8,
    /// Bits remaining in the current byte (0 = need to read next byte).
    bits_remaining: u8,
}

impl BitReader {
    const fn new() -> Self {
        Self {
            byte_buffer: 0,
            bits_remaining: 0,
        }
    }

    /// Reads one bit from the input. Returns 0 if input is exhausted.
    const fn read_bit(&mut self, input: &mut &[u8]) -> u8 {
        if self.bits_remaining == 0 {
            self.byte_buffer = read_byte(input);
            self.bits_remaining = 8;
        }
        self.bits_remaining -= 1;
        (self.byte_buffer >> self.bits_remaining) & 1
    }

    /// Discards remaining bits in the current byte (byte-align after Kireji).
    const fn align_to_byte_boundary(&mut self) {
        self.bits_remaining = 0;
    }
}

/// Reads one byte from `input`, advancing the slice. Returns `0x00` if empty.
const fn read_byte(input: &mut &[u8]) -> u8 {
    match input.split_first() {
        Some((&byte, rest)) => {
            *input = rest;
            byte
        }
        None => 0x00,
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Arithmetic encoder with 16-bit precision and E1/E2/E3 normalization.
///
/// Produces MSB-first bit-packed output. Call [`encode`](Self::encode) for each
/// symbol, then [`finalize`](Self::finalize) to write the Kireji and prepare
/// for the next Haiku.
pub(crate) struct Encoder {
    low: u16,
    high: u16,
    pending_bits: u64,
    writer: BitWriter,
}

impl Encoder {
    /// Creates a new encoder with the full 16-bit interval.
    pub(crate) const fn new() -> Self {
        Self {
            low: 0x0000,
            high: 0xFFFF,
            pending_bits: 0,
            writer: BitWriter::new(),
        }
    }

    /// Encodes `symbol` using `model` and appends compressed bits to `output`.
    ///
    /// Calls `model.update(symbol)` after narrowing the interval (per spec).
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn encode<M: FreqModel>(&mut self, symbol: u8, model: &mut M, output: &mut Vec<u8>) {
        let cum_low = model.cum_freq(symbol);
        let cum_high = cum_low + model.freq(symbol);
        let total = model.total();

        let range = u32::from(self.high) - u32::from(self.low) + 1;
        self.high = (u32::from(self.low) + range * cum_high / total - 1) as u16;
        self.low = (u32::from(self.low) + range * cum_low / total) as u16;

        self.normalize(output);
        model.update(symbol);
    }

    /// Writes Kireji finalization bits, byte-aligns, adds trailing resync
    /// zeros, and resets the interval for the next Haiku.
    ///
    /// The trailing zeros absorb the decoder's 16-bit lookahead so that
    /// after byte-aligning + skipping 2 bytes, the decoder reads fresh
    /// next-Haiku data. Models are NOT reset.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn finalize_haiku(&mut self, output: &mut Vec<u8>) {
        let n_encoded = self.writer.total_bits;
        let p_final = self.pending_bits;

        self.emit_finalization(output);
        self.writer.flush_to_byte_boundary(output);

        // The decoder consumed 16 + N + P bits (16 init + N normalize + P E3
        // lookahead). After byte-aligning, it's at byte ceil((16+N+P) / 8).
        // We add trailing zeros so the decoder can skip exactly 2 bytes and
        // re-read 16 fresh bits from the next Haiku.
        //
        // haiku_bytes is computed from total_bits (not output.len()) because
        // the output Vec may be flushed/cleared between encodes by the Writer.
        let decoder_bits = 16 + n_encoded + p_final;
        let decoder_bytes = decoder_bits.div_ceil(8) as usize;
        let haiku_bits = self.writer.total_bits; // includes finalization, excludes pad
        let haiku_bytes = haiku_bits.div_ceil(8) as usize;
        let trailing = (decoder_bytes + 2).saturating_sub(haiku_bytes);

        for _ in 0..trailing {
            output.push(0x00);
        }

        // Reset for next Haiku.
        self.low = 0x0000;
        self.high = 0xFFFF;
        self.pending_bits = 0;
        self.writer.total_bits = 0;
    }

    /// Writes Kireji finalization bits, pads to byte boundary, and fully resets.
    ///
    /// Used at Sonnet boundaries (EOS/EOO). The byte-alignment enables the
    /// reader to locate the padding and footer.
    pub(crate) fn finalize_sonnet(&mut self, output: &mut Vec<u8>) {
        self.emit_finalization(output);
        self.writer.flush_to_byte_boundary(output);
        self.low = 0x0000;
        self.high = 0xFFFF;
        self.pending_bits = 0;
    }

    /// Emits the finalization bits (FORMAT.md Section 3.4).
    fn emit_finalization(&mut self, output: &mut Vec<u8>) {
        self.pending_bits += 1;
        if self.low < QUARTER {
            self.writer.output_bit(0, output);
            for _ in 0..self.pending_bits {
                self.writer.output_bit(1, output);
            }
        } else {
            self.writer.output_bit(1, output);
            for _ in 0..self.pending_bits {
                self.writer.output_bit(0, output);
            }
        }
    }

    /// Resets all encoder state (for Sonnet boundaries).
    pub(crate) const fn reset(&mut self) {
        self.low = 0x0000;
        self.high = 0xFFFF;
        self.pending_bits = 0;
        self.writer.reset();
    }

    /// Returns the number of pending E3 underflow bits.
    pub(crate) const fn pending_bits(&self) -> u64 {
        self.pending_bits
    }

    /// Returns the number of bits buffered in the partial output byte.
    pub(crate) const fn bits_in_buffer(&self) -> u8 {
        self.writer.bits_in_buffer()
    }

    /// E1/E2/E3 normalization loop (FORMAT.md Section 3.3).
    fn normalize(&mut self, output: &mut Vec<u8>) {
        loop {
            if self.high < HALF {
                // E1: interval in lower half.
                self.writer.output_bit(0, output);
                for _ in 0..self.pending_bits {
                    self.writer.output_bit(1, output);
                }
                self.pending_bits = 0;
            } else if self.low >= HALF {
                // E2: interval in upper half.
                self.writer.output_bit(1, output);
                for _ in 0..self.pending_bits {
                    self.writer.output_bit(0, output);
                }
                self.pending_bits = 0;
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= QUARTER && self.high < THREE_QUARTER {
                // E3: straddle — defer the decision.
                self.pending_bits += 1;
                self.low -= QUARTER;
                self.high -= QUARTER;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// Arithmetic decoder with 16-bit precision.
///
/// Reads bits MSB-first from the input. Call [`decode`](Self::decode) for each
/// symbol, then [`finalize`](Self::finalize) after a terminator tag to align
/// to the byte boundary and prepare for the next Haiku.
pub(crate) struct Decoder {
    low: u16,
    high: u16,
    code: u16,
    reader: BitReader,
    initialized: bool,
}

impl Decoder {
    /// Creates a new decoder. The first [`decode`](Self::decode) call reads
    /// 16 bits from the input to initialize the code register.
    pub(crate) const fn new() -> Self {
        Self {
            low: 0x0000,
            high: 0xFFFF,
            code: 0,
            reader: BitReader::new(),
            initialized: false,
        }
    }

    /// Decodes one symbol using `model`, consuming bits from `input`.
    ///
    /// Calls `model.update(symbol)` after narrowing the interval (per spec).
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn decode<M: FreqModel>(&mut self, model: &mut M, input: &mut &[u8]) -> u8 {
        if !self.initialized {
            self.code = 0;
            for _ in 0..16 {
                self.code = (self.code << 1) | u16::from(self.reader.read_bit(input));
            }
            self.initialized = true;
        }

        let total = model.total();
        let range = u32::from(self.high) - u32::from(self.low) + 1;
        let value = ((u32::from(self.code) - u32::from(self.low) + 1) * total - 1) / range;
        let symbol = model.find(value);

        let cum_low = model.cum_freq(symbol);
        let cum_high = cum_low + model.freq(symbol);

        self.high = (u32::from(self.low) + range * cum_high / total - 1) as u16;
        self.low = (u32::from(self.low) + range * cum_low / total) as u16;

        self.normalize(input);
        model.update(symbol);

        symbol
    }

    /// Byte-aligns the reader, skips 2 trailing resync bytes, and re-reads
    /// 16 bits into `code` for the next Haiku.
    ///
    /// Must be called after decoding an `EOH` tag. Models are NOT reset.
    pub(crate) fn finalize_haiku(&mut self, input: &mut &[u8]) {
        self.reader.align_to_byte_boundary();
        // Skip the 2 resync zero bytes written by the encoder.
        let _ = read_byte(input);
        let _ = read_byte(input);
        // Re-read 16 bits for the next Haiku.
        self.code = 0;
        for _ in 0..16 {
            self.code = (self.code << 1) | u16::from(self.reader.read_bit(input));
        }
        self.low = 0x0000;
        self.high = 0xFFFF;
    }

    /// Returns `true` if the decoder's bit reader has buffered bits remaining.
    pub(crate) const fn has_buffered_bits(&self) -> bool {
        self.reader.bits_remaining > 0
    }

    /// Decoder normalization — mirrors encoder, shifting in bits from input.
    fn normalize(&mut self, input: &mut &[u8]) {
        loop {
            if self.high < HALF {
                // E1: nothing extra.
            } else if self.low >= HALF {
                // E2.
                self.low -= HALF;
                self.high -= HALF;
                self.code -= HALF;
            } else if self.low >= QUARTER && self.high < THREE_QUARTER {
                // E3.
                self.low -= QUARTER;
                self.high -= QUARTER;
                self.code -= QUARTER;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
            self.code = (self.code << 1) | u16::from(self.reader.read_bit(input));
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::model::{Model, TagModel};

    /// Encode then decode `data` through a byte model, verifying roundtrip.
    /// Uses `finalize_sonnet` (byte-aligned, single frame).
    fn roundtrip_bytes(data: &[u8]) -> Vec<u8> {
        let mut enc = Encoder::new();
        let mut model = Model::new();
        let mut compressed = Vec::new();
        for &b in data {
            enc.encode(b, &mut model, &mut compressed);
        }
        enc.finalize_sonnet(&mut compressed);

        let mut dec = Decoder::new();
        let mut model = Model::new();
        let mut input: &[u8] = &compressed;
        let mut decoded = Vec::with_capacity(data.len());
        for _ in 0..data.len() {
            decoded.push(dec.decode(&mut model, &mut input));
        }

        assert_eq!(data, &decoded[..]);
        compressed
    }

    #[test]
    fn empty_roundtrip() {
        roundtrip_bytes(b"");
    }

    #[test]
    fn single_byte() {
        roundtrip_bytes(b"A");
    }

    #[test]
    fn known_data() {
        roundtrip_bytes(b"Hello, world!");
    }

    #[test]
    fn decode_empty_input() {
        let mut dec = Decoder::new();
        let mut model = Model::new();
        let mut input: &[u8] = &[];
        let _ = dec.decode(&mut model, &mut input);
    }

    #[test]
    fn tag_model_roundtrip() {
        let symbols: Vec<u8> = vec![0, 1, 2, 3, 0, 0, 1, 3, 2, 1, 0, 3];
        let mut enc = Encoder::new();
        let mut model = TagModel::new();
        let mut compressed = Vec::new();
        for &s in &symbols {
            enc.encode(s, &mut model, &mut compressed);
        }
        enc.finalize_sonnet(&mut compressed);

        let mut dec = Decoder::new();
        let mut model = TagModel::new();
        let mut input: &[u8] = &compressed;
        let mut decoded = Vec::new();
        for _ in 0..symbols.len() {
            decoded.push(dec.decode(&mut model, &mut input));
        }
        assert_eq!(symbols, decoded);
    }

    #[test]
    fn multi_haiku_roundtrip() {
        let data1 = b"first haiku data";
        let data2 = b"second haiku data";

        let mut enc = Encoder::new();
        let mut model = Model::new();
        let mut compressed = Vec::new();

        for &b in &data1[..] {
            enc.encode(b, &mut model, &mut compressed);
        }
        enc.finalize_haiku(&mut compressed);

        for &b in &data2[..] {
            enc.encode(b, &mut model, &mut compressed);
        }
        enc.finalize_sonnet(&mut compressed);

        let mut dec = Decoder::new();
        let mut dec_model = Model::new();
        let mut input: &[u8] = &compressed;

        let mut decoded1 = Vec::new();
        for _ in 0..data1.len() {
            decoded1.push(dec.decode(&mut dec_model, &mut input));
        }
        assert_eq!(&data1[..], &decoded1[..]);

        dec.finalize_haiku(&mut input);

        let mut decoded2 = Vec::new();
        for _ in 0..data2.len() {
            decoded2.push(dec.decode(&mut dec_model, &mut input));
        }
        assert_eq!(&data2[..], &decoded2[..]);
    }

    #[test]
    fn large_roundtrip_triggers_rescale() {
        // 100K bytes → 100K model updates → triggers rescale multiple times.
        let data = crate::bench::lipsum_bytes(100_000);
        let compressed = roundtrip_bytes(&data);
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn compresses_lipsum() {
        let data = crate::bench::lipsum_bytes(65_536);
        let compressed = roundtrip_bytes(&data);
        assert!(compressed.len() < data.len(), "should compress lipsum");
    }

    proptest! {
        #[test]
        fn roundtrip(data in prop::collection::vec(any::<u8>(), 0..4_096)) {
            roundtrip_bytes(&data);
        }

        #[test]
        fn multi_haiku(data in prop::collection::vec(any::<u8>(), 2..2_048)) {
            let mid = data.len() / 2;
            let (part1, part2) = data.split_at(mid);

            let mut enc = Encoder::new();
            let mut model = Model::new();
            let mut compressed = Vec::new();

            for &b in part1 {
                enc.encode(b, &mut model, &mut compressed);
            }
            enc.finalize_haiku(&mut compressed);

            for &b in part2 {
                enc.encode(b, &mut model, &mut compressed);
            }
            enc.finalize_sonnet(&mut compressed);

            let mut dec = Decoder::new();
            let mut dec_model = Model::new();
            let mut input: &[u8] = &compressed;
            let mut decoded = Vec::new();

            for _ in 0..part1.len() {
                decoded.push(dec.decode(&mut dec_model, &mut input));
            }
            dec.finalize_haiku(&mut input);
            for _ in 0..part2.len() {
                decoded.push(dec.decode(&mut dec_model, &mut input));
            }

            prop_assert_eq!(&data[..], &decoded[..]);
        }

        #[test]
        fn tag_roundtrip(symbols in prop::collection::vec(0u8..4, 1..500)) {
            let mut enc = Encoder::new();
            let mut model = TagModel::new();
            let mut compressed = Vec::new();
            for &s in &symbols {
                enc.encode(s, &mut model, &mut compressed);
            }
            enc.finalize_sonnet(&mut compressed);

            let mut dec = Decoder::new();
            let mut model = TagModel::new();
            let mut input: &[u8] = &compressed;
            let mut decoded = Vec::new();
            for _ in 0..symbols.len() {
                decoded.push(dec.decode(&mut model, &mut input));
            }
            prop_assert_eq!(symbols, decoded);
        }
    }
}
