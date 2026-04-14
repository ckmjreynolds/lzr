//! Byte-aligned arithmetic coding (encoder and decoder).
//!
//! Uses a u64 `[low, high]` interval with E3-style underflow removal.
//! Normalization emits (encoder) or consumes (decoder) whole bytes whenever
//! the top byte of `low` and `high` agree, and removes the second byte when
//! the interval straddles a byte boundary (underflow).
//!
//! The caller is responsible for model management (swapping, resetting) and
//! for knowing how many symbols to decode (there is no EOF sentinel).

use crate::model::Model;

/// Top-byte mask: keeps only byte 7 (most-significant) of a u64.
const TOP: u64 = 0xFF00_0000_0000_0000;

/// Bottom-6-bytes mask: keeps bytes 0–5, clears bytes 6–7.
const BOT: u64 = 0x0000_FFFF_FFFF_FFFF;

/// Computes `(high - low + 1) / total` without overflow.
///
/// When `low == 0` and `high == u64::MAX`, the true range is 2^64 which wraps
/// to 0 in u64 arithmetic.  We detect that and approximate the division.
const fn step(low: u64, high: u64, total: u64) -> u64 {
    let range = high.wrapping_sub(low).wrapping_add(1);
    if range == 0 {
        u64::MAX / total
    } else {
        range / total
    }
}

/// Arithmetic encoder that writes compressed bytes to a `Vec<u8>`.
pub(crate) struct Encoder {
    low: u64,
    high: u64,
    /// Number of deferred underflow bytes waiting to be resolved.
    pending: u64,
    /// Top byte of `low` when the first pending underflow was recorded.
    straddle: u8,
}

impl Encoder {
    /// Creates a new encoder with the full u64 interval.
    pub(crate) const fn new() -> Self {
        Self {
            low: 0,
            high: u64::MAX,
            pending: 0,
            straddle: 0,
        }
    }

    /// Encodes `symbol` using `model` and appends compressed bytes to `output`.
    ///
    /// The caller must call [`Model::update`] (or swap models) as appropriate
    /// *after* this returns — this method calls `model.update(symbol)` internally.
    pub(crate) fn encode(&mut self, model: &mut Model, symbol: u8, output: &mut Vec<u8>) {
        let cum = model.prefix_sum(symbol);
        let freq = model.freq(symbol);

        let step = step(self.low, self.high, model.total());

        // Narrow the interval. Compute high first — both use the old low.
        self.high = self.low.wrapping_add(step.wrapping_mul(cum + freq)).wrapping_sub(1);
        self.low = self.low.wrapping_add(step.wrapping_mul(cum));

        self.normalize(output);
        model.update(symbol);
    }

    /// Flushes remaining state so the decoder can reconstruct the final
    /// interval, then resets the arithmetic interval for the next frame.
    ///
    /// Models are NOT reset — the caller manages model lifetime.
    pub(crate) fn flush(&mut self, output: &mut Vec<u8>) {
        self.normalize(output);

        // Emit the top byte of low, resolving any pending underflow.
        let byte = (self.low >> 56) as u8;
        output.push(byte);
        self.emit_pending(byte, output);

        // Emit the remaining 7 bytes of low so the decoder has enough data.
        for shift in (0..7).rev() {
            #[allow(clippy::cast_possible_truncation)]
            output.push((self.low >> (shift * 8)) as u8);
        }

        // Reset arithmetic interval for the next frame.
        self.low = 0;
        self.high = u64::MAX;
        self.pending = 0;
        self.straddle = 0;
    }

    /// Resets the encoder to its initial state (for block boundaries).
    pub(crate) const fn reset(&mut self) {
        self.low = 0;
        self.high = u64::MAX;
        self.pending = 0;
        self.straddle = 0;
    }

    /// Emits deferred underflow bytes after a resolved byte, then resets the
    /// pending count. The fill value is `0xFF` if `byte` matches the straddle
    /// byte recorded when the underflow began, `0x00` otherwise.
    #[allow(clippy::cast_possible_truncation)]
    fn emit_pending(&mut self, byte: u8, output: &mut Vec<u8>) {
        if self.pending > 0 {
            let fill = if byte == self.straddle {
                0xFF
            } else {
                0x00
            };
            output.resize(output.len() + self.pending as usize, fill);
            self.pending = 0;
        }
    }

    /// Emits bytes while the top bytes of `low` and `high` agree, and removes
    /// the second byte when the interval straddles a byte boundary.
    fn normalize(&mut self, output: &mut Vec<u8>) {
        loop {
            if (self.low ^ self.high) >> 56 == 0 {
                // Top bytes match — safe to emit.
                let byte = (self.low >> 56) as u8;
                output.push(byte);
                self.emit_pending(byte, output);

                self.low <<= 8;
                self.high = (self.high << 8) | 0xFF;
            } else if (self.high >> 56) == (self.low >> 56) + 1
                && (self.low >> 48) & 0xFF == 0xFF
                && (self.high >> 48).trailing_zeros() >= 8
            {
                // Underflow: interval straddles a byte boundary.
                if self.pending == 0 {
                    self.straddle = (self.low >> 56) as u8;
                }
                self.pending += 1;

                self.low = (self.low & TOP) | ((self.low & BOT) << 8);
                self.high = (self.high & TOP) | ((self.high & BOT) << 8) | 0xFF;
            } else {
                break;
            }
        }
    }
}

/// Arithmetic decoder that reads compressed bytes from a `&[u8]` slice.
pub(crate) struct Decoder {
    low: u64,
    high: u64,
    code: u64,
    loaded: bool,
}

impl Decoder {
    /// Creates a new decoder. The first call to [`decode`](Self::decode) will
    /// read 8 bytes from the input to initialise the code register.
    pub(crate) const fn new() -> Self {
        Self {
            low: 0,
            high: u64::MAX,
            code: 0,
            loaded: false,
        }
    }

    /// Resets the decoder to its initial state (for block or frame boundaries).
    pub(crate) const fn reset(&mut self) {
        self.low = 0;
        self.high = u64::MAX;
        self.code = 0;
        self.loaded = false;
    }

    /// Decodes one symbol using `model`, consuming bytes from `input` as needed.
    ///
    /// The caller must call [`Model::update`] (or swap models) as appropriate
    /// *after* this returns — this method calls `model.update(symbol)` internally.
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(crate) fn decode(&mut self, model: &mut Model, input: &mut &[u8]) -> u8 {
        if !self.loaded {
            for _ in 0..8 {
                self.code = (self.code << 8) | u64::from(read_byte(input));
            }
            self.loaded = true;
        }

        let step = step(self.low, self.high, model.total());

        let target = ((self.code - self.low) / step).min(model.total() - 1);
        let symbol = model.find(target);

        let cum = model.prefix_sum(symbol);
        let freq = model.freq(symbol);

        self.high = self.low.wrapping_add(step.wrapping_mul(cum + freq)).wrapping_sub(1);
        self.low = self.low.wrapping_add(step.wrapping_mul(cum));

        self.normalize(input);
        model.update(symbol);

        symbol
    }

    /// Mirrors the encoder's normalization: shifts out agreed-upon top bytes and
    /// removes the second byte on underflow, reading fresh bytes into `code`.
    fn normalize(&mut self, input: &mut &[u8]) {
        loop {
            if (self.low ^ self.high) >> 56 == 0 {
                self.low <<= 8;
                self.high = (self.high << 8) | 0xFF;
                self.code = (self.code << 8) | u64::from(read_byte(input));
            } else if (self.high >> 56) == (self.low >> 56) + 1
                && (self.low >> 48) & 0xFF == 0xFF
                && (self.high >> 48).trailing_zeros() >= 8
            {
                self.low = (self.low & TOP) | ((self.low & BOT) << 8);
                self.high = (self.high & TOP) | ((self.high & BOT) << 8) | 0xFF;
                self.code = (self.code & TOP) | ((self.code & BOT) << 8) | u64::from(read_byte(input));
            } else {
                break;
            }
        }
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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::bench::lipsum_bytes;

    /// Encode-then-decode helper. Returns the compressed bytes.
    fn roundtrip_codec(data: &[u8]) -> Vec<u8> {
        let mut enc = Encoder::new();
        let mut model = Model::new();
        let mut compressed = Vec::new();
        for &b in data {
            enc.encode(&mut model, b, &mut compressed);
        }
        enc.flush(&mut compressed);

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
    fn decode_empty() {
        let mut input: &[u8] = &[];

        let mut decoder = Decoder::new();
        let mut model = Model::new();
        assert_eq!(decoder.decode(&mut model, &mut input), 0x00);
    }

    #[test]
    fn lipsum_roundtrip_compresses() {
        let data = lipsum_bytes(65_536);

        // Encode (keep the model so we can inspect it).
        let mut enc = Encoder::new();
        let mut model = Model::new();
        let mut compressed = Vec::new();
        for &b in &data {
            enc.encode(&mut model, b, &mut compressed);
        }
        enc.flush(&mut compressed);

        #[allow(clippy::cast_precision_loss)]
        let ratio = 100.0 * (1.0 - compressed.len() as f64 / data.len() as f64);
        eprintln!("entropy compression ratio: {ratio:.1}% ({} → {} bytes)", data.len(), compressed.len());
        // eprintln!("{model:?}");

        // Decode and verify roundtrip.
        let mut dec = Decoder::new();
        let mut dec_model = Model::new();
        let mut input: &[u8] = &compressed;
        let mut decoded = Vec::with_capacity(data.len());
        for _ in 0..data.len() {
            decoded.push(dec.decode(&mut dec_model, &mut input));
        }

        assert_eq!(data, &decoded[..]);
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn multi_frame_flush() {
        // Encode two frames with flush between them (models persist).
        let frame1 = lipsum_bytes(1024);
        let frame2 = lipsum_bytes(2048);
        let frame2 = &frame2[1024..];

        let mut enc = Encoder::new();
        let mut model = Model::new();
        let mut compressed = Vec::new();

        // Frame 1
        for &b in &frame1 {
            enc.encode(&mut model, b, &mut compressed);
        }
        enc.flush(&mut compressed);

        // Frame 2 (model continues, encoder interval was reset by flush)
        for &b in frame2 {
            enc.encode(&mut model, b, &mut compressed);
        }
        enc.flush(&mut compressed);

        // Decode both frames from a single stream.
        let mut dec = Decoder::new();
        let mut dec_model = Model::new();
        let mut input: &[u8] = &compressed;
        let mut decoded = Vec::new();

        // Frame 1
        for _ in 0..frame1.len() {
            decoded.push(dec.decode(&mut dec_model, &mut input));
        }
        assert_eq!(&frame1[..], &decoded[..]);

        // Reset decoder for frame 2 (flush wrote finalization bytes).
        dec.reset();

        decoded.clear();
        for _ in 0..frame2.len() {
            decoded.push(dec.decode(&mut dec_model, &mut input));
        }
        assert_eq!(frame2, &decoded[..]);
    }

    proptest! {
        #[test]
        fn roundtrip(data in prop::collection::vec(any::<u8>(), 0..4_096)) {
            roundtrip_codec(&data);
        }

        #[test]
        fn multi_frame_roundtrip(data in prop::collection::vec(any::<u8>(), 1..2_048)) {
            // Split data at midpoint, encode as two frames with flush boundary, verify roundtrip.
            let mid = data.len() / 2;
            let (part1, part2) = data.split_at(mid);

            let mut enc = Encoder::new();
            let mut model = Model::new();
            let mut compressed = Vec::new();

            for &b in part1 {
                enc.encode(&mut model, b, &mut compressed);
            }
            enc.flush(&mut compressed);

            for &b in part2 {
                enc.encode(&mut model, b, &mut compressed);
            }
            enc.flush(&mut compressed);

            let mut dec = Decoder::new();
            let mut dec_model = Model::new();
            let mut input: &[u8] = &compressed;
            let mut decoded = Vec::new();

            for _ in 0..part1.len() {
                decoded.push(dec.decode(&mut dec_model, &mut input));
            }

            dec.reset();

            for _ in 0..part2.len() {
                decoded.push(dec.decode(&mut dec_model, &mut input));
            }

            prop_assert_eq!(&data[..], &decoded[..]);
        }
    }
}
