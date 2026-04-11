//! Test and benchmark utilities. Not part of the public API.

/// Thin wrapper around the internal `Adler32` checksum.
pub struct Adler32(crate::adler32::Adler32);

impl Adler32 {
    #[must_use]
    pub fn new() -> Self {
        Self(crate::adler32::Adler32::new())
    }

    pub fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }

    #[must_use]
    pub fn checksum(&self) -> u32 {
        self.0.checksum()
    }
}

/// Generates `min_bytes` of deterministic lipsum text.
#[must_use]
pub fn lipsum_bytes(min_bytes: usize) -> Vec<u8> {
    use lipsum::lipsum;
    // lipsum counts words, not bytes; average English word is ~5 chars + space.
    let words = min_bytes / 5 + 1;
    let text = lipsum(words);
    let mut bytes = text.into_bytes();
    bytes.truncate(min_bytes);
    bytes
}

/// Thin wrapper around the internal entropy `Model`.
pub struct EntropyModel(crate::entropy::Model);

impl EntropyModel {
    #[must_use]
    pub fn new() -> Self {
        Self(crate::entropy::Model::new())
    }
}

/// Thin wrapper around the internal entropy `Encoder`.
pub struct EntropyEncoder(crate::entropy::Encoder);

impl EntropyEncoder {
    #[must_use]
    pub fn new() -> Self {
        Self(crate::entropy::Encoder::new())
    }

    pub fn encode(&mut self, symbol: u8, model: &mut EntropyModel, output: &mut Vec<u8>) {
        self.0.encode(symbol, &mut model.0, output);
    }

    pub fn finish(self, output: &mut Vec<u8>) {
        self.0.finish(output);
    }
}

/// Thin wrapper around the internal entropy `Decoder`.
pub struct EntropyDecoder(crate::entropy::Decoder);

impl EntropyDecoder {
    #[must_use]
    pub fn new(input: &mut &[u8]) -> Self {
        Self(crate::entropy::Decoder::new(input))
    }

    pub fn decode(&mut self, model: &mut EntropyModel, input: &mut &[u8]) -> u8 {
        self.0.decode(&mut model.0, input)
    }
}
