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

/// Thin wrapper around the internal entropy `Encoder`.
pub struct Encoder(crate::entropy::Encoder);

impl Encoder {
    #[must_use]
    pub fn new() -> Self {
        Self(crate::entropy::Encoder::new())
    }

    pub fn encode(&mut self, model: &mut Model, symbol: u8, output: &mut Vec<u8>) {
        self.0.encode(&mut model.0, symbol, output);
    }

    pub fn finish(self, output: &mut Vec<u8>) {
        self.0.finish(output);
    }
}

/// Thin wrapper around the internal entropy `Decoder`.
pub struct Decoder(crate::entropy::Decoder);

impl Decoder {
    #[must_use]
    pub fn new() -> Self {
        Self(crate::entropy::Decoder::new())
    }

    pub fn decode(&mut self, model: &mut Model, input: &mut &[u8]) -> u8 {
        self.0.decode(&mut model.0, input)
    }
}

/// Thin wrapper around the internal adaptive frequency `Model`.
pub struct Model(crate::model::Model);

impl Model {
    #[must_use]
    pub fn new() -> Self {
        Self(crate::model::Model::new())
    }
}

/// Generates `min_bytes` of deterministic lipsum text.
#[must_use]
pub fn lipsum_bytes(min_bytes: usize) -> Vec<u8> {
    use lipsum::lipsum;
    // lipsum counts words, not bytes; average English word is ~5 chars + space.
    let mut words = min_bytes / 5 + 1;
    loop {
        let text = lipsum(words);
        let mut bytes = text.into_bytes();
        if bytes.len() >= min_bytes {
            bytes.truncate(min_bytes);
            return bytes;
        }
        words *= 2;
    }
}
