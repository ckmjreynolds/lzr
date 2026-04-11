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
    text.into_bytes()[..min_bytes].to_vec()
}
