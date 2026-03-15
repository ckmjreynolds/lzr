//! Error types for the LZR compression library.

/// Errors that can occur during LZR compression or decompression.
#[derive(Debug, thiserror::Error)]
#[allow(clippy::module_name_repetitions)]
pub enum Error {
    /// An I/O error occurred.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// The input data is not valid LZR format.
    #[error("invalid format: {0}")]
    InvalidFormat(String),

    /// The magic number in the header does not match `LZR`.
    #[error("invalid magic number")]
    InvalidMagic,

    /// The format version is not supported by this decoder.
    #[error("unsupported version: {0}")]
    UnsupportedVersion(u8),

    /// The Adler-32 checksum does not match the decompressed data.
    #[error("checksum mismatch: expected {expected:#010X}, actual {actual:#010X}")]
    ChecksumMismatch {
        /// Expected checksum from the footer.
        expected: u32,
        /// Computed checksum from the decompressed data.
        actual: u32,
    },

    /// The decompressed length does not match the footer.
    #[error("length mismatch: expected {expected}, actual {actual}")]
    LengthMismatch {
        /// Expected length from the footer.
        expected: u64,
        /// Actual length of the decompressed data.
        actual: u64,
    },
}

/// A specialized [`Result`] type for LZR operations.
pub type Result<T> = std::result::Result<T, Error>;
