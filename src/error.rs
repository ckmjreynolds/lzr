//! LZR Error type.

/// Errors produced by the LZR library.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A footer field uses a non-canonical ULEB128 encoding (FORMAT.md §7).
    #[error("non-canonical ULEB128 encoding in footer")]
    NonCanonicalUleb128,
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Error>;
