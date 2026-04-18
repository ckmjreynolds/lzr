//! Error type returned by [`Writer`](crate::Writer) and [`Reader`](crate::Reader).
//!
//! `Error` is surfaced by the public constructors and methods on `Reader` and
//! `Writer`. The [`std::io::Read`], [`std::io::Write`], and [`std::io::Seek`]
//! trait implementations convert these errors into [`std::io::Error`] via
//! `From<Error> for io::Error`; consumers wanting structured errors at the I/O
//! boundary can downcast with [`std::io::Error::get_ref`].

use std::io;

/// Errors produced by the LZR library.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The file does not begin with the LZR magic bytes (`b"LZR"`).
    #[error("not an LZR file: invalid magic bytes")]
    InvalidMagic,

    /// The file begins with the LZR magic bytes but uses an unknown version.
    #[error("unsupported LZR version: 0x{0:02X}")]
    UnsupportedVersion(u8),

    /// The file is shorter than the required 4-byte Invocation header.
    #[error("file too small to contain an LZR header")]
    TooSmall,

    /// Attempt to open or append to an already-sealed archive.
    #[error("archive is sealed; cannot append")]
    Sealed,

    /// A `Writer` method was called after the writer was sealed or consumed.
    #[error("writer has already been sealed")]
    WriterClosed,

    /// A decoded `LINE` token has both `literal_len = 0` and `match_len = 0`,
    /// which FORMAT.md §2.2 forbids.
    #[error("invalid LINE token: literal_len and match_len both zero")]
    InvalidLine,

    /// A decoded match uses `match_distance = 0` with `match_len > 0`.
    #[error("invalid match distance: distance must be nonzero when match_len > 0")]
    InvalidMatchDistance,

    /// A decoded match references data beyond what has been emitted in the
    /// current Sonnet.
    #[error("match distance {distance} exceeds output emitted in current Sonnet ({emitted} bytes)")]
    MatchDistanceOutOfRange {
        /// The offending distance.
        distance: u16,
        /// Uncompressed bytes emitted in the current Sonnet so far.
        emitted: u64,
    },

    /// A Sonnet footer's Adler-32 checksum did not match the recomputed value.
    #[error("checksum mismatch: footer says {expected:#010x}, computed {computed:#010x}")]
    ChecksumMismatch {
        /// Checksum stored in the Couplet/Coda footer.
        expected: u32,
        /// Checksum recomputed from the decompressed data.
        computed: u32,
    },

    /// A footer field uses a non-canonical ULEB128 encoding (FORMAT.md §7).
    #[error("non-canonical ULEB128 encoding in footer")]
    NonCanonicalUleb128,

    /// A seek would move the cursor to a negative offset.
    #[error("seek to negative offset")]
    NegativeSeek,

    /// A line number passed to [`Reader::seek_to_line`](crate::Reader::seek_to_line)
    /// is past the end of the file.
    #[error("line {line} out of range (file has {total} lines)")]
    LineOutOfRange {
        /// The requested line.
        line: u64,
        /// Total lines in the file.
        total: u64,
    },

    /// The encoded stream ended mid-token or mid-footer.
    #[error("truncated input")]
    TruncatedInput,

    /// An underlying I/O error from the wrapped `Read`/`Write`/`Seek`.
    #[error(transparent)]
    Io(#[from] io::Error),
}

impl From<Error> for io::Error {
    fn from(err: Error) -> Self {
        match err {
            Error::Io(e) => e,
            Error::InvalidMagic
            | Error::UnsupportedVersion(_)
            | Error::TooSmall
            | Error::Sealed
            | Error::WriterClosed
            | Error::InvalidLine
            | Error::InvalidMatchDistance
            | Error::MatchDistanceOutOfRange {
                ..
            }
            | Error::ChecksumMismatch {
                ..
            }
            | Error::NonCanonicalUleb128
            | Error::TruncatedInput => Self::new(io::ErrorKind::InvalidData, err),
            Error::NegativeSeek
            | Error::LineOutOfRange {
                ..
            } => Self::new(io::ErrorKind::InvalidInput, err),
        }
    }
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Error>;
