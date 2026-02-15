use std::{fmt, io};

/// Errors from LZR compression or decompression.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// An I/O error occurred.
    Io(io::Error),
    /// The input is not valid LZR format.
    InvalidFormat,
    /// Footer checksum did not match.
    ChecksumMismatch,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::InvalidFormat => write!(f, "invalid LZR format"),
            Self::ChecksumMismatch => write!(f, "checksum mismatch"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::InvalidFormat | Self::ChecksumMismatch => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        // Recover wrapped LZR errors that round-tripped through io::Error.
        if err.get_ref().is_some_and(<dyn std::error::Error + Send + Sync>::is::<Self>) {
            return *err.into_inner().expect("checked above").downcast::<Self>().expect("checked above");
        }
        Self::Io(err)
    }
}

impl From<Error> for io::Error {
    fn from(err: Error) -> Self {
        match err {
            Error::Io(e) => e,
            other => Self::new(io::ErrorKind::InvalidData, other),
        }
    }
}

/// A type alias for `Result<T, lzr::Error>`.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use std::io;

    use super::*;

    #[test]
    fn error_display() {
        let io_err = Error::Io(io::Error::new(io::ErrorKind::NotFound, "gone"));
        assert_eq!(io_err.to_string(), "I/O error: gone");

        assert_eq!(Error::InvalidFormat.to_string(), "invalid LZR format");
        assert_eq!(Error::ChecksumMismatch.to_string(), "checksum mismatch");
    }

    #[test]
    fn error_from_io() {
        let io_err = io::Error::new(io::ErrorKind::BrokenPipe, "broken");
        let lzr_err: Error = io_err.into();
        assert!(matches!(lzr_err, Error::Io(_)));
    }

    #[test]
    fn error_into_io() {
        let lzr_err = Error::InvalidFormat;
        let io_err: io::Error = lzr_err.into();
        assert_eq!(io_err.kind(), io::ErrorKind::InvalidData);

        let lzr_err = Error::ChecksumMismatch;
        let io_err: io::Error = lzr_err.into();
        assert_eq!(io_err.kind(), io::ErrorKind::InvalidData);

        let original = io::Error::new(io::ErrorKind::BrokenPipe, "broken");
        let lzr_err = Error::Io(original);
        let io_err: io::Error = lzr_err.into();
        assert_eq!(io_err.kind(), io::ErrorKind::BrokenPipe);
    }
}
