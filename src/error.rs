/// Crate-level error type.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[allow(dead_code)]
pub(crate) enum Error {
    /// Attempted to read past the end of the input.
    #[error("unexpected end of input")]
    UnexpectedEnd,
}

/// Crate-level result alias.
#[allow(dead_code)]
pub(crate) type Result<T> = std::result::Result<T, Error>;
