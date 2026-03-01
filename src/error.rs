/// Crate-level error type (only used by `NibbleReader` in tests).
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum Error {
    /// Attempted to read past the end of the input.
    #[error("unexpected end of input")]
    UnexpectedEnd,
}
