//! Shared helpers for the integration-test binaries.

use std::path::Path;

/// A corpus file under the crate root, or `None` if absent (corpora are not part
/// of the packaged crate, so a `cargo package` build skips corpus-backed tests).
pub(crate) fn corpus(rel: &str) -> Option<Vec<u8>> {
    std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)).ok()
}
