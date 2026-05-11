//! Core `Codec` trait and `Decomposition` accounting struct.
//!
//! Every encoder/decoder in v2 implements `Codec` so the eval panel
//! can run any mix of codecs through the same machinery and compare
//! them on equal footing. `Decomposition` records bits emitted per
//! named component during an encode pass — the substrate for the
//! bit-budget discipline that all v2 work is anchored on.

use std::collections::BTreeMap;

use anyhow::Result;

/// Per-component bit accounting for an encode pass.
///
/// Components are identified by stable short strings like `"xml_tag"`,
/// `"prose_ppm"`, `"lz_match"`. Sum of values across components should
/// equal `8 * archive_bytes` (modulo a small framing overhead) — when
/// it doesn't, a codec is failing to attribute its emissions and the
/// audit table will be wrong.
#[derive(Clone, Debug, Default)]
pub(crate) struct Decomposition {
    pub by_component: BTreeMap<String, u64>,
}

impl Decomposition {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Add `bits` to the named component's running total.
    pub(crate) fn add(&mut self, component: &str, bits: u64) {
        *self.by_component.entry(component.to_string()).or_insert(0) += bits;
    }

    pub(crate) fn total(&self) -> u64 {
        self.by_component.values().sum()
    }
}

/// Stateless compression/decompression configuration. Each `encode_window`
/// or `decode_window` call manages its own internal adaptive state, so
/// the same `Codec` instance can be safely reused across windows in a
/// bench panel without explicit reset.
///
/// The `warm` prefix primes adaptive state on both sides identically
/// without contributing to the output bit count. `decode_window`'s
/// `warm` argument must match the encode-side `warm` exactly for
/// roundtrip to succeed.
pub(crate) trait Codec {
    /// Stable short identifier for reports (`"null"`, `"xml"`, …).
    fn name(&self) -> &'static str;

    /// Encode `measure` preceded by `warm` priming bytes. Returns the
    /// archive bytes (containing only `measure`'s worth of payload, no
    /// `warm` bytes) and a per-component decomposition of the bits
    /// emitted for `measure`.
    fn encode_window(&self, warm: &[u8], measure: &[u8]) -> Result<(Vec<u8>, Decomposition)>;

    /// Decode the archive produced by `encode_window`. Returns the
    /// `measure` bytes; `warm` must match the encoder's `warm`.
    fn decode_window(&self, warm: &[u8], archive: &[u8]) -> Result<Vec<u8>>;
}
