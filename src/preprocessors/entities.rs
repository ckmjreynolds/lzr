//! XML/HTML entity-folding byte preprocessor.

use super::{TEXT_FRACTION, Transform, is_text_byte};

/// The five predefined XML entities this stage folds. The array index is the
/// entity's spare-byte slot (slot `i` ↔ `ENTITIES[i]`); the ordering is part of
/// the on-stream format, so it is append-only — never reorder or remove an entry.
/// None is a prefix of another, so matching is order-independent.
const ENTITIES: [&[u8]; 5] = [b"&lt;", b"&gt;", b"&amp;", b"&quot;", b"&apos;"];

/// XML/HTML entity-folding byte preprocessor.
///
/// Replaces each of the five predefined XML entities (`&lt;`, `&gt;`, `&amp;`,
/// `&quot;`, `&apos;`) with a single dedicated byte, giving the downstream
/// tokenizer and entropy models one compact symbol instead of a 4–6 byte entity
/// spelling. Everything else — including literal `<`, `>`, `&`, `"`, `'` that are
/// *not* part of an entity — is left verbatim.
///
/// Like [`CaseFolding`](super::CaseFolding), the substitute byte *values* are
/// chosen per input as bytes that never occur in it, so a substitute byte in the
/// payload unambiguously marks one of our insertions. This makes the transform
/// exactly reversible for *any* input with no state machine and no
/// violation-checking: a literal `&lt;` that was never meant as an entity still
/// round-trips, because folding it and then unfolding it is the identity. When the
/// input is not text, contains no recognized entity, or has fewer than five spare
/// byte values, the stage falls back to an exact pass-through.
///
/// The stream is self-describing (there is no side channel to
/// [`Transform::inverse`]): a leading mode byte records which path was taken, and
/// the folded path stores its five chosen substitute bytes in the header.
pub(crate) struct EntityFolding;

/// Header mode byte: the input was passed through unchanged.
const MODE_PASSTHROUGH: u8 = 0;
/// Header mode byte: the payload is entity-folded (followed by the five substitute bytes).
const MODE_FOLDED: u8 = 1;

/// The slot of the entity that begins at the start of `rest`, if any. `rest`
/// should be the input sliced from a candidate `&`.
fn match_entity(rest: &[u8]) -> Option<usize> {
    ENTITIES.iter().position(|entity| rest.starts_with(entity))
}

impl EntityFolding {
    /// Picks the five lowest byte values absent from `present`, if at least five
    /// exist. These become the entities' substitute bytes (slot `i` → element `i`).
    fn spare_bytes(present: &[bool; 256]) -> Option<[u8; 5]> {
        let mut unused = (0u8..=255).filter(|&b| !present[usize::from(b)]);
        Some([unused.next()?, unused.next()?, unused.next()?, unused.next()?, unused.next()?])
    }
}

impl Transform<u8> for EntityFolding {
    #[expect(clippy::cast_precision_loss, reason = "byte counts are far under 2^53")]
    fn forward(&self, input: &[u8]) -> Vec<u8> {
        // One pass: presence map, text-byte count, and whether any entity appears.
        let mut present = [false; 256];
        let mut text_bytes = 0usize;
        let mut has_entity = false;
        for (i, &byte) in input.iter().enumerate() {
            present[usize::from(byte)] = true;
            if is_text_byte(byte) {
                text_bytes += 1;
            }
            if byte == b'&' && !has_entity && match_entity(&input[i..]).is_some() {
                has_entity = true;
            }
        }

        // Fall back to pass-through unless the input looks like text, is XML/HTML
        // (a recognized entity is present — which also means there is something to
        // fold), and five spare byte values exist for the substitutes.
        let is_text = !input.is_empty() && text_bytes as f64 >= input.len() as f64 * TEXT_FRACTION;
        let Some(spares) = (has_entity && is_text).then(|| Self::spare_bytes(&present)).flatten() else {
            let mut out = Vec::with_capacity(input.len() + 1);
            out.push(MODE_PASSTHROUGH);
            out.extend_from_slice(input);
            return out;
        };

        let mut out = Vec::with_capacity(input.len() + 6);
        out.push(MODE_FOLDED);
        out.extend_from_slice(&spares);
        let mut i = 0;
        while i < input.len() {
            if input[i] == b'&'
                && let Some(slot) = match_entity(&input[i..])
            {
                out.push(spares[slot]);
                i += ENTITIES[slot].len();
            } else {
                out.push(input[i]);
                i += 1;
            }
        }
        out
    }

    fn inverse(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        let Some((&mode, rest)) = input.split_first() else {
            // Empty input is not something `forward` ever produces; treat it as an
            // empty restoration rather than panicking on corrupt decode data.
            return Ok(Vec::new());
        };
        match mode {
            MODE_PASSTHROUGH => Ok(rest.to_vec()),
            MODE_FOLDED => {
                let [b0, b1, b2, b3, b4, payload @ ..] = rest else {
                    anyhow::bail!("entity-folding header truncated: missing substitute bytes");
                };
                // Map each substitute byte value back to its entity slot.
                let mut slot_of: [Option<usize>; 256] = [None; 256];
                for (slot, &byte) in [*b0, *b1, *b2, *b3, *b4].iter().enumerate() {
                    slot_of[usize::from(byte)] = Some(slot);
                }
                let mut out = Vec::with_capacity(payload.len());
                for &byte in payload {
                    if let Some(slot) = slot_of[usize::from(byte)] {
                        out.extend_from_slice(ENTITIES[slot]);
                    } else {
                        out.push(byte);
                    }
                }
                Ok(out)
            }
            other => anyhow::bail!("unknown entity-folding mode byte {other}"),
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::*;

    /// Token alphabet for the XML-ish round-trip generator: tags, all five
    /// entities, prose, a bare `&`, and a non-entity `&…;` sequence.
    const XMLISH_TOKENS: &[&[u8]] = &[
        b"&lt;", b"&gt;", b"&amp;", b"&quot;", b"&apos;", b"<page>", b"</page>", b" hello ", b"& ",
        b"&notreal;", b"a",
    ];

    proptest! {
        /// Entity folding must round-trip *any* byte input exactly — text and
        /// binary, via both the folded and pass-through paths, including inputs
        /// that already literally contain entity spellings.
        #[test]
        fn entity_roundtrip(data in prop::collection::vec(any::<u8>(), 0..1024)) {
            let s = EntityFolding;
            prop_assert_eq!(s.inverse(&s.forward(&data)).unwrap(), data);
        }

        /// A generator biased toward XML so the folded path is exercised often:
        /// random tokens drawn from tags, entities, and prose.
        #[test]
        fn entity_roundtrip_xmlish(
            tokens in prop::collection::vec(prop::sample::select(XMLISH_TOKENS), 0..200),
        ) {
            let data: Vec<u8> = tokens.concat();
            let s = EntityFolding;
            prop_assert_eq!(s.inverse(&s.forward(&data)).unwrap(), data);
        }

        /// Inverse runs on decoded, possibly-corrupt data — it may error but must
        /// never panic.
        #[test]
        fn entity_inverse_never_panics(data in prop::collection::vec(any::<u8>(), 0..1024)) {
            let s = EntityFolding;
            drop(s.inverse(&data));
        }

        /// Entity folding must round-trip inputs that saturate the whole byte range
        /// (no five spare values → forced pass-through).
        #[test]
        fn entity_dense_roundtrip(extra in prop::collection::vec(any::<u8>(), 0..256)) {
            let mut data: Vec<u8> = (0..=255).collect();
            data.extend_from_slice(b"&lt; &amp;");
            data.extend(extra);
            let s = EntityFolding;
            prop_assert_eq!(s.inverse(&s.forward(&data)).unwrap(), data);
        }
    }

    #[test]
    fn entity_known_values_roundtrip() {
        let s = EntityFolding;
        let inputs: [&[u8]; 7] = [
            b"",
            b"plain text, no entities",
            b"a &lt; b",
            b"&amp; and &apos; share the &a prefix",
            b"a bare & is left alone",
            b"a &notreal; entity is not folded",
            b"<a href=\"x\">1 &lt; 2 &amp; 3</a>",
        ];
        for input in inputs {
            assert_eq!(s.inverse(&s.forward(input)).unwrap(), input, "roundtrip failed for {input:?}");
        }
    }

    #[test]
    fn entity_folds_markup_but_passes_binary_and_entityless() {
        let s = EntityFolding;
        // Text containing a recognized entity takes the folded path (mode byte 1).
        let folded = s.forward(b"<a>1 &lt; 2 &amp; 3</a>");
        assert_eq!(folded[0], MODE_FOLDED);
        // Text with no entity has nothing to fold → pass-through (mode byte 0).
        let entityless = s.forward(b"just some plain prose here");
        assert_eq!(entityless[0], MODE_PASSTHROUGH);
        // A non-text byte stream → pass-through (mode byte 0).
        let binary = s.forward(&[0u8, 1, 2, 255, 254, b'&', b'l', b't', b';', 0, 200]);
        assert_eq!(binary[0], MODE_PASSTHROUGH);
    }

    #[test]
    fn entity_folded_payload_has_no_entity_spellings() {
        // The whole point: no entity spelling survives in the folded payload.
        let folded = EntityFolding.forward(b"<a>1 &lt; 2 &amp; 3 &quot;q&quot; &gt; &apos;</a>");
        assert_eq!(folded[0], MODE_FOLDED);
        let payload = &folded[6..]; // mode byte + five substitute bytes
        for entity in ENTITIES {
            assert!(
                !payload.windows(entity.len()).any(|w| w == entity),
                "folded payload still contains {:?}",
                std::str::from_utf8(entity).unwrap(),
            );
        }
    }
}
