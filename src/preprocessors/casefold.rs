//! ASCII case-folding byte preprocessor.

use super::{TEXT_FRACTION, Transform, is_text_byte, spare_bytes};

/// ASCII case-folding byte preprocessor.
///
/// Lowercases `A`–`Z` and re-encodes the discarded case information as two inline
/// control symbols — a **shift** (capitalize the next letter) and a **caps-lock**
/// (toggle capitalize-all until seen again). Folding the alphabet to mostly
/// lowercase gives the downstream tokenizer and entropy models fewer, more
/// predictable symbols to model; the two extra control symbols cost a little on
/// this stage's own byte count but pay off in the coded stream.
///
/// The two control-symbol byte *values* are chosen per input as bytes that never
/// occur in it, so they are unambiguous inside the transformed payload. When the
/// input is not text, has no uppercase letter to fold, or has fewer than two spare
/// byte values, the stage falls back to an exact pass-through. The stream is
/// self-describing (there is no side channel to [`Transform::inverse`]): a leading
/// mode byte records which path was taken, and the folded path stores its two
/// chosen symbol bytes in the header.
pub(crate) struct CaseFolding;

/// Header mode byte: the input was passed through unchanged.
const MODE_PASSTHROUGH: u8 = 0;
/// Header mode byte: the payload is case-folded (followed by the shift/caps bytes).
const MODE_FOLDED: u8 = 1;

impl Transform<u8> for CaseFolding {
    #[expect(clippy::cast_precision_loss, reason = "byte counts are far under 2^53")]
    fn forward(&self, input: &[u8]) -> Vec<u8> {
        // One pass: presence map, uppercase-letter flag, and text-byte count.
        let mut present = [false; 256];
        let mut has_upper = false;
        let mut text_bytes = 0usize;
        for &byte in input {
            present[usize::from(byte)] = true;
            if byte.is_ascii_uppercase() {
                has_upper = true;
                // The folded payload emits this letter lowercased, so its lowercase
                // form must not be chosen as a control symbol (it may be a byte the
                // original input never contained).
                present[usize::from(byte.to_ascii_lowercase())] = true;
            }
            if is_text_byte(byte) {
                text_bytes += 1;
            }
        }

        // Fall back to pass-through unless there is something to fold, the input
        // looks like text, and two spare byte values exist for the control symbols.
        let is_text = !input.is_empty() && text_bytes as f64 >= input.len() as f64 * TEXT_FRACTION;
        let Some([shift, caps]) = (has_upper && is_text).then(|| spare_bytes::<2>(&present)).flatten() else {
            let mut out = Vec::with_capacity(input.len() + 1);
            out.push(MODE_PASSTHROUGH);
            out.extend_from_slice(input);
            return out;
        };

        let mut out = Vec::with_capacity(input.len() + 3);
        out.push(MODE_FOLDED);
        out.push(shift);
        out.push(caps);
        let mut i = 0;
        while i < input.len() {
            if input[i].is_ascii_uppercase() {
                // Measure the maximal run of consecutive uppercase letters.
                let start = i;
                while i < input.len() && input[i].is_ascii_uppercase() {
                    i += 1;
                }
                let run = &input[start..i];
                if run.len() == 1 {
                    out.push(shift);
                    out.push(run[0].to_ascii_lowercase());
                } else {
                    out.push(caps);
                    out.extend(run.iter().map(u8::to_ascii_lowercase));
                    out.push(caps);
                }
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
                let [shift, caps, payload @ ..] = rest else {
                    anyhow::bail!("case-folding header truncated: missing shift/caps symbols");
                };
                let (shift, caps) = (*shift, *caps);
                let mut out = Vec::with_capacity(payload.len());
                let mut caps_on = false;
                let mut shift_pending = false;
                for &byte in payload {
                    if byte == caps {
                        caps_on = !caps_on;
                    } else if byte == shift {
                        shift_pending = true;
                    } else {
                        let upper = (caps_on || shift_pending) && byte.is_ascii_lowercase();
                        out.push(if upper {
                            byte.to_ascii_uppercase()
                        } else {
                            byte
                        });
                        shift_pending = false;
                    }
                }
                Ok(out)
            }
            other => anyhow::bail!("unknown case-folding mode byte {other}"),
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        /// Case folding must round-trip *any* byte input exactly — text and binary,
        /// via both the folded and pass-through paths.
        #[test]
        fn casefold_roundtrip(data in prop::collection::vec(any::<u8>(), 0..1024)) {
            let s = CaseFolding;
            prop_assert_eq!(s.inverse(&s.forward(&data)).unwrap(), data);
        }

        /// Inverse runs on decoded, possibly-corrupt data — it may error but must
        /// never panic.
        #[test]
        fn casefold_inverse_never_panics(data in prop::collection::vec(any::<u8>(), 0..1024)) {
            let s = CaseFolding;
            drop(s.inverse(&data));
        }

        /// Case folding must round-trip inputs that saturate the whole byte range
        /// (no two spare values → forced pass-through).
        #[test]
        fn casefold_dense_roundtrip(extra in prop::collection::vec(any::<u8>(), 0..256)) {
            let mut data: Vec<u8> = (0..=255).collect();
            data.extend(extra);
            let s = CaseFolding;
            prop_assert_eq!(s.inverse(&s.forward(&data)).unwrap(), data);
        }
    }

    #[test]
    fn casefold_known_values_roundtrip() {
        let s = CaseFolding;
        let inputs: [&[u8]; 6] =
            [b"", b"A", b"HELLO world", b"NASA and HTML", b"MixedCaseABCdef", b"lower only, no caps"];
        for input in inputs {
            assert_eq!(s.inverse(&s.forward(input)).unwrap(), input, "roundtrip failed for {input:?}");
        }
    }

    #[test]
    fn casefold_folds_text_but_passes_binary() {
        let s = CaseFolding;
        // Mostly-uppercase text takes the folded path (mode byte 1).
        let text = s.forward(b"THE QUICK BROWN Fox");
        assert_eq!(text[0], MODE_FOLDED);
        // A byte stream that is not text takes the pass-through path (mode byte 0).
        let binary = s.forward(&[0u8, 1, 2, 255, 254, b'A', 0, 200]);
        assert_eq!(binary[0], MODE_PASSTHROUGH);
    }

    #[test]
    fn casefold_lowers_the_alphabet() {
        // The folded payload should carry no ASCII uppercase letters — the whole point.
        let folded = CaseFolding.forward(b"NASA Reads XML");
        assert_eq!(folded[0], MODE_FOLDED);
        assert!(!folded[3..].iter().any(u8::is_ascii_uppercase), "folded payload still has uppercase");
    }
}
