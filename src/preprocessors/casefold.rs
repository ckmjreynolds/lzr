//! Case-folding preprocessor.
//!
//! Folds ASCII letters to lowercase and records their original case with two
//! marker bytes. Downstream models then see merged contexts ("the"/"The") and,
//! because the output is ALWAYS lowercase, the 26 uppercase codes `0x41..=0x5A`
//! become absent — free for any later stage. Case is reconstructed on the decode
//! side from the marker stream, which is what earns those freed codes.
//!
//! Markers (both in the C0-control range absent from enwik8/enwik9):
//! - [`UPPER_ONE`] (one-shot): invert the case of the next letter relative to
//!   the current run mode. In the default lower mode this makes a single letter
//!   uppercase (`"The"` → `ONE 't' 'h' 'e'`); inside an upper run it makes one
//!   letter lowercase.
//! - [`UPPER_TOGGLE`]: flip the persistent upper-run mode, used for spans of two
//!   or more same-case letters (`"NASA"`, returning to `"world"`).
//!
//! Only ASCII `A`-`Z`/`a`-`z` are folded; bytes ≥ 0x80 (UTF-8) pass through.

use super::Preprocessor;

/// One-shot case inversion for the next letter.
pub(crate) const UPPER_ONE: u8 = 0x00;
/// Persistent upper-run mode toggle.
pub(crate) const UPPER_TOGGLE: u8 = 0x01;

/// Minimum same-case run length that earns a persistent toggle over one-shots.
const RUN_MIN: usize = 2;

/// Folds ASCII letters to lowercase, signalling case with [`UPPER_ONE`] and
/// [`UPPER_TOGGLE`].
#[derive(Debug, Default)]
pub(crate) struct CaseFold;

/// Count leading letters of case `want_upper` in `rest`, skipping non-letters
/// (the run persists across them), stopping at the first opposite-case letter or
/// once [`RUN_MIN`] is reached — enough to decide one-shot vs. toggle.
fn run_len(rest: &[u8], want_upper: bool) -> usize {
    let mut count = 0;
    for &b in rest {
        if b.is_ascii_alphabetic() {
            if b.is_ascii_uppercase() == want_upper {
                count += 1;
                if count >= RUN_MIN {
                    return count;
                }
            } else {
                return count;
            }
        }
    }
    count
}

impl Preprocessor for CaseFold {
    fn forward(&self, input: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(input.len());
        let mut run_upper = false;
        let mut i = 0;
        while i < input.len() {
            let b = input[i];
            if b.is_ascii_alphabetic() {
                let want_upper = b.is_ascii_uppercase();
                if want_upper != run_upper {
                    if run_len(&input[i..], want_upper) >= RUN_MIN {
                        out.push(UPPER_TOGGLE);
                        run_upper = want_upper;
                    } else {
                        out.push(UPPER_ONE);
                    }
                }
                out.push(b.to_ascii_lowercase());
            } else {
                out.push(b);
            }
            i += 1;
        }
        out
    }

    fn inverse(&self, input: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(input.len());
        let mut run_upper = false;
        let mut flip_next = false;
        for &b in input {
            if b == UPPER_TOGGLE {
                run_upper = !run_upper;
                continue;
            }
            if b == UPPER_ONE {
                flip_next = true;
                continue;
            }
            if b.is_ascii_lowercase() && (run_upper ^ flip_next) {
                out.push(b.to_ascii_uppercase());
            } else {
                out.push(b);
            }
            flip_next = false;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(s: &[u8]) {
        let cf = CaseFold;
        let folded = cf.forward(s);
        assert_eq!(cf.inverse(&folded), s, "roundtrip failed for {s:?}");
        assert!(
            folded.iter().all(|&b| !b.is_ascii_uppercase()),
            "folded stream still contains an ASCII uppercase byte: {s:?}"
        );
    }

    #[test]
    fn roundtrips_mixed_case() {
        roundtrip(b"");
        roundtrip(b"the quick brown fox");
        roundtrip(b"The Quick Brown Fox");
        roundtrip(b"NASA and the FBI; U.S.A. report");
        roundtrip(b"iPhone, mRNA, MeV, PhD, ABcDE");
        roundtrip(b"ALLCAPS then lower then MoRE");
        roundtrip(b"camelCaseAndPascalCase");
        roundtrip("caf\u{e9} r\u{e9}sum\u{e9} \u{2014} UTF8 \u{c4}\u{d6}\u{dc}".as_bytes());
    }

    #[test]
    fn roundtrips_enwik8_slice() {
        let Ok(bytes) = std::fs::read("assets/enwik8") else {
            return; // skip when the corpus is absent
        };
        roundtrip(&bytes[5_000_000..5_200_000]);
    }
}
