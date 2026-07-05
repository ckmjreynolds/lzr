//! Re-Pair grammar tokenization as a byte→byte transform.
//!
//! [`RepairTokenizer`] implements [`crate::transform::Transform`]: it builds a Re-Pair grammar over
//! the input and serializes it (uleb128-u22) back to a byte stream, so it composes as an ordinary
//! pipeline stage between the byte preprocessors and the entropy coder. "Tokenization off" is simply
//! the absence of this stage from the pipeline — there is no identity tokenizer.

mod repair;

#[cfg_attr(feature = "bench-internals", visibility::make(pub))]
pub(crate) use self::repair::RepairTokenizer;
pub(crate) use self::repair::{DEFAULT_NUM_TOKENS, MIN_NUM_TOKENS};
