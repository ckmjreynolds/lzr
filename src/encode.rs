//! LZR compression encoder.

use std::io::{Read, Write};

use crate::error::Result;
use crate::options::EncodeOptions;

#[allow(missing_docs)]
#[allow(clippy::missing_errors_doc)]
pub fn encode(_input: &mut impl Read, _output: &mut impl Write, _options: &EncodeOptions) -> Result<()> {
    todo!()
}
