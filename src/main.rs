//! LZ77-based compression CLI.

#![allow(unused_features)]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use std::process::ExitCode;

mod cli;

fn main() -> ExitCode {
    cli::run()
}
