//! LZ77-based compression program.

// Enable coverage attributes for nightly builds.
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
// False positive: deps used by the library or integration tests, not the binary.
#![allow(unused_crate_dependencies)]

use std::io;

use clap::Parser;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Decompress.
    #[arg(short)]
    d: bool,
}

fn main() -> io::Result<()> {
    let _args = Args::parse();
    io::copy(&mut io::stdin().lock(), &mut io::stdout().lock())?;
    Ok(())
}
