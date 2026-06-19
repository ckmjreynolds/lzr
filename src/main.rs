//! lzr — a context-mixing compressor (Hutter Prize attempt, v9).
//!
//! Usage: `lzr <input> <output>`. The direction is chosen by the input
//! extension: a `.lzr` input decodes; anything else encodes. The pass reports
//! bits-per-byte (on encode) and elapsed time / throughput.

// `unreachable_pub = "deny"` (Cargo.toml) wants crate-internal items declared
// `pub(crate)`; clippy's nursery `redundant_pub_crate` wants `pub` because the
// modules are private. The two directly conflict, so the project keeps
// `unreachable_pub` and silences the redundant-pub-crate noise here (the
// Cargo.toml-documented home for per-lint exceptions).
#![allow(clippy::redundant_pub_crate)]

mod codec;
mod coder;
mod mixer;
mod models;
mod preprocessors;

use std::path::Path;
use std::time::Instant;

use clap::Parser;

/// Command-line arguments.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// Input file. A `.lzr` extension decodes; anything else encodes.
    input: String,
    /// Output file.
    output: String,
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();
    let decoding = Path::new(&cli.input)
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("lzr"));

    let input = std::fs::read(&cli.input)?;
    let start = Instant::now();
    let output = if decoding {
        codec::decode(&input)
    } else {
        codec::encode(&input)
    };
    let elapsed = start.elapsed().as_secs_f64();
    std::fs::write(&cli.output, &output)?;

    report(decoding, input.len(), output.len(), elapsed);
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn report(decoding: bool, in_len: usize, out_len: usize, elapsed: f64) {
    let processed = if decoding { out_len } else { in_len };
    let mbps = (processed as f64 / 1e6) / elapsed.max(1e-9);
    if decoding {
        println!("decoded {out_len} bytes in {elapsed:.2}s ({mbps:.2} MB/s)");
    } else {
        let bpb = out_len as f64 * 8.0 / in_len as f64;
        println!(
            "encoded {in_len} -> {out_len} bytes  {bpb:.4} bpb  in {elapsed:.2}s ({mbps:.2} MB/s)"
        );
    }
}
