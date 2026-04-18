//! LZR compression CLI.
//!
//! Subcommands: `compress`, `decompress`, `info`. Use `-` for stdin/stdout.

#![allow(unused_features)]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use lzr::{Reader, Writer};

#[derive(Parser)]
#[command(name = "lzr", version, about = "LZR compression CLI")]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compress INPUT to OUTPUT. Use `-` for stdin or stdout.
    Compress {
        /// Compression level (1-9). Levels 1-3 write raw tokens (no entropy
        /// coding); levels 4-9 enable adaptive arithmetic coding.
        #[arg(short, long, default_value_t = lzr::DEFAULT_LEVEL, value_parser = clap::value_parser!(u8).range(1..=9))]
        level: u8,
        /// Input path, or `-` for stdin.
        input: PathBuf,
        /// Output path, or `-` for stdout.
        output: PathBuf,
    },
    /// Decompress INPUT to OUTPUT. Use `-` for stdin or stdout.
    Decompress {
        /// Input path, or `-` for stdin.
        input: PathBuf,
        /// Output path, or `-` for stdout.
        output: PathBuf,
    },
    /// Print metadata about an LZR file.
    Info {
        /// LZR file to inspect.
        file: PathBuf,
    },
}

fn main() -> ExitCode {
    match Cli::parse().cmd {
        Command::Compress {
            level,
            input,
            output,
        } => run(compress(level, &input, &output)),
        Command::Decompress {
            input,
            output,
        } => run(decompress(&input, &output)),
        Command::Info {
            file,
        } => run(info(&file)),
    }
}

fn run(result: io::Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("lzr: {e}");
            ExitCode::FAILURE
        }
    }
}

fn compress(level: u8, input: &Path, output: &Path) -> io::Result<()> {
    let mut src = open_input(input)?;
    let dst = open_output(output)?;
    let mut w = Writer::with_level(dst, level).map_err(io::Error::from)?;
    io::copy(&mut src, &mut w)?;
    w.seal().map_err(io::Error::from)?;
    Ok(())
}

fn decompress(input: &Path, output: &Path) -> io::Result<()> {
    // Reader requires Read + Seek, so stdin isn't supported here.
    if input == Path::new("-") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "decompress does not support stdin (-); LZR requires a seekable source",
        ));
    }
    let file = File::open(input)?;
    let mut r = Reader::new(file).map_err(io::Error::from)?;
    let mut dst = open_output(output)?;
    io::copy(&mut r, &mut dst)?;
    Ok(())
}

fn info(file: &Path) -> io::Result<()> {
    let r = Reader::new(File::open(file)?).map_err(io::Error::from)?;
    let state = if r.is_sealed() {
        "sealed"
    } else {
        "unsealed"
    };
    let encoding = if r.is_entropy_coded() {
        "entropy"
    } else {
        "raw"
    };
    println!("file:     {}", file.display());
    println!("state:    {state}");
    println!("encoding: {encoding}");
    println!("bytes:    {}", r.len());
    println!("lines:    {}", r.lines());
    Ok(())
}

fn open_input(path: &Path) -> io::Result<Box<dyn Read>> {
    if path == Path::new("-") {
        Ok(Box::new(io::stdin().lock()))
    } else {
        Ok(Box::new(File::open(path)?))
    }
}

fn open_output(path: &Path) -> io::Result<Box<dyn Write>> {
    if path == Path::new("-") {
        Ok(Box::new(io::stdout().lock()))
    } else {
        Ok(Box::new(File::create(path)?))
    }
}
