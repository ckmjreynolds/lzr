//! CLI interface for the LZR compression tool.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;

use lzr::error::Result;
use lzr::options::{DEFAULT_LEVEL, EncodeOptions, MAX_LEVEL, MIN_LEVEL};
use lzr::{decode, encode};

/// I/O buffer capacity (64 KiB, matching the decoder window size).
const IO_BUF_CAP: usize = 65_536;

/// Fast LZ77-based compression.
#[derive(Parser, Debug)]
#[command(name = "lzr", version)]
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct Args {
    /// Decompress mode.
    #[arg(short, long)]
    decompress: bool,

    /// Compression level (1-9).
    #[arg(short, long, default_value_t = DEFAULT_LEVEL as u8,
          value_parser = clap::value_parser!(u8).range(1..=9))]
    level: u8,

    /// Alias for --level 1.
    #[arg(long)]
    fast: bool,

    /// Alias for --level 9.
    #[arg(long)]
    best: bool,

    /// Thread count (0 = auto).
    #[arg(short = 'T', long, default_value_t = 0)]
    threads: usize,

    /// Write to stdout instead of a file.
    #[arg(short = 'c', long)]
    stdout: bool,

    /// Keep input files after processing.
    #[arg(short, long)]
    keep: bool,

    /// Force overwrite of output files.
    #[arg(short, long)]
    force: bool,

    /// Print filename and ratio to stderr.
    #[arg(short, long)]
    verbose: bool,

    /// Suppress non-error messages.
    #[arg(short, long)]
    quiet: bool,

    /// Test compressed file integrity.
    #[arg(short, long)]
    test: bool,

    /// Input files (reads stdin if empty or `-`).
    files: Vec<PathBuf>,
}

pub(crate) fn run() -> ExitCode {
    let args = Args::parse();

    if let Err(e) = execute(&args) {
        if !args.quiet {
            eprintln!("lzr: {e}");
        }
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

const fn effective_level(args: &Args) -> usize {
    if args.fast {
        MIN_LEVEL
    } else if args.best {
        MAX_LEVEL
    } else {
        args.level as usize
    }
}

fn is_stdin(files: &[PathBuf]) -> bool {
    files.is_empty() || (files.len() == 1 && files[0].as_os_str() == "-")
}

fn output_path(input: &Path, decompress: bool) -> Option<PathBuf> {
    if decompress {
        input.to_string_lossy().strip_suffix(".lzr").map(PathBuf::from)
    } else {
        let mut name = input.as_os_str().to_owned();
        name.push(".lzr");
        Some(PathBuf::from(name))
    }
}

fn execute(args: &Args) -> Result<()> {
    let options = EncodeOptions::new().level(effective_level(args)).threads(args.threads).verbose(args.verbose);

    if is_stdin(&args.files) {
        process_stdin(args, &options)
    } else {
        for file in &args.files {
            process_file(args, &options, file)?;
        }
        Ok(())
    }
}

fn process_stdin(args: &Args, options: &EncodeOptions) -> Result<()> {
    let mut reader = io::BufReader::with_capacity(IO_BUF_CAP, io::stdin().lock());

    if args.decompress || args.test {
        if args.test {
            decode::decode(&mut reader, &mut io::sink())?;
        } else {
            let mut writer = io::BufWriter::with_capacity(IO_BUF_CAP, io::stdout().lock());
            decode::decode(&mut reader, &mut writer)?;
        }
    } else {
        let mut writer = io::BufWriter::with_capacity(IO_BUF_CAP, io::stdout().lock());
        encode::encode(&mut reader, &mut writer, options)?;
    }

    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn process_file(args: &Args, options: &EncodeOptions, input_path: &Path) -> Result<()> {
    if args.test {
        let file = fs::File::open(input_path)?;
        let mut input = io::BufReader::new(file);
        decode::decode(&mut input, &mut io::sink())?;
        if args.verbose {
            eprintln!("{}: ok", input_path.display());
        }
        return Ok(());
    }

    if args.stdout {
        let file = fs::File::open(input_path)?;
        let mut input = io::BufReader::new(file);
        let stdout = io::stdout().lock();
        let mut output = io::BufWriter::new(stdout);
        if args.decompress {
            decode::decode(&mut input, &mut output)?;
        } else {
            encode::encode(&mut input, &mut output, options)?;
        }
        return Ok(());
    }

    let out_path = output_path(input_path, args.decompress).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, format!("{}: unknown suffix -- ignored", input_path.display()))
    })?;

    if out_path.exists() && !args.force {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("{}: already exists; use -f to overwrite", out_path.display()),
        )
        .into());
    }

    let input_len = fs::metadata(input_path)?.len();
    let file_in = fs::File::open(input_path)?;
    let mut input = io::BufReader::new(file_in);
    let file_out = fs::File::create(&out_path)?;
    let mut output = io::BufWriter::new(file_out);

    if args.decompress {
        decode::decode(&mut input, &mut output)?;
    } else {
        encode::encode(&mut input, &mut output, options)?;
    }

    if args.verbose {
        let output_len = fs::metadata(&out_path)?.len();
        if input_len > 0 {
            let ratio = output_len as f64 / input_len as f64;
            eprintln!("{}: {ratio:.1}%", input_path.display());
        } else {
            eprintln!("{}: 0 -> {output_len}", input_path.display());
        }
    }

    if !args.keep {
        fs::remove_file(input_path)?;
    }

    Ok(())
}
