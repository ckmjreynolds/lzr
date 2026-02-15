//! LZ77-based compression program.

// Enable coverage attributes for nightly builds.
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;

#[derive(Parser)]
#[command(
    version = about(),
    about = about(),
)]
#[allow(clippy::struct_excessive_bools)]
struct Args {
    /// Decompress (default is compress).
    #[arg(short, long = "decompress", visible_alias = "uncompress")]
    d: bool,

    /// Write to stdout, keep original files.
    #[arg(short, long = "stdout", visible_alias = "to-stdout")]
    c: bool,

    /// Keep original files.
    #[arg(short, long)]
    keep: bool,

    /// Force overwrite of output files.
    #[arg(short, long)]
    force: bool,

    /// Test compressed file integrity.
    #[arg(short, long)]
    test: bool,

    /// List compressed file contents (not yet implemented).
    #[arg(short, long)]
    list: bool,

    /// Verbose: display compression statistics.
    #[arg(short, long)]
    verbose: bool,

    /// Quiet: suppress warnings.
    #[arg(short, long)]
    quiet: bool,

    /// Compress fastest.
    #[arg(short = '1', long = "fast")]
    fast: bool,

    /// Compress best.
    #[arg(short = '9', long = "best")]
    best: bool,

    /// Filename suffix (default: .lzr).
    #[arg(short = 'S', long, default_value = ".lzr")]
    suffix: String,

    /// Files to process (stdin if none).
    files: Vec<PathBuf>,
}

fn main() -> ExitCode {
    let args = Args::parse();

    if args.list {
        eprintln!("lzr: --list is not yet implemented");
        return ExitCode::FAILURE;
    }

    let decompress = args.d || args.test;

    if args.files.is_empty() {
        if let Err(err) = filter_mode(decompress, &args) {
            eprintln!("lzr: {err}");
            return ExitCode::FAILURE;
        }
        return ExitCode::SUCCESS;
    }

    let mut failed = false;
    for path in &args.files {
        if let Err(err) = process_file(path, decompress, &args) {
            eprintln!("lzr: {}: {err}", path.display());
            failed = true;
        }
    }

    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn filter_mode(decompress: bool, args: &Args) -> io::Result<()> {
    let mut stdin = io::stdin().lock();
    let mut stdout = io::stdout().lock();

    let bytes = if decompress {
        lzr::decode(&mut stdin, &mut stdout)?
    } else {
        lzr::encode(&mut stdin, &mut stdout)?
    };

    if args.verbose {
        eprintln!("{bytes} bytes written");
    }

    Ok(())
}

fn process_file(path: &Path, decompress: bool, args: &Args) -> io::Result<()> {
    let to_stdout = args.c;
    let keep = args.keep || args.c || args.test;
    let testing = args.test;

    let output_path = if to_stdout || testing {
        None
    } else if decompress {
        let name = path.to_string_lossy();
        if let Some(stripped) = name.strip_suffix(&args.suffix) {
            Some(PathBuf::from(stripped))
        } else {
            if !args.quiet {
                eprintln!("lzr: {}: unknown suffix -- ignored", path.display());
            }
            return Ok(());
        }
    } else {
        let mut name = path.as_os_str().to_owned();
        name.push(&args.suffix);
        Some(PathBuf::from(name))
    };

    // Check output exists
    if let Some(ref out) = output_path {
        if out.exists() && !args.force {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("{} already exists; use -f to overwrite", out.display()),
            ));
        }
    }

    let input = File::open(path)?;
    let input_len = input.metadata()?.len();
    let mut reader = BufReader::new(input);

    let result = if testing {
        // Decode to sink to test integrity.
        let bytes = lzr::decode(&mut reader, &mut io::sink())?;
        if args.verbose {
            eprintln!("{}: ok", path.display());
        }
        bytes
    } else if to_stdout {
        let mut stdout = BufWriter::new(io::stdout().lock());
        if decompress {
            lzr::decode(&mut reader, &mut stdout)?
        } else {
            lzr::encode(&mut reader, &mut stdout)?
        }
    } else {
        let out = output_path.as_ref().expect("output path must be set");
        let output_file = File::create(out)?;
        let mut writer = BufWriter::new(output_file);
        let result = if decompress {
            lzr::decode(&mut reader, &mut writer)
        } else {
            lzr::encode(&mut reader, &mut writer)
        };

        if let Err(err) = result {
            // Clean up partial output.
            let _ = fs::remove_file(out);
            return Err(err.into());
        }

        result?
    };

    // Print stats.
    if args.verbose && !testing {
        eprintln!("{}: {} -> {} bytes", path.display(), input_len, result);
    }

    // Remove original unless keeping.
    if !keep {
        fs::remove_file(path)?;
    }

    Ok(())
}

const fn about() -> &'static str {
    concat!(
        env!("CARGO_PKG_NAME"),
        " - v",
        env!("CARGO_PKG_VERSION"),
        " - ",
        env!("CARGO_PKG_DESCRIPTION"),
        "\n",
        "Copyright (C) ",
        env!("COPYRIGHT_YEARS"),
        " ",
        env!("CARGO_PKG_AUTHORS"),
        "\n",
        "License: ",
        env!("CARGO_PKG_LICENSE")
    )
}
