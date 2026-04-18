# LZR

[![Crates.io](https://img.shields.io/crates/v/lzr.svg)](https://crates.io/crates/lzr)
[![Documentation](https://docs.rs/lzr/badge.svg)](https://docs.rs/lzr)
[![CI](https://github.com/ckmjreynolds/lzr/actions/workflows/ci.yml/badge.svg)](https://github.com/ckmjreynolds/lzr/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ckmjreynolds/lzr/graph/badge.svg)](https://codecov.io/gh/ckmjreynolds/lzr)
[![License](https://img.shields.io/crates/l/lzr.svg)](https://github.com/ckmjreynolds/lzr#license)

LZR is a compression format and library that pairs an LZ77 stage with an adaptive
arithmetic coder. Data is packed into fixed 256 KiB blocks ("Sonnets") so that an
archive can be:

- **Streamed** end-to-end.
- **Seeked** to any byte offset or line number without decompressing preceding blocks.
- **Appended** to while unsealed, then sealed once complete.

The wire format is specified in [`docs/FORMAT.md`](docs/FORMAT.md).

## Usage

```rust
use std::io::{Cursor, Read, Write};

// Compress
let mut w = lzr::Writer::new(Vec::new()).unwrap();
w.write_all(b"Hello, world!\n").unwrap();
let compressed = w.seal().unwrap();

// Decompress
let mut r = lzr::Reader::new(Cursor::new(compressed)).unwrap();
let mut output = String::new();
r.read_to_string(&mut output).unwrap();
assert_eq!(output, "Hello, world!\n");
```

### Random access

`Reader` implements `std::io::Seek` and adds a line-based seek. Block footers are
read lazily and cached.

```rust,no_run
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

let mut r = lzr::Reader::new(File::open("data.lzr").unwrap()).unwrap();
r.seek_to_line(10_000).unwrap();          // jump to line 10 000
let mut chunk = [0u8; 256];
r.read_exact(&mut chunk).unwrap();        // read 256 bytes from that line
r.seek(SeekFrom::Start(1_048_576)).unwrap();
```

### Append

Open an existing unsealed archive and continue writing. `open` refuses to touch
already-sealed archives.

```rust,no_run
use std::fs::OpenOptions;
use std::io::Write;

let file = OpenOptions::new().read(true).write(true).open("data.lzr").unwrap();
let mut w = lzr::Writer::open(file).unwrap();
w.write_all(b"more data\n").unwrap();
w.seal().unwrap();
```

### Compression level

`Writer::with_level(dest, level)` accepts levels `1..=9`. L1 uses a single
hash-table lookup (lz4-style); L2–L6 walk a hash chain; L7–L9 fall back to the
[`wabi_tree`](https://crates.io/crates/wabi_tree) B+tree finder. Higher levels
are slower but compress more tightly.

## Status

Pre-1.0. The format version is `0x00` and the encoder/decoder interoperate
bit-exactly with the spec. Breaking changes to the Rust API may still happen.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contributing

Contributions welcome. Please run `bash build.sh` before opening a PR — it runs
fmt, clippy, the test suite (including a proptest corpus), and coverage.
