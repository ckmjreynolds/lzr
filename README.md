# LZR

[![Crates.io](https://img.shields.io/crates/v/lzr.svg)](https://crates.io/crates/lzr)
[![Documentation](https://docs.rs/lzr/badge.svg)](https://docs.rs/lzr)
[![CI](https://github.com/ckmjreynolds/lzr/actions/workflows/ci.yml/badge.svg)](https://github.com/ckmjreynolds/lzr/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ckmjreynolds/lzr/graph/badge.svg)](https://codecov.io/gh/ckmjreynolds/lzr)
[![License](https://img.shields.io/crates/l/lzr.svg)](https://github.com/ckmjreynolds/lzr#license)

A general-purpose compression library and CLI written in Rust. What sets it apart is
**random access** into compressed streams (seek to any offset without decoding everything
before it) and **tailing** (read a stream that is still being appended to).

**Status:** pre-release. The wire format is unstable and will change without notice until 1.0.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contributing

Contributions welcome. Please run `bash build.sh` before opening a PR — it runs
fmt, clippy, the test suite (including a proptest corpus), and coverage.
