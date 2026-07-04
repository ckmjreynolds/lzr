# Changelog

All notable changes to this project will be documented in this file.

The format is loosely based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Added

- Pluggable compression pipeline skeleton: byte preprocessors (`Vec<u8> -> Vec<u8>`),
  tokenizers (`Vec<u8> -> Vec<u15>`), token preprocessors (`Vec<u15> -> Vec<u15>`),
  and context-mixed entropy models, each with NULL/identity default stages.
- A carryless binary range coder and a logistic context mixer over order-0 and
  order-1 token models.
- Self-describing container format (magic, version, reserved pipeline profile, and
  an Adler-32 footer over the original bytes); see `docs/FORMAT.md`.
- Public `compress`/`decompress` API and an `lzr <input> <output>` CLI (`-d` to
  decompress), backed by integration and round-trip tests plus a `divan` codec bench.
