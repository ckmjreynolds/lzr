# LZR Format Specification

**Version:** 1.0
**Extension:** `.lzr`

## Overview

LZR is an LZ77-based compression format. A stream has three parts:

```
┌────Header────┐┌───Frame────┐┌───Frame────┐     ┌────────Footer────────┐
│┌────────────┐││┌──────────┐││┌──────────┐│     │┌────────┐┌──────────┐│
││ LZR | 0x00 ││││ 3+ bytes ││││ 3+ bytes ││ ... ││ Length ││ Adler-32 ││
│└────────────┘││└──────────┘││└──────────┘│     │└────────┘└──────────┘│
└──────────────┘└────────────┘└────────────┘     └──────────────────────┘
```

All multi-byte integers are little-endian unless noted otherwise. All frames are byte-aligned.

## Header

| Offset | Size | Value | Description |
|-|-|-|-|
| 0 | 3 bytes | `0x4C 0x5A 0x52` (`LZR`) | Magic number |
| 3 | 1 byte | `0x00` | Format version |

Version `0x00` is format version 1.0. All other values are reserved. A decoder must reject unrecognized magic numbers or versions.

## Frames

Every frame has the same structure:

```
Token (1B) │ Distance (2B LE)    │ Lit Ext* │ Match Ext* │ Literals*
LLLMMMMM   │ DDDDDDDD | DDDDDDDD │ 0+ bytes │ 0+ bytes   │ 0+ bytes
```

Fields marked `*` are conditionally present. A frame is always at least 3 bytes (token + distance).

### Token Byte

```
  7   6   5   4   3   2   1   0
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ L   L   L │ M   M   M   M   M │
└───┴───┴───┴───┴───┴───┴───┴───┘
```

- **L** (bits 7–5): literal count, `0..=7`. A value of 7 triggers a literal length extension.
- **M** (bits 4–0): match length field, decoded as follows:

| MMMMM | Match Length | Type |
|-|-|-|
| `0..=5` | `-9..=-4` | Reverse match |
| `6` | `0` | Literal-only (no match) |
| `7..=31` | `4..=28` | Forward match |

Match length is a signed 16-bit integer (i16). MMMMM=0 (length -9) triggers a negative match extension. MMMMM=31 (length 28) triggers a positive match extension.

### Distance

Bytes 1–2 of the frame. A 16-bit little-endian unsigned integer (`0..=65,535`). Distance 0 is reserved for the end-of-stream sentinel; valid match distances are `1..=65,535`.

When match length is 0 (literal-only frame), the distance is ignored but still present. Encoders may write any value; decoders must not interpret it.

### End of Stream

Token `0x00` with distance `0x0000`. This encodes zero literals with match length -9 at distance 0 — a reserved sentinel that signals the end of the frame sequence.

### Length Extension

When a length field is at its maximum base value, one or more extension bytes follow. The extension chain adds to the base value:

1. Read one byte.
2. Add its value to the total.
3. If the byte equals 255, repeat from step 1.
4. If the byte is less than 255, the chain is complete.

**Literal length extension** (L=7): appears after the distance bytes, before any match extension.

`total_literals = 7 + sum(ext_bytes)`

Literal length is a signed 16-bit quantity (i16). Encoders must not produce literal lengths outside 0..=32,767; decoder behavior for out-of-range lengths is undefined.

**Positive match extension** (MMMMM=31, length 28): appears after literal extension bytes (if any), before literal data.

`total_match = 28 + sum(ext_bytes)`

**Negative match extension** (MMMMM=0, length -9): appears after literal extension bytes (if any), before literal data. Each extension byte increases the magnitude:

`total_match = -9 - sum(ext_bytes)`

Match length is a signed 16-bit quantity (i16). Encoders must not produce match lengths outside -32,768..=32,767; decoder behavior for out-of-range lengths is undefined.

### Frame Order Summary

```
┌────────┐┌──────────┐┌───────────────┐┌───────────────┐┌──────────┐
│ Token  ││ Distance ││ Lit Ext (L=7) ││ Match Ext     ││ Literals │
│ 1 byte ││ 2 bytes  ││ 0+ bytes      ││ (M=0 or M=31) ││ L bytes  │
│        ││          ││               ││ 0+ bytes      ││          │
└────────┘└──────────┘└───────────────┘└───────────────┘└──────────┘
```

## Copy Semantics

**Forward copy (positive length):**

```
for i in 0..length:
    output[pos + i] = output[pos - distance + i]
```

When length > distance, the copy overlaps its own output. Each byte is copied individually — this naturally provides run-length encoding (e.g., distance 1, length 33 repeats the last byte 33 times).

**Reverse copy (negative length):**

```
for i in 0..|length|:
    output[pos + i] = output[pos - distance - i]
```

This matches reversed patterns. For example, `stressed` in the output can produce `desserts` via a reverse copy.

## Sliding Window

The decoder maintains a **65,536 byte** (64 KiB) sliding window of decompressed output. The window must be filled with `0x00` before decompression begins. Any read from a position before the start of decompressed output returns `0x00` (from the zero-initialized window).

Encoders must not emit copies whose read range falls outside the window. Decoder behavior for such copies is undefined.

## Conformance

Encoders are the gatekeepers of validity. Where this specification states an encoder "must not" produce a particular encoding, that encoding is **malformed**. Decoder behavior when processing malformed input is undefined unless explicitly stated otherwise.

The footer (uncompressed length + Adler-32 checksum) is the authoritative integrity check. A decoder that processes a malformed encoding and produces incorrect output will detect the corruption at footer verification. Decoders are not required to validate individual frames beyond what is necessary to decode them.

## Footer

Immediately after the end-of-stream marker.

| Field | Encoding | Description |
|-|-|-|
| Uncompressed length | [ULEB128](https://en.wikipedia.org/wiki/LEB128)-encoded u64 (max 9 bytes) | Original byte count |
| Checksum | 4 bytes, little-endian | [Adler-32](https://en.wikipedia.org/wiki/Adler-32) of uncompressed data |

### ULEB128 Encoding

The uncompressed length uses ULEB128 (Unsigned Little-Endian Base 128) encoding, bounded to 9 bytes maximum:

- Bytes 1–8 use standard ULEB128: 7 payload bits (bits 6–0) and 1 continuation bit (bit 7). If the continuation bit is set, another byte follows.
- If a 9th byte is needed, all 8 bits are payload (no continuation bit).
- Total capacity: 8 × 7 + 8 = 64 bits, sufficient for a full u64.

The decoder must verify that the decompressed byte count matches the stored length and that the Adler-32 matches. A mismatch indicates corruption.

**Note:** The uncompressed length is not authenticated until the checksum is verified. Decoders that use it for pre-allocation or space checks should treat it as untrusted input.

## Stream Concatenation

A file may contain multiple concatenated streams. If data remains after a footer, the decoder must treat it as a new stream beginning with a header.

## Worked Example

Compressing `Hi` (0x48 0x69):

**Frame 1 — Literal-only:**

Token = `0x46` = `010_00110`: L=2 (2 literal bytes), M=6 (match length 0, literal-only).
Distance = `0x00 0x00` (ignored).
Literals = `0x48 0x69`.

**Frame 2 — End of Stream:**

Token = `0x00` = `000_00000`: L=0, M=0 (match length -9).
Distance = `0x00 0x00` (0, reserved sentinel).

**Footer:**

- ULEB128(2) = `0x02` (1 byte, no continuation needed).
- Adler-32("Hi"): A = 1 + 72 + 105 = 178 (0xB2), B = 0 + 73 + 178 = 251 (0xFB). Checksum = 0x00FB00B2, little-endian: `B2 00 FB 00`.

**Complete stream (17 bytes):**

```
4C 5A 52 00  46 00 00 48 69  00 00 00  02 B2 00 FB 00
├─ Header ─┤ ├── Frame 1 ──┤ ├─ EOS ─┤ ├── Footer ──┤
```
