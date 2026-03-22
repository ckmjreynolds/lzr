# LZR Format Specification

**Version:** 1.0
**Extension:** `.lzr`

## Overview

LZR is an LZ77-based compression format. A stream has three parts:

```
┌────Header────┐┌────Frame────┐┌────Frame────┐     ┌────────Footer────────┐
│┌────────────┐││┌───────────┐││┌───────────┐│     │┌────────┐┌──────────┐│
││ LZR | 0x00 ││││ 1-3 bytes ││││ 1-3 bytes ││ ... ││ Length ││ Adler-32 ││
│└────────────┘││└───────────┘││└───────────┘│     │└────────┘└──────────┘│
└──────────────┘└─────────────┘└─────────────┘     └──────────────────────┘
```

All multi-byte integers are little-endian unless noted otherwise. All frames are byte-aligned.

## Header

| Offset | Size | Value | Description |
|-|-|-|-|
| 0 | 3 bytes | `0x4C 0x5A 0x52` (`LZR`) | Magic number |
| 3 | 1 byte | `0x00` | Format version |

Version `0x00` is format version 1.0. All other values are reserved. A decoder must reject unrecognized magic numbers or versions.

## Frames

There are three frame types, identified by the tag bits of the first byte:

```
Short  (1B):  110LLLLD                          L  = 0..=15,   D = 0/1
Medium (2B):  SLLLLDDD | DDDDDDDD              |L| = 2..=17/9, D = 1..=2,048
Long   (3B):  111SLLLL | DDDDDDDD | DDDDDDDD   |L| = 3..=18,   D = 1..=65,536
```

- The format is most easily understood if the Medium frame type is considered the baseline.
- Length is a signed 16-bit integer (i16). Distance is an unsigned 16-bit integer (u16).
- When the magnitude of L is the maximum permitted value, a length extension chain follows (see below).

### Frame Type Dispatch

The decoder determines the frame type from the first byte `b`:

| Condition | Frame type |
| - | - |
| `b = 0xC0` | End of Stream |
| `b & 0xE0 = 0xC0` | Short (`110xxxxx`, excluding EOS) |
| `b & 0xE0 = 0xE0` | Long (`111xxxxx`) |
| otherwise | Medium (`0x00`–`0xBF`) |

### End of Stream

A single `0xC0` byte. This is the Short frame encoding with L=0, D=0, reserved as the end-of-stream marker.

### Medium Frame (2 bytes)

The default/most common frame type. Bits 7–6 of byte 0 determine whether this is a forward match, reverse match, or an escape to Short/Long:

```
  Byte 0                              Byte 1
  7   6   5   4   3   2   1   0       7   6   5   4   3   2   1   0
┌───┬───┬───┬───┬───┬───┬───┬───┐   ┌───┬───┬───┬───┬───┬───┬───┬───┐
│ S │ L   L   L   L │ D   D   D │   │ D   D   D   D   D   D   D   D │
└───┴───┴───┴───┴───┴───┴───┴───┘   └───┴───┴───┴───┴───┴───┴───┴───┘
```

- **S** (bit 7): sign bit, 0/1 => positive/negative length => forward/reverse match.
- **L** (bits 6–3): length magnitude, encoding -9..=-2,2..=17 (LLLL + 2).
- **D** (bits 2–0 of byte 0 : byte 1): 11-bits, encoding 1..=2,048. The 3 bits from byte 0 are the high bits: ((byte0 & 0x07) << 8) | byte1

A length of -9 or 17 triggers a length extension. A Medium frame cannot have bits 7 and 6 of byte 0 set.

### Short Frame (1 byte)

Short frames are used to encode literals and for RLE.

```
  7   6   5   4   3   2   1   0
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 1   1   0 │ L   L   L   L │ D │
└───┴───┴───┴───┴───┴───┴───┴───┘
```

- **L** (bits 4–1): length/count encoding 0..=15.
- **D** (bit 0): 0 = literal, 1 = RLE of the last byte.
- **EOS**: L=0, D=0 (`0xC0`).
- **Reserved**: L=0, D=1 (`0xC1`). Encoders must not emit this byte. Decoder behavior is undefined.

**When D=0 (literal):** L raw bytes follow. L is the number of literal bytes (1..=15).

**When D=1 (RLE):** Repeats the last output byte L times (1..=15).

A length of 15 triggers a length extension.

### Long Frame (3 bytes)

Long frames are the simplest to understand.

```
  Byte 0                           Byte 1     Byte 2
  7   6   5   4   3   2   1   0    7 ───── 0  7 ────── 0
┌───┬───┬───┬───┬───┬───┬───┬───┐ ┌─────────┐┌──────────┐
│ 1   1   1 │ S │ L   L   L   L │ │ D (low) ││ D (high) │
└───┴───┴───┴───┴───┴───┴───┴───┘ └─────────┘└──────────┘
```

- **S** (bit 7): sign bit, 0/1 => positive/negative length => forward/reverse match.
- **L** (bits 3–0): encoding |L| = 3..=18
- **D** (bytes 1–2): 16-bit little-endian, encoding 1..=65,536.

A length of +/- 18 triggers a length extension.

### Length Extension

When the L field is the maximum magnitude, one or more extension bytes follow immediately after the frame's base bytes. The extension chain adds to the base magnitude:

1. Read one byte.
2. Add its value to the magnitude.
3. If the byte equals 255, repeat from step 1.
4. If the byte is less than 255, the chain is complete.

`total = original_magnitude + sum(ext_bytes)`

The total length (base + extensions) is a signed 16-bit quantity. Lengths must fall within -32,768..=32,767 (i16). Encoders must not produce lengths outside this range; decoder behavior for out-of-range lengths is undefined.

For literals, the extension chain appears after the frame byte and before the literal bytes.

## Copy Semantics

**Forward copy (positive length):**

```
for i in 0..length:
    output[pos + i] = output[pos - distance + i]
```

When length > distance, the copy overlaps its own output. Each byte is copied individually — this enables run-length encoding (e.g., distance 1, length 33 repeats the last byte 33 times).

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

**Frames:**

| Frame | Type | Header Byte | Payload | Description |
|-|-|-|-|-|
| 1 | Short Literal | `0xC4` (L=2, D=0) | `0x48 0x69` | 2 literal bytes |
| 2 | EOS | `0xC0` | — | End of stream |

`0xC4` = `1100_0100`: tag `110`, LLLL=0010 (2), D=0 → literal, 2 bytes follow.

**Footer:**

- ULEB128(2) = `0x02` (1 byte, no continuation needed).
- Adler-32("Hi"): A = 1 + 72 + 105 = 178 (0xB2), B = 0 + 73 + 178 = 251 (0xFB). Checksum = 0x00FB00B2, little-endian: `B2 00 FB 00`.

**Complete stream (13 bytes):**

```
4C 5A 52 00  C4 48 69 C0  02 B2 00 FB 00
├─ Header ─┤ ├─ Frames ─┤ ├── Footer ──┤
```
