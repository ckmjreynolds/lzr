# LZR Format Specification

**Version:** 1.0 \
**File Extension:** `.lzr`

## Overview

LZR is a byte-oriented LZ77 compression format. A compressed stream consists of a header, a sequence of frames, and a footer:

```
 Header      Frames                          EOS    Footer
┌─────────┐ ┌───────┬───────┬─────┬───────┐ ┌──────┐ ┌───────────────────┐
│ 4 bytes │ │ frame │ frame │ ... │ frame │ │ 0x00 │ │ length + checksum │
└─────────┘ └───────┴───────┴─────┴───────┘ └──────┘ └───────────────────┘
```

The encoder compresses data by replacing repeated byte sequences with back-references into a sliding window of prior output. Each frame either carries literal (uncompressed) bytes or a back-reference described by a *distance* (how far back to look) and a *length* (how many bytes to copy). Lengths are signed: positive means copy forward, negative means copy in reverse.

## Conventions

All multi-byte integers are **little-endian**. Bit numbering is LSB-0 (bit 0 is the least significant).

## Header

Every stream begins with a 4-byte header:

| Offset | Size | Value | Description |
|-|-|-|-|
| 0 | 3 bytes | `0x4C 0x5A 0x52` (`LZR`) | Magic number |
| 3 | 1 byte | `0x00` | Format version |

Version `0x00` indicates format version 1.0. All other values are reserved. A decoder **must** reject streams with an unrecognized magic number or version.

## Sliding Window

The decoder maintains the most recent **1,048,576 bytes** (1 MiB) of decompressed output as its sliding window. The window is logically filled with `0x00` before any output is produced. Match distances reference positions within this window. An encoder may exploit this to match into the zero-initialized region.

## Token Byte

Every frame begins with a single token byte:

```
  7   6   5   4   3   2   1   0
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ L   L   L   L   L   L │ T   T │
└───┴───┴───┴───┴───┴───┴───┴───┘
      length field         type
```

**TT** (bits 1-0) selects the frame type. **LLLLLL** (bits 7-2) is a 6-bit length field whose interpretation depends on the frame type.

The special token `0x00` (TT=00, LLLLLL=0) is the **end-of-stream** (EOS) marker. It carries no data; the decoder proceeds directly to the footer.

## Frame Types

| TT | Type | Payload | Total Size |
|-|-|-|-|
| 00 | Literal | LLLLLL raw bytes | 1 + LLLLLL |
| 01 | Short match | 1 byte distance | 2 |
| 10 | Medium match | 2 byte distance | 3 |
| 11 | Extended match | 3 bytes (packed length + distance) | 4 |

### Literal Frame (TT = 00)

LLLLLL (`token >> 2`) gives the byte count (1-63). That many uncompressed bytes follow the token. The decoder appends them directly to output.

### Short Match Frame (TT = 01)

```
 byte 0        byte 1
┌───────────┐ ┌──────────┐
│ LLLLLL 01 │ │ dist_raw │
└───────────┘ └──────────┘
```

- **Length:** Decoded from the 6-bit LLLLLL field (`token >> 2`) with minimum magnitude **2** (see [Signed Length Encoding](#signed-length-encoding)). Range: **+2 to +33** or **-2 to -33**.
- **Distance:** `dist_raw + 1`. Range: **1 to 256**.

As a little-endian `u16`: `raw = 0x01 | (length_raw << 2) | (dist_raw << 8)`.

### Medium Match Frame (TT = 10)

```
 byte 0        byte 1         byte 2
┌───────────┐ ┌──────────────────────┐
│ LLLLLL 10 │ │ dist_raw (16-bit LE) │
└───────────┘ └──────────────────────┘
```

- **Length:** Decoded from the 6-bit LLLLLL field (`token >> 2`) with minimum magnitude **3**. Range: **+3 to +34** or **-3 to -34**.
- **Distance:** `dist_raw + 1` where dist_raw is a 16-bit little-endian unsigned integer. Range: **1 to 65,536**.

As the first 3 bytes of a little-endian `u32`: `raw = 0x02 | (length_raw << 2) | (dist_raw << 8)`.

### Extended Match Frame (TT = 11)

The extended frame packs a 10-bit length and a 20-bit distance across all 4 bytes:

```
 byte 0        byte 1          byte 2       byte 3
┌───────────┐ ┌─────────────┐ ┌──────────┐ ┌──────────┐
│ LLLLLL 11 │ │ DDDD | LLLL │ │ DDDDDDDD │ │ DDDDDDDD │
└───────────┘ └─────────────┘ └──────────┘ └──────────┘
```

When read as a little-endian `u32`, `length_raw` occupies the contiguous 10-bit field at bits `[11:2]` and `distance_raw` occupies the contiguous 20-bit field at bits `[31:12]`.

**Decode:**

```
raw          = u32 from bytes 0–3 (little-endian)
length_raw   = (raw >> 2) & 0x3FF
distance_raw = raw >> 12
distance     = distance_raw + 1
```

**Encode:**

```
raw = 0x03 | (length_raw << 2) | (distance_raw << 12)
```

- **Length:** Decoded from the 10-bit field with minimum magnitude **4**. Range: **+4 to +515** or **-4 to -515**.
- **Distance:** `distance_raw + 1`. Range: **1 to 1,048,576**.

## Signed Length Encoding

Match lengths are signed. Positive lengths copy forward; negative lengths copy in reverse. Each match frame type encodes its length as a **sign bit** followed by a **magnitude offset**:

```
 MSB                  LSB
┌──────┬──────────────────┐
│ sign │ magnitude offset │
└──────┴──────────────────┘
  0 = forward (+)      offset = |length| - M
  1 = reverse (-)
```

The sign bit is the most significant bit of the length field (bit 5 for short/medium, bit 9 for extended). The remaining bits encode `|length| - M`, where **M** is the frame type's minimum magnitude.

**Decode:**

```
sign      = length_raw >> (bits - 1)
offset    = length_raw & ((1 << (bits - 1)) - 1)
magnitude = offset + M
length    = if sign == 0 then +magnitude else -magnitude
```

**Encode:**

```
offset     = |length| - M
length_raw = if length > 0 then offset else offset | (1 << (bits - 1))
```

### Parameters by Frame Type

| Frame | Field Width | Sign Bit | Offset Bits | M | Forward Range | Reverse Range |
|-|-|-|-|-|
| Short | 6 bits | bit 5 | bits 4-0 | 2 | +2 to +33 | -2 to -33 |
| Medium | 6 bits | bit 5 | bits 4-0 | 3 | +3 to +34 | -3 to -34 |
| Extended | 10 bits | bit 9 | bits 8-0 | 4 | +4 to +515 | -4 to -515 |

### Worked Example

Encoding length **+10** as a short match (6-bit field, M=2):

```
offset     = 10 - 2 = 8
sign       = 0 (positive)
length_raw = 8             → 0b001000
token      = (8 << 2) | 1 → 0x21
```

Encoding length **-5** as a short match:

```
offset     = 5 - 2 = 3
sign       = 1 (negative)
length_raw = 3 | 0b100000 = 35   → 0b100011
token      = (35 << 2) | 1       → 0x8D
```

## Copy Semantics

A match frame tells the decoder to copy `|length|` bytes from a position `distance` bytes back in the output buffer. The sign of the length controls the copy direction.

### Forward Copy (length > 0)

Copy `length` bytes starting at `output[pos - distance]`, reading forward:

```
for i in 0..length:
    output[pos + i] = output[pos - distance + i]
```

When length exceeds distance, the copy overlaps with its own output. This is intentional and well-defined — each byte is copied individually. It enables run-length encoding: for example, distance=1 and length=100 repeats the most recent byte 100 times.

### Reverse Copy (length < 0)

Copy `|length|` bytes starting at `output[pos - distance]`, reading backward:

```
count = |length|
for i in 0..count:
    output[pos + i] = output[pos - distance - i]
```

This enables matching reversed patterns. For example, if the output contains `stressed`, a reverse match can produce `desserts` by referencing it backward.

If any source index falls before the start of the sliding window, the byte reads as `0x00`.

## Distance Encoding

All match frame types store distance as an unsigned integer minus one:

```
stored_value = distance - 1
distance     = stored_value + 1
```

This encoding is lossless for the full distance range of each frame type and avoids wasting a code point on the meaningless value distance=0.

## Footer

The footer immediately follows the EOS token:

| Field | Size | Description |
|-|-|-|
| Uncompressed length | 1-9 bytes | ULEB128-encoded byte count of the original data (unsigned 64-bit). |
| Checksum | 4 bytes | Adler-32 of the uncompressed data, little-endian. |

The decoder **must** verify that the decompressed byte count matches the stored length and that the Adler-32 checksum matches. A mismatch indicates corruption.

### ULEB128

Unsigned integers use a modified [ULEB128](https://en.wikipedia.org/wiki/LEB128#Unsigned_LEB128) encoding. Bytes 1–8 each store 7 data bits in bits 6–0, with bit 7 as a continuation flag (1 = more bytes follow, 0 = final byte). If all 8 of those bytes have the continuation flag set, a 9th byte follows that carries 8 data bits with no continuation flag (8 × 7 + 8 = 64 bits). A conforming implementation must support values up to 2^64 − 1, requiring at most 9 bytes.

### Adler-32

The checksum is a standard [Adler-32](https://en.wikipedia.org/wiki/Adler-32) computed over the entire uncompressed output, stored as a 4-byte little-endian integer.

## Stream Concatenation

A file may contain multiple concatenated streams. If data remains after a footer, the decoder should treat it as the beginning of a new stream (starting with a header). This allows compressed files to be concatenated with `cat`.

## Quick Reference

### Encoding Ranges

| Frame | Length | Distance | Frame Size |
|-|-|-|-|
| Literal | 1-63 bytes | -- | 1 + length |
| Short match | ±(2-33) | 1-256 | 2 bytes |
| Medium match | ±(3-34) | 1-65,536 | 3 bytes |
| Extended match | ±(4-515) | 1-1,048,576 | 4 bytes |

Each match frame's minimum `|length|` equals its size in bytes, so every match is at least as compact as the equivalent literal bytes.

### Token Byte Map

The frame type is determined by the two least-significant bits (`token & 0x03`):

| TT (bits 1-0) | Meaning |
|-|-|
| `0b00` | Literal (1-63 bytes follow) or **EOS** if byte = `0x00` |
| `0b01` | Short match (1 distance byte follows) |
| `0b10` | Medium match (2 distance bytes follow) |
| `0b11` | Extended match (3 extension bytes follow) |
