# LZR Format Specification

**Version:** 1.0
**Extension:** `.lzr`

## Overview

LZR is an LZ77-based compression format. A stream has three parts:

```
┌────Header────┐┌────────Frame────────┐┌────────Frame────────┐     ┌────────Footer────────┐
│┌────────────┐││┌───────┐┌ ─ ─ ─ ─ ─ ││┌───────┐┌ ─ ─ ─ ─ ─ │     │┌────────┐┌──────────┐│
││ LZR | 0x00 ││││ D | L │  Literals ││││ D | L │  Literals ││ ... ││ Length ││ Adler-32 ││
│└────────────┘││└───────┘└ ─ ─ ─ ─ ─ ││└───────┘└ ─ ─ ─ ─ ─ │     │└────────┘└──────────┘│
└──────────────┘└─────────────────────┘└─────────────────────┘     └──────────────────────┘
```

All multi-byte integers are little-endian. The bitstream between the header and footer is nibble-oriented; see [BLEB8](BLEB8.md#nibble-stream) for nibble packing conventions.

## Header

| Offset | Size | Value | Description |
|-|-|-|-|
| 0 | 3 bytes | `0x4C 0x5A 0x52` (`LZR`) | Magic number |
| 3 | 1 byte | `0x00` | Format version |

Version `0x00` is format version 1.0. All other values are reserved. A decoder must reject unrecognized magic numbers or versions.

## Bitstream

The bitstream between the header and footer is a sequence of nibbles encoding frames. Nibbles pack into bytes as defined in [BLEB8](BLEB8.md#nibble-stream): first nibble in bits 7–4, second in bits 3–0. The bitstream must contain an even number of nibbles (byte-aligned).

## Frames

Each frame is a distance (D) followed by a length (L), optionally followed by literal bytes:

- **D** — [UBLEB8](BLEB8.md#ubleb8-unsigned)-encoded u16 (max 5 nibbles).
- **L** — [UBLEB8](BLEB8.md#ubleb8-unsigned)-encoded u16 when D = 0; [SLEB8](BLEB8.md#sleb8-signed)-encoded i16 when D > 0 (max 5 nibbles).

| D | L | Type |
|-|-|-|
| 0 | 0 | **End of stream** |
| 0 | > 0 | **Literal** — L raw bytes follow in the nibble stream |
| > 0 | ≠ 0 | **Match** — copy \|L\| bytes from distance D |
| > 0 | 0 | **No-op** — decoder does nothing |

### End of Stream

D = 0 and L = 0. Both encode as a single UBLEB8 nibble `0x0`, totaling two zero nibbles.

### Literal

D = 0, L > 0. The next L uncompressed bytes follow in the nibble stream. Each byte is stored as two nibbles in little-endian order (low nibble first, high nibble second).

### Match

D > 0, L ≠ 0. Copy |L| bytes from `output[pos - D]`. Distance 1 refers to the most recently emitted byte. The stored distance is the raw value (not distance-minus-one).

The sign of L determines the copy direction:

**Forward copy (L > 0):**

```
for i in 0..|L|:
    output[pos + i] = output[pos - D + i]
```

When L > D, the copy overlaps its own output. Each byte is copied individually — this enables run-length encoding (e.g., D=1, L=100 repeats the last byte 100 times).

**Reverse copy (L < 0):**

```
for i in 0..|L|:
    output[pos + i] = output[pos - D - i]
```

This matches reversed patterns. For example, `stressed` in the output can produce `desserts` via a reverse copy.

### No-op

D > 0, L = 0. Valid; the decoder emits nothing and advances to the next frame. Encoders use no-op frames to pad the nibble stream to a byte boundary.

## Sliding Window

The decoder maintains a **65,536 byte** (64 KiB) sliding window of decompressed output. The window must be filled with `0x00` before decompression begins. Any read from a position before the start of decompressed output returns `0x00` (from the zero-initialized window).

## Footer

Byte-aligned, immediately after the padded bitstream.

| Field | Encoding | Description |
|-|-|-|
| Uncompressed length | [UBLEB8](BLEB8.md#ubleb8-unsigned)-encoded u64 (max 21 nibbles) | Original byte count |
| Checksum | 4 bytes, little-endian | [Adler-32](https://en.wikipedia.org/wiki/Adler-32) of uncompressed data |

The UBLEB8 length packs nibbles into bytes the same way as the bitstream. The nibbles are padded to a byte boundary; at most 21 nibbles = 11 bytes.

The decoder must verify that the decompressed byte count matches the stored length and that the Adler-32 matches. A mismatch indicates corruption.

## Stream Concatenation

A file may contain multiple concatenated streams. If data remains after a footer, the decoder must treat it as a new stream beginning with a header.

## Worked Example

Compressing `Hi` (0x48 0x69):

**Nibble stream:**

| Frame | Field | Value | Nibbles |
|-|-|-|-|
| 1 | D | 0 (literal) | `0` |
| 1 | L | 2 (byte count) | `2` |
| 1 | Byte 0x48 | | `8 4` |
| 1 | Byte 0x69 | | `9 6` |
| 2 | D | 0 (EOS) | `0` |
| 2 | L | 0 (EOS) | `0` |

Eight nibbles = four bytes. Packed: `0x02 0x84 0x96 0x00`.

**Footer:**

- UBLEB8(2) = nibble `0x2`, padded to byte: `0x20`.
- Adler-32("Hi"): A = 1 + 72 + 105 = 178 (0xB2), B = 0 + 73 + 178 = 251 (0xFB). Checksum = 0x00FB00B2, little-endian: `B2 00 FB 00`.

**Complete stream (13 bytes):**

```
4C 5A 52 00  02 84 96 00  20 B2 00 FB 00
├─ Header ─┤ ├ Bitstream ┤ ├── Footer ──┤
```
