# LZR Format Specification

**Version:** 1.0
**File Extension:** `.lzr`

## Overview

LZR is a byte-oriented compression format based on the LZ77 sliding-window algorithm. A compressed stream consists of a **header**, a sequence of **frames**, an **end-of-stream sentinel**, and a **footer**.

> **Note:** A file may contain more than one stream (i.e. files can be concatenated). If a file does not end at the footer, a decoder should continue to process any additional streams.

```
┌─────────────────────────────Stream─────────────────────────────┐
│┌──────────────Header───────────────┐                           │
││┌─Magic─┐ ┌Version┐ ┌────Flags────┐│                           │
│││  LZR  │ │ 0x00  │ │ RRRR | WWWW ││                           │
││└───────┘ └───────┘ └─────────────┘│                           │
│└───────────────────────────────────┘                           │
│┌────────────────────────────Frame─────────────────────────────┐│
││┌────Token────┐ ┌ ─Length ─ ┐ ┌ Distance─ ┐ ┌ ─ ─Literals ─ ─ ││
│││ LLL | DDDDD │   0-2 bytes     0-3 bytes     0-Length bytes │││
││└─────────────┘ └ ─ ─ ─ ─ ─ ┘ └ ─ ─ ─ ─ ─ ┘ └ ─ ─ ─ ─ ─ ─ ─ ─ ││
│└──────────────────────────────────────────────────────────────┘│
│                                                                │
│ ...                                                            │
│                                                                │
│┌─EOS──┐                                                        │
││ 0x00 │                                                        │
│└──────┘                                                        │
│┌───────────────────Footer────────────────────┐                 │
││┌─Uncompressed Length─┐┌──────Checksum──────┐│                 │
│││  1-9 bytes ULEB128  ││ Adler-32 (4-bytes) ││                 │
││└─────────────────────┘└────────────────────┘│                 │
│└─────────────────────────────────────────────┘                 │
└────────────────────────────────────────────────────────────────┘
```

## Byte Order

All multi-byte integer values are stored in **little-endian** byte order.

## ULEB128 Encoding

LZR uses a bounded variant of ULEB128 in the **footer** (for the uncompressed length field). It is identical to standard, unbounded ULEB128 except that the **final byte** uses all 8 bits for data with no continuation bit.

**Decoding procedure** (given a maximum of *N* bytes):

1. Read a byte.
2. If fewer than *N* bytes have been read and bit 7 is set, accumulate bits 0-6 as data and read another byte.
3. If fewer than *N* bytes have been read and bit 7 is clear, accumulate bits 0-6 as data and stop.
4. If this is the *N*th byte, accumulate **all 8 bits** as data and stop.

Data bits are accumulated from least-significant to most-significant (standard ULEB128 order).

## Header

The header is **5 bytes** and must appear at the start of every LZR stream.

| Offset | Size    | Value                      | Description    |
|--------|---------|----------------------------|----------------|
| 0      | 3 bytes | `LZR` (`0x4C 0x5A 0x52`)   | Magic bytes    |
| 3      | 1 byte  | `0x00`                     | Format version |
| 4      | 1 byte  | See below                  | Flags          |

A decoder must reject any stream whose magic bytes do not match or whose version is unrecognized.

### Format Version

| Value         | Version  |
| ------------- | -------- |
| `0x00`        | 1.0      |
| `0x01 - 0xFF` | Reserved |

### Flags Byte

```
   Flags Byte
┌──────────┬──────────┐
│ Reserved │  Window  │
│  bit 7   │  bit 3   │
│   to 4   │   to 0   │
└──────────┴──────────┘
```

| Field    | Bits | Description                                                              |
|----------|------|--------------------------------------------------------------------------|
| Reserved | 7-4  | Reserved for future use. A decoder must ignore what's written here.      |
| Window   | 3-0  | Window size exponent *N*. The sliding window size is 2^(*N*+9) bytes.    |

**Window sizes:**

| N   | Window Size |
|-----|-------------|
| 0   | 512 B       |
| 1   | 1 KiB       |
| ... |             |
| 14  |  8 MiB      |
| 15  | 16 MiB      |

## Sliding Window

The decoder maintains a sliding window of the most recent *W* bytes of output, where *W* = 2^(*N*+9) and *N* is the window exponent from the header flags byte. The window is logically initialized to **0x00** at the start of each stream. Any read that references a position before the start of the window (or before any output has been produced) returns `0x00`.

> **Note:** An encoder may take advantage of this, it is not an error to encode a distance outside of the declared window.

## Frames

Each frame begins with a **token byte** followed by optional extension fields:

```
   Token Byte
┌───────┬────────┐
│  LLL  │ DDDDD  │
│ bit 7 │ bit 4  │
│  to 5 │  to 0  │
└───────┴────────┘
```

- **LLL** (bits 7-5): 3-bit length code (0-7).
- **DDDDD** (bits 4-0): 5-bit distance code (0-31).

The token value is computed as: `token = (LLL << 5) | DDDDD`

**Length decoding:**

| LLL | Meaning                                                                           |
|-----|-----------------------------------------------------------------------------------|
| 0-5 | Length is LLL (direct).                                                           |
| 6   | A 1-byte **signed** extension follows; its value is the length (-128 to 127).     |
| 7   | A 2-byte **signed** (little-endian) extension follows; its value is the length (-32,768 to 32,767). |

**Distance decoding:**

| DDDDD | Meaning                                                                                             |
|-------|-----------------------------------------------------------------------------------------------------|
| 0-28  | Distance is DDDDD (direct).                                                                        |
| 29    | A 1-byte unsigned extension follows; distance = stored value + 1 (1-256).                          |
| 30    | A 2-byte unsigned (little-endian) extension follows; distance = stored value + 1 (1-65,536).       |
| 31    | A 3-byte unsigned (little-endian) extension follows; distance = stored value + 1 (1-16,777,216).   |

### Frame Field Order

1. **Token** (1 byte) — always present.
2. **Length extension** (0, 1, or 2 bytes) — present only when LLL = 6 (1 byte) or LLL = 7 (2 bytes).
3. **Distance extension** (0, 1, 2, or 3 bytes) — present only when DDDDD = 29 (1 byte), 30 (2 bytes), or 31 (3 bytes).
4. **Literals** (0 or more bytes) — present only in literal frames.

### End-of-Stream (EOS)

A token byte of **0x00** (LLL = 0, DDDDD = 0) signals the end of the frame sequence. No extension or literal
fields follow. The decoder must then read the footer.

### Literal Frame (DDDDD = 0, LLL > 0)

When DDDDD = 0 the frame carries raw uncompressed bytes.

| Field    | Description                                                |
|----------|------------------------------------------------------------|
| Length   | Decoded from LLL (see length decoding table above).        |
| Literals | Exactly |*length*| bytes of uncompressed data.             |

If the decoded length is negative, the absolute value is used as the byte count. The decoder appends the literal bytes directly to the output.

### Match Frame (DDDDD > 0)

When DDDDD > 0 the frame describes a back-reference copy.

| Field    | Description                                                  |
|----------|--------------------------------------------------------------|
| Length   | Decoded from LLL (see length decoding table above).          |
| Distance | Decoded from DDDDD (see distance decoding table above).     |

No literal bytes follow a match frame. A length of zero is a no-op (no bytes are copied). Otherwise, the sign of the length determines the copy direction:

**Forward copy** (length > 0): The decoder copies *length* bytes starting from *distance* bytes back in the output buffer, reading forward: `output[pos-D], output[pos-D+1], ..., output[pos-D+L-1]`.

**Reverse copy** (length < 0): The decoder copies |*length*| bytes starting from *distance* bytes back in the output buffer, reading backward: `output[pos-D], output[pos-D-1], ..., output[pos-D-|L|+1]`. This enables matching reversed patterns (e.g., encoding "desserts" by referencing an earlier occurrence of "stressed"). If any position falls before the start of the sliding window, it reads as 0x00 (see [Sliding Window](#sliding-window)).

> **Note (forward copy):** When *length* exceeds *distance*, the copy wraps forward into bytes produced earlier in the same operation. This is well-defined, occurs one byte at a time, and enables run-length encoding (e.g., distance = 1, length = 100 repeats the last byte 100 times).

### Extension Frame (LLL = 0, DDDDD > 0)

When LLL = 0 and DDDDD > 0 this is an extension frame reserved for future use. The distance field encodes a skip count using the normal distance decoding rules (see distance decoding table above). The decoder must skip the next *distance* bytes in the **input stream** and continue to the next frame. The contents of the skipped bytes are undefined.

### Encoding Ranges

| Value    | Inline Range | 1-byte Extension       | 2-byte Extension          | 3-byte Extension     |
|----------|--------------|------------------------|---------------------------|----------------------|
| Length   | 0 - 5        | -128 - 127             | -32,768 - 32,767          | —                    |
| Distance | 0 - 28       | 1 - 256                | 1 - 65,536                | 1 - 16,777,216       |

## Footer

The footer immediately follows the EOS token.

| Field               | Size       | Description                                       |
|---------------------|------------|---------------------------------------------------|
| Uncompressed length | 1-9 bytes  | ULEB128-encoded original data size (u64, N=9).    |
| Checksum            | 4 bytes    | Adler-32 of the uncompressed data, little-endian. |

The **Adler-32** checksum is computed over the original uncompressed data. The decoder should verify it after decompression and report an error on mismatch.
