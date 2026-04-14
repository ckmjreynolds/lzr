# LZR Format Specification - v1.0

```text
+----------------------------------------------Mythos----------------------------------------------+
|                                                                                                  |
|  +--------------------------------------------Opus--------------------------------------------+  |
|  |                                                                                            |# |
|  |                                    +----Invocation----+                                    |# |
|  |                                    | b"LZR" | version |                                    |# |
|  |                                    +------------------+                                    |# |
|  |                                                                                            |# |
|  |  +---------------------------------Sonnet (fixed size)----------------------------------+  |# |
|  |  |                                                                                      |# |# |
|  |  |  +----------------------------Haiku (arithmetic coded)----------------------------+  |# |# |
|  |  |  | +-----------------------------------Line------------------------------------+  |# |# |# |
|  |  |  | |+------------------++ - - - - - - - - - - - - - - - - - - - - - - - - - - +|# |# |# |# |
|  |  |  | || LINE/EOH/EOS/EOO |  literal_len | match_len | match_distance | literals  |# |# |# |# |
|  |  |  | |+------------------++ - - - - - - - - - - - - - - - - - - - - - - - - - - +|# |# |# |# |
|  |  |  | +---------------------------------------------------------------------------+# |# |# |# |
|  |  |  |  ############################################################################# |# |# |# |
|  |  |  |                                                                                |# |# |# |
|  |  |  |                             +------Kireji-------+                              |# |# |# |
|  |  |  |                             | finalization_bits |                              |# |# |# |
|  |  |  |                             +-------------------+                              |# |# |# |
|  |  |  +--------------------------------------------------------------------------------+# |# |# |
|  |  |   ################################################################################## |# |# |
|  |  |                                                                                      |# |# |
|  |  |           +------------------------Couplet/Coda-------------------------+            |# |# |
|  |  |           | cumulative_bytes | cumulative_lines | checksum | footer_len |            |# |# |
|  |  |           +-------------------------------------------------------------+            |# |# |
|  |  +--------------------------------------------------------------------------------------+# |# |
|  |   ######################################################################################## |# |
|  |                                                                                            |# |
|  +--------------------------------------------------------------------------------------------+# |
|   ############################################################################################## |
|                                                                                                  |
+--------------------------------------------------------------------------------------------------+
```

## Introduction

LZR is a block-based compression format combining LZ77 dictionary coding with adaptive
arithmetic entropy coding. It is designed to support three modes of operation:

- **Streaming compression/decompression** — process data sequentially from start to end.
- **Random-access reads** — seek to an arbitrary byte offset or line number and decompress
  from that point without decompressing preceding data.
- **Append-only writes** — append new data to an existing archive without rewriting it.

Additionally, sealed (complete) archives MAY be concatenated for streaming decompression.

This specification defines the LZR format to the level of detail required to build a
compliant encoder and decoder from scratch. The key words "MUST", "MUST NOT", "SHOULD",
"SHOULD NOT", and "MAY" are to be interpreted as described in RFC 2119.

## Version History

| Version | Byte   | Status  | Notes                      |
|---------|--------|---------|----------------------------|
| v1.0    | `0x00` | Current | Initial format definition  |

Decoders MUST reject files whose version byte is not recognized.

## Nomenclature

We have a little fun with our nomenclature, taking inspiration from Anthropic's model naming
in a nod to Claude's contributions to this project.

| Name         | Technical Meaning                                          |
|--------------|------------------------------------------------------------|
| **Mythos**   | Concatenation of one or more sealed Opus archives          |
| **Opus**     | A single LZR archive                                      |
| **Invocation** | File header: magic bytes and version                     |
| **Sonnet**   | Fixed-size block (256 KiB); the unit of random access      |
| **Haiku**    | Arithmetic-coded region within a Sonnet                    |
| **Line**     | A single LZ77 token (tag + optional fields)                |
| **Kireji**   | Arithmetic coder finalization bits at the end of a Haiku   |
| **Couplet**  | Footer of a non-final Sonnet                               |
| **Coda**     | Footer of the final Sonnet in a sealed Opus                |

## Conventions

- **Byte order**: Little-endian unless otherwise noted.
- **Bit packing**: MSB-first within bytes (the arithmetic coder emits the most significant
  bit of each byte first).
- **Integer types**: `u8`, `u16`, `u32`, `u64` denote unsigned integers of the given bit width.
- **Pseudocode**: Uses Python-like syntax with explicit types. `//` denotes integer division.

---

## 1. Format Layers

### 1.1 Mythos

A **Mythos** is zero or more sealed Opus archives concatenated end-to-end. There are no
framing bytes at this layer; a decoder simply processes each Opus in sequence until it reaches
end-of-input.

A Mythos containing a single Opus is the common case. Concatenation is supported so that
sealed archives can be joined (e.g., `cat a.lzr b.lzr > combined.lzr`) and decoded as a
single stream.

Random-access reads are NOT supported on a concatenated Mythos (more than one Opus), because
block offsets are relative to the start of each Opus.

### 1.2 Opus

An **Opus** is a single LZR archive. It begins with an Invocation header and contains one or
more Sonnets.

An Opus is in one of two states:

- **Sealed**: The final Sonnet ends with an `EOO` tag followed by a Coda footer. The archive
  is complete and immutable.
- **Unsealed**: The final Sonnet is incomplete (no `EOO` or Coda). The archive is appendable.

An Opus representing empty input (zero bytes) is valid: it contains a single Sonnet with only
an `EOO` tag and a Coda footer (`cumulative_bytes = 0`, `cumulative_lines = 0`,
`checksum = 0x00000001`).

### 1.3 Invocation

The **Invocation** is the first 4 bytes of every Opus:

| Offset | Size    | Value          | Description          |
|--------|---------|----------------|----------------------|
| 0      | 3 bytes | `0x4C 0x5A 0x52` | Magic bytes: ASCII `"LZR"` |
| 3      | 1 byte  | `0x00`         | Version byte (v1.0)  |

Decoders MUST validate the magic bytes and MUST reject unknown version bytes.

### 1.4 Sonnet

A **Sonnet** is a block of compressed data. It is the fundamental unit of random access.

- **Non-final Sonnets** are exactly **262,144 compressed bytes** (256 KiB) on disk.
- **The final Sonnet** in a sealed Opus MAY be shorter than 256 KiB.
- **The last Sonnet** in an unsealed Opus has no `EOO` tag or Coda footer.

#### Sonnet layout

```
+-------+-------+-----+-------+---------+-----------------+
| Haiku | Haiku | ... | Haiku | padding | Couplet or Coda |
+-------+-------+-----+-------+---------+-----------------+
```

- One or more Haiku regions, each terminated by a tag (`EOH`, `EOS`, or `EOO`).
- Zero or more **padding bytes** (`0x00`) fill the space between the last Kireji and the
  footer.
- A **Couplet** (non-final Sonnet) or **Coda** (final Sonnet, sealed) footer occupies the
  last `footer_len` bytes of the Sonnet.

#### Resets at Sonnet boundaries

At the start of each Sonnet, the following state MUST be reset:

- All six adaptive frequency models (see [Section 5](#5-model-set)) are reset to their
  initial uniform distributions.
- The LZ77 sliding window is cleared.
- The arithmetic coder is initialized fresh (this also happens at each Haiku boundary).

### 1.5 Haiku

A **Haiku** is a contiguous region of arithmetic-coded tokens within a Sonnet.

#### Haiku layout

```
+-------------------------------+--------+
| arithmetic-coded token stream | Kireji |
+-------------------------------+--------+
```

The token stream is a sequence of Line tokens (see [Section 2](#2-token-encoding)) encoded
through the arithmetic coder, terminated by one of:

- **`EOH`** — End of Haiku. The arithmetic coder is finalized (Kireji written) and a new
  Haiku begins. The adaptive models are **not** reset.
- **`EOS`** — End of Sonnet. The arithmetic coder is finalized, and the decoder reads and
  validates the Couplet footer.
- **`EOO`** — End of Opus. The arithmetic coder is finalized, and the decoder reads and
  validates the Coda footer. The Opus is sealed.

At the start of each Haiku, the arithmetic coder state (low, high, pending bits) is
initialized fresh. The adaptive frequency models carry over from the previous Haiku within the
same Sonnet.

### 1.6 Kireji

The **Kireji** is the finalization output of the arithmetic encoder at the end of a Haiku.
It consists of the bits needed to unambiguously identify the final interval, padded to a byte
boundary with zero bits. See [Section 3.4](#34-finalization-kireji) for the exact algorithm.

### 1.7 Couplet and Coda

A **Couplet** is the footer of a non-final Sonnet. A **Coda** is the footer of the final
Sonnet in a sealed Opus. They are structurally identical.

All footer values are computed over **uncompressed (original) data** and are **cumulative**
from the start of the Opus.

#### Footer layout

The footer occupies the last `footer_len` bytes of the Sonnet. The very last byte of the
Sonnet is always `footer_len`, so a decoder reads that byte first and backs up accordingly.

Fields are stored in the following order:

| Field              | Encoding     | Size        | Description                              |
|--------------------|--------------|-------------|------------------------------------------|
| `cumulative_bytes` | ULEB128      | 1–9 bytes   | Total uncompressed bytes so far           |
| `cumulative_lines` | ULEB128      | 1–9 bytes   | Total `0x0A` bytes in uncompressed data   |
| `checksum`         | Adler-32, LE | 4 bytes     | Adler-32 of all uncompressed bytes so far |
| `footer_len`       | u8           | 1 byte      | Total size of this footer, including this byte |

- `footer_len` ranges from 7 (minimum: 1+1+4+1) to 23 (maximum: 9+9+4+1).
- `cumulative_lines` counts occurrences of the byte `0x0A` (LF) in the uncompressed data.
- Decompressing an entire sealed Opus and recomputing byte count, line count, and Adler-32
  MUST produce values matching the Coda.

---

## 2. Token Encoding

### 2.1 Tag Alphabet

Every token begins with a **tag** symbol, encoded through the 4-symbol tag model:

| Symbol | Value | Meaning                                                       |
|--------|-------|---------------------------------------------------------------|
| `LINE` | 0     | Normal LZ77 sequence (literals and/or a match)                |
| `EOH`  | 1     | End of Haiku — finalize arithmetic coder, begin new Haiku     |
| `EOS`  | 2     | End of Sonnet — finalize, validate Couplet, reset all state   |
| `EOO`  | 3     | End of Opus — finalize, validate Coda, archive is sealed      |

### 2.2 LINE Token

A `LINE` token encodes a sequence of literal bytes optionally followed by a backward match
copy. The fields are encoded in order through the arithmetic coder, each using its own
adaptive model:

| #  | Field              | Model        | Condition          |
|----|--------------------|--------------|--------------------|
| 1  | `tag`              | tag (4 sym)  | Always (= `LINE`)  |
| 2  | `literal_len`      | literal_len  | Always             |
| 3  | `match_len`        | match_len    | Always             |
| 4  | `match_distance_lo`| dist_lo      | Only if `match_len > 0` |
| 5  | `match_distance_hi`| dist_hi      | Only if `match_len > 0` |
| 6  | `literals[0..literal_len]` | literals | Each byte individually |

**Constraints:**

- A `LINE` with `literal_len = 0` AND `match_len = 0` is **INVALID**. Decoders MUST reject
  such tokens.
- `match_distance` is reconstructed as `(match_distance_hi << 8) | match_distance_lo`.
- `match_distance` is **1-based**: a distance of 1 means the byte immediately preceding the
  current output position.

**Semantics:**

1. Emit `literal_len` literal bytes to the output (and into the LZ77 window).
2. If `match_len > 0`: copy `match_len` bytes from the LZ77 window at offset
   `match_distance` bytes back from the current position.

### 2.3 Terminator Tokens (EOH, EOS, EOO)

Terminator tokens encode **only** the tag symbol. No additional fields are encoded.

An encoder MUST flush any pending literal bytes as a `LINE` token (with `match_len = 0`)
before emitting a terminator tag.

---

## 3. Arithmetic Coding

LZR uses the classic Witten/Neal/Cleary adaptive arithmetic coder with 16-bit precision and
E3 (underflow) scaling. The algorithm is specified here to the level of detail required for
bit-exact interoperability between encoders and decoders.

### 3.1 State and Constants

```
HALF          = 0x8000      // 32768
QUARTER       = 0x4000      // 16384
THREE_QUARTER = 0xC000      // 49152

// Encoder state
low:  u16 = 0x0000
high: u16 = 0xFFFF
pending_bits: u64 = 0

// Decoder state (in addition to low, high)
code: u16                   // current code value read from input
```

Both encoder and decoder initialize `low = 0x0000` and `high = 0xFFFF` at the start of each
Haiku.

### 3.2 Encoding a Symbol

Given a symbol `s` with cumulative frequency `cum_low` (sum of frequencies of symbols
`0..s`), `cum_high` (sum of frequencies of symbols `0..=s`), and `total` (sum of all
frequencies):

```
range: u32 = (high as u32) - (low as u32) + 1
high = low + ((range * cum_high) / total) as u16 - 1
low  = low + ((range * cum_low)  / total) as u16
normalize()
```

The intermediate products `range * cum_high` and `range * cum_low` require u32 arithmetic
to avoid overflow. Since `range` is at most 65536 and `total` is at most 16383 (see
[Section 4](#4-adaptive-frequency-model)), the products fit in u32.

After encoding, the model is updated (see [Section 4.3](#43-adaptation)).

### 3.3 Normalization

Normalization is performed after each symbol encoding (and the mirror operation during
decoding). It consists of a loop that runs until the interval `[low, high]` straddles the
midpoint without being containable in a quarter:

```
loop:
    if high < HALF:                                   // E1: interval in lower half
        output_bit(0)
        for _ in 0..pending_bits: output_bit(1)       // flush pending
        pending_bits = 0

    else if low >= HALF:                              // E2: interval in upper half
        output_bit(1)
        for _ in 0..pending_bits: output_bit(0)       // flush pending
        pending_bits = 0
        low  -= HALF
        high -= HALF

    else if low >= QUARTER and high < THREE_QUARTER:  // E3: straddle
        pending_bits += 1
        low  -= QUARTER
        high -= QUARTER

    else:
        break                                         // interval is wide enough

    low  = low << 1                                   // shift in 0
    high = (high << 1) | 1                            // shift in 1
```

### 3.4 Finalization (Kireji)

When all tokens for a Haiku have been encoded, the encoder finalizes by emitting enough bits
to place the decoder's code value within the final interval:

```
pending_bits += 1
if low < QUARTER:
    output_bit(0)
    for _ in 0..pending_bits: output_bit(1)
else:
    output_bit(1)
    for _ in 0..pending_bits: output_bit(0)
flush_to_byte_boundary()    // pad remaining bits in the current byte with 0
```

The resulting bytes (from the start of this Haiku's arithmetic output through the final
padded byte) are the Kireji.

### 3.5 Bit Packing

Bits are packed into bytes **MSB-first**:

```
// State
byte_buffer: u8 = 0
bits_in_buffer: u8 = 0

fn output_bit(bit: u1):
    byte_buffer = (byte_buffer << 1) | bit
    bits_in_buffer += 1
    if bits_in_buffer == 8:
        emit(byte_buffer)
        byte_buffer = 0
        bits_in_buffer = 0

fn flush_to_byte_boundary():
    if bits_in_buffer > 0:
        byte_buffer = byte_buffer << (8 - bits_in_buffer)
        emit(byte_buffer)
        byte_buffer = 0
        bits_in_buffer = 0
```

### 3.6 Decoding a Symbol

The decoder maintains `low`, `high`, and `code` (a 16-bit value read from the input stream).
At the start of each Haiku, the decoder sets `low = 0x0000`, `high = 0xFFFF`, and reads the
first 16 bits from the bitstream into `code`.

To decode a symbol:

```
range: u32 = (high as u32) - (low as u32) + 1
value: u32 = ((code as u32 - low as u32 + 1) * total - 1) / range

// Find symbol s such that cum_freq(s) <= value < cum_freq(s + 1)
s = model.find(value)

// Update interval (same formula as encoder)
cum_low  = model.cum_freq(s)
cum_high = model.cum_freq(s + 1)   // = cum_low + freq(s)
high = low + ((range * cum_high) / total) as u16 - 1
low  = low + ((range * cum_low)  / total) as u16

// Normalize (mirror of encoder, shifting in bits from input)
normalize_decoder()
```

#### Decoder normalization

```
loop:
    if high < HALF:
        // E1: nothing extra
    else if low >= HALF:
        // E2
        low  -= HALF
        high -= HALF
        code -= HALF
    else if low >= QUARTER and high < THREE_QUARTER:
        // E3
        low  -= QUARTER
        high -= QUARTER
        code -= QUARTER
    else:
        break

    low  = low << 1
    high = (high << 1) | 1
    code = (code << 1) | read_bit()
```

After decoding, the model is updated identically to the encoder (see [Section 4.3](#43-adaptation)).

### 3.7 Decoder Haiku Finalization

After decoding a terminator tag (`EOH`, `EOS`, or `EOO`), the decoder MUST advance its bit
reader to the next byte boundary, discarding any remaining bits in the current byte. This is
the mirror of the encoder's `flush_to_byte_boundary()` in [Section 3.4](#34-finalization-kireji).

After byte-aligning:

- **`EOH`**: The decoder re-initializes `low = 0x0000`, `high = 0xFFFF`, and reads the next
  16 bits from the byte-aligned position into `code`. Decoding continues with the next Haiku.
- **`EOS`** or **`EOO`**: The decoder proceeds to read the Sonnet footer. See
  [Section 10.1](#101-streaming-decode) for the footer-reading procedure.

---

## 4. Adaptive Frequency Model

The adaptive frequency model tracks symbol frequencies and provides cumulative distribution
function (CDF) queries to the arithmetic coder. The model MUST be implemented identically in
encoders and decoders to maintain synchronization.

### 4.1 State

- **Alphabet size**: `N` symbols (4 for the tag model, 256 for all byte models).
- **Frequency table**: `freq[0..N]` — per-symbol counts.
- **Total**: sum of all frequencies.

### 4.2 Initialization

```
for i in 0..N:
    freq[i] = 1
total = N
```

All symbols start with equal probability (uniform distribution).

### 4.3 Adaptation

After encoding or decoding a symbol `s`:

```
freq[s] += 1
total += 1
if total >= 16384:
    rescale()
```

### 4.4 Rescaling

When `total` reaches or exceeds 16384:

```
total = 0
for i in 0..N:
    freq[i] = max(freq[i] / 2, 1)    // integer division, floor 1
    total += freq[i]
```

The floor of 1 ensures no symbol ever reaches zero probability.

### 4.5 CDF Queries

The arithmetic coder requires:

- **`cum_freq(s)`**: The cumulative frequency of symbols `0..s` (exclusive of `s`).
  Equivalently, `cum_freq(0) = 0`, `cum_freq(s) = freq[0] + freq[1] + ... + freq[s-1]`.
- **`freq(s)`**: The frequency of symbol `s`. Note: `cum_freq(s+1) = cum_freq(s) + freq(s)`.
- **`total`**: The sum of all frequencies, i.e., `cum_freq(N)`.
- **`find(value)`**: The symbol `s` such that `cum_freq(s) <= value < cum_freq(s) + freq(s)`.

A Fenwick tree (binary indexed tree) provides O(log N) performance for these operations but
is not required. A linear scan is equally valid.

---

## 5. Model Set

Six adaptive models are used. All are reset to their initial uniform distributions at the
start of each Sonnet. Models are **not** reset at Haiku boundaries.

| # | Name          | Alphabet Size | Description                           |
|---|---------------|---------------|---------------------------------------|
| 1 | `tag`         | 4             | Token type: LINE, EOH, EOS, EOO      |
| 2 | `literal_len` | 256           | Number of literal bytes (0–255)       |
| 3 | `match_len`   | 256           | Match copy length (0–255)             |
| 4 | `dist_lo`     | 256           | Low byte of match distance            |
| 5 | `dist_hi`     | 256           | High byte of match distance           |
| 6 | `literals`    | 256           | Individual literal byte values        |

---

## 6. LZ77 Sliding Window

The LZ77 stage uses a sliding window for backward-reference matching.

- **Window size**: 65,536 bytes (64 KiB). The maximum representable distance is 65,535
  (since distance 0 is invalid), so the oldest byte in a completely full window is not
  addressable. In practice this has negligible impact on compression.
- **`match_distance`**: 1-based. A distance of 1 refers to the byte immediately before the
  current output position. Valid range: 1–65,535. A distance of 0 is invalid and MUST be
  rejected by decoders. A distance exceeding the number of bytes emitted so far within the
  current Sonnet is also invalid and MUST be rejected.
- **`match_len`**: 0 means no match (literals only). When nonzero, valid range: 1–255.
  The encoder SHOULD use a minimum match length of 3 for compression efficiency, but the
  format permits any nonzero value.
- **Overlapping matches**: A match MAY extend beyond the source region (i.e.,
  `match_len > match_distance`). This enables run-length encoding: the decoder copies one
  byte at a time from the window, and previously copied bytes become available as source.
- **Window reset**: The sliding window MUST be cleared at the start of each Sonnet.

The choice of match-finding algorithm (greedy, lazy evaluation, optimal parsing, etc.) is
left to the encoder. Any algorithm that produces valid `(literal_len, match_len,
match_distance)` tuples conforming to the constraints above is compliant.

---

## 7. ULEB128

Footer fields `cumulative_bytes` and `cumulative_lines` are encoded as unsigned LEB128,
bounded to `u64`.

### Encoding

```
fn encode_uleb128(value: u64) -> bytes:
    out = []
    for _ in 0..8:                     // up to 8 continuation bytes
        byte = value & 0x7F            // 7 payload bits
        value >>= 7
        if value == 0:
            out.append(byte)           // final byte: MSB = 0
            return out
        out.append(byte | 0x80)        // continuation: MSB = 1
    out.append(value as u8)            // 9th byte: all 8 bits, no continuation bit
    return out
```

### Decoding

```
fn decode_uleb128(input: &[u8]) -> (u64, bytes_consumed):
    result: u64 = 0
    for i in 0..8:
        byte = input[i]
        result |= (byte & 0x7F) as u64 << (7 * i)
        if byte & 0x80 == 0:
            return (result, i + 1)
    // 9th byte: all 8 bits
    result |= (input[8] as u64) << 56
    return (result, 9)
```

- Maximum encoding: 9 bytes for a `u64` value.
- The 9th byte (if present) uses all 8 bits — there is no continuation bit.
- Encoders MUST produce canonical (minimal-length) encodings.
- Decoders SHOULD reject non-canonical encodings (e.g., `0x80 0x00` for the value 0).

---

## 8. Adler-32

The checksum in Couplet and Coda footers is a standard Adler-32, computed over
**uncompressed (original) bytes only**.

### Algorithm

```
MOD_ADLER = 65521    // largest prime less than 2^16

a: u32 = 1
b: u32 = 0

for each uncompressed byte:
    a = (a + byte) mod MOD_ADLER
    b = (b + a) mod MOD_ADLER

checksum: u32 = (b << 16) | a
```

### Storage

The 4-byte checksum is stored **little-endian** in the footer.

### Cumulative property

The checksum in each Couplet or Coda covers all uncompressed bytes from the start of the
Opus up to and including the data represented by that Sonnet. Decompressing an entire sealed
Opus and recomputing the Adler-32 over the result MUST produce a value matching the Coda's
`checksum` field.

### Random-access validation

For random-access reads, a decoder can validate a single Sonnet without decompressing all
preceding data. Extract the Adler-32 components from the previous Sonnet's Couplet checksum:
`a = checksum & 0xFFFF`, `b = checksum >> 16`. Initialize the Adler-32 state with these
`(a, b)` values, decompress the target Sonnet, and compare the resulting checksum against
the target Sonnet's Couplet or Coda. For the first Sonnet, use the standard initial state
(`a = 1`, `b = 0`).

---

## 9. Operational Modes

### 9.1 Sealed Archives (.lzr)

A sealed archive is a complete, immutable Opus:

- The final Sonnet contains an `EOO` tag followed by a Coda footer.
- Supports **streaming decompression** from start to end.
- Supports **random-access reads** via Sonnet footers (when the file contains a single Opus).
- Supports **concatenation**: multiple sealed archives may be concatenated into a Mythos for
  streaming decompression. Random-access reads are NOT supported on a concatenated Mythos.
- MUST NOT be appended to. To add data, the archive must be unsealed or a new archive
  concatenated.

### 9.2 Unsealed Archives (.lzra)

An unsealed archive is an in-progress, appendable Opus:

- The final Sonnet is incomplete — there is no `EOO` tag or Coda footer.
- Supports **random-access reads** via the footers of completed Sonnets. The incomplete
  final Sonnet is not randomly accessible.
- Supports **appending** new data, which extends the current Sonnet or begins a new one.
- MUST NOT be concatenated.
- May be **sealed** by writing an `EOO` tag and Coda to finalize the archive.

### 9.3 Detecting Archive State

A decoder can determine the archive state by examining the end of the file:

1. Read the last byte of the file as a candidate `footer_len`.
2. If `footer_len` is in the valid range [7, 23], back up `footer_len` bytes from the end of
   the file and parse the candidate footer (two ULEB128 values, a 4-byte Adler-32, and the
   `footer_len` byte).
3. If the parsed fields are self-consistent (total size matches `footer_len`) and the
   Adler-32 validates against the decompressed data, the archive is **sealed**.
4. Otherwise, the archive is **unsealed** or the file is corrupt.

Decoders SHOULD return an appropriate error if a file opened for random-access or append
operations turns out to be a concatenated Mythos (multiple Opus archives).

---

## 10. Decoder Algorithm

### 10.1 Streaming Decode

```
1.  Read and validate Invocation (4 bytes).
2.  Initialize all six models to uniform distributions.
3.  Clear the LZ77 window.
4.  Initialize arithmetic decoder: low = 0, high = 0xFFFF, read 16 bits into code.
5.  Loop:
      a. Decode tag from the tag model.
      b. If tag == LINE:
           i.   Decode literal_len from literal_len model.
           ii.  Decode match_len from match_len model.
           iii. If match_len > 0: decode dist_lo, dist_hi from their models.
           iv.  Decode literal_len literal bytes from literals model.
           v.   Emit literals to output and LZ77 window.
           vi.  If match_len > 0: copy match_len bytes from window at match_distance.
      c. If tag == EOH:
           i.   Finalize arithmetic decoder (see [Section 3.7](#37-decoder-haiku-finalization)).
           ii.  Re-initialize arithmetic decoder for next Haiku.
           iii. Continue loop (models are NOT reset).
      d. If tag == EOS:
           i.   Finalize arithmetic decoder (see [Section 3.7](#37-decoder-haiku-finalization)).
           ii.  Read `footer_len` from the last byte of the Sonnet, then parse and
                validate the Couplet from the last `footer_len` bytes. Padding bytes
                between the Kireji and footer are skipped implicitly.
           iii. Reset all six models to uniform distributions.
           iv.  Clear LZ77 window.
           v.   Re-initialize arithmetic decoder.
           vi.  Continue loop.
      e. If tag == EOO:
           i.   Finalize arithmetic decoder (see [Section 3.7](#37-decoder-haiku-finalization)).
           ii.  Read `footer_len` from the last byte of the Sonnet, then parse and
                validate the Coda from the last `footer_len` bytes. Padding bytes
                between the Kireji and footer are skipped implicitly.
           iii. Opus is complete. Check for another Invocation (Mythos) or EOF.
6.  Done.
```

### 10.2 Random-Access Seek

To seek to a target byte offset or line number within a single (non-concatenated) Opus:

```
1.  Compute the block index: target is in Sonnet N if the Couplet of Sonnet N-1 shows
    cumulative_bytes < target <= cumulative_bytes of Sonnet N's Couplet.
    (Use interpolation or binary search over Sonnet footers.)
2.  Seek to file offset: 4 (Invocation) + N * 262144.
3.  Decompress Sonnet N from the beginning (reset models, clear window, init decoder).
4.  Discard uncompressed bytes until reaching the target offset within the Sonnet.
5.  Begin emitting output from the target position.
```

For line-based seeking, use `cumulative_lines` in the same manner.

---

## 11. Conformance

### Encoder Requirements

- Encoders MUST write a valid Invocation with a recognized version byte.
- Encoders MUST NOT emit a `LINE` token with `literal_len = 0` and `match_len = 0`.
- Encoders MUST flush pending literals as a `LINE` token before emitting any terminator tag
  (`EOH`, `EOS`, `EOO`).
- Encoders MUST write correct cumulative footer values over uncompressed data.
- Encoders MUST produce canonical (minimal-length) ULEB128 encodings in footer fields.
- Encoders MUST pad the space between the last Kireji and the footer with `0x00` bytes.
- Non-final Sonnets MUST be exactly 262,144 compressed bytes.

### Decoder Requirements

- Decoders MUST validate the Invocation magic bytes and reject unknown versions.
- Decoders MUST reject `LINE` tokens with `literal_len = 0` and `match_len = 0`.
- Decoders MUST reject `match_distance = 0` when `match_len > 0`.
- Decoders MUST reject `match_distance` greater than the number of uncompressed bytes emitted
  so far within the current Sonnet.
- Decoders MUST validate the Adler-32 checksum in each Couplet and Coda against the
  cumulative uncompressed data.
- Decoders MUST reset all models and the LZ77 window at Sonnet boundaries.
- Decoders MUST reset the arithmetic coder state at Haiku boundaries.
- Decoders MUST NOT reset adaptive models at Haiku boundaries.
- Decoders MUST byte-align the bit reader after decoding any terminator tag (`EOH`, `EOS`,
  `EOO`), as described in [Section 3.7](#37-decoder-haiku-finalization).
- Decoders SHOULD reject non-canonical ULEB128 encodings in footer fields.