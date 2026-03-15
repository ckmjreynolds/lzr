# BLEB8 Specification

**Bounded Little-Endian Base 8** — a variable-length integer encoding.

## Overview

BLEB8 is to [LEB128](https://en.wikipedia.org/wiki/LEB128) what nibbles are to bytes. LEB128 packs 7 data bits per byte with 1 continuation bit; BLEB8 packs 3 data bits per nibble with 1 continuation bit, emitted least-significant nibble first.

Because the maximum number of nibbles is always known from the data type being encoded, the final nibble needs no continuation bit — all 4 of its bits carry data. This bounds both the wire size and the maximum representable value to the range of the underlying type.

BLEB8 has two variants:
- **UBLEB8** — unsigned (analogous to unsigned LEB128)
- **SBLEB8** — signed (analogous to signed LEB128)

## Nibble Stream

A nibble is a 4-bit value. Two nibbles pack into each byte:

```
┌──────────Byte K────────────┐
│┌───────────┐┌─────────────┐│
││ Nibble 2K ││ Nibble 2K+1 ││
││ bits 7–4  ││  bits 3–0   ││
│└───────────┘└─────────────┘│
└────────────────────────────┘
```

- **Nibble 2K** occupies byte K, bits 7–4 (high nibble).
- **Nibble 2K+1** occupies byte K, bits 3–0 (low nibble).
- **Little-endian:** nibble 0 carries the least significant data bits.

If the total number of nibbles is odd, the final byte's low nibble is padding (zero).

## Nibble Format

Each nibble takes one of two forms:

```
Non-final nibble:          Final nibble (at max position):
  3   2   1   0              3   2   1   0
┌───┬───────────┐          ┌───────────────┐
│ C │  D  D  D  │          │  D  D  D  D   │
└───┴───────────┘          └───────────────┘
C=1: more nibbles follow   all 4 bits are data
C=0: last nibble
```

- **Non-final nibble:** bit 3 is the continuation bit (C). Bits 2–0 are 3 data bits.
  - C=1: more nibbles follow.
  - C=0: this is the last nibble (early termination).
- **Final nibble** (at the maximum position for the type): all 4 bits are data. No continuation bit is needed because the decoder knows this is the last nibble by position alone.

## Valid Types

BLEB8 is defined for types whose bit width satisfies:

```
type_bits = 3N + 1
```

where N is the maximum number of nibbles. Equivalently:

```
N = (type_bits - 1) / 3       (must divide evenly)
```

With N nibbles, BLEB8 carries (N-1) x 3 + 4 = 3N + 1 data bits, exactly covering the type's range. Types whose bit width does not satisfy this constraint (e.g. u8, u32) are not valid BLEB8 types. A type must be widened or narrowed to a valid width before encoding.

## UBLEB8 (Unsigned)

Data bits from successive nibbles concatenate least-significant first to form an unsigned integer.

**Decoding:**

```
decode_ubleb8(max_nibbles N):
    value = 0, shift = 0
    for i in 0..(N-1):
        nibble = read_nibble()
        value |= (nibble & 0x7) << shift
        shift += 3
        if (nibble & 0x8) == 0: return value    // C=0: done
    nibble = read_nibble()                      // final nibble
    value |= nibble << shift                    // all 4 bits are data
    return value
```

**Encoding:**

```
encode_ubleb8(value, max_nibbles N):
    for i in 0..(N-1):
        nibble = value & 0x7
        value >>= 3
        if value != 0: nibble |= 0x8            // C=1: more follow
        emit_nibble(nibble)
        if value == 0: return                   // C=0: done
    emit_nibble(value & 0xF)                    // final nibble
```

**Minimal encoding:** the encoder must stop at the earliest nibble where the remaining value is zero. This is the natural behavior of the algorithm above — it returns as soon as `value == 0`.

## SBLEB8 (Signed)

Same as UBLEB8 but the decoded value is sign-extended from the number of data bits actually read. Analogous to signed LEB128.

**Decoding:**

```
decode_sleb8(max_nibbles N):
    value = 0, shift = 0, done = false
    for i in 0..(N-1):
        nibble = read_nibble()
        value |= (nibble & 0x7) << shift
        shift += 3
        if (nibble & 0x8) == 0:                 // C=0: done
            done = true
            break
    if not done:                                // final nibble
        nibble = read_nibble()
        value |= nibble << shift
        shift += 4
    if value & (1 << (shift - 1)):              // sign-extend
        value |= ~0 << shift
    return value
```

**Encoding** follows the [signed LEB128](https://en.wikipedia.org/wiki/LEB128#Signed_LEB128) termination rule. After extracting each nibble's 3 data bits, arithmetic-shift the value right by 3. Stop (C=0) when sign extension would reconstruct the remaining value — that is, when the remaining value is 0 and the data MSB is 0, or the remaining value is -1 and the data MSB is 1:

```
encode_sleb8(value, max_nibbles N):
    for i in 0..(N-1):
        nibble = value & 0x7
        value >>= 3                            // arithmetic shift
        done = (value ==  0 && (nibble & 0x4) == 0) ||
               (value == -1 && (nibble & 0x4) != 0)
        if not done: nibble |= 0x8             // C=1: more follow
        emit_nibble(nibble)
        if done: return
    emit_nibble(value & 0xF)                   // final nibble
```

**Minimal encoding:** the encoder must stop at the earliest nibble where sign extension of the data bits already emitted would reconstruct the full value. A signed value sometimes needs an extra nibble compared to unsigned when the data MSB conflicts with the sign.

## Worked Examples

All examples use u16/i16 (max 5 nibbles).

### 31,415 as u16

```
31,415 = 0b111_1010_1011_0111

Nibble 1: 31415 & 0x7 = 7, 31415 >> 3 = 3926, more → C=1 → 0b1111 (0xF)
Nibble 2:  3926 & 0x7 = 6,  3926 >> 3 =  490, more → C=1 → 0b1110 (0xE)
Nibble 3:   490 & 0x7 = 2,   490 >> 3 =   61, more → C=1 → 0b1010 (0xA)
Nibble 4:    61 & 0x7 = 5,    61 >> 3 =    7, more → C=1 → 0b1101 (0xD)
Nibble 5 (final):  7 & 0xF = 7                     →        0b0111 (0x7)

Emitted nibbles (LE): 0xF 0xE 0xA 0xD 0x7

Check: 7 + (6 << 3) + (2 << 6) + (5 << 9) + (7 << 12) = 31,415 ✓

Byte layout:
  Byte 0: (0xF << 4) | 0xE = 0xFE
  Byte 1: (0xA << 4) | 0xD = 0xAD
  Byte 2: (0x7 << 4) | 0x0 = 0x70  (padded)
```

This value requires all 5 nibbles. The final nibble at the max position uses all 4 data bits with no continuation bit.

### 5 as u16

```
Nibble 1: 5 & 0x7 = 5, 5 >> 3 = 0, done → C=0 → 0b0101 (0x5)

Emitted: 0x5

Check: 5 ✓
```

Early termination after 1 nibble.

### 0 as u16

```
Nibble 1: 0 & 0x7 = 0, 0 >> 3 = 0, done → C=0 → 0b0000 (0x0)

Emitted: 0x0

Check: 0 ✓
```

The smallest possible encoding: a single zero nibble.

### -5 as i16 (signed)

```
-5 = ...1111_1011

Nibble 1: -5 & 0x7 = 3, -5 >> 3 = -1, data MSB=0 ≠ sign → C=1 → 0b1011 (0xB)
Nibble 2: -1 & 0x7 = 7, -1 >> 3 = -1, data MSB=1 = sign  → C=0 → 0b0111 (0x7)

Emitted nibbles (LE): 0xB 0x7

Check: 3 + (7 << 3) = 59 = 0b111011, bit 5 set → sign-extend → -5 ✓
```

Two nibbles are needed because the first nibble's data MSB (bit 2 = 0) does not match the sign (negative), so sign extension from 3 bits would give +3, not -5.

### +5 as i16 (signed)

```
Nibble 1: 5 & 0x7 = 5, 5 >> 3 = 0, data MSB=1 ≠ sign → C=1 → 0b1101 (0xD)
Nibble 2: 0 & 0x7 = 0, 0 >> 3 = 0, data MSB=0 = sign  → C=0 → 0b0000 (0x0)

Emitted nibbles (LE): 0xD 0x0

Check: 5 + (0 << 3) = 5, bit 5 clear → positive → +5 ✓
```

+5 (0b101) cannot fit in one signed nibble because bit 2 being set would sign-extend to -3. An extra nibble with data=0 resolves the ambiguity.

## Minimal Encoding

Encoders **MUST** produce the minimal encoding — the fewest nibbles that represent the value. Decoders **MAY** reject non-minimal encodings.

**Unsigned rule:** stop at the first nibble where the remaining value (after shifting) is zero.

**Signed rule:** stop at the first nibble where sign extension of the bits emitted so far would reconstruct the full value. Concretely, stop when the remaining value is 0 and the last data MSB is 0, or the remaining value is -1 and the last data MSB is 1.

**Non-minimal example:** encoding 5 as u16 using 2 nibbles:

```
Non-minimal: 0xD 0x0   (C=1, data=5) (C=0, data=0)
Minimal:     0x5        (C=0, data=5)
```

Both decode to 5, but the 2-nibble form is non-minimal because the encoder could have stopped after nibble 1 (the remaining value was already 0). A conforming encoder must emit `0x5` — a single nibble.
