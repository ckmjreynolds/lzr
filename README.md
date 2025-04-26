# The LZR Specification
This project started as an initial project to learn [Rust](https://www.rust-lang.org/). There were no specific goals other than learning and by extension readability. Since this was first and foremost a learning experience, I avoided referencing any existing code. Though "based on" existing algorithms, no attempt was made to faithfully follow any existing algorithm.

### LZR File Format
**Note:** All fields are big-endian.

| `magic` | `sequences` | `length` | `checksum` |
| - | - | - | - |
| `0x4C5A5200` | See below. | 32-bit | [Adler-32](https://en.wikipedia.org/wiki/Adler-32) |

**`magic`** - magic number ASCII "LZR" plus a version byte, currently `0x00`.
**`sequences`** - is a list of LZR sequences.
**`length`** - is the lower 32-bits of the actual length of the original file, a file of length 2^32 bytes will have `0` here.
**`checksum`** - is the Adler-32 checksum of the original file.

### LZR Sequence Format
This stage is roughly LZ77, at least as described by [Wikipedia](https://en.wikipedia.org/wiki/LZ77_and_LZ78#LZ77). This specification assumes existing knowledge of LZ77. There are four sequence formats which can be determined by the first one to three bits. There is always at least one literal byte. When `distance` is `0`, the `length` encodes the number of literal bytes included. When `distance` is not `0`, the `length` encodes the number of bytes to repeat and then one literal byte is also included.

#### Short Repeat Sequence
| `format` | `distance` | `length` | `literal(s)` |
| - | - | - | - |
| 1-bit(`0b0`) | 5-bits | 2-bits | (1-4)-bytes |

**`format`** - is `0b0` for short repeat sequences.
**`distance`** - encodes a distance within the last 31 bytes for a repeat or `0` for a literal sequence.
**`length`** - encodes a repeat or literal length of 1...4 bytes.
**`literal(s)`** - are the literal bytes to be copied to the output.

#### Medium Repeat Sequence
| `format` | `distance` | `length` | `literal(s)` |
| - | - | - | - |
| 2-bits(`0b10`) | 10-bits | 4-bits | (2-17)-bytes |

**`format`** - is `0b10` for medium repeat sequences.
**`distance`** - encodes a distance within the last 1,023 bytes for a repeat or `0` for a literal sequence.
**`length`** - encodes a repeat or literal length of 2...17 bytes.
**`literal(s)`** - are the literal bytes to be copied to the output.

#### Long Repeat Sequence
| `format` | `distance` | `length` | `literal(s)` |
| - | - | - | - |
| 3-bits(`0b110`) | 13-bits | 8-bits | (3-258)-bytes |

**`format`** - is `0b110` for long repeat sequences.
**`distance`** - encodes a distance within the last 8,191 bytes for a repeat or `0` for a literal sequence.
**`length`** - encodes a repeat or literal length of 3...258 bytes.
**`literal(s)`** - are the literal bytes to be copied to the output.

#### Extended Repeat Sequence
| `format` | `distance` | `length` | `literal(s)` |
| - | - | - | - |
| 3-bits(`111b`) | 19-bits | 10-bits | (4-1,027)-bytes |

**`format`** - is `0b111` for extended repeat sequences.
**`distance`** - encodes a distance within the last 524,287 bytes for a repeat or `0` for a literal sequence.
**`length`** - encodes a repeat or literal length of 4...1,027 bytes.
**`literal(s)`** - are the literal bytes to be copied to the output.
