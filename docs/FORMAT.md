# LZR wire format

This document specifies the on-disk format written by `lzr::compress` and read by
`lzr::decompress`. All multi-byte integers are unsigned. The format is a small,
self-describing container wrapped around a framing-agnostic **codec core**, so the
core can later be re-wrapped in a near-zero-header framing (for the Hutter-Prize
branch) without changing the codec.

## 1. Container layout

| Offset      | Field       | Size          | Value / meaning                                   |
| ----------- | ----------- | ------------- | ------------------------------------------------- |
| 0           | `magic`     | 3             | ASCII `LZR`                                        |
| 3           | `version`   | 1             | `0x02` (`0x00` = abandoned block format, `0x01` = the earlier 15-bit-token core) |
| 4           | `profile`   | ULEB128       | pipeline feature bitmask (see §3)                  |
| 4 + p       | `core`      | variable      | codec core (§2)                                   |
| len − 4     | `adler32`   | 4, big-endian | Adler-32 of the **original** (uncompressed) bytes |

The minimum container is 9 bytes (`4` prefix + `1` profile + at least `0` core + `4` footer).

Decoding validates, in order: minimum length, `magic`, `version`, `profile`; then
decodes the core; then recomputes the Adler-32 of the output and compares it to the
stored footer. Any mismatch is an error — a container never decodes to silently
wrong bytes.

## 2. Codec core

The core is the output of a **pipeline of byte→byte transforms**, each an optional stage selected by
the `profile` (§3). Every stage transforms a `Vec<u8>` into a `Vec<u8>` and exactly inverts it on
decode; the decoder runs the same stages in reverse order. The canonical stage order is:

```
casefold → entities → repair → entropy
```

with a future LZ77 byte stage slotting in before `entropy`. Any subset may be enabled: turning
`repair` off still compresses if `entropy` is on, and turning `entropy` off still compresses because
the `repair` grammar shrinks the stream on its own. With no stages enabled the core is the original
bytes verbatim.

The two stages with their own sub-formats:

- **`repair`** serializes its grammar *inline* ahead of the reduced symbol sequence, all as
  ULEB128-**u22** varints (§4):
  ```
  [V-1] [(left_0, right_0)] … [(left_{V-1}, right_{V-1})] [sequence …]
  ```
  `V` is the symbol count. Every symbol is a uniform `(left, right)` pair — a terminal is
  `(byte, NIL)` where `NIL = 0x3F_FFFF` (`u22::MAX`, the one reserved value, never a symbol id, so
  `V ≤ u22::MAX − 1 = 0x3F_FFFE`), a non-terminal is a pair of earlier symbol ids. Symbol ids are
  renumbered by descending reference frequency (id 0 = most used), so the vocabulary is self-describing
  and the decoder needs no side channel. The encode-side vocabulary cap (`--num-tokens`, default
  `0x3F_FFFE`) is **not** stored — the decoder reads the actual `V` from `[V-1]`.

- **`entropy`** is the final stage — a context-mixing range coder over bytes:
  ```
  byte_count (ULEB128) || range-coded payload
  ```
  `byte_count` is the number of input bytes; it is the decoder's stop condition and sizes the hashed
  model tables identically on both sides. The payload codes each byte's 8 bits MSB-first through a
  **carryless binary range coder** driven by a logistic mix of the always-on order-0 model and
  whichever optional models the `profile` enables (§3). The decoder rejects a `byte_count` larger
  than the payload could plausibly encode (a decompression-bomb guard) and stops early if it reads
  past the payload into zero-padding, bounding work on corrupt input.

## 3. Pipeline profile

`profile` is a **ULEB128 feature bitmask** recording which pipeline features produced the core, so
the decoder can rebuild the exact pipeline. Each bit is one toggleable feature; the CLI toggles them
by name with `--enable`/`--disable`, and each feature carries its own default. This is a pre-1.0
format: bit assignments may change between versions, but the container `version` (§1) is bumped
whenever they do, so an old stream is rejected rather than misread.

- bit `0` (`0x01`) — **`casefold`** (default on): ASCII case-folding byte preprocessor. Lowercases
  `A`–`Z` and re-encodes the case as two inline control symbols — a **shift** (capitalize the next
  letter) and a **caps-lock** (toggle capitalize-all until seen again) — whose byte values are chosen
  per input as two values absent from it. Self-describing: a leading mode byte is `0x00` for a
  pass-through (input not text, no uppercase, or fewer than two spare bytes) or `0x01` for a folded
  payload, in which case the two chosen symbol bytes follow the mode byte.
- bit `1` (`0x02`) — **`entities`** (default on): XML/HTML entity-folding byte preprocessor. Applied
  after `casefold`. Replaces each of `&lt; &gt; &amp; &quot; &apos;` with one dedicated byte (chosen
  per input as a value absent from it). Same self-describing `0x00`/`0x01` mode-byte header, followed
  by the five chosen substitute bytes on the folded path.
- bit `2` (`0x04`) — **`repair`** (default on): the capped Re-Pair grammar tokenizer, serialized as in
  §2. When clear, the stage is simply absent (bytes pass straight through to the next stage).
- bit `3` (`0x08`) — **`entropy`** (default on): the byte-level context-mixing range coder, serialized
  as in §2. Always the last stage. When clear, the core is whatever the earlier stages produced.
- bit `4` (`0x10`) — **`null`** (default off): an optional entropy model that predicts ½ and never
  learns (a no-op placeholder for the toggleable-model framework). Only participates when `entropy`
  is enabled.

The order-0 entropy model is **always** present (when `entropy` is enabled) and is not a feature.
Bits with no assigned feature are reserved; a `profile` with any reserved bit set is an error.

## 4. ULEB128 (§7 canonical rule)

Unsigned LEB128 stores 7 payload bits per byte with a high continuation bit; a value ≤ 127 fits in one
byte and `u64::MAX` takes nine. Two codecs are used:

- **`u64`** — the general codec (1–9 bytes), used for the `profile` field and the `entropy`
  `byte_count`.
- **`u22`** — a fixed-width-bounded codec (1–3 bytes) for Re-Pair symbol ids. It packs 7 + 7 + 8 = 22
  bits: bytes 0 and 1 carry 7 payload bits plus a continuation bit, and the terminating third byte
  carries all 8 remaining bits. A value < `0x80` takes one byte, < `0x4000` two, otherwise three.

Encodings must be **canonical**: the terminating byte's payload must be non-zero unless a shorter
encoding is impossible. A non-canonical (overlong) encoding — e.g. `0x80 0x00` for the value `0` — is
rejected, as is any encoding that ends before its terminating byte (a truncated input errors rather
than panics).
