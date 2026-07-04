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
| 3           | `version`   | 1             | `0x01` (`0x00` denoted the abandoned block format) |
| 4           | `profile`   | ULEB128       | pipeline feature bitmask (see §3)                  |
| 4 + p       | `core`      | variable      | codec core (§2)                                   |
| len − 4     | `adler32`   | 4, big-endian | Adler-32 of the **original** (uncompressed) bytes |

The minimum container is 10 bytes (`4` prefix + `1` profile + at least `1` core + `4` footer).

Decoding validates, in order: minimum length, `magic`, `version`, `profile`; then
decodes the core; then recomputes the Adler-32 of the output and compares it to the
stored footer. Any mismatch is an error — a container never decodes to silently
wrong bytes.

## 2. Codec core

```
core = token_count (ULEB128) || range-coded payload
```

- `token_count` — the number of `u15` tokens, ULEB128-encoded (§4). It is the
  decoder's stop condition and also sizes the hashed model tables identically on
  both sides, keeping their predictions in lock step.
- `range-coded payload` — the tokens' bits, coded MSB-first (15 bits per token)
  through a **carryless binary range coder** driven by a logistic mix of the
  always-on order-0 model and whichever optional models the `profile` enables (§3).
  The coder emits a byte only once the top byte of its range is settled, so no carry
  propagates and no output is buffered.

The decoder rejects a `token_count` larger than the payload could plausibly encode
(a decompression-bomb guard) and stops early if it reads past the payload into
zero-padding, bounding work on corrupt input.

## 3. Pipeline profile

`profile` is a **ULEB128 feature bitmask** recording which pipeline features produced the core, so
the decoder can rebuild the exact pipeline. Each bit is one toggleable feature; the CLI toggles them
by name with `--enable`/`--disable`, and each feature carries its own default. Bit positions are
append-only (never reorder or reuse a bit, or old containers become unreadable):

- bit `0` (`0x01`) — **`repair`** (default on): the capped Re-Pair grammar tokenizer. When clear, the
  identity (NULL) tokenizer is used (each byte → the equal token). Re-Pair replaces frequent adjacent
  digrams with new symbols, renumbers all symbols by descending reference frequency, and emits its
  grammar *inline* ahead of the reduced sequence: `[V-1] [(left, right) × V] [sequence]`, all `u15`.
  Every symbol is a uniform `(left, right)` pair — a terminal is `(byte, NIL)` where `NIL = 0x7FFF`
  (`u15::MAX`, a reserved value that is never a symbol id, so `V ≤ 32767`), a non-terminal is a pair
  of symbol ids. The codec core framing (§2) is unchanged — the grammar rides in the token stream.
- bit `1` (`0x02`) — **`null`** (default off): an optional entropy model that predicts ½ and never
  learns (a no-op placeholder for the toggleable-model framework).
- bit `2` (`0x04`) — **`casefold`** (default on): an ASCII case-folding byte preprocessor, applied
  before tokenization. It lowercases `A`–`Z` and re-encodes the case as two inline control symbols —
  a **shift** (capitalize the next letter) and a **caps-lock** (toggle capitalize-all until seen
  again) — whose byte values are chosen per input as two values absent from it. The transformed
  stream is self-describing: a leading mode byte is `0x00` for a pass-through (used when the input is
  not text, has no uppercase letter, or has fewer than two spare byte values) or `0x01` for a folded
  payload, in which case the two chosen symbol bytes follow the mode byte. When clear, the identity
  (NULL) byte preprocessor is used.

The order-0 entropy model is **always** present and is not a feature. The NULL token preprocessor is
always present too, as is the NULL byte preprocessor (the `casefold` bit only *adds* a byte stage
ahead of it). Bits with no assigned feature are reserved for future stages (more models, tokenizers,
preprocessors, coders); a `profile` with any reserved bit set is an error.

## 4. ULEB128 (§7 canonical rule)

Unsigned LEB128 stores 7 payload bits per byte with a high continuation bit; a value
≤ 127 fits in one byte and `u64::MAX` takes nine. The `u15` token-count uses the same
scheme.

Encodings must be **canonical**: the terminating byte's payload must be non-zero
unless the value is a single byte. A non-canonical (overlong) encoding — e.g.
`0x80 0x00` for the value `0` — is rejected, as is any encoding that ends before its
terminating byte (a truncated input errors rather than panics).
