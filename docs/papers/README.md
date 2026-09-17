# Referenced papers

Vendored copies of papers whose algorithms this crate implements, kept in-repo so the code's
citations are self-contained (offline, and pinned to the version we implemented against).

## `bille-gortz-prezza-2017-space-efficient-repair.pdf`

**Space-Efficient Re-Pair Compression** — Philip Bille, Inge Li Gørtz, Nicola Prezza.
Data Compression Conference (DCC) 2017.

- arXiv: [1611.01479](https://arxiv.org/abs/1611.01479) (PDF: <https://arxiv.org/pdf/1611.01479>)
- Provenance: fetched from arXiv, 2026-07-04.

Implemented by the Re-Pair grammar builder in
[`src/preprocessors/repair/builder.rs`](../../src/preprocessors/repair/builder.rs): **Theorem 1(i)** — exact
frequency-based Re-Pair in O(n/ε) expected time using (1+ε)n + √n words of working space on top of the
text. We use it to keep Re-Pair tokenization of gigabyte inputs (enwik9) under a ~10 GB memory budget,
where the classic Larsson–Moffat layout needs ~24–28 GB. See the module docs in `builder.rs` for the
per-function mapping to the paper's Algorithms 1–2 and Lemma 3.
