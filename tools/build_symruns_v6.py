#!/usr/bin/env python3
"""Build the v6 symbol-run vocabulary from the article-body byte stream.

A "symbol run" is a maximal run of bytes that are neither ASCII letters
nor ASCII digits (matching the Rust tokenizer's symbol class) — i.e.
punctuation, whitespace, wiki markup, and UTF-8 continuation bytes. We
take the most frequent multi-byte runs (length 2..=MAXLEN); single-byte
symbols are always representable as a fallback in the tokenizer, so they
need not be listed.

Output: tools/symruns_v6.txt, one run per line as lowercase hex
(e.g. "5b5b" = "[["), most-frequent first.
"""
import re, sys, collections

CORPUS = sys.argv[1] if len(sys.argv) > 1 else "assets/enwik9"
TOP_K = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
MAXLEN = 8
OUT = "tools/symruns_v6.txt"

data = open(CORPUS, "rb").read()
TEXT = re.compile(rb"<text[^>]*>(.*?)</text>", re.DOTALL)
content = b"".join(m.group(1) for m in TEXT.finditer(data))

RUN = re.compile(rb"[^A-Za-z0-9]+")
runs = collections.Counter()
for m in RUN.finditer(content):
    r = m.group(0)
    if 2 <= len(r) <= MAXLEN:
        runs[r] += 1

top = runs.most_common(TOP_K)
with open(OUT, "w") as f:
    for r, _ in top:
        f.write(r.hex() + "\n")

print(f"content bytes: {len(content):,}")
print(f"distinct multi-byte symbol runs (len 2..{MAXLEN}): {len(runs):,}")
print(f"wrote top {len(top)} to {OUT}")
print("top 15:", [(bytes.fromhex(r.hex()).decode('latin1'), c) for r, c in top[:15]])
