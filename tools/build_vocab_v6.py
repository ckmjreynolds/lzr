#!/usr/bin/env python3
"""v6 candidate tokenizer: word / digit / symbol vocabulary builder + measurement.

Design (CDR, v6 branch):
  - WORDS: lowercase maximal letter-runs, frequency-capped (default 32K).
    Capitalization is a separate side-channel, NOT a vocab axis.
  - DIGITS: the 10 digits, each its own token.
  - SYMBOLS: distinct single non-letter/non-digit chars.
  - <esc>: out-of-vocab word escape -> word spelled with letter tokens (a-z),
    the bounded tail-handler (NOT a general byte crutch).

Single streaming pass over the corpus; everything else derived from the counts.
Emits a human-readable vocab and reports effective bytes/token under three
tokenization assumptions, plus compressed dictionary ship-cost (xz -9 proxy for
the UPX/LZMA-compressed submission).
"""
import re, sys, os, collections, subprocess, json

CORPUS = sys.argv[1] if len(sys.argv) > 1 else "assets/enwik9"
WORD_CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 32768
OUT_TXT = "tools/vocab_v6.txt"
OUT_JSON = "tools/vocab_v6.json"

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
DIGITS = "0123456789"

words_cs = collections.Counter()      # case-sensitive (for capitalization stats)
sym_chars = collections.Counter()     # distinct single symbol chars
n_word_occ = 0
n_digit = 0
n_symbol = 0
n_space = 0                           # single ' ' occurrences (side-channel candidate)

with open(CORPUS, "rb") as f:
    for line in f:
        s = line.decode("utf-8", "replace")
        ws = WORD.findall(s)
        n_word_occ += len(ws)
        words_cs.update(ws)
        lc = sum(map(len, ws))
        dc = sum(s.count(d) for d in DIGITS)
        n_word_chars = lc
        n_symbol += len(s) - lc - dc
        n_digit += dc
        n_space += s.count(" ")
        # symbol chars: everything not letter, not digit
        for ch, c in collections.Counter(s).items():
            if not ch.isalpha() and not ch.isdigit():
                sym_chars[ch] += c

file_bytes = os.path.getsize(CORPUS)

# fold case
folded = collections.Counter()
for w, c in words_cs.items():
    folded[w.lower()] += c

top = folded.most_common(WORD_CAP)
top_set = set(w for w, _ in top)
in_vocab_occ = sum(c for _, c in top)
esc_word_occ = n_word_occ - in_vocab_occ
esc_letter_tokens = sum(c * len(w) for w, c in folded.items() if w not in top_set)

# capitalization side-channel distribution over word OCCURRENCES
cap_lower = cap_title = cap_upper = cap_other = 0
for w, c in words_cs.items():
    if w.islower():
        cap_lower += c
    elif w.isupper():
        cap_upper += c
    elif w[0].isupper() and w[1:].islower():
        cap_title += c
    else:
        cap_other += c

# ---- effective bytes/token under three assumptions ----
# A: literal spec — each digit + each symbol char + word(cap+esc) is a token
tok_words = in_vocab_occ + esc_word_occ + esc_letter_tokens  # +esc handled below
tok_A = in_vocab_occ + esc_word_occ + esc_letter_tokens + n_digit + n_symbol
# B: single space is a free separator side-channel (not emitted as a token)
tok_B = tok_A - n_space
# C: B, and digit-runs collapse to one token per run instead of per digit
#    (approximate: count maximal digit runs)
# (computed in a cheap second consideration below if needed; report A and B firmly)

vocab_size = 1 + 10 + 26 + len(sym_chars) + len(top)  # esc + digits + letters + symbols + words

def fmt_sym(ch):
    if ch == " ": return "<SP>"
    if ch == "\n": return "<NL>"
    if ch == "\t": return "<TAB>"
    if ch == "\r": return "<CR>"
    o = ord(ch)
    if 0x21 <= o <= 0x7e: return ch
    return f"<U+{o:04X}>"

# ---- write human-readable vocab ----
lines = []
lines.append("# v6 candidate tokenizer vocabulary")
lines.append(f"# corpus: {CORPUS} ({file_bytes:,} bytes)")
lines.append(f"# classes: <esc> + 10 digits + 26 escape-letters + {len(sym_chars)} symbols + {len(top)} words")
lines.append(f"# total vocab entries: {vocab_size:,}")
lines.append("#")
lines.append("# columns: id <TAB> class <TAB> token <TAB> corpus_freq")
lines.append("")
i = 0
def emit(cls, tok, freq):
    global i
    lines.append(f"{i}\t{cls}\t{tok}\t{freq}")
    i += 1
emit("special", "<esc>", esc_word_occ)
for d in DIGITS:
    emit("digit", d, "")
for ch in "abcdefghijklmnopqrstuvwxyz":
    emit("letter", ch, "")
lines.append("# --- symbols (single chars, by freq) ---")
for ch, c in sym_chars.most_common():
    emit("symbol", fmt_sym(ch), c)
lines.append(f"# --- words (lowercase, by freq) — top {len(top)} ---")
for w, c in top:
    emit("word", w, c)

with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines) + "\n")

with open(OUT_JSON, "w") as f:
    json.dump({
        "corpus": CORPUS, "word_cap": WORD_CAP, "vocab_size": vocab_size,
        "symbols": [ch for ch, _ in sym_chars.most_common()],
        "words": [w for w, _ in top],
    }, f, ensure_ascii=False)

def xz_size(path):
    return len(subprocess.run(["xz", "-9", "-c", path], capture_output=True).stdout)

vocab_txt_xz = xz_size(OUT_TXT)
# words-only ship cost: newline-joined word list (what actually must ship for the word axis)
words_only = "tools/.words_only.tmp"
with open(words_only, "w") as f:
    f.write("\n".join(w for w, _ in top) + "\n")
words_raw = os.path.getsize(words_only)
words_xz = xz_size(words_only)
bpe_xz = xz_size("/Users/creynolds/Programming/lzr-neural/ckpts/bpe_16k.bin") \
    if os.path.exists("/Users/creynolds/Programming/lzr-neural/ckpts/bpe_16k.bin") else None
bpe_raw = os.path.getsize("/Users/creynolds/Programming/lzr-neural/ckpts/bpe_16k.bin") \
    if os.path.exists("/Users/creynolds/Programming/lzr-neural/ckpts/bpe_16k.bin") else None
os.remove(words_only)

P = print
P(f"\n=== v6 candidate tokenizer on {CORPUS} ({file_bytes:,} bytes), word cap = {WORD_CAP:,} ===\n")
P(f"vocab entries        : {vocab_size:,}  (esc + 10 digits + 26 letters + {len(sym_chars)} symbols + {len(top)} words)")
P(f"distinct words total : {len(folded):,} folded  -> capped to {len(top):,}")
P(f"word coverage by cap : {100*in_vocab_occ/n_word_occ:.3f}%  (escape rate {100*esc_word_occ/n_word_occ:.3f}% of word occ)")
P()
P("token counts (enwik9-wide):")
P(f"  in-vocab words     : {in_vocab_occ:,}")
P(f"  escaped words      : {esc_word_occ:,}  -> {esc_word_occ:,} <esc> + {esc_letter_tokens:,} letter tokens")
P(f"  digits             : {n_digit:,}")
P(f"  symbols            : {n_symbol:,}   (of which single-space: {n_space:,} = {100*n_space/n_symbol:.1f}%)")
P()
P("effective bytes/token (BPE-16K baseline = 4.58):")
P(f"  A  literal spec (space is a token)        : {file_bytes/tok_A:.3f}   ({tok_A:,} tokens)")
P(f"  B  single-space as side-channel           : {file_bytes/tok_B:.3f}   ({tok_B:,} tokens)")
P()
P("capitalization side-channel (over word occurrences):")
tot = cap_lower+cap_title+cap_upper+cap_other
P(f"  all-lower {100*cap_lower/tot:.1f}%   Title {100*cap_title/tot:.1f}%   "
  f"ALLCAPS {100*cap_upper/tot:.1f}%   other {100*cap_other/tot:.1f}%")
P()
P("dictionary ship-cost (xz -9 = LZMA/UPX proxy):")
P(f"  v6 words-only list : {words_raw:,} raw  ->  {words_xz:,} xz  ({100*words_xz/words_raw:.1f}%)")
P(f"  v6 full vocab .txt : {os.path.getsize(OUT_TXT):,} raw  ->  {vocab_txt_xz:,} xz")
if bpe_xz is not None:
    P(f"  BPE-16K table .bin : {bpe_raw:,} raw  ->  {bpe_xz:,} xz")
P()
P(f"wrote {OUT_TXT} and {OUT_JSON}")
