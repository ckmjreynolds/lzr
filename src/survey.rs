//! v3 capability survey: information-theoretic entropy bounds across
//! tokenization schemes.
//!
//! Answers the architectural question "what's the lowest bpb floor
//! achievable under each tokenization choice?" before committing
//! more codec engineering to a specific path. Each row in the survey
//! reports Order-0 and Order-1 entropy of the tokenized stream,
//! expressed in bits-per-byte-of-original-input so comparisons are
//! direct.
//!
//! Schemes surveyed:
//! - **bytes**: raw u8 alphabet (Order-0/1/2).
//! - **chars**: same as bytes, listed separately for clarity since
//!   most enwik8 bytes are ASCII so they coincide.
//! - **word tokens**: letter/non-letter alternating runs, retained
//!   as variable-size dictionaries sized 8k, 16k, 32k, 64k, 128k,
//!   256k (frequency-pruned at the top). Tokens beyond the cap
//!   pay per-byte OOV cost.
//! - **BPE tokens**: trained byte-pair encoding at vocab sizes 4k,
//!   8k, 16k, 32k, 64k. No OOV (BPE always falls back to byte
//!   tokens at the base).
//!
//! Entropy computed as `-Σ p(t) log₂ p(t)` for Order-0 and
//! `Σ p(t-1) · H(t | t-1)` for Order-1. OOV cost for word
//! tokens accounted as the full byte-by-byte spelling cost
//! (estimated from byte Order-1 entropy on the OOV bytes).

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};

use crate::bpe;

/// CLI entry point. Reads `bytes` bytes from the start of `corpus`
/// and runs every survey row, printing a final summary table.
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
pub(crate) fn run_survey(corpus: &Path, bytes: usize) -> Result<()> {
    let mut f =
        File::open(corpus).with_context(|| format!("opening corpus {}", corpus.display()))?;
    let mut buf = vec![0u8; bytes];
    let read = f.read(&mut buf)?;
    buf.truncate(read);
    let n_bytes = buf.len();
    eprintln!(
        "Surveying {} bytes ({:.1} MiB) of {}",
        n_bytes,
        n_bytes as f64 / (1024.0 * 1024.0),
        corpus.display(),
    );

    let mut rows: Vec<SurveyRow> = Vec::new();

    // === Byte-level ===
    eprintln!("byte Order-0..2 ...");
    let (bo0, bo1, bo2) = byte_entropy_orders_0_1_2(&buf);
    rows.push(SurveyRow {
        scheme: "bytes".into(),
        params: "Order-0".into(),
        n_tokens: u64::try_from(n_bytes).expect("byte count fits u64"),
        order0_bpb: bo0,
        order1_bpb: bo0, // n/a; report Order-0 again
        notes: String::new(),
    });
    rows.push(SurveyRow {
        scheme: "bytes".into(),
        params: "Order-1".into(),
        n_tokens: u64::try_from(n_bytes).expect("byte count fits u64"),
        order0_bpb: bo0,
        order1_bpb: bo1,
        notes: String::new(),
    });
    rows.push(SurveyRow {
        scheme: "bytes".into(),
        params: "Order-2".into(),
        n_tokens: u64::try_from(n_bytes).expect("byte count fits u64"),
        order0_bpb: bo0,
        order1_bpb: bo2,
        notes: "Order-2".into(),
    });

    // === Word tokens at various dict sizes ===
    for &cap in &[8_192_usize, 16_384, 32_768, 65_536, 131_072, 262_144] {
        eprintln!("word tokens, dict cap {cap} ...");
        let (n_tok, h0, h1, oov_bytes) = word_token_entropy(&buf, cap);
        let oov_bits = (oov_bytes as f64) * 8.0; // upper bound, full byte cost
        let total0_bits = h0.mul_add(n_tok as f64, oov_bits);
        let total1_bits = h1.mul_add(n_tok as f64, oov_bits);
        rows.push(SurveyRow {
            scheme: "word".into(),
            params: format!("dict={cap}"),
            n_tokens: n_tok,
            order0_bpb: total0_bits / (n_bytes as f64 * 8.0) * 8.0,
            order1_bpb: total1_bits / (n_bytes as f64 * 8.0) * 8.0,
            notes: format!(
                "{} OOV bytes ({:.1}% of corpus)",
                oov_bytes,
                100.0 * oov_bytes as f64 / n_bytes as f64,
            ),
        });
    }

    // === BPE at various vocab sizes ===
    // BPE is trained on a 4 MiB prefix regardless of survey buffer
    // size; the merges generalize and keeping training tractable
    // matters at 16 MiB+ corpora. Entropy is measured on the full
    // buffer, so this is a slight pessimization (no test-time
    // overfit) — still apples-to-apples for the survey question.
    let bpe_train_bytes = buf.len().min(4 * 1024 * 1024);
    for &vsize in &[4096_usize, 8192, 16_384, 32_768, 65_536] {
        eprintln!("BPE vocab {vsize} ...");
        let merges = bpe::train(&buf[..bpe_train_bytes], vsize);
        let tokens = bpe::tokenize(&buf, &merges);
        let h0 = token_entropy_order0(&tokens);
        let h1 = token_entropy_order1(&tokens);
        let n_tok = tokens.len() as u64;
        let total0_bits = h0 * n_tok as f64;
        let total1_bits = h1 * n_tok as f64;
        rows.push(SurveyRow {
            scheme: "BPE".into(),
            params: format!("vocab={vsize}"),
            n_tokens: n_tok,
            order0_bpb: total0_bits / (n_bytes as f64 * 8.0) * 8.0,
            order1_bpb: total1_bits / (n_bytes as f64 * 8.0) * 8.0,
            notes: format!(
                "{} merges; ~{:.2} bytes/token",
                merges.len(),
                n_bytes as f64 / n_tok as f64
            ),
        });
    }

    // === Print table ===
    println!();
    println!(
        "Capability survey on {} ({} bytes scanned)",
        corpus.display(),
        n_bytes
    );
    println!();
    println!(
        "{:<8} {:<14} {:>11} {:>10} {:>10}    notes",
        "scheme", "params", "n_tokens", "Ord-0 bpb", "Ord-1 bpb",
    );
    println!(
        "{:-<8} {:-<14} {:->11} {:->10} {:->10}    {:-<40}",
        "", "", "", "", "", "",
    );
    for r in &rows {
        println!(
            "{:<8} {:<14} {:>11} {:>10.3} {:>10.3}    {}",
            r.scheme, r.params, r.n_tokens, r.order0_bpb, r.order1_bpb, r.notes
        );
    }
    println!();
    println!(
        "Best Order-0 floor: {:.3} bpb",
        rows.iter()
            .map(|r| r.order0_bpb)
            .fold(f64::INFINITY, f64::min)
    );
    println!(
        "Best Order-1 floor: {:.3} bpb",
        rows.iter()
            .map(|r| r.order1_bpb)
            .fold(f64::INFINITY, f64::min)
    );

    Ok(())
}

struct SurveyRow {
    scheme: String,
    params: String,
    n_tokens: u64,
    order0_bpb: f64,
    order1_bpb: f64,
    notes: String,
}

// =========================================================================
// Byte-level entropy
// =========================================================================

#[allow(clippy::cast_precision_loss)]
fn byte_entropy_orders_0_1_2(buf: &[u8]) -> (f64, f64, f64) {
    // Order-0: 256-symbol histogram.
    let mut c0 = [0u64; 256];
    for &b in buf {
        c0[b as usize] += 1;
    }
    let n = buf.len() as f64;
    let h0: f64 = c0
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.log2()
        })
        .sum();

    // Order-1: P(b | prev). Dense 256x256 counts.
    let mut c1 = vec![[0u64; 256]; 256];
    let mut prev = buf[0] as usize;
    for &b in &buf[1..] {
        c1[prev][b as usize] += 1;
        prev = b as usize;
    }
    let mut h1 = 0.0_f64;
    for (i, row) in c1.iter().enumerate() {
        let row_sum: u64 = row.iter().sum();
        if row_sum == 0 {
            continue;
        }
        let p_prev = c0[i] as f64 / n;
        let row_h: f64 = row
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p_given = c as f64 / row_sum as f64;
                -p_given * p_given.log2()
            })
            .sum();
        h1 = p_prev.mul_add(row_h, h1);
    }

    // Order-2: P(b | prev2, prev1). Sparse to keep memory in check.
    let mut c2: HashMap<(u8, u8), [u64; 256]> = HashMap::new();
    let mut p2 = buf[0];
    let mut p1 = buf[1];
    let mut ctx_marginal: HashMap<(u8, u8), u64> = HashMap::new();
    for &b in &buf[2..] {
        let row = c2.entry((p2, p1)).or_insert([0u64; 256]);
        row[b as usize] += 1;
        *ctx_marginal.entry((p2, p1)).or_insert(0) += 1;
        p2 = p1;
        p1 = b;
    }
    let total_ctx: u64 = ctx_marginal.values().sum();
    let mut h2 = 0.0_f64;
    for (k, row) in &c2 {
        let row_sum = ctx_marginal[k];
        let p_ctx = row_sum as f64 / total_ctx as f64;
        let row_h: f64 = row
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p_given = c as f64 / row_sum as f64;
                -p_given * p_given.log2()
            })
            .sum();
        h2 = p_ctx.mul_add(row_h, h2);
    }

    (h0, h1, h2)
}

// =========================================================================
// Word-token entropy
// =========================================================================

/// Tokenize letter / non-letter runs; pick the top `cap` by
/// frequency as the in-dict words and pay full byte-cost on the
/// remainder. Returns
/// `(in_dict_token_count, order0_bits_per_in_dict_token,
///   order1_bits_per_in_dict_token, oov_bytes_total)`.
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn word_token_entropy(buf: &[u8], cap: usize) -> (u64, f64, f64, u64) {
    // Pass 1: tokenize and count word frequencies (we count the
    // lowercased form, matching our case-fold convention).
    let mut p = 0;
    let mut all_freq: HashMap<Vec<u8>, u64> = HashMap::new();
    let mut all_tokens: Vec<Vec<u8>> = Vec::new();
    while p < buf.len() {
        let class_is_letter = buf[p].is_ascii_alphabetic();
        let mut q = p + 1;
        while q < buf.len() && buf[q].is_ascii_alphabetic() == class_is_letter {
            q += 1;
        }
        let token: Vec<u8> = if class_is_letter {
            buf[p..q].iter().map(u8::to_ascii_lowercase).collect()
        } else {
            buf[p..q].to_vec()
        };
        *all_freq.entry(token.clone()).or_insert(0) += 1;
        all_tokens.push(token);
        p = q;
    }

    // Pick top `cap` by frequency.
    let mut sorted: Vec<(Vec<u8>, u64)> = all_freq.into_iter().collect();
    sorted.sort_by_key(|e| std::cmp::Reverse(e.1));
    let mut in_dict: HashMap<Vec<u8>, u32> = HashMap::new();
    for (i, (k, _)) in sorted.iter().take(cap).enumerate() {
        in_dict.insert(k.clone(), u32::try_from(i).expect("dict index fits u32"));
    }

    // Pass 2: stream the tokenized sequence, separating in-dict
    // tokens (counted in Order-0/1 tables) and OOV tokens (charged
    // their full byte spelling).
    let mut in_dict_seq: Vec<u32> = Vec::new();
    let mut oov_bytes: u64 = 0;
    for tok in &all_tokens {
        if let Some(&id) = in_dict.get(tok) {
            in_dict_seq.push(id);
        } else {
            oov_bytes += u64::try_from(tok.len()).expect("token length fits u64");
        }
    }

    let n_tok = in_dict_seq.len() as u64;
    if n_tok == 0 {
        return (0, 0.0, 0.0, oov_bytes);
    }
    let h0 = token_entropy_order0_u32(&in_dict_seq);
    let h1 = token_entropy_order1_u32(&in_dict_seq);

    (n_tok, h0, h1, oov_bytes)
}

// =========================================================================
// Token entropy helpers (u32-keyed; same impl applies to BPE)
// =========================================================================

#[allow(clippy::cast_precision_loss)]
pub(crate) fn token_entropy_order0(tokens: &[u32]) -> f64 {
    token_entropy_order0_u32(tokens)
}

#[allow(clippy::cast_precision_loss)]
fn token_entropy_order0_u32(tokens: &[u32]) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }
    let mut counts: HashMap<u32, u64> = HashMap::new();
    for &t in tokens {
        *counts.entry(t).or_insert(0) += 1;
    }
    let n = tokens.len() as f64;
    counts
        .values()
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.log2()
        })
        .sum()
}

#[allow(clippy::cast_precision_loss)]
pub(crate) fn token_entropy_order1(tokens: &[u32]) -> f64 {
    token_entropy_order1_u32(tokens)
}

#[allow(clippy::cast_precision_loss)]
fn token_entropy_order1_u32(tokens: &[u32]) -> f64 {
    if tokens.len() < 2 {
        return 0.0;
    }
    // Marginal P(prev).
    let mut marginal: HashMap<u32, u64> = HashMap::new();
    for &t in &tokens[..tokens.len() - 1] {
        *marginal.entry(t).or_insert(0) += 1;
    }
    // Joint counts P(prev, curr).
    let mut joint: HashMap<u32, HashMap<u32, u64>> = HashMap::new();
    let mut prev = tokens[0];
    for &t in &tokens[1..] {
        *joint.entry(prev).or_default().entry(t).or_insert(0) += 1;
        prev = t;
    }
    let total_pairs = (tokens.len() - 1) as f64;
    let mut h = 0.0_f64;
    for (p, row) in &joint {
        let prev_count = marginal[p];
        let p_prev = prev_count as f64 / total_pairs;
        let row_h: f64 = row
            .values()
            .map(|&c| {
                let p_given = c as f64 / prev_count as f64;
                -p_given * p_given.log2()
            })
            .sum();
        h += p_prev * row_h;
    }
    h
}
