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
use std::io::{Read, Seek, SeekFrom};
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
        h = p_prev.mul_add(row_h, h);
    }
    h
}

// =========================================================================
// Panel survey — matches the bench panel structure (warm + measure)
// =========================================================================

const PANEL_WARM_BYTES: usize = 4 * 1024 * 1024;
const PANEL_MEASURE_BYTES: usize = 256 * 1024;
const PANEL_N_OFFSETS: usize = 20;

/// CLI entry: run the panel survey on the canonical 20×256 KiB
/// panel and report panel-OOV rate plus Order-1 entropy attributed
/// to measure bytes only. This is the calibration-corrected version
/// of the prefix survey — it measures what the codec actually sees
/// under prewarm-then-measure, not what a fresh-walk-from-byte-0
/// would observe.
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
pub(crate) fn run_panel_survey(corpus: &Path) -> Result<()> {
    let mut f =
        File::open(corpus).with_context(|| format!("opening corpus {}", corpus.display()))?;
    let corpus_len = f.metadata()?.len();
    let offsets = pick_panel_offsets(corpus_len);
    if offsets.is_empty() {
        anyhow::bail!(
            "corpus too small for the canonical panel (need ≥ {} bytes)",
            PANEL_WARM_BYTES + PANEL_MEASURE_BYTES,
        );
    }

    eprintln!(
        "Panel survey: {} windows × ({} KiB warm + {} KiB measure)",
        offsets.len(),
        PANEL_WARM_BYTES / 1024,
        PANEL_MEASURE_BYTES / 1024,
    );
    eprintln!();

    let mut total_measure_bytes: u64 = 0;
    let mut total_measure_tokens: u64 = 0;
    let mut total_oov_tokens: u64 = 0;
    let mut total_oov_bytes: u64 = 0;
    let mut total_order0_bits: f64 = 0.0;
    let mut total_order1_bits: f64 = 0.0;
    let mut total_order1_laplace_bits: f64 = 0.0;

    println!(
        "{:>11} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "offset", "OOV%", "Ord-0", "Ord-1", "Ord-1+L", "tok/byte",
    );
    println!(
        "{:->11} {:->10} {:->10} {:->10} {:->10} {:->10}",
        "", "", "", "", "", "",
    );

    for off in offsets {
        let warm_start = off.saturating_sub(PANEL_WARM_BYTES as u64);
        let warm_len =
            usize::try_from(off - warm_start).expect("warm length bounded by PANEL_WARM_BYTES");
        let mut warm = vec![0u8; warm_len];
        f.seek(SeekFrom::Start(warm_start))?;
        f.read_exact(&mut warm)?;
        let mut measure = vec![0u8; PANEL_MEASURE_BYTES];
        f.seek(SeekFrom::Start(off))?;
        f.read_exact(&mut measure)?;

        let warm_tokens = tokenize_words(&warm);
        let measure_tokens = tokenize_words(&measure);

        // Build the "warm dictionary": every distinct token that
        // appeared in warm. This is the set against which we measure
        // panel-OOV — a token is OOV only if it's truly first-of-its-
        // kind in this window (not just below a frequency cap).
        let mut warm_dict: HashMap<Vec<u8>, u32> = HashMap::new();
        for tok in &warm_tokens {
            let next_id = u32::try_from(warm_dict.len()).expect("id fits u32");
            warm_dict.entry(tok.clone()).or_insert(next_id);
        }

        // Panel OOV: measure tokens whose key isn't in warm_dict.
        // (Dict still grows during measure in the real codec; for
        // entropy purposes we hold it static at warm_end so OOV
        // means "the model hasn't seen this token yet when it
        // starts encoding measure.")
        let mut window_oov_tokens: u64 = 0;
        let mut window_oov_bytes: u64 = 0;
        for tok in &measure_tokens {
            if !warm_dict.contains_key(tok) {
                window_oov_tokens += 1;
                window_oov_bytes += tok.len() as u64;
            }
        }

        // Build bigram counts over the combined warm+measure token
        // stream. This is the "static" Order-1 model — the codec's
        // adaptive model would be slightly worse than this floor
        // due to Laplace smoothing on undersampled contexts.
        // Use a single combined id space.
        let mut combined_dict: HashMap<Vec<u8>, u32> = HashMap::new();
        let mut combined_ids: Vec<u32> =
            Vec::with_capacity(warm_tokens.len() + measure_tokens.len());
        for tok in warm_tokens.iter().chain(measure_tokens.iter()) {
            let next_id = u32::try_from(combined_dict.len()).expect("id fits u32");
            let id = *combined_dict.entry(tok.clone()).or_insert(next_id);
            combined_ids.push(id);
        }
        let warm_len_tokens = warm_tokens.len();

        // Static bigram + marginal counts (full window).
        let mut marginal: HashMap<u32, u64> = HashMap::new();
        let mut joint: HashMap<u32, HashMap<u32, u64>> = HashMap::new();
        for w in combined_ids.windows(2) {
            *marginal.entry(w[0]).or_insert(0) += 1;
            *joint.entry(w[0]).or_default().entry(w[1]).or_insert(0) += 1;
        }
        let total_tokens_in_window = combined_ids.len() as u64;

        // Order-0 + Order-1 entropy attributed to measure tokens
        // only. For each measure token at position i, look up its
        // bigram probability conditioned on token at i-1.
        let mut window_order0_bits: f64 = 0.0;
        let mut window_order1_bits: f64 = 0.0;
        let mut window_order1_laplace_bits: f64 = 0.0;
        let vocab_size = combined_dict.len() as u64;

        let mut measure_token_marginal: HashMap<u32, u64> = HashMap::new();
        for &t in &combined_ids[warm_len_tokens..] {
            *measure_token_marginal.entry(t).or_insert(0) += 1;
        }
        let measure_token_count = combined_ids.len() - warm_len_tokens;

        // Order-0: -log P(t) using full-window frequencies.
        for &t in &combined_ids[warm_len_tokens..] {
            let count = marginal.get(&t).copied().unwrap_or(1).max(1);
            let p = count as f64 / total_tokens_in_window as f64;
            window_order0_bits += -p.log2();
        }

        // Order-1: -log P(curr | prev) using full-window bigram
        // and marginal counts.
        for i in 0..measure_token_count {
            let pos = warm_len_tokens + i;
            if pos == 0 {
                continue; // no prev
            }
            let prev = combined_ids[pos - 1];
            let curr = combined_ids[pos];
            let prev_total = marginal.get(&prev).copied().unwrap_or(0);
            let joint_count = joint
                .get(&prev)
                .and_then(|r| r.get(&curr))
                .copied()
                .unwrap_or(0);
            // No-Laplace (asymptotic floor): if count is 0, skip
            // by attributing 1 bit (unrealistic but bounded).
            let p_no_laplace = if joint_count == 0 || prev_total == 0 {
                1.0 / vocab_size as f64
            } else {
                joint_count as f64 / prev_total as f64
            };
            window_order1_bits += -p_no_laplace.log2();
            // Laplace +1: matches what the codec's adaptive Order-1
            // model would charge for this token.
            let p_laplace = (joint_count + 1) as f64 / (prev_total + vocab_size) as f64;
            window_order1_laplace_bits += -p_laplace.log2();
        }

        let measure_bytes = measure.len() as u64;
        let bpb_oov = 8.0 * (window_oov_bytes as f64) / (measure_bytes as f64);
        let bpb_order0 = window_order0_bits / measure_bytes as f64;
        let bpb_order1 = window_order1_bits / measure_bytes as f64;
        let bpb_order1_laplace = window_order1_laplace_bits / measure_bytes as f64;
        let tok_per_byte = measure_token_count as f64 / measure_bytes as f64;
        let oov_pct = 100.0 * window_oov_tokens as f64 / measure_token_count as f64;
        // OOV cost is approximately a per-byte byte-spelling
        // overhead; surface alongside the Order-N entropy.
        let _ = bpb_oov;

        println!(
            "{off:>11} {oov_pct:>9.2}% {bpb_order0:>10.3} {bpb_order1:>10.3} {bpb_order1_laplace:>10.3} {tok_per_byte:>10.3}",
        );

        total_measure_bytes += measure_bytes;
        total_measure_tokens += measure_token_count as u64;
        total_oov_tokens += window_oov_tokens;
        total_oov_bytes += window_oov_bytes;
        total_order0_bits += window_order0_bits;
        total_order1_bits += window_order1_bits;
        total_order1_laplace_bits += window_order1_laplace_bits;
    }

    println!(
        "{:->11} {:->10} {:->10} {:->10} {:->10} {:->10}",
        "", "", "", "", "", "",
    );
    let mean_oov_pct = 100.0 * total_oov_tokens as f64 / total_measure_tokens as f64;
    let mean_order0 = total_order0_bits / total_measure_bytes as f64;
    let mean_order1 = total_order1_bits / total_measure_bytes as f64;
    let mean_order1_laplace = total_order1_laplace_bits / total_measure_bytes as f64;
    println!(
        "{:>11} {:>9.2}% {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
        "mean",
        mean_oov_pct,
        mean_order0,
        mean_order1,
        mean_order1_laplace,
        total_measure_tokens as f64 / total_measure_bytes as f64,
    );

    println!();
    println!("Panel-OOV rate:        {mean_oov_pct:.2}% of measure tokens");
    println!(
        "Panel-OOV byte share:  {:.2}% of measure bytes",
        100.0 * total_oov_bytes as f64 / total_measure_bytes as f64,
    );
    println!("Order-0 (asymptotic):  {mean_order0:.3} bpb on measure");
    println!("Order-1 (asymptotic):  {mean_order1:.3} bpb on measure  ← entropy floor");
    println!("Order-1 (Laplace +1):  {mean_order1_laplace:.3} bpb on measure  ← codec target");

    Ok(())
}

/// Tokenize `buf` into letter / non-letter runs, lowercasing words
/// to match xml-tok's case-fold convention. Allocates a `Vec<u8>`
/// per token; survey-scale only.
fn tokenize_words(buf: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut p = 0;
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
        out.push(token);
        p = q;
    }
    out
}

/// Pick the same 20 panel offsets the bench panel uses.
fn pick_panel_offsets(corpus_len: u64) -> Vec<u64> {
    let prewarm = PANEL_WARM_BYTES as u64;
    let sample = PANEL_MEASURE_BYTES as u64;
    if corpus_len < prewarm + sample {
        return Vec::new();
    }
    let min_off = prewarm;
    let max_off = corpus_len - sample;
    if max_off < min_off {
        return Vec::new();
    }
    let span = max_off - min_off;
    let n_u64 = PANEL_N_OFFSETS as u64;
    (0..PANEL_N_OFFSETS)
        .map(|i| {
            if PANEL_N_OFFSETS == 1 {
                min_off + span / 2
            } else {
                min_off + (span * i as u64) / (n_u64 - 1)
            }
        })
        .collect()
}
