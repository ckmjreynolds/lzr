//! Reconnaissance for v3 next-phase decisions, run on the full corpus.
//!
//! Streams the whole corpus through:
//! - The XML `Classifier` for Content/TagStructure/AttrValue separation.
//! - A fine-grained wiki sub-classifier with five sub-modes inside
//!   Content: `Plain`, `LinkTarget`, `LinkDisplay`, `TemplateName`,
//!   `TemplateArg`. Phase-12's analyzer only distinguished three
//!   (Plain/Link/Template); this one splits at the first `|` inside
//!   `[[..]]` and `{{..}}` because the bytes before vs. after that pipe
//!   have very different distributions (link target ≈ proper noun,
//!   link display ≈ free prose).
//! - A page-id counter incremented on each literal `<page>` byte
//!   sequence in the stream.
//! - Per-Word-token statistics: total occurrences, distinct pages,
//!   max occurrences in any one page, and avg per page (the key
//!   diagnostic for Proposal C's page-local cache).
//!
//! Output answers two questions:
//!
//! 1. **Is wiki sub-mode routing worth pursuing (Proposal B)?** Bytes
//!    inside `[[..]]` and `{{..}}` are reported as a percentage of
//!    Content. Phase-12 on a 16 MB prefix found 76% / 4.5% / 19% for
//!    Link/Template/Plain; this re-runs on the full 1 GB and splits
//!    Link further into target vs display, plus Template into name vs
//!    args.
//! 2. **Is a page-local cache worth pursuing (Proposal C)?** For the
//!    top-N most-frequent tokens, `avg_in_page = total / distinct_pages`
//!    measures in-page repetition. avg ≈ 1 means tokens spread across
//!    many pages with one occurrence each (cache useless). avg ≫ 1
//!    means tokens repeat heavily within a page (cache amortizes the
//!    "first occurrence" cost over many subsequent ones).
//!
//! Streaming — never holds more than 1 MB of corpus in RAM, but the
//! `HashMap` of distinct lowercased word tokens can grow to a few
//! hundred MB on enwik9 (~5 M unique types).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result};

use crate::classifier::{Classifier, Mode};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum WikiFine {
    Plain = 0,
    LinkTarget = 1,
    LinkDisplay = 2,
    TemplateName = 3,
    TemplateArg = 4,
}

const N_WIKI_FINE: usize = 5;

impl WikiFine {
    const fn idx(self) -> usize {
        self as usize
    }
    const fn label(self) -> &'static str {
        match self {
            Self::Plain => "Plain",
            Self::LinkTarget => "LinkTarget",
            Self::LinkDisplay => "LinkDisplay",
            Self::TemplateName => "TemplateName",
            Self::TemplateArg => "TemplateArg",
        }
    }
    const fn from_idx(i: usize) -> Self {
        match i {
            1 => Self::LinkTarget,
            2 => Self::LinkDisplay,
            3 => Self::TemplateName,
            4 => Self::TemplateArg,
            _ => Self::Plain,
        }
    }
}

/// One-byte-memory Moore FSM for fine wiki sub-mode. Approximate:
/// the `|` flag is global per type, not per nesting level. For nested
/// templates with `|` only in the inner one we mis-categorize a few
/// outer bytes as `TemplateArg`. Cheap enough to ignore in recon.
struct WikiFineClassifier {
    link_depth: u32,
    template_depth: u32,
    seen_pipe_in_link: bool,
    seen_pipe_in_template: bool,
    prev: Option<u8>,
}

impl WikiFineClassifier {
    const fn new() -> Self {
        Self {
            link_depth: 0,
            template_depth: 0,
            seen_pipe_in_link: false,
            seen_pipe_in_template: false,
            prev: None,
        }
    }

    const fn current_sub(&self) -> WikiFine {
        if self.link_depth > 0 {
            if self.seen_pipe_in_link {
                WikiFine::LinkDisplay
            } else {
                WikiFine::LinkTarget
            }
        } else if self.template_depth > 0 {
            if self.seen_pipe_in_template {
                WikiFine::TemplateArg
            } else {
                WikiFine::TemplateName
            }
        } else {
            WikiFine::Plain
        }
    }

    fn advance(&mut self, byte: u8) {
        let pair_open_link = self.prev == Some(b'[') && byte == b'[';
        let pair_close_link = self.prev == Some(b']') && byte == b']';
        let pair_open_tpl = self.prev == Some(b'{') && byte == b'{';
        let pair_close_tpl = self.prev == Some(b'}') && byte == b'}';

        if pair_open_link {
            self.link_depth = self.link_depth.saturating_add(1);
            if self.link_depth == 1 {
                self.seen_pipe_in_link = false;
            }
            self.prev = None;
        } else if pair_close_link && self.link_depth > 0 {
            self.link_depth -= 1;
            if self.link_depth == 0 {
                self.seen_pipe_in_link = false;
            }
            self.prev = None;
        } else if pair_open_tpl {
            self.template_depth = self.template_depth.saturating_add(1);
            if self.template_depth == 1 {
                self.seen_pipe_in_template = false;
            }
            self.prev = None;
        } else if pair_close_tpl && self.template_depth > 0 {
            self.template_depth -= 1;
            if self.template_depth == 0 {
                self.seen_pipe_in_template = false;
            }
            self.prev = None;
        } else if byte == b'|' {
            if self.link_depth > 0 {
                self.seen_pipe_in_link = true;
            } else if self.template_depth > 0 {
                self.seen_pipe_in_template = true;
            }
            self.prev = None;
        } else if matches!(byte, b'[' | b']' | b'{' | b'}') {
            self.prev = Some(byte);
        } else {
            self.prev = None;
        }
    }
}

#[derive(Clone, Copy)]
struct TokenStats {
    total: u64,
    max_in_page: u32,
    distinct_pages: u32,
    /// `u64::MAX` sentinel = not yet observed.
    last_page: u64,
    cur_page_count: u32,
}

impl TokenStats {
    const fn new() -> Self {
        Self {
            total: 0,
            max_in_page: 0,
            distinct_pages: 0,
            last_page: u64::MAX,
            cur_page_count: 0,
        }
    }

    const fn observe(&mut self, page_id: u64) {
        self.total += 1;
        if self.last_page == page_id {
            self.cur_page_count += 1;
        } else {
            // Page changed (or first observation): flush previous, start new.
            if self.cur_page_count > self.max_in_page {
                self.max_in_page = self.cur_page_count;
            }
            self.last_page = page_id;
            self.cur_page_count = 1;
            self.distinct_pages += 1;
        }
    }

    const fn finalize(&mut self) {
        if self.cur_page_count > self.max_in_page {
            self.max_in_page = self.cur_page_count;
        }
    }
}

/// 6-byte sliding window match pattern for `<page>`.
const PAGE_OPEN: u64 = ((b'<' as u64) << 40)
    | ((b'p' as u64) << 32)
    | ((b'a' as u64) << 24)
    | ((b'g' as u64) << 16)
    | ((b'e' as u64) << 8)
    | (b'>' as u64);
const TAIL_MASK: u64 = (1u64 << 48) - 1;

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]
pub(crate) fn run_recon(corpus: &Path, top_k: usize) -> Result<()> {
    let file =
        File::open(corpus).with_context(|| format!("opening corpus {}", corpus.display()))?;
    let mut reader = BufReader::with_capacity(1 << 20, file);

    let mut xml_class = Classifier::new();
    let mut wiki_class = WikiFineClassifier::new();

    let mut total_bytes = 0u64;
    let mut content_bytes = 0u64;
    let mut tag_bytes = 0u64;
    let mut attr_bytes = 0u64;
    let mut byte_per_submode = [0u64; N_WIKI_FINE];
    let mut word_per_submode = [0u64; N_WIKI_FINE];

    let mut page_id: u64 = 0;
    let mut tail: u64 = 0;

    let mut token_stats: HashMap<Vec<u8>, TokenStats> = HashMap::with_capacity(1 << 22);

    let mut in_word = false;
    let mut word_buf: Vec<u8> = Vec::with_capacity(64);
    let mut word_submode = WikiFine::Plain;

    let mut chunk = vec![0u8; 1 << 20];
    let mut progress_next: u64 = 100 * 1024 * 1024;

    eprintln!("scanning {} ...", corpus.display());
    loop {
        let n = reader.read(&mut chunk)?;
        if n == 0 {
            break;
        }

        for &byte in &chunk[..n] {
            total_bytes += 1;

            tail = ((tail << 8) | u64::from(byte)) & TAIL_MASK;
            if tail == PAGE_OPEN {
                page_id += 1;
            }

            let mode = xml_class.current_mode();
            match mode {
                Mode::TagStructure => tag_bytes += 1,
                Mode::AttrValue => attr_bytes += 1,
                Mode::Content => {
                    content_bytes += 1;
                    let sub = wiki_class.current_sub();
                    byte_per_submode[sub.idx()] += 1;

                    let is_letter = byte.is_ascii_alphabetic();
                    if is_letter {
                        if !in_word {
                            in_word = true;
                            word_buf.clear();
                            word_submode = sub;
                        }
                        word_buf.push(byte.to_ascii_lowercase());
                    } else if in_word {
                        // Emit the just-finished word.
                        let entry = token_stats
                            .entry(std::mem::take(&mut word_buf))
                            .or_insert_with(TokenStats::new);
                        entry.observe(page_id);
                        word_per_submode[word_submode.idx()] += 1;
                        in_word = false;
                    }

                    wiki_class.advance(byte);
                }
            }
            xml_class.advance(byte);

            if total_bytes >= progress_next {
                eprintln!(
                    "  {:>5} MiB processed, page_id={}",
                    total_bytes / (1024 * 1024),
                    page_id,
                );
                progress_next += 100 * 1024 * 1024;
            }
        }
    }

    if in_word {
        let entry = token_stats.entry(word_buf).or_insert_with(TokenStats::new);
        entry.observe(page_id);
        word_per_submode[word_submode.idx()] += 1;
    }
    for stats in token_stats.values_mut() {
        stats.finalize();
    }

    println!();
    println!("Corpus:        {}", corpus.display());
    println!(
        "Total bytes:   {} ({:.1} MiB)",
        total_bytes,
        total_bytes as f64 / (1024.0 * 1024.0),
    );
    println!("Pages:         {page_id}");
    println!("Distinct word types: {}", token_stats.len());
    println!();

    let total_f = total_bytes.max(1) as f64;
    println!("By XML mode:");
    println!(
        "  Content:      {:>12}  ({:>5.2}%)",
        content_bytes,
        100.0 * content_bytes as f64 / total_f,
    );
    println!(
        "  TagStructure: {:>12}  ({:>5.2}%)",
        tag_bytes,
        100.0 * tag_bytes as f64 / total_f,
    );
    println!(
        "  AttrValue:    {:>12}  ({:>5.2}%)",
        attr_bytes,
        100.0 * attr_bytes as f64 / total_f,
    );
    println!();

    println!("Within Content, by wiki sub-mode:");
    let content_f = content_bytes.max(1) as f64;
    let total_words: u64 = word_per_submode.iter().sum();
    let word_f = total_words.max(1) as f64;
    println!(
        "  {:<14}  {:>14}  {:>8}  {:>14}  {:>8}",
        "sub-mode", "bytes", "%cnt", "words", "%wrd",
    );
    for i in 0..N_WIKI_FINE {
        let mode = WikiFine::from_idx(i);
        println!(
            "  {:<14}  {:>14}  {:>7.2}%  {:>14}  {:>7.2}%",
            mode.label(),
            byte_per_submode[i],
            100.0 * byte_per_submode[i] as f64 / content_f,
            word_per_submode[i],
            100.0 * word_per_submode[i] as f64 / word_f,
        );
    }
    println!();

    // Sort tokens by total occurrences (descending).
    let mut tokens: Vec<(&Vec<u8>, &TokenStats)> = token_stats.iter().collect();
    tokens.sort_unstable_by_key(|&(_, s)| std::cmp::Reverse(s.total));

    let show_top = top_k.min(tokens.len());
    println!("Top {show_top} word types by total occurrences:");
    println!(
        "{:>6}  {:<24}  {:>12}  {:>11}  {:>10}  {:>10}  {:>9}",
        "rank", "token", "total", "distinct_pg", "max_in_pg", "avg_in_pg", "max/tot",
    );
    for (i, (token, stats)) in tokens.iter().take(show_top).enumerate() {
        let avg = stats.total as f64 / f64::from(stats.distinct_pages.max(1));
        let conc = f64::from(stats.max_in_page) / stats.total as f64;
        let disp = String::from_utf8_lossy(token);
        let disp: &str = &disp;
        let disp = if disp.len() > 22 { &disp[..22] } else { disp };
        println!(
            "  {:>4}  {:<24}  {:>12}  {:>11}  {:>10}  {:>10.2}  {:>9.4}",
            i + 1,
            disp,
            stats.total,
            stats.distinct_pages,
            stats.max_in_page,
            avg,
            conc,
        );
    }
    println!();

    // Aggregate: histogram of avg_in_page for top-N by frequency.
    let summary_k = 1000.min(tokens.len());
    let mut buckets = [0u64; 11];
    let mut sum_avg = 0.0f64;
    let mut sum_total: u64 = 0;
    for (_, stats) in tokens.iter().take(summary_k) {
        let avg = stats.total as f64 / f64::from(stats.distinct_pages.max(1));
        sum_avg += avg;
        sum_total += stats.total;
        let bucket = (avg.floor() as usize).min(10);
        buckets[bucket] += 1;
    }
    println!("Top-{summary_k} tokens: avg occurrences per page (histogram):");
    let max_bar = 50.0f64;
    let max_count = buckets.iter().copied().max().unwrap_or(1).max(1);
    for (k, &count) in buckets.iter().enumerate() {
        let label = if k == 10 {
            "10+".to_string()
        } else {
            format!("{k}-{}", k + 1)
        };
        let bar_len = (count as f64 / max_count as f64 * max_bar) as usize;
        let bar = "#".repeat(bar_len);
        println!("  {label:>5}: {count:>6}  {bar}");
    }
    println!(
        "  Mean avg_in_page across top-{summary_k}: {:.2}",
        sum_avg / summary_k as f64,
    );
    println!(
        "  Top-{summary_k} contribute {sum_total} word obs / {total_words} total Content words ({:.1}%)",
        100.0 * sum_total as f64 / total_words.max(1) as f64,
    );
    println!();

    // Sub-mode link analysis: count of Content-words that fall in
    // each sub-mode, in absolute and relative terms. Decisive for
    // Proposal B: how big is "link-interior" as a word slice?
    let plain_w = word_per_submode[WikiFine::Plain.idx()];
    let link_target_w = word_per_submode[WikiFine::LinkTarget.idx()];
    let link_display_w = word_per_submode[WikiFine::LinkDisplay.idx()];
    let template_w = word_per_submode[WikiFine::TemplateName.idx()]
        + word_per_submode[WikiFine::TemplateArg.idx()];
    println!("Proposal-B signal: word-token slice by sub-mode");
    println!(
        "  Plain words:        {plain_w} ({:>5.2}%)",
        100.0 * plain_w as f64 / word_f
    );
    println!(
        "  LinkTarget words:   {link_target_w} ({:>5.2}%)",
        100.0 * link_target_w as f64 / word_f
    );
    println!(
        "  LinkDisplay words:  {link_display_w} ({:>5.2}%)",
        100.0 * link_display_w as f64 / word_f
    );
    println!(
        "  Template* words:    {template_w} ({:>5.2}%)",
        100.0 * template_w as f64 / word_f
    );

    Ok(())
}
