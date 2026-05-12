//! Reconnaissance for the wiki-template recognizer (Phase 12).
//!
//! Walks a corpus prefix through the existing `Classifier`, counts how
//! many Content bytes live inside wiki sub-structures (internal links,
//! templates, headings), and reports byte coverage plus the top
//! distinct strings in each category. The output is the basis for the
//! Phase-12 bpb-gain prediction: without measured coverage we can't
//! distinguish a 0.05 bpb opportunity from a 0.4 bpb one.
//!
//! Wiki sub-state FSM (per Content byte, in addition to the XML
//! classifier):
//! - `Plain` — Content byte outside any wiki structure.
//! - `Link` — between `[[` and the matching `]]`. `MediaWiki` internal
//!   links nest rarely; we track depth to handle the few cases.
//! - `Template` — between `{{` and matching `}}`. Templates nest
//!   heavily (`{{infobox|param={{convert|5|km}}}}`).
//! - `Heading` — line spanning `==..==` (any level). Treated as one
//!   sub-mode regardless of `==` vs `===` vs `====`.
//!
//! Categorization is greedy and approximate — a `[[` inside a
//! `{{template}}` argument counts as Link, not Template. The goal is
//! prediction, not perfect parsing.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};

use crate::classifier::{Classifier, Mode};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WikiSub {
    Plain,
    Link,
    Template,
    Heading,
}

#[derive(Default)]
struct Counts {
    plain: u64,
    link: u64,
    template: u64,
    heading: u64,
}

/// CLI entry point: read `bytes` bytes from `corpus` and report wiki
/// sub-mode byte coverage plus the top distinct link targets and
/// template names.
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
pub(crate) fn run_analyze_wiki(corpus: &Path, bytes: usize) -> Result<()> {
    let mut f =
        File::open(corpus).with_context(|| format!("opening corpus {}", corpus.display()))?;
    let mut buf = vec![0u8; bytes];
    let read = f.read(&mut buf)?;
    buf.truncate(read);

    let mut classifier = Classifier::new();
    let mut content_bytes: u64 = 0;
    let mut tag_bytes: u64 = 0;
    let mut attr_bytes: u64 = 0;
    let mut counts = Counts::default();

    let mut link_depth: u32 = 0;
    let mut template_depth: u32 = 0;
    let mut heading_open = false;
    let mut at_line_start = true;

    let mut link_strings: HashMap<Vec<u8>, u64> = HashMap::new();
    let mut template_strings: HashMap<Vec<u8>, u64> = HashMap::new();
    let mut current_capture: Option<(WikiSub, Vec<u8>)> = None;

    let mut total_link_spans: u64 = 0;
    let mut total_template_spans: u64 = 0;
    let mut total_heading_spans: u64 = 0;

    let n = buf.len();
    let mut i = 0;
    while i < n {
        let mode = classifier.current_mode();
        let byte = buf[i];

        match mode {
            Mode::TagStructure => tag_bytes += 1,
            Mode::AttrValue => attr_bytes += 1,
            Mode::Content => {
                content_bytes += 1;

                // Detect closers before openers so [[]] doesn't double-count.
                let close_link = byte == b']' && i + 1 < n && buf[i + 1] == b']' && link_depth > 0;
                let close_template =
                    byte == b'}' && i + 1 < n && buf[i + 1] == b'}' && template_depth > 0;
                let close_heading = byte == b'\n' && heading_open;

                let open_link = byte == b'[' && i + 1 < n && buf[i + 1] == b'[';
                let open_template = byte == b'{' && i + 1 < n && buf[i + 1] == b'{';
                let open_heading = at_line_start
                    && byte == b'='
                    && i + 1 < n
                    && buf[i + 1] == b'='
                    && !heading_open;

                let sub = if link_depth > 0 {
                    WikiSub::Link
                } else if template_depth > 0 {
                    WikiSub::Template
                } else if heading_open {
                    WikiSub::Heading
                } else {
                    WikiSub::Plain
                };

                match sub {
                    WikiSub::Plain => counts.plain += 1,
                    WikiSub::Link => counts.link += 1,
                    WikiSub::Template => counts.template += 1,
                    WikiSub::Heading => counts.heading += 1,
                }

                if let Some((cap_sub, ref mut s)) = current_capture
                    && cap_sub == sub
                    && s.len() < 256
                {
                    s.push(byte);
                }

                if close_link {
                    if let Some((WikiSub::Link, s)) = current_capture.take() {
                        *link_strings.entry(s).or_insert(0) += 1;
                    }
                    link_depth -= 1;
                    total_link_spans += 1;
                    // Skip the second ']'.
                    classifier.advance(byte);
                    i += 1;
                    classifier.advance(buf[i]);
                    counts.link += 1;
                    content_bytes += 1;
                    i += 1;
                    at_line_start = false;
                    continue;
                }
                if close_template {
                    if let Some((WikiSub::Template, s)) = current_capture.take() {
                        *template_strings.entry(s).or_insert(0) += 1;
                    }
                    template_depth -= 1;
                    total_template_spans += 1;
                    classifier.advance(byte);
                    i += 1;
                    classifier.advance(buf[i]);
                    counts.template += 1;
                    content_bytes += 1;
                    i += 1;
                    at_line_start = false;
                    continue;
                }
                if close_heading {
                    heading_open = false;
                    total_heading_spans += 1;
                }

                if open_link {
                    if link_depth == 0 && template_depth == 0 && !heading_open {
                        current_capture = Some((WikiSub::Link, Vec::new()));
                    }
                    link_depth += 1;
                    classifier.advance(byte);
                    i += 1;
                    classifier.advance(buf[i]);
                    counts.link += 1;
                    content_bytes += 1;
                    i += 1;
                    at_line_start = false;
                    continue;
                }
                if open_template {
                    if link_depth == 0 && template_depth == 0 && !heading_open {
                        current_capture = Some((WikiSub::Template, Vec::new()));
                    }
                    template_depth += 1;
                    classifier.advance(byte);
                    i += 1;
                    classifier.advance(buf[i]);
                    counts.template += 1;
                    content_bytes += 1;
                    i += 1;
                    at_line_start = false;
                    continue;
                }
                if open_heading {
                    heading_open = true;
                }

                at_line_start = byte == b'\n';
            }
        }

        classifier.advance(byte);
        i += 1;
    }

    let total = content_bytes + tag_bytes + attr_bytes;
    let total_f = total as f64;
    let content_f = content_bytes.max(1) as f64;

    println!("Corpus:  {}", corpus.display());
    println!(
        "Scanned: {total} bytes ({:.1} MiB)",
        total_f / (1024.0 * 1024.0)
    );
    println!();
    println!("By XML mode:");
    println!(
        "  Content      {content_bytes:>12}  ({:>5.2}%)",
        100.0 * content_bytes as f64 / total_f,
    );
    println!(
        "  TagStructure {tag_bytes:>12}  ({:>5.2}%)",
        100.0 * tag_bytes as f64 / total_f,
    );
    println!(
        "  AttrValue    {attr_bytes:>12}  ({:>5.2}%)",
        100.0 * attr_bytes as f64 / total_f,
    );
    println!();
    println!("Within Content, by wiki sub-mode:");
    let report = |label: &str, n: u64| {
        println!(
            "  {label:<10} {n:>12}  ({:>5.2}% of Content, {:>5.2}% of total)",
            100.0 * n as f64 / content_f,
            100.0 * n as f64 / total_f,
        );
    };
    report("Plain", counts.plain);
    report("Link", counts.link);
    report("Template", counts.template);
    report("Heading", counts.heading);
    println!();
    println!("Span counts:");
    println!("  Link spans:     {total_link_spans}");
    println!("  Template spans: {total_template_spans}");
    println!("  Heading spans:  {total_heading_spans}");
    println!();

    print_top("Top 20 link targets (captured prefix)", &link_strings, 20);
    println!();
    print_top(
        "Top 20 template names (captured prefix)",
        &template_strings,
        20,
    );

    Ok(())
}

fn print_top(label: &str, m: &HashMap<Vec<u8>, u64>, k: usize) {
    println!("{label}:");
    let mut v: Vec<(&Vec<u8>, &u64)> = m.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1));
    let distinct = v.len();
    println!("  distinct strings: {distinct}");
    for (s, count) in v.iter().take(k) {
        // Show only the head up to the first '|' (link/template arg
        // separator), and only the first ~40 bytes, to keep output readable.
        let head = s.split(|&b| b == b'|').next().unwrap_or(s);
        let head = &head[..head.len().min(40)];
        let disp = String::from_utf8_lossy(head);
        println!("    {count:>8}  {disp}");
    }
}
