//! Multi-offset eval panel and per-component decomposition.
//!
//! Reads fixed offsets from a corpus, runs each through a codec's
//! encode → decode roundtrip, and reports per-offset bpb plus mean.
//! Optionally dumps the per-component decomposition to CSV so
//! every change can be attributed to specific predictors.
//!
//! Two panel sizes:
//! - **full** (default): 20 windows × 256 KiB. Detects ~0.001 bpb
//!   changes reliably.
//! - **quick** (`--quick`): 5 windows × 64 KiB. For tight iteration
//!   loops; noise floor ~0.005 bpb.
//!
//! Pre-warm is fixed at 4 MiB before each window, matching the LZR v1
//! convergence point for PPM/LZ adaptive state.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::classifier_stats::ClassifierStats;
use crate::codec::{Codec, Decomposition};
use crate::null::NullCodec;
use crate::xml_codec::XmlCodec;
use crate::xml_lz_cp::XmlLzCpCodec;
use crate::xml_lz_ord3::XmlLzOrd3Codec;
use crate::xml_lz_ppm::XmlLzPpmCodec;
use crate::xml_lz_ppmc::XmlLzPpmcCodec;
use crate::xml_lz_word::XmlLzWordCodec;
use crate::xml_ppm::XmlPpmCodec;

const SAMPLE_BYTES_FULL: usize = 256 * 1024;
const SAMPLE_BYTES_QUICK: usize = 64 * 1024;
const PREWARM_BYTES: usize = 4 * 1024 * 1024;
const N_OFFSETS_FULL: usize = 20;
const N_OFFSETS_QUICK: usize = 5;

#[derive(Clone, Debug)]
pub(crate) struct WindowResult {
    pub offset: u64,
    pub measured_bytes: usize,
    pub archive_bytes: usize,
    pub bpb: f64,
    pub decomposition: Decomposition,
}

#[derive(Clone, Debug)]
pub(crate) struct BenchReport {
    pub codec_name: String,
    pub corpus: String,
    pub per_window: Vec<WindowResult>,
    pub mean_bpb: f64,
}

/// CLI entry point: parse codec name, run panel, optionally dump CSV.
pub(crate) fn run_bench(
    corpus: &Path,
    codec_name: &str,
    quick: bool,
    decompose_out: Option<&Path>,
) -> Result<()> {
    let codec: Box<dyn Codec> = make_codec(codec_name)?;
    let report = bench(corpus, codec.as_ref(), quick)?;
    print_report(&report);
    if let Some(p) = decompose_out {
        dump_decomposition_csv(&report, p)?;
        eprintln!("decomposition written to {}", p.display());
    }
    Ok(())
}

fn make_codec(name: &str) -> Result<Box<dyn Codec>> {
    match name {
        "null" => Ok(Box::new(NullCodec)),
        "classifier-stats" => Ok(Box::new(ClassifierStats)),
        "xml" => Ok(Box::new(XmlCodec)),
        "xml-ppm" => Ok(Box::new(XmlPpmCodec)),
        "xml-lz-ppm" => Ok(Box::new(XmlLzPpmCodec)),
        "xml-lz-word" => Ok(Box::new(XmlLzWordCodec)),
        "xml-lz-ord3" => Ok(Box::new(XmlLzOrd3Codec)),
        "xml-lz-cp" => Ok(Box::new(XmlLzCpCodec)),
        "xml-lz-ppmc" => Ok(Box::new(XmlLzPpmcCodec)),
        other => bail!(
            "unknown codec '{other}' (known: null, classifier-stats, xml, xml-ppm, xml-lz-ppm, xml-lz-word, xml-lz-ord3, xml-lz-cp, xml-lz-ppmc)"
        ),
    }
}

/// Run the panel against `codec`. The panel positions windows evenly
/// across the corpus with `PREWARM_BYTES` headroom on the leading edge.
pub(crate) fn bench(corpus: &Path, codec: &dyn Codec, quick: bool) -> Result<BenchReport> {
    let mut f =
        File::open(corpus).with_context(|| format!("opening corpus {}", corpus.display()))?;
    let corpus_len = f.metadata()?.len();
    let sample_bytes = if quick {
        SAMPLE_BYTES_QUICK
    } else {
        SAMPLE_BYTES_FULL
    };
    let offsets = pick_offsets(corpus_len, sample_bytes, quick);
    if offsets.is_empty() {
        bail!(
            "corpus {} too small for the chosen panel (need ≥ {} bytes)",
            corpus.display(),
            PREWARM_BYTES + sample_bytes,
        );
    }

    let mut per_window = Vec::with_capacity(offsets.len());

    for off in offsets {
        let warm_start = off.saturating_sub(PREWARM_BYTES as u64);
        let warm_len = usize::try_from(off - warm_start)
            .expect("PREWARM_BYTES bounds the difference to usize range");
        let warm = if warm_len > 0 {
            f.seek(SeekFrom::Start(warm_start))?;
            let mut buf = vec![0u8; warm_len];
            f.read_exact(&mut buf)?;
            buf
        } else {
            Vec::new()
        };

        f.seek(SeekFrom::Start(off))?;
        let mut measure = vec![0u8; sample_bytes];
        f.read_exact(&mut measure)?;

        let (archive, decomp) = codec
            .encode_window(&warm, &measure)
            .with_context(|| format!("encoding window at offset {off}"))?;
        let decoded = codec
            .decode_window(&warm, &archive)
            .with_context(|| format!("decoding window at offset {off}"))?;
        if decoded != measure {
            bail!("roundtrip mismatch at offset {off}");
        }

        let archive_bytes = archive.len();
        // Sanity invariant: each codec must attribute every emitted bit
        // to a named component, so the sum of the decomposition equals
        // the framed archive's bit count exactly. Catches "lost" bits
        // in component accounting before they distort the audit table.
        let total_attributed = decomp.total();
        let archive_bits = 8 * u64::try_from(archive_bytes).expect("archive fits u64");
        if total_attributed != archive_bits {
            bail!(
                "decomposition mismatch at offset {off}: components attribute {total_attributed} bits, archive has {archive_bits} bits",
            );
        }

        let archive_bytes_f =
            f64::from(u32::try_from(archive_bytes).expect("archive bytes per window fits u32"));
        let sample_bytes_f = f64::from(u32::try_from(sample_bytes).expect("sample bytes fits u32"));
        let bpb = 8.0 * archive_bytes_f / sample_bytes_f;
        per_window.push(WindowResult {
            offset: off,
            measured_bytes: sample_bytes,
            archive_bytes,
            bpb,
            decomposition: decomp,
        });
    }

    let n_windows = f64::from(u32::try_from(per_window.len()).expect("≤ N_OFFSETS_FULL windows"));
    let mean_bpb = per_window.iter().map(|r| r.bpb).sum::<f64>() / n_windows;

    Ok(BenchReport {
        codec_name: codec.name().to_string(),
        corpus: corpus.display().to_string(),
        per_window,
        mean_bpb,
    })
}

/// Evenly-spaced offsets strictly inside the corpus, leaving
/// `PREWARM_BYTES` headroom at the leading edge and `sample_bytes` at
/// the trailing edge so all reads stay in-bounds.
fn pick_offsets(corpus_len: u64, sample_bytes: usize, quick: bool) -> Vec<u64> {
    let n = if quick {
        N_OFFSETS_QUICK
    } else {
        N_OFFSETS_FULL
    };
    let prewarm = PREWARM_BYTES as u64;
    let sample = sample_bytes as u64;
    if corpus_len < prewarm + sample {
        return Vec::new();
    }
    let min_off = prewarm;
    let max_off = corpus_len - sample;
    if max_off < min_off {
        return Vec::new();
    }
    let span = max_off - min_off;
    let n_u64 = n as u64;
    (0..n)
        .map(|i| {
            if n == 1 {
                min_off + span / 2
            } else {
                min_off + (span * i as u64) / (n_u64 - 1)
            }
        })
        .collect()
}

fn print_report(r: &BenchReport) {
    println!("Codec:  {}", r.codec_name);
    println!("Corpus: {}", r.corpus);
    println!();
    println!("  offset (B)       bpb     archive (B)");
    println!("  ----------       -----   -----------");
    for w in &r.per_window {
        println!(
            "  {:>10}     {:.3}   {:>11}",
            w.offset, w.bpb, w.archive_bytes,
        );
    }
    println!("  ----------       -----   -----------");
    println!("  mean             {:.3}", r.mean_bpb);
    println!();
}

fn dump_decomposition_csv(r: &BenchReport, path: &Path) -> Result<()> {
    let mut all_components: BTreeSet<String> = BTreeSet::new();
    for w in &r.per_window {
        for k in w.decomposition.by_component.keys() {
            all_components.insert(k.clone());
        }
    }

    let mut f = File::create(path)
        .with_context(|| format!("creating decomposition CSV {}", path.display()))?;
    write!(f, "offset,measured_bytes,archive_bytes,bpb")?;
    for c in &all_components {
        write!(f, ",bits_{c}")?;
    }
    writeln!(f)?;
    for w in &r.per_window {
        write!(
            f,
            "{},{},{},{:.4}",
            w.offset, w.measured_bytes, w.archive_bytes, w.bpb,
        )?;
        for c in &all_components {
            let bits = w.decomposition.by_component.get(c).copied().unwrap_or(0);
            write!(f, ",{bits}")?;
        }
        writeln!(f)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_offsets_full_spans_corpus() {
        let len = 1_000_000_000;
        let off = pick_offsets(len, SAMPLE_BYTES_FULL, false);
        assert_eq!(off.len(), N_OFFSETS_FULL);
        assert_eq!(off[0], PREWARM_BYTES as u64);
        assert_eq!(off[N_OFFSETS_FULL - 1], len - SAMPLE_BYTES_FULL as u64);
        for w in off.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    #[test]
    fn pick_offsets_quick_picks_five() {
        let len = 100_000_000;
        let off = pick_offsets(len, SAMPLE_BYTES_QUICK, true);
        assert_eq!(off.len(), N_OFFSETS_QUICK);
        assert!(off[0] >= PREWARM_BYTES as u64);
        assert!(off[N_OFFSETS_QUICK - 1] + SAMPLE_BYTES_QUICK as u64 <= len);
    }

    #[test]
    fn pick_offsets_rejects_tiny_corpus() {
        let off = pick_offsets(100, SAMPLE_BYTES_FULL, false);
        assert!(off.is_empty());
    }
}
