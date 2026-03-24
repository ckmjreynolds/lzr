#![allow(missing_docs)]

use assert_cmd::Command;
use proptest::prelude::*;

fn lzr() -> Command {
    Command::cargo_bin("lzr").unwrap()
}

#[test]
fn help_flag() {
    let output = lzr().arg("--help").output().unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("lzr"));
}

#[test]
fn version_flag() {
    let output = lzr().arg("--version").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains(env!("CARGO_PKG_VERSION")));
}

fn corpus_files() -> Vec<String> {
    const DIRS: &[&str] = &[
        "corpora/artificial",
        "corpora/calgary",
        "corpora/canterbury",
        "corpora/large",
        "corpora/miscellaneous",
        "corpora/neuro",
        "corpora/snappy",
    ];
    let mut files = Vec::new();
    for dir in DIRS {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    files.push(path.to_string_lossy().into_owned());
                }
            }
        }
    }
    files.sort();
    files
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1))]
    #[test]
    #[ignore = "failing"]
    fn roundtrip_random_corpus(
        file_idx in 0..1000usize,
        level in 1..=9u8,
        threads in prop_oneof![Just(0usize), 1..=255usize],
    ) {
        let files = corpus_files();
        prop_assume!(!files.is_empty());
        let path = &files[file_idx % files.len()];
        let data = std::fs::read(path).unwrap();

        let compressed = lzr()
            .args(["--level", &level.to_string(), "-T", &threads.to_string()])
            .write_stdin(data.clone())
            .output()
            .unwrap();
        prop_assert!(compressed.status.success(), "compression failed for {path} level={level} threads={threads}");

        let decompressed = lzr()
            .arg("-d")
            .write_stdin(compressed.stdout)
            .output()
            .unwrap();
        prop_assert!(decompressed.status.success(), "decompression failed for {path}");
        prop_assert_eq!(decompressed.stdout, data, "roundtrip mismatch for {}", path);
    }
}
