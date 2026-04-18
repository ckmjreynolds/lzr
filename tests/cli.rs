//! Integration tests for the `lzr` CLI binary.

use assert_cmd::Command;
use pretty_assertions::assert_eq;

#[test]
fn stdin_roundtrip_via_compress_then_decompress() {
    let tmp = tempdir();
    let archive = tmp.join("roundtrip.lzr");
    let input = b"hello from the lzr CLI\n".repeat(1000);

    // Stage 1: stdin -> compress -> file.
    Command::cargo_bin("lzr")
        .unwrap()
        .args(["compress", "-"])
        .arg(&archive)
        .write_stdin(input.clone())
        .assert()
        .success();

    // Stage 2: file -> decompress -> stdout.
    let decompressed = Command::cargo_bin("lzr")
        .unwrap()
        .arg("decompress")
        .arg(&archive)
        .arg("-")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    assert_eq!(decompressed, input);
}

#[test]
fn info_reports_sealed_state_and_counts() {
    let tmp = tempdir();
    let archive = tmp.join("info.lzr");
    let input = b"line one\nline two\nline three\n";

    Command::cargo_bin("lzr")
        .unwrap()
        .args(["compress", "-"])
        .arg(&archive)
        .write_stdin(input.as_slice())
        .assert()
        .success();

    let stdout = String::from_utf8(
        Command::cargo_bin("lzr").unwrap().arg("info").arg(&archive).assert().success().get_output().stdout.clone(),
    )
    .unwrap();

    assert!(stdout.contains("state:    sealed"), "info output: {stdout}");
    assert!(stdout.contains(&format!("bytes:    {}", input.len())), "info output: {stdout}");
    assert!(stdout.contains("lines:    3"), "info output: {stdout}");
}

#[test]
fn decompress_rejects_stdin() {
    Command::cargo_bin("lzr").unwrap().args(["decompress", "-", "-"]).assert().failure();
}

fn tempdir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("lzr-cli-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
