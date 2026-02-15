#![allow(missing_docs)]

use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::process::Command;

use assert_cmd::cargo_bin;
use clap as _;
use pretty_assertions::assert_eq;

fn shasum(cmd: &str) -> String {
    let lzr = cargo_bin!("lzr").display().to_string();
    let shell_cmd = cmd.replace("lzr", &lzr);
    let output = Command::new("bash").args(["-c", &shell_cmd]).output().unwrap();
    assert!(output.status.success(), "failed: {shell_cmd}");
    String::from_utf8(output.stdout).unwrap()
}

fn test_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn lzr_cmd() -> Command {
    Command::new(cargo_bin!("lzr"))
}

// --- Pipe-based round-trip tests ---

#[test]
fn round_trip_a_txt() {
    let original = shasum("cat corpora/artificial/a.txt | shasum -a 256");
    let round_trip = shasum("cat corpora/artificial/a.txt | lzr | lzr -d | shasum -a 256");
    assert_eq!(original, round_trip);
}

#[test]
fn round_trip_aaa_txt() {
    let original = shasum("cat corpora/artificial/aaa.txt | shasum -a 256");
    let round_trip = shasum("cat corpora/artificial/aaa.txt | lzr | lzr -d | shasum -a 256");
    assert_eq!(original, round_trip);
}

#[test]
fn round_trip_alphabet_txt() {
    let original = shasum("cat corpora/artificial/alphabet.txt | shasum -a 256");
    let round_trip = shasum("cat corpora/artificial/alphabet.txt | lzr | lzr -d | shasum -a 256");
    assert_eq!(original, round_trip);
}

#[test]
fn round_trip_random_txt() {
    let original = shasum("cat corpora/artificial/random.txt | shasum -a 256");
    let round_trip = shasum("cat corpora/artificial/random.txt | lzr | lzr -d | shasum -a 256");
    assert_eq!(original, round_trip);
}

#[test]
fn round_trip_alice29_txt() {
    let original = shasum("cat corpora/canterbury/alice29.txt | shasum -a 256");
    let round_trip = shasum("cat corpora/canterbury/alice29.txt | lzr | lzr -d | shasum -a 256");
    assert_eq!(original, round_trip);
}

#[test]
fn round_trip_empty() {
    let original = shasum("echo -n '' | shasum -a 256");
    let round_trip = shasum("echo -n '' | lzr | lzr -d | shasum -a 256");
    assert_eq!(original, round_trip);
}

// --- Library struct round-trip ---

#[test]
fn round_trip_via_structs() {
    let data = b"round trip test data";
    let mut compressed = Vec::new();
    lzr::Encoder::new(&data[..]).read_to_end(&mut compressed).unwrap();

    let mut decompressed = Vec::new();
    lzr::Decoder::new(&compressed[..]).read_to_end(&mut decompressed).unwrap();

    assert_eq!(decompressed, data);
}

// --- CLI flag tests ---

#[test]
fn version_flag() {
    let output = lzr_cmd().arg("--version").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("lzr"), "expected 'lzr' in version output");
    assert!(stdout.contains("Copyright"), "expected 'Copyright' in version output");
}

#[test]
fn short_version_flag() {
    let output = lzr_cmd().arg("-V").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("lzr"), "expected 'lzr' in version output");
}

#[test]
fn help_flag() {
    let output = lzr_cmd().arg("--help").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("decompress"), "expected 'decompress' in help output");
    assert!(stdout.contains("--keep"), "expected '--keep' in help output");
    assert!(stdout.contains("--force"), "expected '--force' in help output");
    assert!(stdout.contains("--test"), "expected '--test' in help output");
    assert!(stdout.contains("--verbose"), "expected '--verbose' in help output");
    assert!(stdout.contains("--fast"), "expected '--fast' in help output");
    assert!(stdout.contains("--best"), "expected '--best' in help output");
    assert!(stdout.contains("--suffix"), "expected '--suffix' in help output");
}

#[test]
fn decompress_long_flag() {
    let original = shasum("cat corpora/artificial/a.txt | shasum -a 256");
    let round_trip = shasum("cat corpora/artificial/a.txt | lzr | lzr --decompress | shasum -a 256");
    assert_eq!(original, round_trip);
}

#[test]
fn uncompress_alias() {
    let original = shasum("cat corpora/artificial/a.txt | shasum -a 256");
    let round_trip = shasum("cat corpora/artificial/a.txt | lzr | lzr --uncompress | shasum -a 256");
    assert_eq!(original, round_trip);
}

#[test]
fn fast_flag() {
    let original = shasum("cat corpora/artificial/a.txt | shasum -a 256");
    let round_trip = shasum("cat corpora/artificial/a.txt | lzr -1 | lzr -d | shasum -a 256");
    assert_eq!(original, round_trip);

    let round_trip2 = shasum("cat corpora/artificial/a.txt | lzr --fast | lzr -d | shasum -a 256");
    assert_eq!(original, round_trip2);
}

#[test]
fn best_flag() {
    let original = shasum("cat corpora/artificial/a.txt | shasum -a 256");
    let round_trip = shasum("cat corpora/artificial/a.txt | lzr -9 | lzr -d | shasum -a 256");
    assert_eq!(original, round_trip);

    let round_trip2 = shasum("cat corpora/artificial/a.txt | lzr --best | lzr -d | shasum -a 256");
    assert_eq!(original, round_trip2);
}

// --- File operation tests ---

#[test]
fn file_compress_decompress() {
    let dir = test_dir("file_compress_decompress");
    let input = dir.join("hello.txt");
    fs::write(&input, b"hello world").unwrap();

    // Compress
    let output = lzr_cmd().arg(&input).output().unwrap();
    assert!(output.status.success(), "compress failed: {}", String::from_utf8_lossy(&output.stderr));

    let compressed = dir.join("hello.txt.lzr");
    assert!(compressed.exists(), "compressed file should exist");
    assert!(!input.exists(), "original should be removed");

    // Decompress
    let output = lzr_cmd().arg("-d").arg(&compressed).output().unwrap();
    assert!(output.status.success(), "decompress failed: {}", String::from_utf8_lossy(&output.stderr));

    let restored = dir.join("hello.txt");
    assert!(restored.exists(), "restored file should exist");
    assert!(!compressed.exists(), "compressed file should be removed");
    assert_eq!(fs::read(&restored).unwrap(), b"hello world");
}

#[test]
fn file_compress_keep() {
    let dir = test_dir("file_compress_keep");
    let input = dir.join("data.txt");
    fs::write(&input, b"keep me").unwrap();

    let output = lzr_cmd().arg("-k").arg(&input).output().unwrap();
    assert!(output.status.success());

    assert!(input.exists(), "original should be kept with -k");
    assert!(dir.join("data.txt.lzr").exists(), "compressed file should exist");
}

#[test]
fn file_decompress_keep() {
    let dir = test_dir("file_decompress_keep");
    let input = dir.join("data.txt");
    fs::write(&input, b"keep compressed").unwrap();

    // First compress
    let output = lzr_cmd().arg("-k").arg(&input).output().unwrap();
    assert!(output.status.success());

    let compressed = dir.join("data.txt.lzr");
    // Remove original to test -dk
    fs::remove_file(&input).unwrap();

    let output = lzr_cmd().arg("-dk").arg(&compressed).output().unwrap();
    assert!(output.status.success());

    assert!(compressed.exists(), "compressed should be kept with -dk");
    assert!(input.exists(), "decompressed file should exist");
}

#[test]
fn stdout_flag() {
    let dir = test_dir("stdout_flag");
    let input = dir.join("data.txt");
    fs::write(&input, b"to stdout").unwrap();

    let output = lzr_cmd().arg("-c").arg(&input).output().unwrap();
    assert!(output.status.success());
    assert!(input.exists(), "original should be kept with -c");
    assert_eq!(output.stdout, b"to stdout", "stdout should contain compressed data");
}

#[test]
fn stdout_decompress() {
    let dir = test_dir("stdout_decompress");
    let input = dir.join("data.txt");
    fs::write(&input, b"cd test").unwrap();

    // Compress keeping original
    let output = lzr_cmd().arg("-k").arg(&input).output().unwrap();
    assert!(output.status.success());

    let compressed = dir.join("data.txt.lzr");
    let output = lzr_cmd().arg("-cd").arg(&compressed).output().unwrap();
    assert!(output.status.success());
    assert!(compressed.exists(), "compressed should be kept with -cd");
    assert_eq!(output.stdout, b"cd test");
}

#[test]
fn force_overwrite() {
    let dir = test_dir("force_overwrite");
    let input = dir.join("data.txt");
    let output_file = dir.join("data.txt.lzr");
    fs::write(&input, b"force test").unwrap();
    fs::write(&output_file, b"old data").unwrap();

    let output = lzr_cmd().arg("-fk").arg(&input).output().unwrap();
    assert!(output.status.success(), "force overwrite failed: {}", String::from_utf8_lossy(&output.stderr));
    assert!(output_file.exists());
}

#[test]
fn output_exists_error() {
    let dir = test_dir("output_exists_error");
    let input = dir.join("data.txt");
    let output_file = dir.join("data.txt.lzr");
    fs::write(&input, b"exists test").unwrap();
    fs::write(&output_file, b"already here").unwrap();

    let output = lzr_cmd().arg("-k").arg(&input).output().unwrap();
    assert!(!output.status.success(), "should fail when output exists");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("already exists"), "expected 'already exists' in stderr: {stderr}");
}

#[test]
fn test_flag() {
    let dir = test_dir("test_flag");
    let input = dir.join("data.txt");
    fs::write(&input, b"test integrity").unwrap();

    // Compress keeping original
    lzr_cmd().arg("-k").arg(&input).output().unwrap();
    let compressed = dir.join("data.txt.lzr");

    let output = lzr_cmd().arg("-t").arg(&compressed).output().unwrap();
    assert!(output.status.success(), "test flag failed: {}", String::from_utf8_lossy(&output.stderr));
    assert!(compressed.exists(), "compressed should be kept with -t");
}

#[test]
fn test_flag_verbose() {
    let dir = test_dir("test_flag_verbose");
    let input = dir.join("data.txt");
    fs::write(&input, b"verbose test").unwrap();

    lzr_cmd().arg("-k").arg(&input).output().unwrap();
    let compressed = dir.join("data.txt.lzr");

    let output = lzr_cmd().arg("-tv").arg(&compressed).output().unwrap();
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("ok"), "expected 'ok' in stderr: {stderr}");
}

#[test]
fn list_stub() {
    let output = lzr_cmd().arg("-l").output().unwrap();
    assert!(!output.status.success(), "list should exit non-zero");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("not yet implemented"), "expected 'not yet implemented' in stderr: {stderr}");
}

#[test]
fn custom_suffix() {
    let dir = test_dir("custom_suffix");
    let input = dir.join("data.txt");
    fs::write(&input, b"custom suffix").unwrap();

    let output = lzr_cmd().args(["-k", "-S", ".z"]).arg(&input).output().unwrap();
    assert!(output.status.success(), "custom suffix failed: {}", String::from_utf8_lossy(&output.stderr));
    assert!(dir.join("data.txt.z").exists(), "file with custom suffix should exist");
}

#[test]
fn unknown_suffix_skip() {
    let dir = test_dir("unknown_suffix_skip");
    let input = dir.join("data.txt");
    fs::write(&input, b"no suffix").unwrap();

    let output = lzr_cmd().arg("-d").arg(&input).output().unwrap();
    assert!(output.status.success(), "should succeed (skip file)");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unknown suffix"), "expected 'unknown suffix' warning: {stderr}");
    assert!(input.exists(), "original should still exist");
}

#[test]
fn unknown_suffix_with_stdout() {
    let dir = test_dir("unknown_suffix_stdout");
    let input = dir.join("data.txt");
    fs::write(&input, b"suffix irrelevant").unwrap();

    let output = lzr_cmd().arg("-dc").arg(&input).output().unwrap();
    assert!(output.status.success());
    assert_eq!(output.stdout, b"suffix irrelevant");
}

#[test]
fn multiple_files() {
    let dir = test_dir("multiple_files");
    let a = dir.join("a.txt");
    let b = dir.join("b.txt");
    fs::write(&a, b"aaa").unwrap();
    fs::write(&b, b"bbb").unwrap();

    let output = lzr_cmd().arg(&a).arg(&b).output().unwrap();
    assert!(output.status.success());
    assert!(dir.join("a.txt.lzr").exists());
    assert!(dir.join("b.txt.lzr").exists());
    assert!(!a.exists(), "originals should be removed");
    assert!(!b.exists());
}

#[test]
fn multiple_files_partial_failure() {
    let dir = test_dir("multiple_partial_failure");
    let a = dir.join("a.txt");
    let b = dir.join("nonexistent.txt");
    fs::write(&a, b"aaa").unwrap();

    let output = lzr_cmd().arg(&a).arg(&b).output().unwrap();
    assert!(!output.status.success(), "should fail when a file is missing");
    // First file should still have been processed.
    assert!(dir.join("a.txt.lzr").exists(), "first file should still be compressed");
}

#[test]
fn verbose_output() {
    let dir = test_dir("verbose_output");
    let input = dir.join("data.txt");
    fs::write(&input, b"verbose data here").unwrap();

    let output = lzr_cmd().arg("-vk").arg(&input).output().unwrap();
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("bytes"), "expected byte stats in stderr: {stderr}");
}

#[test]
fn quiet_suppresses() {
    let dir = test_dir("quiet_suppresses");
    let input = dir.join("data.txt");
    fs::write(&input, b"quiet mode").unwrap();

    let output = lzr_cmd().arg("-dq").arg(&input).output().unwrap();
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("unknown suffix"), "quiet should suppress warnings");
}

#[test]
fn help_shows_flags() {
    let output = lzr_cmd().arg("--help").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    for flag in [
        "--decompress",
        "--stdout",
        "--keep",
        "--force",
        "--test",
        "--verbose",
        "--quiet",
        "--fast",
        "--best",
        "--suffix",
    ] {
        assert!(stdout.contains(flag), "expected '{flag}' in help output");
    }
}
