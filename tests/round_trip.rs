#![allow(unused_crate_dependencies, missing_docs)]

use assert_cmd::cargo_bin;
use clap as _;
use std::process::Command;

fn shasum(cmd: &str) -> String {
    let lzr = cargo_bin!("lzr").display().to_string();
    let shell_cmd = cmd.replace("lzr", &lzr);
    let output = Command::new("bash").args(["-c", &shell_cmd]).output().unwrap();
    assert!(output.status.success(), "failed: {shell_cmd}");
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn round_trip_a_txt() {
    let original = shasum("cat corpora/artificial/a.txt | shasum -a 256");
    let round_trip = shasum("cat corpora/artificial/a.txt | lzr | lzr -d | shasum -a 256");
    assert_eq!(original, round_trip);
}
