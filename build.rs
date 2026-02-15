//! Build script — sets `COPYRIGHT_YEARS` env var for compile-time use.

use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    let secs = SystemTime::now().duration_since(UNIX_EPOCH).expect("system clock before Unix epoch").as_secs();

    // Walk years from 1970 with proper leap-year rules.
    let mut year: u64 = 1970;
    let mut remaining = secs / 86400;
    loop {
        let days = if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
            366
        } else {
            365
        };
        if remaining < days {
            break;
        }
        remaining -= days;
        year += 1;
    }

    let copyright_years = if year > 2025 {
        format!("2025-{year}")
    } else {
        "2025".to_string()
    };
    println!("cargo::rustc-env=COPYRIGHT_YEARS={copyright_years}");
}
