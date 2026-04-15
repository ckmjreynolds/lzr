use std::io::Write as _;

/// Generates numbered lines with pseudo-random padding totaling at least `min_bytes`.
///
/// Uses a simple LCG to produce high-entropy padding that resists LZ77 compression,
/// ensuring the encoded output exceeds the raw input threshold for Sonnet boundaries.
pub(crate) fn numbered_lines(min_bytes: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(min_bytes + 256);
    let mut rng: u64 = 0xDEAD_BEEF_CAFE_BABE;
    let mut line = 0u64;
    while buf.len() < min_bytes {
        // 8 pseudo-random hex u64s per line ~ 160 bytes of high-entropy content.
        write!(buf, "line {line:08}: ").unwrap();
        for _ in 0..8 {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            write!(buf, "{rng:016x} ").unwrap();
        }
        writeln!(buf).unwrap();
        line += 1;
    }
    buf
}
