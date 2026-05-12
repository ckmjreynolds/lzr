//! Move-to-front transform.
//!
//! Maintains a list of all 256 byte values. For each input byte:
//! - find its position in the list, emit the position;
//! - move the byte to the front (position 0).
//!
//! After BWT, similar-context bytes cluster, so adjacent emitted
//! positions tend to be 0 (or small). The arithmetic coder
//! downstream then assigns very few bits to those small values.

/// Forward MTF: input byte stream → position stream.
pub(crate) fn forward(s: &[u8]) -> Vec<u8> {
    let mut list: [u8; 256] = std::array::from_fn(|i| u8::try_from(i).unwrap());
    let mut out = Vec::with_capacity(s.len());
    for &b in s {
        let pos = list
            .iter()
            .position(|&x| x == b)
            .expect("byte must be in the 256-entry list");
        out.push(u8::try_from(pos).expect("pos < 256 fits u8"));
        // Slide list[0..pos] right by one, place `b` at front.
        if pos > 0 {
            list.copy_within(0..pos, 1);
            list[0] = b;
        }
    }
    out
}

/// Inverse MTF: position stream → byte stream.
pub(crate) fn inverse(positions: &[u8]) -> Vec<u8> {
    let mut list: [u8; 256] = std::array::from_fn(|i| u8::try_from(i).unwrap());
    let mut out = Vec::with_capacity(positions.len());
    for &p in positions {
        let p = p as usize;
        let b = list[p];
        out.push(b);
        if p > 0 {
            list.copy_within(0..p, 1);
            list[0] = b;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(s: &[u8]) {
        let mtf = forward(s);
        let back = inverse(&mtf);
        assert_eq!(back, s);
    }

    #[test]
    fn mtf_roundtrip_empty() {
        roundtrip(b"");
    }

    #[test]
    fn mtf_roundtrip_single_byte() {
        roundtrip(b"a");
    }

    #[test]
    fn mtf_roundtrip_text() {
        roundtrip(b"the quick brown fox jumps over the lazy dog");
    }

    #[test]
    fn mtf_clustered_input_produces_zero_runs() {
        // "aaaab" → first 'a' at position 97, subsequent 'a's at 0,
        // then 'b' at position 98 (since 'a' moved to front, all
        // bytes >= 'a' shifted by one). The zero-run is the win.
        let positions = forward(b"aaaab");
        assert_eq!(positions[0], 97);
        assert_eq!(positions[1], 0);
        assert_eq!(positions[2], 0);
        assert_eq!(positions[3], 0);
        assert_eq!(positions[4], 98); // 'b' was at index 98 after 'a' moved to front
    }

    #[test]
    fn mtf_roundtrip_random_ish() {
        let s: Vec<u8> = (0..5000)
            .map(|i| u8::try_from((i * 37 + 13) % 256).unwrap())
            .collect();
        roundtrip(&s);
    }
}
