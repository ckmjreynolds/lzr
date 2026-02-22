const EOS: u8 = 0x00;
const LITERAL: u8 = 0x00;
const SHORT: u8 = 0x01;
const SHORT_REV: u8 = 0x81;
const MEDIUM: u8 = 0x02;
const MEDIUM_REV: u8 = 0x82;
const EXTENDED: u8 = 0x03;
const EXTENDED_REV: u32 = 0x803;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Frame {
    Eos,
    Literal(isize),
    Match(isize, usize),
}

impl Frame {
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_lossless)]
    #[allow(clippy::cast_sign_loss)]
    pub(crate) fn encode(self, buf: &mut Vec<u8>) -> usize {
        match self {
            Self::Eos => {
                buf.push(EOS);
                1
            }
            Self::Literal(l) => {
                debug_assert!((1..=63).contains(&l));
                buf.push(LITERAL | ((l << 2) as u8));
                1
            }
            Self::Match(l, d) => {
                let s = l.is_negative();
                let m = l.unsigned_abs();

                match (s, m, d) {
                    (false, (2..=33), (1..=256)) => {
                        buf.push(SHORT | (m as u8 - 2) << 2);
                        buf.push((d - 1) as u8);
                        2
                    }
                    (true, (2..=33), (1..=256)) => {
                        buf.push(SHORT_REV | (m as u8 - 2) << 2);
                        buf.push((d - 1) as u8);
                        2
                    }
                    (false, (3..=34), (1..=65_536)) => {
                        buf.push(MEDIUM | (m as u8 - 3) << 2);
                        buf.extend_from_slice(&((d - 1) as u16).to_le_bytes());
                        3
                    }
                    (true, (3..=34), (1..=65_536)) => {
                        buf.push(MEDIUM_REV | (m as u8 - 3) << 2);
                        buf.extend_from_slice(&((d - 1) as u16).to_le_bytes());
                        3
                    }
                    (false, (4..=515), (1..=1_048_576)) => {
                        let raw: u32 = EXTENDED as u32 | ((d as u32 - 1) << 12) | ((m as u32 - 4) << 2);
                        buf.extend_from_slice(&raw.to_le_bytes());
                        4
                    }
                    (true, (4..=515), (1..=1_048_576)) => {
                        let raw: u32 = EXTENDED_REV | ((d as u32 - 1) << 12) | ((m as u32 - 4) << 2);
                        buf.extend_from_slice(&raw.to_le_bytes());
                        4
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    pub(crate) fn decode(input: &[u8]) -> (Self, usize) {
        todo!()
    }
}
/*
#[cfg(test)]
mod tests {
    use super::*;clear
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    // ── Specific value tests ──────────────────────────────────────────

    #[test]
    fn decode_eos() {
        assert_eq!(decode(&[0x00]), (Frame::Eos, 1));
    }

    #[test]
    fn decode_literal() {
        assert_eq!(decode(&[0x01]), (Frame::Literal(1), 1));
        assert_eq!(decode(&[0x3F]), (Frame::Literal(63), 1));
    }

    #[test]
    fn encode_decode_short_forward() {
        // Min: length=+2, distance=1
        let mut buf = Vec::new();
        let n = encode(2, 1, &mut buf);
        assert_eq!(n, 2);
        assert_eq!(decode(&buf), (Frame::Match(2, 1), 2));

        // Max: length=+33, distance=256
        buf.clear();
        let n = encode(33, 256, &mut buf);
        assert_eq!(n, 2);
        assert_eq!(decode(&buf), (Frame::Match(33, 256), 2));
    }

    #[test]
    fn encode_decode_short_reverse() {
        let mut buf = Vec::new();
        let n = encode(-2, 1, &mut buf);
        assert_eq!(n, 2);
        assert_eq!(decode(&buf), (Frame::Match(-2, 1), 2));

        buf.clear();
        let n = encode(-33, 256, &mut buf);
        assert_eq!(n, 2);
        assert_eq!(decode(&buf), (Frame::Match(-33, 256), 2));
    }

    #[test]
    fn encode_decode_medium_forward() {
        // distance=257 forces medium (exceeds short range)
        let mut buf = Vec::new();
        let n = encode(3, 257, &mut buf);
        assert_eq!(n, 3);
        assert_eq!(decode(&buf), (Frame::Match(3, 257), 3));

        // Max: length=+34, distance=65536
        buf.clear();
        let n = encode(34, 65_536, &mut buf);
        assert_eq!(n, 3);
        assert_eq!(decode(&buf), (Frame::Match(34, 65_536), 3));
    }

    #[test]
    fn encode_decode_medium_reverse() {
        let mut buf = Vec::new();
        let n = encode(-3, 257, &mut buf);
        assert_eq!(n, 3);
        assert_eq!(decode(&buf), (Frame::Match(-3, 257), 3));

        buf.clear();
        let n = encode(-34, 65_536, &mut buf);
        assert_eq!(n, 3);
        assert_eq!(decode(&buf), (Frame::Match(-34, 65_536), 3));
    }

    #[test]
    fn encode_decode_extended_forward() {
        // distance=65537 forces extended (exceeds medium range)
        let mut buf = Vec::new();
        let n = encode(4, 65_537, &mut buf);
        assert_eq!(n, 4);
        assert_eq!(decode(&buf), (Frame::Match(4, 65_537), 4));

        // Max: length=+515, distance=1048576
        buf.clear();
        let n = encode(515, 1_048_576, &mut buf);
        assert_eq!(n, 4);
        assert_eq!(decode(&buf), (Frame::Match(515, 1_048_576), 4));
    }

    #[test]
    fn encode_decode_extended_reverse() {
        let mut buf = Vec::new();
        let n = encode(-4, 65_537, &mut buf);
        assert_eq!(n, 4);
        assert_eq!(decode(&buf), (Frame::Match(-4, 65_537), 4));

        buf.clear();
        let n = encode(-515, 1_048_576, &mut buf);
        assert_eq!(n, 4);
        assert_eq!(decode(&buf), (Frame::Match(-515, 1_048_576), 4));
    }

    #[test]
    fn encode_picks_smallest_frame() {
        let mut buf = Vec::new();

        // Should pick short (2 bytes)
        let n = encode(2, 1, &mut buf);
        assert_eq!(n, 2);

        // Should pick medium (3 bytes) — distance forces it
        buf.clear();
        let n = encode(3, 257, &mut buf);
        assert_eq!(n, 3);

        // Should pick extended (4 bytes) — distance forces it
        buf.clear();
        let n = encode(4, 65_537, &mut buf);
        assert_eq!(n, 4);
    }

    #[test]
    fn worked_example_from_spec() {
        // +10 as short: offset = 10 - 2 = 8, sign = 0, length_raw = 8 = 0b001000
        // token = 0x40 | 0x08 = 0x48
        let mut buf = Vec::new();
        encode(10, 1, &mut buf);
        assert_eq!(buf[0], 0x48);

        // -5 as short: offset = 5 - 2 = 3, sign = 1, length_raw = 3 | 0b100000 = 35 = 0x23
        // token = 0x40 | 0x23 = 0x63
        buf.clear();
        encode(-5, 1, &mut buf);
        assert_eq!(buf[0], 0x63);
    }

    // ── Proptest roundtrips ───────────────────────────────────────────

    #[allow(clippy::cast_possible_wrap)]
    fn signed_length(min: u16, max: u16) -> impl Strategy<Value = i16> {
        prop_oneof![(min..=max).prop_map(|v| v as i16), (min..=max).prop_map(|v| -(v as i16)),]
    }

    proptest! {
        #[test]
        fn roundtrip_short(length in signed_length(2, 33), distance in 1u32..=256) {
            let mut buf = Vec::new();
            let n = encode(length, distance, &mut buf);

            prop_assert_eq!(n, 2);
            prop_assert_eq!(decode(&buf), (Frame::Match(length, distance), 2));
        }

        #[test]
        fn roundtrip_medium(length in signed_length(3, 34), distance in 1u32..=65_536) {
            let mut buf = Vec::new();
            let n = encode(length, distance, &mut buf);

            prop_assert!(n <= 3);
            let (frame, consumed) = decode(&buf);
            prop_assert_eq!(frame, Frame::Match(length, distance));
            prop_assert_eq!(consumed, n);
        }

        #[test]
        fn roundtrip_extended(length in signed_length(4, 515), distance in 1u32..=1_048_576) {
            let mut buf = Vec::new();
            let n = encode(length, distance, &mut buf);

            prop_assert!(n <= 4);
            let (frame, consumed) = decode(&buf);
            prop_assert_eq!(frame, Frame::Match(length, distance));
            prop_assert_eq!(consumed, n);
        }

        #[test]
        #[allow(clippy::cast_possible_wrap)]
        fn roundtrip_any_valid_match(
            frame_type in 0u8..3,
            mag_offset in 0u16..512,
            dist_offset in 0u32..1_048_576,
        ) {
            let (length, distance) = match frame_type {
                0 => {
                    let mag = 2 + (mag_offset % 32);
                    let dist = 1 + (dist_offset % 256);
                    (mag as i16, dist)
                }
                1 => {
                    let mag = 3 + (mag_offset % 32);
                    let dist = 1 + (dist_offset % 65_536);
                    (mag as i16, dist)
                }
                _ => {
                    let mag = 4 + (mag_offset % 512);
                    let dist = 1 + (dist_offset % 1_048_576);
                    (mag as i16, dist)
                }
            };

            // Randomly negate
            let length = if mag_offset % 2 == 0 { length } else { -length };

            let mut buf = Vec::new();
            encode(length, distance, &mut buf);

            let (frame, _) = decode(&buf);
            prop_assert_eq!(frame, Frame::Match(length, distance));
        }
    }
}
*/
