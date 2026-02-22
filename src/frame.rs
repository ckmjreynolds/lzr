const EOS: u8 = 0x00;
const LITERAL: u8 = 0x00;
const SHORT: u8 = 0x01;
const SHORT_REV: u8 = 0x81;
const MEDIUM: u8 = 0x02;
const MEDIUM_REV: u8 = 0x82;
const EXTENDED: u8 = 0x03;
const EXTENDED_REV: u32 = 0x803;
const TOKEN_TYPE_MASK: u8 = 0x03;

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

    #[allow(clippy::cast_possible_wrap)]
    pub(crate) fn decode(input: &[u8]) -> (Self, usize) {
        let token = input[0];
        match token & TOKEN_TYPE_MASK {
            LITERAL => {
                if token == EOS {
                    (Self::Eos, 1)
                } else {
                    (Self::Literal((token >> 2) as isize), 1)
                }
            }
            SHORT => {
                let sign = token & 0x80 != 0;
                let m = ((token >> 2) & 0x1F) as usize + 2;
                let d = input[1] as usize + 1;
                let l = if sign {
                    -(m as isize)
                } else {
                    m as isize
                };
                (Self::Match(l, d), 2)
            }
            MEDIUM => {
                let sign = token & 0x80 != 0;
                let m = ((token >> 2) & 0x1F) as usize + 3;
                let d = u16::from_le_bytes([input[1], input[2]]) as usize + 1;
                let l = if sign {
                    -(m as isize)
                } else {
                    m as isize
                };
                (Self::Match(l, d), 3)
            }
            EXTENDED => {
                let raw = u32::from_le_bytes([input[0], input[1], input[2], input[3]]);
                let sign = raw & 0x800 != 0;
                let m = ((raw >> 2) & 0x1FF) as usize + 4;
                let d = ((raw >> 12) & 0xF_FFFF) as usize + 1;
                let l = if sign {
                    -(m as isize)
                } else {
                    m as isize
                };
                (Self::Match(l, d), 4)
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    #[test]
    fn eos() {
        let mut buf = Vec::new();

        assert_eq!(Frame::Eos.encode(&mut buf), 1);
        assert_eq!(Frame::decode(&buf), (Frame::Eos, 1));
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0], 0x00);
    }

    #[test]
    #[should_panic(expected = "assertion failed: (1..=63).contains(&l)")]
    fn unencodable_literal() {
        let mut buf = Vec::new();

        Frame::Literal(64).encode(&mut buf);
    }

    #[test]
    #[should_panic(expected = "entered unreachable code")]
    fn unencodable_match() {
        let mut buf = Vec::new();

        Frame::Match(3, 65_567).encode(&mut buf);
    }

    #[allow(clippy::match_like_matches_macro)]
    fn is_encodable(length: isize, distance: usize) -> bool {
        match (length, distance) {
            (0, 0) => true,
            (1..=63, 0) => true,
            (-33..=-2, 1..=256) => true,
            (2..=33, 1..=256) => true,
            (-34..=-3, 1..=65_536) => true,
            (3..=34, 1..=65_536) => true,
            (-515..=-4, 1..=1_048_576) => true,
            (4..=515, 1..=1_048_576) => true,
            _ => false,
        }
    }

    proptest! {
        #[test]
        fn roundtrip_literal(length in 1isize..=63isize) {
            prop_assume!(is_encodable(length, 0));

            let mut buf = Vec::new();

            prop_assert_eq!(Frame::Literal(length).encode(&mut buf), 1);
            prop_assert_eq!(Frame::decode(&buf), (Frame::Literal(length), 1));
        }

        #[test]
        fn roundtrip_small_match(length in -33isize..=33isize, distance in 1usize..=256) {
            prop_assume!(is_encodable(length, distance));

            let mut buf = Vec::new();

            prop_assert_eq!(Frame::Match(length, distance).encode(&mut buf), 2);
            prop_assert_eq!(Frame::decode(&buf), (Frame::Match(length, distance), 2));
        }

        #[test]
        fn roundtrip_medium_match(length in -34isize..=34isize, distance in 257usize..=65_536) {
            prop_assume!(is_encodable(length, distance));

            let mut buf = Vec::new();

            prop_assert_eq!(Frame::Match(length, distance).encode(&mut buf), 3);
            prop_assert_eq!(Frame::decode(&buf), (Frame::Match(length, distance), 3));
        }

        #[test]
        fn roundtrip_extended_match(length in -515isize..=515isize, distance in 65_537usize..=1_048_576) {
            prop_assume!(is_encodable(length, distance));

            let mut buf = Vec::new();

            prop_assert_eq!(Frame::Match(length, distance).encode(&mut buf), 4);
            prop_assert_eq!(Frame::decode(&buf), (Frame::Match(length, distance), 4));
        }

        #[test]
        fn roundtrip_any_valid(length in -515isize..=515, distance in 0usize..=1_048_576) {
            prop_assume!(is_encodable(length, distance));

            let mut buf = Vec::new();

            match (length, distance) {
                (0, 0) => {
                    prop_assert_eq!(Frame::Eos.encode(&mut buf), 1);
                    prop_assert_eq!(Frame::decode(&buf), (Frame::Eos, 1));
                },
                (1..=63, 0) => {
                    prop_assert_eq!(Frame::Literal(length).encode(&mut buf), 1);
                    prop_assert_eq!(Frame::decode(&buf), (Frame::Literal(length), 1));
                }
                _ => {
                    prop_assert!(Frame::Match(length, distance).encode(&mut buf) > 1);
                    prop_assert_eq!(Frame::decode(&buf).0, Frame::Match(length, distance));
                }
            }
        }
    }
}
