use crate::nibble::{NibbleWriter, ReadNibble};

/// BLEB8 requires `type_bits = 3N + 1`; returns N (the max nibble count).
const fn max_nibbles(bits: u32) -> usize {
    assert!((bits - 1).is_multiple_of(3), "bit width must equal 3N+1");
    (bits as usize - 1) / 3
}

/// Trait for types that can be encoded/decoded as unsigned BLEB8.
pub(crate) trait Ubleb8: Sized {
    /// Maximum number of nibbles this type can occupy.
    const MAX_NIBBLES: usize;

    /// Encodes `self` as UBLEB8, appending nibbles to `w`.
    fn encode_ubleb8(self, w: &mut NibbleWriter);

    /// Decodes a UBLEB8 value from `r`.
    fn decode_ubleb8<R: ReadNibble>(r: &mut R) -> Result<Self, R::Error>;
}

/// Trait for types that can be encoded/decoded as signed BLEB8.
pub(crate) trait Sleb8: Sized {
    /// Maximum number of nibbles this type can occupy.
    const MAX_NIBBLES: usize;

    /// Encodes `self` as SLEB8, appending nibbles to `w`.
    fn encode_sleb8(self, w: &mut NibbleWriter);

    /// Decodes an SLEB8 value from `r`.
    fn decode_sleb8<R: ReadNibble>(r: &mut R) -> Result<Self, R::Error>;
}

macro_rules! impl_ubleb8 {
    ($T:ty) => {
        impl Ubleb8 for $T {
            const MAX_NIBBLES: usize = max_nibbles(<$T>::BITS);

            #[allow(clippy::cast_possible_truncation)]
            fn encode_ubleb8(mut self, w: &mut NibbleWriter) {
                for _ in 0..Self::MAX_NIBBLES - 1 {
                    let mut nibble = (self & 0x7) as u8;
                    self >>= 3;
                    if self != 0 {
                        nibble |= 0x8;
                    }
                    w.push(nibble);
                    if self == 0 {
                        return;
                    }
                }
                w.push((self & 0xF) as u8);
            }

            fn decode_ubleb8<R: ReadNibble>(r: &mut R) -> Result<Self, R::Error> {
                let mut value: Self = 0;
                let mut shift: u32 = 0;
                for _ in 0..Self::MAX_NIBBLES - 1 {
                    let nibble = r.read_nibble()?;
                    value |= Self::from(nibble & 0x7) << shift;
                    shift += 3;
                    if nibble & 0x8 == 0 {
                        return Ok(value);
                    }
                }
                let nibble = r.read_nibble()?;
                value |= Self::from(nibble) << shift;
                Ok(value)
            }
        }
    };
}

macro_rules! impl_sleb8 {
    ($T:ty, $U:ty) => {
        impl Sleb8 for $T {
            const MAX_NIBBLES: usize = {
                assert!(<$T>::BITS == <$U>::BITS, "signed/unsigned bit-width mismatch");
                max_nibbles(<$T>::BITS)
            };

            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            fn encode_sleb8(mut self, w: &mut NibbleWriter) {
                for _ in 0..Self::MAX_NIBBLES - 1 {
                    let mut nibble = (self & 0x7) as u8;
                    self >>= 3;
                    let done = (self == 0 && nibble & 0x4 == 0) || (self == -1 && nibble & 0x4 != 0);
                    if !done {
                        nibble |= 0x8;
                    }
                    w.push(nibble);
                    if done {
                        return;
                    }
                }
                w.push((self & 0xF) as u8);
            }

            #[allow(clippy::cast_possible_wrap)]
            fn decode_sleb8<R: ReadNibble>(r: &mut R) -> Result<Self, R::Error> {
                let mut value: $U = 0;
                let mut shift: u32 = 0;
                let mut done = false;
                for _ in 0..Self::MAX_NIBBLES - 1 {
                    let nibble = r.read_nibble()?;
                    value |= <$U>::from(nibble & 0x7) << shift;
                    shift += 3;
                    if nibble & 0x8 == 0 {
                        done = true;
                        break;
                    }
                }
                if !done {
                    let nibble = r.read_nibble()?;
                    value |= <$U>::from(nibble) << shift;
                    shift += 4;
                }
                // Sign-extend if the sign bit of the decoded data is set.
                if shift < <$T>::BITS as u32 && value & (1 << (shift - 1)) != 0 {
                    value |= !0 << shift;
                }
                Ok(value as $T)
            }
        }
    };
}

impl_ubleb8!(u16);
impl_ubleb8!(u64);
impl_sleb8!(i16, u16);

/// Returns the number of nibbles needed to encode `value` as UBLEB8.
///
/// UBLEB8 encodes 3 bits per nibble; thresholds at 2^3, 2^6, 2^9, 2^12.
pub(crate) const fn ubleb8_len(value: u16) -> usize {
    if value < 8 {
        1
    } else if value < 64 {
        2
    } else if value < 512 {
        3
    } else if value < 4096 {
        4
    } else {
        5
    }
}

/// Returns the number of nibbles needed to encode `value` as SLEB8.
///
/// SLEB8 signed thresholds at +/-4, +/-32, +/-256, +/-2048.
pub(crate) const fn sleb8_len(value: i16) -> usize {
    if value >= -4 && value < 4 {
        1
    } else if value >= -32 && value < 32 {
        2
    } else if value >= -256 && value < 256 {
        3
    } else if value >= -2048 && value < 2048 {
        4
    } else {
        5
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    use super::*;
    use crate::nibble::NibbleReader;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn nibbles_from_ubleb8<T: Ubleb8>(value: T) -> Vec<u8> {
        let mut w = NibbleWriter::new();
        value.encode_ubleb8(&mut w);
        let count = w.len();
        let packed = w.finish();
        let mut r = NibbleReader::new(&packed);
        (0..count).map(|_| r.read().unwrap()).collect()
    }

    fn nibbles_from_sleb8<T: Sleb8>(value: T) -> Vec<u8> {
        let mut w = NibbleWriter::new();
        value.encode_sleb8(&mut w);
        let count = w.len();
        let packed = w.finish();
        let mut r = NibbleReader::new(&packed);
        (0..count).map(|_| r.read().unwrap()).collect()
    }

    // ── Spec Worked Examples ────────────────────────────────────────────

    #[test]
    fn ubleb8_31415_u16() {
        assert_eq!(nibbles_from_ubleb8(31415u16), vec![0xF, 0xE, 0xA, 0xD, 0x7]);
    }

    #[test]
    fn sleb8_neg5_i16() {
        assert_eq!(nibbles_from_sleb8(-5i16), vec![0xB, 0x7]);
    }

    #[test]
    fn sleb8_pos5_i16() {
        assert_eq!(nibbles_from_sleb8(5i16), vec![0xD, 0x0]);
    }

    proptest! {
        #[test]
        fn round_trip_u16(value in any::<u16>()) {
            let mut w = NibbleWriter::new();
            value.encode_ubleb8(&mut w);
            let packed = w.finish();
            let mut r = NibbleReader::new(&packed);
            prop_assert_eq!(u16::decode_ubleb8(&mut r).unwrap(), value);
        }

        #[test]
        fn round_trip_u64(value in any::<u64>()) {
            let mut w = NibbleWriter::new();
            value.encode_ubleb8(&mut w);
            let packed = w.finish();
            let mut r = NibbleReader::new(&packed);
            prop_assert_eq!(u64::decode_ubleb8(&mut r).unwrap(), value);
        }

        #[test]
        fn round_trip_i16(value in any::<i16>()) {
            let mut w = NibbleWriter::new();
            value.encode_sleb8(&mut w);
            let packed = w.finish();
            let mut r = NibbleReader::new(&packed);
            prop_assert_eq!(i16::decode_sleb8(&mut r).unwrap(), value);
        }

        #[test]
        fn ubleb8_len_matches_encode(value in any::<u16>()) {
            let mut w = NibbleWriter::new();
            value.encode_ubleb8(&mut w);
            prop_assert_eq!(ubleb8_len(value), w.len());
        }

        #[test]
        fn sleb8_len_matches_encode(value in any::<i16>()) {
            let mut w = NibbleWriter::new();
            value.encode_sleb8(&mut w);
            prop_assert_eq!(sleb8_len(value), w.len());
        }
    }
}
