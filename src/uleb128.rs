use tinyvec::ArrayVec;

const MAX_ULEB128_LEN: usize = 9;

// This is a bounded version of ULEB128 (see docs/FORMAT.md).
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn encode(mut value: u64) -> ArrayVec<[u8; MAX_ULEB128_LEN]> {
    let mut output = ArrayVec::<[u8; MAX_ULEB128_LEN]>::default();
    let mut byte;

    loop {
        // Consume the lower 7 or 8 bits
        if output.len() < (MAX_ULEB128_LEN - 1) {
            byte = (value & 0x7f) as u8;
            value >>= 7;
        } else {
            byte = value as u8;
            value >>= 8;
        }

        // Add a continuation bit, if required.
        if value > 0 {
            byte |= 0x80;
        }

        output.push(byte);

        if value == 0 {
            break;
        }
    }

    output
}

// This is a bounded version of ULEB128 (see docs/FORMAT.md).
pub(crate) fn decode(input: &[u8]) -> u64 {
    let mut result = 0u64;
    let mut shift = 0u32;
    let mut i = 0usize;

    loop {
        if i < (MAX_ULEB128_LEN - 1) {
            result |= u64::from(input[i] & 0x7f) << shift;

            if input[i] & 0x80 == 0 {
                break;
            }

            shift += 7;
            i += 1;
        } else {
            result |= u64::from(input[i]) << shift;
            break;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;
    use tinyvec::array_vec;

    #[test]
    fn encode_zero() {
        let expected = array_vec!([u8; 9] => 0x00);
        assert_eq!(encode(0), expected);
    }

    #[test]
    fn decode_zero() {
        let data = array_vec!([u8; 9] => 0x00);
        assert_eq!(decode(data.as_slice()), 0);
    }

    #[test]
    fn encode_max() {
        let expected = array_vec!([u8; 9] => 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        assert_eq!(encode(u64::MAX), expected);
    }

    #[test]
    fn decode_max() {
        let data = array_vec!([u8; 9] => 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        assert_eq!(decode(data.as_slice()), u64::MAX);
    }

    proptest! {
        #[test]
        fn roundtrip_any_u64(val: u64) {
            prop_assert_eq!(val, decode(encode(val).as_slice()));
        }
    }
}
