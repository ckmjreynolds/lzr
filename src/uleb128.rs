const MAX_ULEB128_LEN: usize = 9;

// This is a bounded version of ULEB128 (see docs/FORMAT.md).
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn encode(mut value: u64, buf: &mut Vec<u8>) -> usize {
    let mut byte;
    let mut count = 0usize;

    loop {
        // Consume the lower 7 or 8 bits
        if count < (MAX_ULEB128_LEN - 1) {
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

        buf.push(byte);
        count += 1;

        if value == 0 {
            break;
        }
    }

    count
}

// This is a bounded version of ULEB128 (see docs/FORMAT.md).
pub(crate) fn decode(input: &[u8]) -> (u64, usize) {
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

    (result, i + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    #[test]
    fn encode_zero() {
        let mut buf = Vec::new();
        let n = encode(0, &mut buf);

        assert_eq!(buf, vec![0x00]);
        assert_eq!(n, 1);
    }

    #[test]
    fn decode_zero() {
        assert_eq!(decode(&[0x00]), (0, 1));
    }

    #[test]
    fn encode_max() {
        let mut buf = Vec::new();
        let n = encode(u64::MAX, &mut buf);

        assert_eq!(buf, vec![0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
        assert_eq!(n, 9);
    }

    #[test]
    fn decode_max() {
        assert_eq!(decode(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]), (u64::MAX, 9));
    }

    proptest! {
        #[test]
        fn roundtrip_any_u64(val: u64) {
            let mut buf = Vec::new();

            encode(val, &mut buf);

            let (decoded, bytes_consumed) = decode(&buf);

            prop_assert_eq!(val, decoded);
            prop_assert_eq!(buf.len(), bytes_consumed);
        }
    }
}
