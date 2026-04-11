//! Low-level memory primitives used by the encoder and decoder hot paths.
//!
//! Two implementations are provided, selected at compile time by the `unsafe`
//! feature:
//!
//! - `feature = "unsafe"` (default): pointer-based unaligned reads and
//!   `ptr::copy_nonoverlapping`-based wildcopy. These avoid the bounds checks
//!   that the safe variants would otherwise perform on every byte access in
//!   the hot loops.
//! - `not(feature = "unsafe")`: slice-based equivalents that the optimizer can
//!   often vectorize, but with bounds checks the optimizer cannot always elide.
//!
//! The two implementations are observationally identical: they read and write
//! the same bytes given the same inputs. Output is byte-identical between the
//! two builds; only speed differs.
//!
//! All caller-visible functions document a "wildcopy contract": the caller is
//! responsible for ensuring the read and write ranges are in-bounds. Wildcopy
//! routines deliberately copy more bytes than the logical match length so the
//! over-copied tail can be safely overwritten by the next operation.

#[cfg(feature = "unsafe")]
#[allow(unsafe_code)]
mod imp {
    use std::ptr;

    /// Reads a little-endian `u32` from `buf[pos..pos+4]`.
    ///
    /// Caller must guarantee `pos + 4 <= buf.len()`.
    #[inline]
    pub(crate) fn read_u32_le(buf: &[u8], pos: usize) -> u32 {
        debug_assert!(pos + 4 <= buf.len());
        // SAFETY: caller guarantees `pos + 4 <= buf.len()`. The pointer is
        // therefore in-bounds and `read_unaligned` allows any alignment.
        let raw = unsafe { ptr::read_unaligned(buf.as_ptr().add(pos).cast::<u32>()) };
        u32::from_le(raw)
    }

    /// Reads a big-endian `u32` from `buf[pos..pos+4]`.
    ///
    /// Caller must guarantee `pos + 4 <= buf.len()`.
    #[inline]
    pub(crate) fn read_u32_be(buf: &[u8], pos: usize) -> u32 {
        debug_assert!(pos + 4 <= buf.len());
        // SAFETY: see read_u32_le.
        let raw = unsafe { ptr::read_unaligned(buf.as_ptr().add(pos).cast::<u32>()) };
        u32::from_be(raw)
    }

    /// Reads a native-endian `u64` from `buf[pos..pos+8]`.
    ///
    /// Used for u64-XOR match expansion.
    /// Caller must guarantee `pos + 8 <= buf.len()`.
    #[inline]
    pub(crate) fn read_u64_ne(buf: &[u8], pos: usize) -> u64 {
        debug_assert!(pos + 8 <= buf.len());
        // SAFETY: caller guarantees `pos + 8 <= buf.len()`.
        unsafe { ptr::read_unaligned(buf.as_ptr().add(pos).cast::<u64>()) }
    }

    /// Reads a little-endian `u16` from `buf[pos..pos+2]`.
    ///
    /// Caller must guarantee `pos + 2 <= buf.len()`.
    #[inline]
    pub(crate) fn read_u16_le(buf: &[u8], pos: usize) -> u16 {
        debug_assert!(pos + 2 <= buf.len());
        // SAFETY: caller guarantees `pos + 2 <= buf.len()`.
        let raw = unsafe { ptr::read_unaligned(buf.as_ptr().add(pos).cast::<u16>()) };
        u16::from_le(raw)
    }

    /// Copies exactly 16 bytes from `src[src_pos..src_pos+16]` to
    /// `dst[dst_pos..dst_pos+16]`.
    ///
    /// Wildcopy contract: caller has reserved at least 16 bytes of slop past
    /// the logical end so over-copying is safe. The copied bytes past the
    /// logical end will be overwritten by the next operation.
    #[inline]
    pub(crate) fn wildcopy_16(dst: &mut [u8], dst_pos: usize, src: &[u8], src_pos: usize) {
        debug_assert!(dst_pos + 16 <= dst.len());
        debug_assert!(src_pos + 16 <= src.len());
        // SAFETY: caller-guaranteed bounds; src and dst are distinct slices.
        unsafe {
            ptr::copy_nonoverlapping(src.as_ptr().add(src_pos), dst.as_mut_ptr().add(dst_pos), 16);
        }
    }

    /// In-buffer copy from `src_pos` to `dst_pos`, exactly 16 bytes.
    ///
    /// Caller must guarantee non-overlap (typically: `dst_pos - src_pos >= 16`).
    #[inline]
    pub(crate) fn wildcopy_16_within(buf: &mut [u8], dst_pos: usize, src_pos: usize) {
        debug_assert!(dst_pos + 16 <= buf.len());
        debug_assert!(src_pos + 16 <= buf.len());
        debug_assert!(dst_pos >= src_pos + 16 || src_pos >= dst_pos + 16);
        let p = buf.as_mut_ptr();
        // SAFETY: caller-guaranteed in-bounds and non-overlapping.
        unsafe {
            ptr::copy_nonoverlapping(p.add(src_pos), p.add(dst_pos), 16);
        }
    }

    /// In-buffer forward copy that may overlap (LZ77 RLE-style).
    ///
    /// Bytes are copied one at a time so `distance < length` produces the
    /// standard repeating pattern (e.g., distance 1, length 33 repeats the
    /// last byte 33 times).
    ///
    /// Caller must guarantee `dst_pos + len <= buf.len()` and
    /// `src_pos + len <= buf.len()`.
    #[inline]
    pub(crate) fn copy_forward(buf: &mut [u8], dst_pos: usize, src_pos: usize, len: usize) {
        debug_assert!(dst_pos + len <= buf.len());
        debug_assert!(src_pos + len <= buf.len());
        let p = buf.as_mut_ptr();
        // SAFETY: caller-guaranteed bounds; pointer arithmetic stays in-bounds.
        unsafe {
            for i in 0..len {
                *p.add(dst_pos + i) = *p.add(src_pos + i);
            }
        }
    }

    /// In-buffer reverse copy used for negative-length (reverse) matches.
    ///
    /// `dst[dst_pos + i] = buf[src_start - i]` for `i in 0..len`. Always
    /// byte-by-byte; reverse matches don't benefit from wildcopy.
    ///
    /// Caller must guarantee `dst_pos + len <= buf.len()` and that
    /// `src_start - i` does not underflow for `i in 0..len`.
    #[inline]
    pub(crate) fn copy_reverse(buf: &mut [u8], dst_pos: usize, src_start: usize, len: usize) {
        debug_assert!(dst_pos + len <= buf.len());
        debug_assert!(src_start + 1 >= len);
        let p = buf.as_mut_ptr();
        // SAFETY: caller-guaranteed bounds.
        unsafe {
            for i in 0..len {
                *p.add(dst_pos + i) = *p.add(src_start - i);
            }
        }
    }
}

#[cfg(not(feature = "unsafe"))]
mod imp {
    #[inline]
    pub(crate) fn read_u32_le(buf: &[u8], pos: usize) -> u32 {
        u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]])
    }

    #[inline]
    pub(crate) fn read_u32_be(buf: &[u8], pos: usize) -> u32 {
        u32::from_be_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]])
    }

    #[inline]
    pub(crate) fn read_u64_ne(buf: &[u8], pos: usize) -> u64 {
        let bytes: [u8; 8] = buf[pos..pos + 8].try_into().unwrap();
        u64::from_ne_bytes(bytes)
    }

    #[inline]
    pub(crate) fn read_u16_le(buf: &[u8], pos: usize) -> u16 {
        u16::from_le_bytes([buf[pos], buf[pos + 1]])
    }

    #[inline]
    pub(crate) fn wildcopy_16(dst: &mut [u8], dst_pos: usize, src: &[u8], src_pos: usize) {
        dst[dst_pos..dst_pos + 16].copy_from_slice(&src[src_pos..src_pos + 16]);
    }

    #[inline]
    pub(crate) fn wildcopy_16_within(buf: &mut [u8], dst_pos: usize, src_pos: usize) {
        buf.copy_within(src_pos..src_pos + 16, dst_pos);
    }

    #[inline]
    pub(crate) fn copy_forward(buf: &mut [u8], dst_pos: usize, src_pos: usize, len: usize) {
        for i in 0..len {
            buf[dst_pos + i] = buf[src_pos + i];
        }
    }

    #[inline]
    pub(crate) fn copy_reverse(buf: &mut [u8], dst_pos: usize, src_start: usize, len: usize) {
        for i in 0..len {
            buf[dst_pos + i] = buf[src_start - i];
        }
    }
}

// Some primitives are only used by the decoder, which is rewritten in a later
// step. Suppress unused-import warnings until then.
#[allow(unused_imports)]
pub(crate) use imp::{
    copy_forward, copy_reverse, read_u16_le, read_u32_be, read_u32_le, read_u64_ne, wildcopy_16, wildcopy_16_within,
};

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn read_u32_le_basic() {
        let buf = [0x01, 0x02, 0x03, 0x04, 0xFF];
        assert_eq!(read_u32_le(&buf, 0), 0x0403_0201);
        assert_eq!(read_u32_le(&buf, 1), 0xFF04_0302);
    }

    #[test]
    fn read_u32_be_basic() {
        let buf = [0x01, 0x02, 0x03, 0x04, 0xFF];
        assert_eq!(read_u32_be(&buf, 0), 0x0102_0304);
        assert_eq!(read_u32_be(&buf, 1), 0x0203_04FF);
    }

    #[test]
    fn read_u64_ne_basic() {
        let buf = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let expected = u64::from_ne_bytes(buf);
        assert_eq!(read_u64_ne(&buf, 0), expected);
    }

    #[test]
    fn read_u16_le_basic() {
        let buf = [0x01, 0x02, 0xFF];
        assert_eq!(read_u16_le(&buf, 0), 0x0201);
        assert_eq!(read_u16_le(&buf, 1), 0xFF02);
    }

    #[test]
    fn wildcopy_16_basic() {
        let mut dst = vec![0u8; 32];
        let src: Vec<u8> = (0..32u8).collect();
        wildcopy_16(&mut dst, 8, &src, 0);
        assert_eq!(&dst[8..24], (0..16u8).collect::<Vec<_>>().as_slice());
        assert_eq!(&dst[..8], &[0; 8]);
    }

    #[test]
    fn wildcopy_16_within_basic() {
        let mut buf: Vec<u8> = (0..32u8).collect();
        wildcopy_16_within(&mut buf, 16, 0);
        assert_eq!(&buf[16..32], (0..16u8).collect::<Vec<_>>().as_slice());
    }

    #[test]
    fn copy_forward_overlapping_rle() {
        // Distance 1, length 5 → repeats the byte at src_pos five times.
        let mut buf = vec![0u8; 16];
        buf[0] = 0xAA;
        copy_forward(&mut buf, 1, 0, 5);
        assert_eq!(&buf[..6], &[0xAA; 6]);
    }

    #[test]
    fn copy_reverse_basic() {
        // buf = [A, B, C, D, _, _, _, _]; reverse from src_start=3 to dst_pos=4 length 4.
        // Should write D, C, B, A.
        let mut buf = vec![b'A', b'B', b'C', b'D', 0, 0, 0, 0];
        copy_reverse(&mut buf, 4, 3, 4);
        assert_eq!(&buf[4..8], b"DCBA");
    }
}
