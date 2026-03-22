//! Power-of-two circular buffer for sliding-window compression.
//!
//! [`Buffer`] wraps a flat `Vec<u8>` of size `N` (which must be a power of two) and uses
//! bitmask indexing (`index & (N - 1)`) so that logical positions wrap automatically.
//! This avoids explicit modulo arithmetic in the encoder and decoder hot paths.

use std::ops::{Index, IndexMut, Range};

/// A fixed-size circular byte buffer with power-of-two masking.
///
/// `N` **must** be a power of two (enforced at compile time). All index operations
/// automatically wrap via `index & (N - 1)`, so callers can use unbounded logical
/// positions without manual modulo.
///
/// # Examples
///
/// ```text
/// let mut buf = Buffer::<256>::new();
///
/// // Writes wrap around transparently.
/// buf[0] = 0xAA;
/// buf[256] = 0xBB; // wraps to index 0
/// assert_eq!(buf[0], 0xBB);
///
/// // Bulk copy from a slice.
/// buf.copy_from_slice(b"hello", 250);
/// let (a, b) = buf.slices(250, 5);
/// assert_eq!([a, b].concat(), b"hello");
/// ```
pub(crate) struct Buffer<const N: usize> {
    buf: Vec<u8>,
}

impl<const N: usize> Buffer<N> {
    /// Bitmask for wrapping indices: `N - 1` (valid because `N` is a power of two).
    const MASK: usize = {
        assert!(N.is_power_of_two());
        N - 1
    };
}

impl<const N: usize> Buffer<N> {
    /// Creates a new buffer with all bytes initialized to zero.
    ///
    /// # Examples
    ///
    /// ```text
    /// let buf = Buffer::<1024>::new();
    /// assert_eq!(buf[0], 0);
    /// ```
    pub(crate) fn new() -> Self {
        Self {
            buf: vec![0u8; N],
        }
    }

    /// Copies `src.len()` bytes from positions `src` to position `dest`, in forward order.
    ///
    /// Both `src` and `dest` are logical (unwrapped) positions — masking is handled by
    /// the `Index`/`IndexMut` impls. Because bytes are copied one at a time from low to
    /// high, overlapping regions replicate earlier bytes (the standard LZ77 run-length
    /// trick).
    ///
    /// # Examples
    ///
    /// ```text
    /// let mut buf = Buffer::<256>::new();
    /// buf.copy_from_slice(b"ABCD", 0);
    /// buf.copy_within(0..4, 10);
    /// assert_eq!(buf[10], b'A');
    /// assert_eq!(buf[13], b'D');
    /// ```
    pub(crate) fn copy_within(&mut self, src: Range<usize>, dest: usize) {
        for i in 0..src.len() {
            self[dest + i] = self[src.start + i];
        }
    }

    /// Copies `src.len()` bytes to `dest`, reading backwards from `src.start`.
    ///
    /// Byte `i` of the output is read from `src.start.wrapping_sub(i)`, allowing
    /// reverse-order copies for back-reference patterns that read history in reverse.
    ///
    /// # Examples
    ///
    /// ```text
    /// let mut buf = Buffer::<256>::new();
    /// buf.copy_from_slice(b"ABCD", 0);
    /// buf.copy_within_rev(3..3 + 4, 10);
    /// // Reads indices 3, 2, 1, 0 → 'D', 'C', 'B', 'A'
    /// assert_eq!(buf[10], b'D');
    /// assert_eq!(buf[13], b'A');
    /// ```
    pub(crate) fn copy_within_rev(&mut self, src: Range<usize>, dest: usize) {
        for i in 0..src.len() {
            self[dest + i] = self[src.start.wrapping_sub(i)];
        }
    }

    /// Copies the bytes of `src` into the buffer starting at logical position `dest`.
    ///
    /// # Examples
    ///
    /// ```text
    /// let mut buf = Buffer::<256>::new();
    /// buf.copy_from_slice(b"hello", 0);
    /// assert_eq!(buf[0], b'h');
    /// assert_eq!(buf[4], b'o');
    /// ```
    pub(crate) fn copy_from_slice(&mut self, src: &[u8], dest: usize) {
        for (i, &b) in src.iter().enumerate() {
            self[dest + i] = b;
        }
    }

    /// Returns one or two contiguous slices covering `len` bytes starting at `start`.
    ///
    /// If the range does not wrap around the end of the underlying storage, the first
    /// slice contains all `len` bytes and the second slice is empty. If the range wraps,
    /// the first slice covers `start..N` and the second covers `0..(start + len - N)`.
    ///
    /// # Examples
    ///
    /// ```text
    /// let mut buf = Buffer::<256>::new();
    /// buf.copy_from_slice(b"hello", 254);
    ///
    /// let (a, b) = buf.slices(254, 5);
    /// // 'h','e' at end of buffer; 'l','l','o' wrap to the beginning.
    /// assert_eq!(a, b"he");
    /// assert_eq!(b, b"llo");
    /// ```
    pub(crate) fn slices(&self, start: usize, len: usize) -> (&[u8], &[u8]) {
        let start = start & Self::MASK;
        let end = start + len;

        if end <= N {
            (&self.buf[start..end], &[])
        } else {
            (&self.buf[start..], &self.buf[..end - N])
        }
    }

    /// Mutable variant of [`Buffer::slices`].
    ///
    /// Returns one or two mutable slices covering `len` bytes starting at `start`.
    pub(crate) fn slices_mut(&mut self, start: usize, len: usize) -> (&mut [u8], &mut [u8]) {
        let start = start & Self::MASK;
        let end = start + len;

        if end <= N {
            (&mut self.buf[start..end], &mut [])
        } else {
            let (left, right) = self.buf.split_at_mut(start);
            (right, &mut left[..end - N])
        }
    }
}

impl<const N: usize> Index<usize> for Buffer<N> {
    type Output = u8;

    fn index(&self, index: usize) -> &u8 {
        &self.buf[index & Self::MASK]
    }
}

impl<const N: usize> IndexMut<usize> for Buffer<N> {
    fn index_mut(&mut self, index: usize) -> &mut u8 {
        &mut self.buf[index & Self::MASK]
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::ops::Range;

    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    use super::*;

    fn wrap(n: usize, i: usize) -> usize {
        i & (n - 1)
    }

    #[derive(Debug, Clone)]
    enum Op {
        Write(usize, u8),
        CopyWithin {
            src: Range<usize>,
            dest: usize,
        },
        CopyWithinRev {
            src: Range<usize>,
            dest: usize,
        },
        CopyFromSlice {
            data: Vec<u8>,
            dest: usize,
        },
        Slices {
            start: usize,
            len: usize,
        },
    }

    fn op_strategy() -> impl Strategy<Value = Op> {
        prop_oneof![
            (any::<usize>(), any::<u8>()).prop_map(|(i, v)| Op::Write(i, v)),
            (any::<usize>(), 0..64usize, any::<usize>()).prop_map(|(start, len, dest)| Op::CopyWithin {
                src: start..start + len,
                dest
            }),
            (any::<usize>(), 0..64usize, any::<usize>()).prop_map(|(start, len, dest)| Op::CopyWithinRev {
                src: start..start + len,
                dest
            }),
            (prop::collection::vec(any::<u8>(), 0..64), any::<usize>()).prop_map(|(data, dest)| Op::CopyFromSlice {
                data,
                dest
            }),
            (any::<usize>(), 0..=256usize).prop_map(|(start, len)| Op::Slices {
                start,
                len
            }),
        ]
    }

    proptest! {
        #[test]
        fn matches_oracle(ops in prop::collection::vec(op_strategy(), 0..=128)) {
            const N: usize = 256;
            let mut buf = Buffer::<N>::new();
            let mut oracle = vec![0u8; N];

            for op in ops {
                match op {
                    Op::Write(i, v) => {
                        buf[i] = v;
                        oracle[wrap(N, i)] = v;
                    }
                    Op::CopyWithin { src, dest } => {
                        buf.copy_within(src.clone(), dest);
                        for i in 0..src.len() {
                            oracle[wrap(N, dest + i)] = oracle[wrap(N, src.start + i)];
                        }
                    }
                    Op::CopyWithinRev { src, dest } => {
                        buf.copy_within_rev(src.clone(), dest);
                        for i in 0..src.len() {
                            oracle[wrap(N, dest + i)] = oracle[wrap(N, src.start.wrapping_sub(i))];
                        }
                    }
                    Op::CopyFromSlice { data, dest } => {
                        buf.copy_from_slice(&data, dest);
                        for (i, &b) in data.iter().enumerate() {
                            oracle[wrap(N, dest + i)] = b;
                        }
                    }
                    Op::Slices { start, len } => {
                        let (a, b) = buf.slices(start, len);
                        let mut actual = Vec::new();
                        actual.extend_from_slice(a);
                        actual.extend_from_slice(b);

                        let masked = wrap(N, start);
                        let expected: Vec<u8> = (0..len).map(|i| oracle[wrap(N, masked + i)]).collect();
                        assert_eq!(actual, expected);
                    }
                }
                assert_eq!(&buf.buf[..], &oracle[..]);
            }
        }
    }
}
