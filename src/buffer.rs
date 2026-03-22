use std::ops::{Index, IndexMut, Range};

pub(crate) struct Buffer<const N: usize> {
    buf: [u8; N],
}

impl<const N: usize> Buffer<N> {
    const MASK: usize = {
        assert!(N.is_power_of_two());
        N - 1
    };
}

impl<const N: usize> Buffer<N> {
    pub(crate) const fn new() -> Self {
        Self {
            buf: [0u8; N],
        }
    }

    pub(crate) fn copy_within(&mut self, src: Range<usize>, dest: usize) {
        for i in 0..src.len() {
            self[dest + i] = self[src.start + i];
        }
    }

    pub(crate) fn copy_within_rev(&mut self, src: Range<usize>, dest: usize) {
        for i in 0..src.len() {
            self[dest + i] = self[src.start.wrapping_sub(i)];
        }
    }

    pub(crate) fn copy_from_slice(&mut self, src: &[u8], dest: usize) {
        for (i, &b) in src.iter().enumerate() {
            self[dest + i] = b;
        }
    }

    pub(crate) fn slices(&self, start: usize, len: usize) -> (&[u8], &[u8]) {
        let start = start & Self::MASK;
        let end = start + len;

        if end <= N {
            (&self.buf[start..end], &[])
        } else {
            (&self.buf[start..], &self.buf[..end - N])
        }
    }

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
