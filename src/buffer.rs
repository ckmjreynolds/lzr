use std::ops::{Index, IndexMut, Range};

struct Buffer<const N: usize> {
    buf: [u8; N],
}

impl<const N: usize> Buffer<N> {
    const MASK: usize = {
        assert!(N.is_power_of_two());
        N - 1
    };
}

impl<const N: usize> Index<usize> for Buffer<N> {
    type Output = u8;
    #[inline]
    fn index(&self, index: usize) -> &u8 {
        &self.buf[index & Self::MASK]
    }
}

impl<const N: usize> IndexMut<usize> for Buffer<N> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut u8 {
        &mut self.buf[index & Self::MASK]
    }
}

impl<const N: usize> Buffer<N> {
    #[inline]
    fn copy_within(&mut self, src: Range<usize>, dest: usize) {
        for i in 0..src.len() {
            self[dest + i] = self[src.start + i];
        }
    }

    #[inline]
    fn copy_within_rev(&mut self, src: Range<usize>, dest: usize) {
        for i in 0..src.len() {
            self[dest + i] = self[src.start.wrapping_sub(i)];
        }
    }

    #[inline]
    fn copy_from_slice(&mut self, src: &[u8], dest: usize) {
        for (i, &b) in src.iter().enumerate() {
            self[dest + i] = b;
        }
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
        ]
    }

    proptest! {
        #[test]
        fn matches_oracle(ops in prop::collection::vec(op_strategy(), 0..=128)) {
            const N: usize = 256;
            let mut buf = Buffer::<N> { buf: [0; N] };
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
                }
                assert_eq!(&buf.buf[..], &oracle[..]);
            }
        }
    }
}
