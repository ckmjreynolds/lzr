//! Adaptive arithmetic coding using u64 range coding with byte-oriented I/O.

use static_assertions::const_assert;

const_assert!(TOTAL_COUNTS.is_power_of_two());
const_assert!(TOTAL_COUNTS == NUM_SYMBOLS * INITIAL_COUNT);

const TOTAL_COUNTS: usize = 1 << 12;
const INITIAL_COUNT: usize = TOTAL_COUNTS / NUM_SYMBOLS;
const NUM_SYMBOLS: usize = 1 << 8;

// These const arrays are pre-computed based on a known values for TOTAL_COUNTS, INITIAL_COUNT, and NUM_SYMBOLS.
const_assert!(TOTAL_COUNTS == 4_096);
const_assert!(INITIAL_COUNT == 16);
const_assert!(NUM_SYMBOLS == 256);

#[rustfmt::skip]
pub(crate) const UNIFORM_TREE: [i16; NUM_SYMBOLS + 1] = [
    0,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 256,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 512,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 256,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 1024,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 256,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 512,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 256,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 2048,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 256,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 512,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 256,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 1024,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 256,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 512,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 256,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 4096,
];

const UNIFORM_FREQ: [u16; NUM_SYMBOLS] = [16u16; NUM_SYMBOLS];

pub(crate) struct Model {
    tree: [i16; NUM_SYMBOLS + 1],
    freq: [u16; NUM_SYMBOLS],
    cursor: u8,
}

impl Model {
    pub(crate) const fn new() -> Self {
        Self {
            tree: UNIFORM_TREE,
            freq: UNIFORM_FREQ,
            cursor: 0xFF,
        }
    }

    pub(crate) const fn reset(&mut self) {
        *self = Self::new();
    }

    pub(crate) const fn total() -> u64 {
        TOTAL_COUNTS as u64
    }

    pub(crate) const fn freq(&self, byte: u8) -> u64 {
        self.freq[byte as usize] as u64
    }

    pub(crate) const fn update(&mut self, byte: u8) {
        // Cursor is the _LAST_ victim, bump it first.
        self.cursor = self.cursor.wrapping_add(1);

        // Find a victim that _CAN_ be decremented.
        while self.freq[self.cursor as usize] <= 1 {
            self.cursor = self.cursor.wrapping_add(1);
        }

        // Just return if this is going to be a no-op.
        if self.cursor == byte {
            return;
        }

        // Update raw frequencies.
        self.freq[self.cursor as usize] -= 1;
        self.freq[byte as usize] += 1;

        // Update the tree.
        self.update_inner(self.cursor as usize, -1);
        self.update_inner(byte as usize, 1);
    }

    const fn update_inner(&mut self, mut i: usize, delta: i16) {
        // Fenwick trees are 1-based for more efficient index math.
        i += 1;

        while i <= NUM_SYMBOLS {
            self.tree[i] = self.tree[i].wrapping_add(delta);
            i += i & i.wrapping_neg();
        }
    }

    pub(crate) const fn prefix_sum(&self, byte: u8) -> u64 {
        self.prefix_sum_inner(byte as usize).cast_unsigned() as u64
    }

    const fn prefix_sum_inner(&self, mut i: usize) -> i16 {
        let mut sum = 0i16;

        while i > 0 {
            sum += self.tree[i];
            i -= i & i.wrapping_neg();
        }

        sum
    }

    #[allow(clippy::cast_possible_truncation)]
    pub(crate) const fn find(&self, target: u64) -> u8 {
        let mut target = target.cast_signed() as i16;
        let mut bit = NUM_SYMBOLS >> 1;
        let mut i = 0;

        while bit > 0 {
            let next = i + bit;

            if next < self.tree.len() && self.tree[next] <= target {
                target -= self.tree[next];
                i = next;
            }

            bit >>= 1;
        }

        i as u8
    }
}
