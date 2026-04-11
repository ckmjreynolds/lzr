use static_assertions::const_assert;

const TOTAL: usize = 4_096;
const SYMBOLS: usize = 256;
const INITIAL_COUNT: usize = TOTAL / SYMBOLS;

const_assert!(TOTAL.is_power_of_two());
const_assert!(TOTAL == SYMBOLS * INITIAL_COUNT);

pub(crate) struct Model {
    counts: [u16; SYMBOLS],
    cursor: u8,
}

impl Model {
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) const fn new() -> Self {
        Self {
            counts: [INITIAL_COUNT as u16; SYMBOLS],
            cursor: 0,
        }
    }

    pub(crate) const fn update(&mut self, symbol: u8) {
        // Find a victim and decrement their count.
        if self.counts[self.cursor as usize] <= 1 || self.cursor == symbol {
            while self.counts[self.cursor as usize] <= 1 {
                self.cursor = self.cursor.wrapping_add(1);
            }
        }

        self.counts[self.cursor as usize] -= 1;

        // Now, we know that the increment here cannot overflow.
        self.counts[symbol as usize] += 1;
    }

    pub(crate) const fn count(&self, symbol: u8) -> u16 {
        self.counts[symbol as usize]
    }

    pub(crate) fn symbol_to_cumulative(&self, symbol: u8) -> u16 {
        let mut cumulative = 0u16;

        for i in 0..symbol {
            cumulative += self.counts[i as usize];
        }

        cumulative
    }

    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn cumulative_to_symbol(&self, value: u16) -> u8 {
        let mut cumulative = 0u16;

        for i in 0..SYMBOLS {
            cumulative += self.counts[i];
            if cumulative > value {
                return i as u8;
            }
        }

        unreachable!()
    }

    #[allow(clippy::cast_possible_truncation)]
    pub(crate) const fn total() -> u16 {
        TOTAL as u16
    }
}
