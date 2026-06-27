//! Bit-prediction models and the shared [`Context`] they read.
//!
//! A model is queried once per bit ([`Model::predict`]) and then told the actual
//! bit ([`Model::update`]). Predictions are in the stretched (logit) domain so
//! the mixer can combine them directly.

pub(crate) mod context;
pub(crate) mod finder;
pub(crate) mod indirect;
#[cfg(feature = "arm")]
pub(crate) mod lstm;
#[cfg(test)]
mod lstm_spike;
pub(crate) mod match_model;
pub(crate) mod pretrained;
pub(crate) mod statemap;

/// Mutable per-stream prediction context shared by every model.
///
/// Holds the full finalized-byte history (so long-range models like the match
/// model can read arbitrarily far back), plus the partial current byte and a
/// 4-byte rolling window the low-order context models index directly.
#[derive(Debug)]
pub(crate) struct Context {
    history: Vec<u8>,
    /// Partial current byte: a leading-1 sentinel followed by the bits coded so far.
    pub(crate) c0: u32,
    /// Number of bits of the current byte already coded (`0..=7`).
    pub(crate) bpos: u8,
    /// The last four finalized bytes; the most recent is in the low 8 bits.
    pub(crate) c4: u32,
    /// Rolling hash of the current word's letters so far (`0` between words).
    /// A "letter" is `a..=z` only: the stream is post-fold, so `A..=Z` never
    /// means a letter — those bytes are dictionary codes and act as boundaries.
    pub(crate) word_hash: u64,
    /// Digit-run state for the numeric model: `num_pos` = digits seen so far in
    /// the current run (0 outside a run), `num_field` = the byte that preceded
    /// the run's first digit (the field tag, e.g. `>` after `<id>`).
    pub(crate) num_pos: u8,
    pub(crate) num_field: u8,
    /// Letters in the current word so far (0 between words) — a regime selector.
    pub(crate) word_pos: u8,
    /// Bytes since the last newline (column) — a line-position regime selector.
    pub(crate) col: u16,
}

/// Mixing multiplier for folding a letter into the rolling word hash.
const WORD_PRIME: u64 = 0x0100_0000_01b3;

impl Context {
    /// New, empty context with room reserved for `capacity` finalized bytes.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            history: Vec::with_capacity(capacity),
            c0: 1,
            bpos: 0,
            c4: 0,
            word_hash: 0,
            num_pos: 0,
            num_field: 0,
            word_pos: 0,
            col: 0,
        }
    }

    /// The finalized byte `i` positions back (`i` ≥ 1); `0` before that much
    /// history exists.
    pub(crate) fn byte_back(&self, i: usize) -> u8 {
        let n = self.history.len();
        if i <= n { self.history[n - i] } else { 0 }
    }

    /// All finalized bytes so far, in order. The most recent is the last element.
    pub(crate) fn history(&self) -> &[u8] {
        &self.history
    }

    /// Append one freshly-coded bit to the partial current byte.
    pub(crate) fn push_bit(&mut self, bit: u8) {
        self.c0 = (self.c0 << 1) | u32::from(bit);
        self.bpos += 1;
    }

    /// Finalize the current byte once all 8 bits are in, and reset for the next.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn push_byte(&mut self) {
        let b = self.c0 as u8; // low 8 bits are the byte; the sentinel is bit 8
        self.history.push(b);
        self.c4 = (self.c4 << 8) | u32::from(b);
        self.col = if b == b'\n' {
            0
        } else {
            self.col.saturating_add(1)
        };
        if b.is_ascii_digit() {
            if self.num_pos == 0 {
                self.num_field = self.byte_back(2); // the byte before this run
            }
            self.num_pos = self.num_pos.saturating_add(1);
        } else {
            self.num_pos = 0;
        }
        if b.is_ascii_lowercase() {
            self.word_hash = self
                .word_hash
                .wrapping_mul(WORD_PRIME)
                .wrapping_add(u64::from(b) + 1);
            self.word_pos = self.word_pos.saturating_add(1);
        } else {
            self.word_hash = 0;
            self.word_pos = 0;
        }
        self.c0 = 1;
        self.bpos = 0;
    }
}

/// A model that predicts the next bit from the [`Context`].
pub(crate) trait Model {
    /// Predict P(next bit == 1) in the stretched (logit) domain, clamped to
    /// roughly `[-2047, 2047]`. Called before the bit is known.
    fn predict(&mut self, ctx: &Context) -> i32;
    /// Observe the actual `bit`. `ctx` still reflects the pre-bit state.
    fn update(&mut self, ctx: &Context, bit: u8);
    /// Optional coarse state this model contributes as a mixer weight-set
    /// selector (e.g. the match model's current run-length bucket). `None` (the
    /// default) means the model offers no selector. Constant within a byte.
    fn selector(&self) -> Option<usize> {
        None
    }
}

/// Static-dispatch wrapper over the concrete model types. The production model
/// set is stored as `Vec<AnyModel>` rather than `Vec<Box<dyn Model>>`, so the
/// per-bit predict/update loops dispatch through a `match` the compiler can
/// inline — removing the ~66 vtable indirections per byte and letting the
/// independent per-model table loads overlap (matters most at enwik9 scale,
/// where each is a DRAM miss). `AnyModel` also `impl Model`, and `From` impls
/// exist for each variant, so the ablation helpers compose models unchanged.
pub(crate) enum AnyModel {
    Context(context::ContextModel),
    Indirect(indirect::IndirectModel),
    Match(match_model::MatchModel),
    // Boxed: the online-neural LSTM arm is far larger than the deterministic
    // variants, so an unboxed variant would bloat every `AnyModel` slot.
    #[cfg(feature = "arm")]
    Arm(Box<lstm::ArmModel>),
    // Frozen pretrained MLP byte-LM, shipped as a weight blob (paid as L(D)). A
    // nonlinear function of the raw context window, orthogonal to the per-context
    // deterministic models. Boxed for the same reason as the arm.
    Pretrained(Box<pretrained::PretrainedMlp>),
}

impl From<context::ContextModel> for AnyModel {
    fn from(m: context::ContextModel) -> Self {
        Self::Context(m)
    }
}
impl From<indirect::IndirectModel> for AnyModel {
    fn from(m: indirect::IndirectModel) -> Self {
        Self::Indirect(m)
    }
}
impl From<match_model::MatchModel> for AnyModel {
    fn from(m: match_model::MatchModel) -> Self {
        Self::Match(m)
    }
}
#[cfg(feature = "arm")]
impl From<lstm::ArmModel> for AnyModel {
    fn from(m: lstm::ArmModel) -> Self {
        Self::Arm(Box::new(m))
    }
}
impl From<pretrained::PretrainedMlp> for AnyModel {
    fn from(m: pretrained::PretrainedMlp) -> Self {
        Self::Pretrained(Box::new(m))
    }
}

impl Model for AnyModel {
    #[inline]
    fn predict(&mut self, ctx: &Context) -> i32 {
        match self {
            Self::Context(m) => m.predict(ctx),
            Self::Indirect(m) => m.predict(ctx),
            Self::Match(m) => m.predict(ctx),
            #[cfg(feature = "arm")]
            Self::Arm(m) => m.predict(ctx),
            Self::Pretrained(m) => m.predict(ctx),
        }
    }
    #[inline]
    fn update(&mut self, ctx: &Context, bit: u8) {
        match self {
            Self::Context(m) => m.update(ctx, bit),
            Self::Indirect(m) => m.update(ctx, bit),
            Self::Match(m) => m.update(ctx, bit),
            #[cfg(feature = "arm")]
            Self::Arm(m) => m.update(ctx, bit),
            Self::Pretrained(m) => m.update(ctx, bit),
        }
    }
    #[inline]
    fn selector(&self) -> Option<usize> {
        match self {
            Self::Context(m) => m.selector(),
            Self::Indirect(m) => m.selector(),
            Self::Match(m) => m.selector(),
            #[cfg(feature = "arm")]
            Self::Arm(m) => m.selector(),
            Self::Pretrained(m) => m.selector(),
        }
    }
}

impl AnyModel {
    /// The pretrained net's warming-head logit for this bit (a SEPARATE mixer
    /// input alongside the frozen logit), `None` for every other model and when
    /// the head is disabled. Valid after `predict` ran for the bit.
    #[inline]
    pub(crate) fn head_out(&self) -> Option<i32> {
        match self {
            Self::Pretrained(m) => m.head_out(),
            _ => None,
        }
    }

    /// Whether this model contributes a warming-head extra input (static — drives
    /// the codec's extra mixer-input slot reservation).
    pub(crate) fn has_head(&self) -> bool {
        matches!(self, Self::Pretrained(m) if m.has_head())
    }
}
