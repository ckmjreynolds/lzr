//! Encoder configuration options.
//!
//! # Examples
//!
//! ```
//! use lzr::options::EncodeOptions;
//!
//! let opts = EncodeOptions::new().level(6).threads(4);
//! ```

use rayon::current_num_threads;

/// Minimum compression level.
pub const MIN_LEVEL: usize = 1;

/// Maximum compression level.
pub const MAX_LEVEL: usize = 9;

/// Default compression level.
pub const DEFAULT_LEVEL: usize = 9;

/// Options for controlling the LZR encoder.
///
/// Use the builder methods to customize compression behavior.
///
/// # Examples
///
/// ```
/// use lzr::options::EncodeOptions;
///
/// let opts = EncodeOptions::new().level(6).threads(4);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct EncodeOptions {
    level: usize,
    threads: usize,
    verbose: bool,
}

impl EncodeOptions {
    /// Creates encoder options with default settings (level 9, auto threads).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            level: DEFAULT_LEVEL,
            threads: 0,
            verbose: false,
        }
    }

    /// Sets the compression level (1–9).
    ///
    /// Level 1 is fastest; level 9 provides the best compression ratio.
    #[must_use]
    pub fn level(mut self, level: usize) -> Self {
        self.level = level.clamp(1, 9);
        self
    }

    /// Sets the number of threads for compression.
    ///
    /// A value of `0` means auto-detect based on available CPU cores.
    #[must_use]
    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = threads.clamp(0, current_num_threads());
        self
    }

    /// Enables verbose diagnostic output to stderr.
    #[must_use]
    pub const fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Returns the compression level.
    pub(crate) const fn get_level(&self) -> usize {
        self.level
    }

    /// Returns the thread count.
    pub(crate) const fn get_threads(&self) -> usize {
        self.threads
    }

    /// Returns whether verbose output is enabled.
    pub(crate) const fn get_verbose(&self) -> bool {
        self.verbose
    }
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self::new()
    }
}
