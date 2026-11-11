use std::ops::Deref;
use std::{fmt::Display, num::NonZero};

use crate::{Error, Result};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
/// The number of past noise reference samples used when processing
/// each input sample.
/// Also determines the number of weights used by the filter.
pub struct WindowSize {
    inner: usize,
}
impl WindowSize {
    /// Creates a new `WindowSize` from a `usize`.
    /// The window size must be greater than zero.
    ///
    /// # Errors
    ///
    /// Returns an error if `window_size == 0`.
    pub fn new(window_size: usize) -> Result<Self> {
        if window_size == 0 {
            Err(Error::WindowSizeZero)
        } else {
            Ok(WindowSize { inner: window_size })
        }
    }
}
impl Deref for WindowSize {
    type Target = usize;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl From<WindowSize> for NonZero<usize> {
    fn from(value: WindowSize) -> Self {
        NonZero::new(*value).expect("WindowSize should be guaranteed non-zero.")
    }
}
impl Display for WindowSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}
