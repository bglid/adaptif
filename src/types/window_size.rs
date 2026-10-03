use std::num::NonZero;
use std::ops::Deref;

use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowSize(usize);
impl WindowSize {
    /// # Errors
    ///
    /// Returns an error if `window_size == 0`.
    pub fn new(window_size: usize) -> Result<Self> {
        if window_size == 0 {
            Err(Error::WindowSizeZero)
        } else {
            Ok(WindowSize(window_size))
        }
    }
}
impl Deref for WindowSize {
    type Target = usize;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl From<WindowSize> for NonZero<usize> {
    fn from(value: WindowSize) -> Self {
        #[allow(clippy::unwrap_used, reason = "WindowSize is guaranteed non-zero.")]
        NonZero::new(*value).unwrap()
    }
}
