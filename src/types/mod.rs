#![allow(
    clippy::exhaustive_structs,
    reason = "Simple wrappers for primitives, so adding fields is highly unlikely."
)]

mod filter_weights;
pub use filter_weights::FilterWeights;

pub mod buffers;
pub mod signals;

use std::{num::NonZero, ops::Deref};

use crate::error::{Error, Result};

// TODO: use pub(crate) to limit public exports to only the types needed for the public API

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockSize(usize);
impl BlockSize {
    /// # Errors
    ///
    /// Returns an error if `block_size == 0`.
    pub fn new(block_size: usize) -> Result<Self> {
        if block_size == 0 {
            Err(Error::BlockSizeZero)
        } else {
            Ok(BlockSize(block_size))
        }
    }
}
impl Deref for BlockSize {
    type Target = usize;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl From<BlockSize> for NonZero<usize> {
    fn from(value: BlockSize) -> Self {
        #[allow(clippy::unwrap_used, reason = "BlockSize is guaranteed non-zero.")]
        NonZero::new(*value).unwrap()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NoiseEstimate(pub f64);
impl Deref for NoiseEstimate {
    type Target = f64;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
