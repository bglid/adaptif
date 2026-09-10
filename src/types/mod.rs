#![allow(
    clippy::exhaustive_structs,
    reason = "Simple wrappers for primitives, so adding fields is highly unlikely."
)]

mod filter_weights;
pub use filter_weights::FilterWeights;

pub mod buffers;
pub mod signals;

use std::ops::Deref;

#[derive(Debug, Clone, Copy)]
pub struct WindowSize(usize);
impl WindowSize {
    pub fn new(window_size: usize) -> Option<Self> {
        if window_size == 0 {
            None
        } else {
            Some(WindowSize(window_size))
        }
    }
}
impl Deref for WindowSize {
    type Target = usize;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BlockSize(usize);
impl BlockSize {
    pub fn new(block_size: usize) -> Option<Self> {
        if block_size == 0 {
            None
        } else {
            Some(BlockSize(block_size))
        }
    }
}
impl Deref for BlockSize {
    type Target = usize;
    fn deref(&self) -> &Self::Target {
        &self.0
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
