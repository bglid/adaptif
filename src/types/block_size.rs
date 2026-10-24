use std::num::NonZero;
use std::ops::Deref;

use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockSize {
    inner: usize,
}
impl BlockSize {
    /// # Errors
    ///
    /// Returns an error if `block_size == 0`.
    pub fn new(block_size: usize) -> Result<Self> {
        if block_size == 0 {
            Err(Error::BlockSizeZero)
        } else {
            Ok(BlockSize { inner: block_size })
        }
    }
}
impl Deref for BlockSize {
    type Target = usize;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl From<BlockSize> for NonZero<usize> {
    fn from(value: BlockSize) -> Self {
        NonZero::new(*value).expect("BlockSize should be guaranteed non-zero")
    }
}
