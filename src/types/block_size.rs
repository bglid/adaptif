use std::num::NonZero;
use std::ops::Deref;

use crate::{Error, Result};

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
