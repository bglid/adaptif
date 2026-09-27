use std::ops::{Deref, DerefMut};

use crate::types::signals::OutputSample;

use super::{BlockSize, SampleBuffer};

pub struct BlockError(SampleBuffer);
impl BlockError {
    pub fn new(block_size: BlockSize) -> Self {
        BlockError(SampleBuffer::new(block_size.into()))
    }

    pub fn push(&mut self, item: OutputSample) {
        self.0.push(*item);
    }
}
impl Deref for BlockError {
    type Target = SampleBuffer;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for BlockError {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;
    use crate::test_utils::all_approx_equal;

    #[test]
    fn error_buffer_init_to_zero() {
        let buffer = BlockError::new(BlockSize::new(2).unwrap());
        assert!(all_approx_equal(buffer.iter(), [0_f64; 2].iter()));
    }
}
