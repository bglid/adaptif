use std::ops::{Deref, DerefMut};

use crate::types::buffers::SampleBuffer;
use crate::types::signals::OutputSample;
use crate::types::{BlockSize, Float};

pub struct BlockError<F: Float>(SampleBuffer<F>);
impl<F: Float> BlockError<F> {
    pub fn new(block_size: BlockSize) -> Self {
        BlockError(SampleBuffer::new(block_size.into()))
    }

    pub fn push(&mut self, item: OutputSample<F>) {
        self.0.push(*item);
    }
}
impl<F: Float> Deref for BlockError<F> {
    type Target = SampleBuffer<F>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<F: Float> DerefMut for BlockError<F> {
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
        assert!(all_approx_equal(buffer.iter(), [0.0; 2].iter()));
    }
}
