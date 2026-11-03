use std::ops::{Deref, DerefMut};

use crate::types::buffers::SampleBuffer;
use crate::types::signals::OutputSample;
use crate::types::{BlockSize, Float};

pub struct BlockError<F: Float> {
    inner: SampleBuffer<F>,
}
impl<F: Float> BlockError<F> {
    pub fn new(block_size: BlockSize) -> Self {
        BlockError {
            inner: SampleBuffer::new(block_size.into()),
        }
    }

    pub fn push(&mut self, item: OutputSample<F>) {
        self.inner.push(*item);
    }
    /// Create an error buffer filled with the values in `arr`.
    /// Only to be used for setting up variables in unit tests.
    ///
    /// The samples in the buffer will be stored in the same order
    /// as they are in `arr`. This means that `arr` should be ordered from
    /// most to least recent sample, i.e. `arr[0]` is the most recent sample.
    #[cfg(test)]
    #[allow(
        clippy::unwrap_used,
        clippy::missing_panics_doc,
        reason = "Testing only"
    )]
    pub fn from_slice(value: &[F]) -> Self {
        let mut buffer = BlockError::new(BlockSize::new(value.len()).unwrap());

        // The underlying buffer implementation pushes new samples to the front and pops
        // old samples from the back.
        // We therefore process `arr` in reverse so that the orders of the samples is preserved.
        for val in value.iter().rev() {
            buffer.push(OutputSample(*val));
        }

        buffer
    }
}
impl<F: Float> Deref for BlockError<F> {
    type Target = SampleBuffer<F>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<F: Float> DerefMut for BlockError<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;
    use crate::test_utils::all_approx_equal;

    #[test]
    fn block_error_init_to_zero() {
        let buffer = BlockError::new(BlockSize::new(2).unwrap());
        assert!(all_approx_equal(buffer.iter(), [0.0; 2].iter()));
    }
}
