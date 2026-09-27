use num_traits::Float;

use std::num::NonZero;
use std::ops::{Deref, DerefMut};

use super::SampleBuffer;
use crate::types::{BlockSize, FilterWeights};

pub struct NoiseBuffer<F: Float>(SampleBuffer<F>);
impl<F: Float> NoiseBuffer<F> {
    pub fn new(weights: &FilterWeights<F>) -> Self {
        NoiseBuffer(SampleBuffer::new(weights.window_size().into()))
    }
}
impl<F: Float> Deref for NoiseBuffer<F> {
    type Target = SampleBuffer<F>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<F: Float> DerefMut for NoiseBuffer<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Noise reference buffer for block processing.
pub struct BlockNoiseBuffer<F: Float>(SampleBuffer<F>);
impl<F: Float> BlockNoiseBuffer<F> {
    /// Creates a buffer of length `window_size` + `block_size` - 1 for block processing.
    pub fn new(weights: &FilterWeights<F>, block_size: BlockSize) -> Self {
        #[allow(
            clippy::unwrap_used,
            clippy::missing_panics_doc,
            reason = "FilterWeights and BlockSize types ensure that capacity > 0"
        )]
        let capacity = NonZero::new(*weights.window_size() + *block_size - 1).unwrap();
        let buffer = SampleBuffer::new(capacity);

        BlockNoiseBuffer(buffer)
    }
}
impl<F: Float> Deref for BlockNoiseBuffer<F> {
    type Target = SampleBuffer<F>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<F: Float> DerefMut for BlockNoiseBuffer<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;
    use crate::test_utils::all_approx_equal;
    use crate::types::WindowSize;

    #[test]
    fn noise_buffer_init_to_zero() {
        let weights = FilterWeights::new(WindowSize::new(3).unwrap());

        let buffer = NoiseBuffer::new(&weights);
        assert!(all_approx_equal(buffer.iter(), [0.0; 3].iter()));
    }

    #[test]
    fn block_noise_buffer_init_to_zero() {
        let weights = FilterWeights::new(WindowSize::new(3).unwrap());

        let buffer = BlockNoiseBuffer::new(&weights, BlockSize::new(2).unwrap());
        assert!(all_approx_equal(buffer.iter(), [0.0; 4].iter()));
    }
}
