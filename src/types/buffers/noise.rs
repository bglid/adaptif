use super::SampleBuffer;
use crate::types::{BlockSize, FilterWeights};

use std::num::NonZero;
use std::ops::{Deref, DerefMut};

pub struct NoiseBuffer(SampleBuffer);
impl NoiseBuffer {
    pub fn new(weights: &FilterWeights) -> Self {
        NoiseBuffer(SampleBuffer::new(weights.window_size().into()))
    }
}
impl Deref for NoiseBuffer {
    type Target = SampleBuffer;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for NoiseBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Noise reference buffer for block processing.
pub struct BlockNoiseBuffer(SampleBuffer);
impl BlockNoiseBuffer {
    /// Creates a buffer of length `window_size` + `block_size` - 1 for block processing.
    pub fn new(weights: &FilterWeights, block_size: BlockSize) -> Self {
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
impl Deref for BlockNoiseBuffer {
    type Target = SampleBuffer;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for BlockNoiseBuffer {
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
        assert!(all_approx_equal(buffer.iter(), [0_f64; 3].iter()));
    }

    #[test]
    fn block_noise_buffer_init_to_zero() {
        let weights = FilterWeights::new(WindowSize::new(3).unwrap());

        let buffer = BlockNoiseBuffer::new(&weights, BlockSize::new(2).unwrap());
        assert!(all_approx_equal(buffer.iter(), [0_f64; 4].iter()));
    }
}
