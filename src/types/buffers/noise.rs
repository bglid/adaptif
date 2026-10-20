use std::num::NonZero;
use std::ops::{Deref, DerefMut};

use super::SampleBuffer;
use crate::types::{BlockSize, FilterWeights, Float, WindowSize};

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
#[derive(Debug, Clone, PartialEq)]
pub struct BlockNoiseBuffer<F: Float> {
    inner: SampleBuffer<F>,
    window_size: WindowSize,
    block_size: BlockSize,
}
impl<F: Float> BlockNoiseBuffer<F> {
    /// Creates a buffer of length `window_size` + `block_size` - 1 for block processing.
    pub fn new(weights: &FilterWeights<F>, block_size: BlockSize) -> Self {
        let window_size = weights.window_size();
        #[allow(
            clippy::unwrap_used,
            clippy::missing_panics_doc,
            reason = "FilterWeights and BlockSize types ensure that capacity > 0"
        )]
        let capacity = NonZero::new(*window_size + *block_size - 1).unwrap();
        let buffer = SampleBuffer::new(capacity);

        BlockNoiseBuffer {
            inner: buffer,
            window_size,
            block_size,
        }
    }

    pub fn get_window(&self, window_start: usize) -> Option<NoiseWindow<'_, F>> {
        if window_start >= *self.block_size {
            None
        } else {
            Some(NoiseWindow {
                buffer: self,
                start_idx: window_start,
            })
        }
    }
}
impl<F: Float> Deref for BlockNoiseBuffer<F> {
    type Target = SampleBuffer<F>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<F: Float> DerefMut for BlockNoiseBuffer<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[allow(
    clippy::len_without_is_empty,
    reason = "Buffer has a fixed size and can't be empty"
)]
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseWindow<'a, F: Float> {
    buffer: &'a BlockNoiseBuffer<F>,
    start_idx: usize,
}
impl<F: Float> NoiseWindow<'_, F> {
    pub fn get(&self, index: usize) -> Option<&F> {
        if index >= *self.buffer.window_size {
            None
        } else {
            self.buffer.get(self.start_idx + index)
        }
    }

    pub fn len(&self) -> usize {
        *self.buffer.window_size
    }

    pub fn iter(&self) -> impl Iterator<Item = F> {
        self.buffer
            .iter()
            .copied()
            .skip(self.start_idx)
            .take(*self.buffer.window_size)
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

    #[test]
    fn noise_window_indexing_works() {
        let window_size = 16;
        let block_size = 8;
        let noise = (0..(u32::try_from(window_size + block_size - 1)).unwrap())
            .map(f64::from)
            .collect::<Vec<f64>>();

        let buffer = {
            let weights = FilterWeights::new(WindowSize::new(window_size).unwrap());
            let mut buffer = BlockNoiseBuffer::new(&weights, BlockSize::new(block_size).unwrap());

            for x in noise {
                buffer.push(x);
            }

            buffer
        };

        assert_eq!(buffer.get_window(block_size), None);
        assert_eq!(buffer.get_window(block_size + 1), None);

        for window_start in [0, 3, block_size - 1] {
            let noise_window = buffer.get_window(window_start).unwrap();

            assert_eq!(noise_window.get(window_size), None);
            assert_eq!(noise_window.get(window_size + 1), None);

            for i in 0..window_size {
                assert_eq!(noise_window.get(i), buffer.get(window_start + i));
            }
        }
    }
}
