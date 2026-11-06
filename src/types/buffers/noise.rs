use std::num::NonZero;
use std::ops::{Deref, DerefMut};

use super::SampleBuffer;
use crate::types::signals::NoiseSample;
use crate::types::{BlockSize, FilterWeights, Float, WindowSize};

/// Noise reference buffer for block processing.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NoiseBuffer<F: Float> {
    inner: SampleBuffer<F>,
    window_size: WindowSize,
    block_size: BlockSize,
}
impl<F: Float> NoiseBuffer<F> {
    /// Creates a buffer of length `window_size` + `block_size` - 1 for block processing.
    pub fn new(weights: &FilterWeights<F>, block_size: BlockSize) -> Self {
        let window_size = weights.window_size();

        #[allow(clippy::missing_panics_doc, reason = "independent of fn input")]
        let capacity = NonZero::new(*window_size + *block_size - 1)
            .expect("FilterWeights and BlockSize types should ensure that capacity > 0");

        let buffer = SampleBuffer::new(capacity);

        NoiseBuffer {
            inner: buffer,
            window_size,
            block_size,
        }
    }

    pub fn window(&self, start_idx: usize) -> Option<NoiseWindow<'_, F>> {
        if start_idx >= *self.block_size {
            None
        } else {
            Some(NoiseWindow {
                buffer: self,
                start_idx,
            })
        }
    }

    pub fn push(&mut self, value: NoiseSample<F>) {
        self.inner.push(*value);
    }

    /// Create a noise buffer filled with the values in `arr`.
    /// Only to be used for setting up variables in unit tests.
    ///
    /// The samples in the buffer will be stored in the same order
    /// as they are in `arr`. This means that `arr` should be ordered from
    /// most to least recent sample, i.e. `arr[0]` is the most recent sample.
    #[cfg(test)]
    #[allow(clippy::missing_panics_doc, reason = "Testing only")]
    pub fn from_slice(arr: &[F], window_size: WindowSize, block_size: BlockSize) -> Self {
        assert_eq!(arr.len(), *window_size + *block_size - 1);

        let weights = FilterWeights::new(window_size);
        let mut buffer = NoiseBuffer::new(&weights, block_size);

        // The underlying buffer implementation pushes new samples to the front and pops
        // old samples from the back.
        // We therefore process `arr` in reverse so that the orders of the samples is preserved.
        for val in arr.iter().rev() {
            buffer.push(NoiseSample(*val));
        }

        buffer
    }
}
impl<F: Float> Deref for NoiseBuffer<F> {
    type Target = SampleBuffer<F>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<F: Float> DerefMut for NoiseBuffer<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[allow(
    clippy::len_without_is_empty,
    reason = "Buffer has a fixed size and can't be empty"
)]
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NoiseWindow<'a, F: Float> {
    buffer: &'a NoiseBuffer<F>,
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

        let buffer = NoiseBuffer::new(&weights, BlockSize::new(2).unwrap());
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
            let mut buffer = NoiseBuffer::new(&weights, BlockSize::new(block_size).unwrap());

            for x in noise {
                buffer.push(NoiseSample(x));
            }

            buffer
        };

        assert_eq!(buffer.window(block_size), None);
        assert_eq!(buffer.window(block_size + 1), None);

        for window_start in [0, 3, block_size - 1] {
            let noise_window = buffer.window(window_start).unwrap();

            assert_eq!(noise_window.get(window_size), None);
            assert_eq!(noise_window.get(window_size + 1), None);

            for i in 0..window_size {
                assert_eq!(noise_window.get(i), buffer.get(window_start + i));
            }
        }
    }
}
