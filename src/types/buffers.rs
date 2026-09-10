use std::collections::VecDeque;
use std::num::{NonZero, NonZeroUsize};
use std::ops::{Deref, DerefMut};

use crate::types::FilterWeights;

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

/// Fixed-size ring buffer for processing samples.
/// Functions must ensure that `samples.len()` is the same before and after function calls
/// to enforce the invariant it is the same as the number of weights, and equal to the filter's window size.
#[allow(
    clippy::len_without_is_empty,
    reason = "Buffer has a fixed size and can't be empty"
)]
#[derive(Debug, Clone)]
pub struct SampleBuffer {
    samples: VecDeque<f64>,
    capacity: NonZeroUsize,
}
impl SampleBuffer {
    // We get the capacity directly from the weights to guarantee
    // that the buffer length and the number of weights are the same.
    pub fn new(capacity: NonZeroUsize) -> Self {
        SampleBuffer {
            samples: std::iter::repeat_n(0.0, capacity.into()).collect(),
            capacity,
        }
    }

    pub fn push(&mut self, sample: f64) {
        // have to bind this because pyo3 adds extra impl of PartialEq
        let capacity: usize = self.capacity.into();

        if self.samples.len() == capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    pub fn get(&self, index: usize) -> Option<&f64> {
        self.samples.get(index)
    }

    pub fn len(&self) -> usize {
        self.capacity.into()
    }

    pub fn iter(&self) -> SampleIter<'_> {
        SampleIter {
            buffer: self,
            next_idx: 0,
        }
    }
}
impl<'a> IntoIterator for &'a SampleBuffer {
    type Item = &'a f64;
    type IntoIter = SampleIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct SampleIter<'a> {
    buffer: &'a SampleBuffer,
    next_idx: usize,
}
impl<'a> Iterator for SampleIter<'a> {
    type Item = &'a f64;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.buffer.get(self.next_idx);
        self.next_idx += 1;
        item
    }
}
impl ExactSizeIterator for SampleIter<'_> {
    fn len(&self) -> usize {
        self.buffer.len()
    }
}

pub struct NoiseBuffer(SampleBuffer);
impl NoiseBuffer {
    pub fn new(weights: &FilterWeights) -> Self {
        #[allow(
            clippy::unwrap_used,
            clippy::missing_panics_doc,
            reason = "FilterWeights::new() checks that the number of weights is greater than 0"
        )]
        NoiseBuffer(SampleBuffer::new(NonZero::new(weights.len()).unwrap()))
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
        let capacity = NonZero::new(weights.len() + *block_size - 1).unwrap();
        BlockNoiseBuffer(SampleBuffer::new(capacity))
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

pub struct ErrorBuffer(SampleBuffer);
impl ErrorBuffer {
    pub fn new(block_size: BlockSize) -> Self {
        #[allow(
            clippy::unwrap_used,
            clippy::missing_panics_doc,
            reason = "BlockSize type cannot be zero"
        )]
        ErrorBuffer(SampleBuffer::new(NonZero::new(*block_size).unwrap()))
    }
}
impl Deref for ErrorBuffer {
    type Target = SampleBuffer;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for ErrorBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;
    use crate::test_utils::{all_approx_equal, noise_buffer_from};

    #[test]
    fn noise_buffer_init_to_zero() {
        let weights = FilterWeights::new(WindowSize::new(3).unwrap(), 0.0, 5e-5).unwrap();

        let buffer = NoiseBuffer::new(&weights);
        assert!(all_approx_equal(buffer.iter(), [0_f64; 3].iter()));
    }

    #[test]
    fn block_noise_buffer_init_to_zero() {
        let weights = FilterWeights::new(WindowSize::new(3).unwrap(), 0.0, 5e-5).unwrap();

        let buffer = BlockNoiseBuffer::new(&weights, BlockSize::new(2).unwrap());
        assert!(all_approx_equal(buffer.iter(), [0_f64; 4].iter()));
    }

    #[test]
    fn error_buffer_init_to_zero() {
        let buffer = ErrorBuffer::new(BlockSize::new(2).unwrap());
        assert!(all_approx_equal(buffer.iter(), [0_f64; 2].iter()));
    }

    #[test]
    fn push() {
        let mut buffer = noise_buffer_from(&[0.0; 3]);

        buffer.push(1.0);
        assert_eq!(buffer.len(), 3);
        assert!(all_approx_equal(buffer.iter(), [0.0, 0.0, 1.0].iter()));

        buffer.push(2.0);
        assert_eq!(buffer.len(), 3);
        assert!(all_approx_equal(buffer.iter(), [0.0, 1.0, 2.0].iter()));
    }

    #[test]
    fn buffer_size_invariant() {
        let mut buffer = noise_buffer_from(&[0.0; 3]);

        buffer.push(1.0);
        buffer.push(2.0);
        buffer.push(3.0);

        assert_eq!(buffer.len(), 3);
        assert!(all_approx_equal(buffer.iter(), [1.0, 2.0, 3.0].iter()));

        buffer.push(4.0);

        assert_eq!(buffer.len(), 3);
        assert!(all_approx_equal(buffer.iter(), [2.0, 3.0, 4.0].iter()));
    }

    #[test]
    fn get() {
        let mut buffer = noise_buffer_from(&[0.0; 3]);

        buffer.push(1.0);
        buffer.push(2.0);
        buffer.push(3.0);

        assert_eq!(buffer.get(0), Some(&1.0));
        assert_eq!(buffer.get(1), Some(&2.0));
        assert_eq!(buffer.get(2), Some(&3.0));
        assert_eq!(buffer.get(3), None);
    }
}
