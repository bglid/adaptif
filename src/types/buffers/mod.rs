use std::collections::VecDeque;
use std::num::NonZeroUsize;

use super::BlockSize;

mod error;
pub use error::*;

mod noise;
pub use noise::*;

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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;
    use crate::test_utils::{all_approx_equal, noise_buffer_from};

    #[test]
    fn error_buffer_init_to_zero() {
        let buffer = BlockError::new(BlockSize::new(2).unwrap());
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
