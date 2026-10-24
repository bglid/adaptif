#![allow(unused, reason = "Used in tests for other modules")]
#![allow(
    clippy::unwrap_used,
    clippy::unwrap_in_result,
    reason = "Only used in tests"
)]

use std::num::NonZero;

use crate::types::buffers::{BlockError, BlockNoiseBuffer, NoiseBuffer};
use crate::types::signals::OutputSample;
use crate::types::{BlockSize, FilterWeights, Float, WindowSize};

pub fn approx_equal<F>(a: F, b: F, eps: F) -> bool
where
    F: Float,
{
    (a - b).abs() < eps
}

pub fn approx_equal_iter<'a, 'b, I, J, F>(a: I, b: J, eps: F) -> bool
where
    I: Iterator<Item = &'a F>,
    J: Iterator<Item = &'b F>,
    F: Float + 'a + 'b,
{
    a.zip(b).all(|(x, y)| approx_equal(*x, *y, eps))
}

pub fn all_approx_equal<'a, 'b, I, J, F>(a: I, b: J) -> bool
where
    I: ExactSizeIterator<Item = &'a F>,
    J: ExactSizeIterator<Item = &'b F>,
    F: Float + 'a + 'b,
{
    if a.len() == b.len() {
        approx_equal_iter(a, b, F::from(1e-6).unwrap())
    } else {
        false
    }
}

/// Create a noise buffer filled with the values in `arr`.
///
/// The samples in the buffer will be stored in the same order
/// as they are in `arr`. This means that `arr` should be ordered from
/// most to least recent sample, i.e. `arr[0]` is the most recent sample.
pub fn block_noise_buffer_from<F>(
    arr: &[F],
    window_size: WindowSize,
    block_size: BlockSize,
) -> BlockNoiseBuffer<F>
where
    F: Float,
{
    assert_eq!(arr.len(), *window_size + *block_size - 1);

    let weights = FilterWeights::new(window_size);
    let mut buffer = BlockNoiseBuffer::new(&weights, block_size);

    // The underlying buffer implementation pushes new samples to the front and pops
    // old samples from the back.
    // We therefore process `arr` in reverse so that the orders of the samples is preserved.
    for val in arr.iter().rev() {
        buffer.push(*val);
    }

    buffer
}

/// Create an error buffer filled with the values in `arr`.
///
/// The samples in the buffer will be stored in the same order
/// as they are in `arr`. This means that `arr` should be ordered from
/// most to least recent sample, i.e. `arr[0]` is the most recent sample.
pub fn error_buffer_from<F>(arr: &[F]) -> BlockError<F>
where
    F: Float,
{
    let mut buffer = BlockError::new(BlockSize::new(arr.len()).unwrap());

    // The underlying buffer implementation pushes new samples to the front and pops
    // old samples from the back.
    // We therefore process `arr` in reverse so that the orders of the samples is preserved.
    for val in arr.iter().rev() {
        buffer.push(OutputSample(*val));
    }

    buffer
}
