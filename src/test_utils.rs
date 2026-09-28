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

pub fn noise_buffer_from<F>(arr: &[F]) -> NoiseBuffer<F>
where
    F: Float,
{
    let weights = FilterWeights::new(WindowSize::new(arr.len()).unwrap());
    let mut buffer = NoiseBuffer::new(&weights);

    for val in arr {
        buffer.push(*val);
    }

    buffer
}

pub fn block_noise_buffer_from<F>(arr: &[F]) -> BlockNoiseBuffer<F>
where
    F: Float,
{
    let weights = FilterWeights::new(WindowSize::new(arr.len()).unwrap());
    let mut buffer = BlockNoiseBuffer::new(&weights, BlockSize::new(1).unwrap());

    for val in arr {
        buffer.push(*val);
    }

    buffer
}

pub fn error_buffer_from<F>(arr: &[F]) -> BlockError<F>
where
    F: Float,
{
    let mut buffer = BlockError::new(BlockSize::new(arr.len()).unwrap());

    for val in arr {
        buffer.push(OutputSample(*val));
    }

    buffer
}
