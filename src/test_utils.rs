#![allow(unused, reason = "Used in tests for other modules")]
#![allow(
    clippy::unwrap_used,
    clippy::unwrap_in_result,
    reason = "Only used in tests"
)]

use std::num::NonZero;

use crate::types::{FilterWeights, SampleBuffer, WindowSize};

pub fn approx_equal(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps
}

pub fn approx_equal_iter<'a, 'b, I, J>(a: I, b: J, eps: f64) -> bool
where
    I: Iterator<Item = &'a f64>,
    J: Iterator<Item = &'b f64>,
{
    a.zip(b).all(|(x, y)| approx_equal(*x, *y, eps))
}

pub fn all_approx_equal<'a, 'b, I, J>(a: I, b: J) -> bool
where
    I: ExactSizeIterator<Item = &'a f64>,
    J: ExactSizeIterator<Item = &'b f64>,
{
    if a.len() == b.len() {
        approx_equal_iter(a, b, 1e-6)
    } else {
        false
    }
}

pub fn sample_buffer_from(arr: &[f64]) -> SampleBuffer {
    let weights = FilterWeights::zeros(WindowSize::new(arr.len()).unwrap());
    let mut buffer = SampleBuffer::new(&weights);

    for val in arr {
        buffer.push(*val);
    }

    buffer
}
