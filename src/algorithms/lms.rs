use crate::types::buffers::NoiseWindowView;
use crate::types::{Float, WindowSize};
use crate::{Error, Result};

use crate::algorithms::Algorithm;

/// Least Mean Squares (LMS) adaptive filter.
///
/// Updates the filter weights using the [least mean squares] algorithm.
/// The LMS algorithm uses stochastic gradient descent to minimize
/// the mean squared error between the reference input signal
/// and the filter output.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Lms<F: Float> {
    /// Step size for weight updates.
    step_size: F,
    /// Number of samples processed at a time.
    window_size: WindowSize,
}
impl<F: Float> Lms<F> {
    /// # Errors
    ///
    /// Returns an error if `step_size <= 0.0`.
    pub fn new(step_size: F, window_size: WindowSize) -> Result<Self> {
        if step_size > F::zero() {
            Ok(Lms {
                step_size,
                window_size,
            })
        } else {
            Err(Error::NonPositiveStepSize)
        }
    }
}
#[allow(clippy::doc_paragraphs_missing_punctuation, reason = "maths")]
impl<F: Float> Algorithm<F> for Lms<F> {
    /// Updates the filter weights using the LMS algorithm.
    ///
    /// The weights are updated at each sample according to:
    ///
    /// $$\mathbf{w}(n+1) = \mathbf{w}(n) + \mu e(n)\mathbf{x}(n)$$
    ///
    /// Where:
    /// - $\mathbf{w}(n)$ is the filter weight vector.
    /// - $μ$ is the adaptation step size.
    /// - $e(n)$ is the scalar error.
    /// - $\mathbf{x}(n)$ is the reference input vector.
    fn update_step(&mut self, weights: &mut [F], error: F, noise_window: NoiseWindowView<F>) {
        for (w, x) in weights.iter_mut().zip(noise_window.iter()) {
            *w += self.step_size * error * x;
        }
    }

    fn window_size(&self) -> WindowSize {
        self.window_size
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;
    use crate::{
        test_utils::all_approx_equal,
        types::{BlockSize, FilterWeights, WindowSize, buffers::NoiseBuffer},
    };

    #[test]
    fn update_lms_1() {
        let window_size = WindowSize::new(2).unwrap();
        let block_size = BlockSize::new(1).unwrap();

        let mut lms = Lms::new(0.5, window_size).unwrap();
        let e_n = 2.0;
        let x_n = NoiseBuffer::from_slice(&[1.0, -1.0], window_size, block_size);
        let expected = [1.0, -1.0];
        let mut weights = FilterWeights::new(window_size);

        lms.update_step(&mut weights, e_n, x_n.window(0).unwrap());

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn update_lms_2() {
        let window_size = WindowSize::new(2).unwrap();
        let block_size = BlockSize::new(1).unwrap();

        let mut lms = Lms::new(1.0, window_size).unwrap();
        let e_n = 1.0;
        let x_n = NoiseBuffer::from_slice(&[5.0, 2.0], window_size, block_size);
        let expected = [5.0, 2.0];
        let mut weights = FilterWeights::new(window_size);

        lms.update_step(&mut weights, e_n, x_n.window(0).unwrap());

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn mu_range() {
        let window_size = WindowSize::new(16).unwrap();

        Lms::new(1.0, window_size).unwrap();
        Lms::new(f64::MAX, window_size).unwrap();

        assert!(matches!(
            Lms::new(0.0, window_size),
            Err(Error::NonPositiveStepSize)
        ));
        assert!(matches!(
            Lms::new(-1.0, window_size),
            Err(Error::NonPositiveStepSize)
        ));
    }

    #[test]
    fn window_size_works() {
        let window_size = WindowSize::new(16).unwrap();

        let lms = Lms::new(0.1, window_size).unwrap();

        assert_eq!(lms.window_size(), window_size);
    }
}
