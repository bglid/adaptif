use crate::types::buffers::NoiseWindowView;
use crate::types::{Float, WindowSize};
use crate::{Error, Result};

use crate::algorithms::Algorithm;

/// Normalized Least Mean Squares (NLMS) algorithm.
///
/// Updates the filter weights using the [normalized least mean squares] algorithm.
/// The NLMS algorithm uses a stochastic gradient descent approach similar to the LMS algorithm.
/// However, with the "pure" LMS algorithm, it can be difficult to choose a step size $\mu$
/// that isn't sensitive to the scaling of input $x(n)$.
///
/// The NLMS algorithm normalizes with the power of the input to solve this problem, providing
/// a more stable step size range.
///
/// [normalized least mean squares]: https://en.wikipedia.org/wiki/Least_mean_squares_filter#Normalized_least_mean_squares_filter_(NLMS)
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Nlms<F: Float> {
    /// Step size for weight updates.
    step_size: F,
    /// Regularization term to avoid division by zero.
    eps: F,
    /// Number of noise samples processed at a time.
    window_size: WindowSize,
}
impl<F: Float> Nlms<F> {
    /// # Errors
    ///
    /// Returns an error if `step_size <= 0.0`.
    /// Returns an error if `eps <= 0.0`.
    pub fn new(step_size: F, eps: F, window_size: WindowSize) -> Result<Self> {
        if step_size <= F::zero() {
            return Err(Error::NonPositiveStepSize);
        }

        if eps <= F::zero() {
            return Err(Error::NonPositiveEpsilon);
        }

        Ok(Nlms {
            step_size,
            eps,
            window_size,
        })
    }
}
#[allow(clippy::doc_paragraphs_missing_punctuation, reason = "maths")]
impl<F: Float> Algorithm<F> for Nlms<F> {
    /// Updates the filter weights using the NLMS algorithm.
    ///
    /// The weights are updated at every step according to:
    ///
    /// $$
    /// \mathbf{w}(n+1) = \mathbf{w}(n) +\mu e(n)
    /// \frac{\mathbf{x}(n)}
    /// {\epsilon + \|\|\mathbf{x}(n)\|\|^2}
    /// $$
    ///
    /// Where:
    /// - $\mathbf{w}(n)$ is the filter weight vector.
    /// - $μ$ is the adaptation step size.
    /// - $e(n)$ is the scalar error.
    /// - $\mathbf{x}(n)$ is the reference input vector.
    /// - $\epsilon$ is the normalization regularization term that prevents division by 0.
    /// - $\|\|\mathbf{x}(n)\|\|^2$ is the squared Euclidean norm of the input vector.
    fn update_step(&mut self, weights: &mut [F], error: F, noise_window: NoiseWindowView<F>) {
        let norm_squared = noise_window.iter().map(|x| x * x).sum();
        let step_size_normalized = self.step_size / (self.eps + norm_squared);

        for (w, x) in weights.iter_mut().zip(noise_window.iter()) {
            *w += step_size_normalized * error * x;
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
        types::buffers::NoiseBuffer,
        types::{BlockSize, FilterWeights, WindowSize},
    };

    #[test]
    fn update_nlms_1() {
        let window_size = WindowSize::new(2).unwrap();
        let block_size = BlockSize::new(1).unwrap();

        let mut nlms = Nlms::new(0.5, 1e-8, window_size).unwrap();
        let e_n = 2.0;
        let x_n = NoiseBuffer::from_slice(&[1.0, -1.0], window_size, block_size);
        let expected = [1.0 / (2.0 + nlms.eps), -1.0 / (2.0 + nlms.eps)];
        let mut weights = FilterWeights::new(window_size);

        nlms.update_step(&mut weights, e_n, x_n.window(0).unwrap());

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn update_nlms_2() {
        let window_size = WindowSize::new(2).unwrap();
        let block_size = BlockSize::new(1).unwrap();

        let mut nlms = Nlms::new(1.0, 1e-8, window_size).unwrap();
        let e_n = 1.0;
        let x_n = NoiseBuffer::from_slice(&[5.0, 2.0], window_size, block_size);
        let expected = [(5.0 / (29.0 + nlms.eps)), (2.0 / (29.0 + nlms.eps))];
        let mut weights = FilterWeights::new(window_size);

        nlms.update_step(&mut weights, e_n, x_n.window(0).unwrap());

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn mu_range() {
        let window_size = WindowSize::new(16).unwrap();

        Nlms::new(1.0, 1e-8, window_size).unwrap();
        Nlms::new(f64::MAX, 1e-8, window_size).unwrap();

        assert!(matches!(
            Nlms::new(0.0, 1e-8, window_size),
            Err(Error::NonPositiveStepSize)
        ));
        assert!(matches!(
            Nlms::new(-1.0, 1e-8, window_size),
            Err(Error::NonPositiveStepSize)
        ));
    }

    #[test]
    fn eps_range() {
        let window_size = WindowSize::new(16).unwrap();

        Nlms::new(1.0, 1e-8, window_size).unwrap();

        assert!(matches!(
            Nlms::new(1.0, 0.0, window_size),
            Err(Error::NonPositiveEpsilon)
        ));
        assert!(matches!(
            Nlms::new(1.0, -1.0, window_size),
            Err(Error::NonPositiveEpsilon)
        ));
    }

    #[test]
    fn window_size_works() {
        let window_size = WindowSize::new(16).unwrap();

        let nlms = Nlms::new(0.1, 1e-8, window_size).unwrap();

        assert_eq!(nlms.window_size(), window_size);
    }
}
