use crate::types::buffers::NoiseWindowView;
use crate::types::{Float, WindowSize};
use crate::{Error, Result};

use crate::algorithms::Algorithm;

/// Least mean squares algorithm.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Lms<F: Float> {
    /// Step size for weight updates.
    mu: F,
    /// Number of noise samples processed at a time.
    window_size: WindowSize,
}
impl<F: Float> Lms<F> {
    /// # Errors
    ///
    /// Returns an error if mu <= 0.0.
    pub fn new(mu: F, window_size: WindowSize) -> Result<Self> {
        if mu > F::zero() {
            Ok(Lms { mu, window_size })
        } else {
            Err(Error::NonPositiveStepSize)
        }
    }
}
impl<F: Float> Algorithm<F> for Lms<F> {
    /// Updates the filter weights using the following equation:
    ///
    /// $w_{n+1} = ``w_n`` + \mu ``e_n``^T ``X_n``$
    /// where $``X_n``$ is a matrix with shape `(block_size, window_size)`,
    /// and $``e_n``$ is a vector of length `block_size`.
    fn update_step(&mut self, weights: &mut [F], error: F, noise_window: NoiseWindowView<F>) {
        for (w, x) in weights.iter_mut().zip(noise_window.iter()) {
            *w += self.mu * error * x;
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
