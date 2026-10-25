use crate::types::buffers::NoiseWindow;
use crate::types::signals::OutputSample;
use crate::types::{FilterWeights, Float};
use crate::{Error, Result};

use crate::algorithms::Algorithm;

#[derive(Debug, Clone, PartialEq)]
/// Least mean squares algorithm.
pub struct Lms<F: Float> {
    /// Step size for weight updates.
    mu: F,
}
impl<F: Float> Lms<F> {
    /// # Errors
    ///
    /// Returns an error if mu <= 0.0.
    pub fn new(mu: F) -> Result<Self> {
        if mu > F::zero() {
            Ok(Lms { mu })
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
    fn update_step(
        &mut self,
        weights: &mut FilterWeights<F>,
        error: OutputSample<F>,
        noise_window: NoiseWindow<F>,
    ) {
        for (w, x) in weights.iter_mut().zip(noise_window.iter()) {
            *w += self.mu * *error * x;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;
    use crate::{
        test_utils::{all_approx_equal, block_noise_buffer_from, error_buffer_from},
        types::{BlockSize, FilterWeights, WindowSize},
    };

    #[test]
    fn update_lms_1() {
        let window_size = WindowSize::new(2).unwrap();
        let block_size = BlockSize::new(1).unwrap();

        let mut lms = Lms::new(0.5).unwrap();
        let e_n = error_buffer_from(&[2.0]);
        let x_n = block_noise_buffer_from(&[1.0, -1.0], window_size, block_size);
        let expected = [1.0, -1.0];
        let mut weights = FilterWeights::new(WindowSize::new(2).unwrap());

        lms.update_block(&mut weights, &e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn update_lms_2() {
        let window_size = WindowSize::new(2).unwrap();
        let block_size = BlockSize::new(1).unwrap();

        let mut lms = Lms::new(1.0).unwrap();
        let e_n = error_buffer_from(&[1.0]);
        let x_n = block_noise_buffer_from(&[5.0, 2.0], window_size, block_size);
        let expected = [5.0, 2.0];
        let mut weights = FilterWeights::new(window_size);

        lms.update_block(&mut weights, &e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn update_block_lms_1() {
        let window_size = WindowSize::new(2).unwrap();

        let mut lms = Lms::new(0.5).unwrap();
        let e_n = error_buffer_from(&[5.0, -6.0, 7.0]);

        let block_size = BlockSize::new(e_n.len()).unwrap();

        let x_n = block_noise_buffer_from(&[1.0, -2.0, 3.0, -4.0], window_size, block_size);
        let expected = [19.0, -28.0];
        let mut weights = FilterWeights::new(window_size);

        lms.update_block(&mut weights, &e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn update_block_lms_2() {
        let window_size = WindowSize::new(3).unwrap();

        let mut lms = Lms::new(1.0).unwrap();
        let e_n = error_buffer_from(&[1.0, -1.0, 1.5]);

        let block_size = BlockSize::new(e_n.len()).unwrap();
        let x_n = block_noise_buffer_from(&[1.0, -2.0, 3.0, -4.0, 5.0], window_size, block_size);
        let expected = [7.5, -11.0, 14.5];
        let mut weights = FilterWeights::new(window_size);

        lms.update_block(&mut weights, &e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn mu_range() {
        Lms::new(1.0).unwrap();
        Lms::new(f64::MAX).unwrap();

        assert!(matches!(Lms::new(0.0), Err(Error::NonPositiveStepSize)));
        assert!(matches!(Lms::new(-1.0), Err(Error::NonPositiveStepSize)));
    }
}
