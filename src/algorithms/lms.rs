use crate::types::{FilterWeights, OutputSample, SampleBuffer};
use crate::{Error, Result};

use crate::algorithms::{Algorithm, BlockAlgorithm};

#[derive(Debug, Clone)]
#[allow(clippy::exhaustive_structs, reason = "No more fields have to be added")]
pub struct LeastMeanSquares {
    /// Step size for weight updates.
    mu: f64,
}
impl LeastMeanSquares {
    /// # Errors
    ///
    /// Returns an error if mu <= 0.0.
    pub fn new(mu: f64) -> Result<Self> {
        if mu > 0.0 {
            Ok(LeastMeanSquares { mu })
        } else {
            Err(Error::NonPositiveStepSize)
        }
    }
}
impl Algorithm for LeastMeanSquares {
    /// Updates the filter weights using the following equation:
    ///
    /// $w_{n+1} = \mu ``e_n`` ``x_n``$
    /// where $``e_n``$ is the scalar error for the current sample,
    /// and $``x_n``$ is a vector of length `window_size` of the
    /// most recent noise reference samples.
    fn update_step(
        &self,
        weights: &mut FilterWeights,
        error: OutputSample,
        noise_ref: &SampleBuffer,
    ) {
        for (w, x) in weights.iter_mut().zip(noise_ref.iter()) {
            *w += self.mu * (*error) * x;
        }
    }
}
impl BlockAlgorithm for LeastMeanSquares {
    /// Updates the filter weights using the following equation:
    ///
    /// $w_{n+1} = \mu ``X_n``^T ``e_n``$
    /// where $``X_n``$ is a matrix with shape `(block_size, window_size)`,
    /// and $``e_n``$ is a vector of length `block_size`.
    fn update_block(
        &self,
        weights: &mut FilterWeights,
        error: &SampleBuffer,
        noise_ref: &SampleBuffer,
    ) {
        // TODO: make sure the buffer access patterns share the same time ordering (i.e. either
        // oldest to newest or newest to oldest, but not both)
        for (n, w) in weights.iter_mut().enumerate() {
            let mut acc = 0_f64;

            #[allow(clippy::unwrap_used, reason = "TODO")]
            for (b, e) in error.iter().enumerate() {
                // This is equivalent to a matrix multiplication.
                // Since the noise references for the samples in the block overlap,
                // we can save space by keeping them in a linear array of length
                // `block_size + window_size - 1`.
                // Thus, instead of indexing with `b * window_size + n` like in
                // a (row-ordered) matrix, we use `n + b` to get the noise sample
                // for block index `b` in window `n`.

                acc += self.mu * e * noise_ref.get(n + b).unwrap();
            }
            *w += acc;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;
    use crate::{
        test_utils::{all_approx_equal, sample_buffer_from},
        types::{FilterWeights, WindowSize},
    };

    #[test]
    fn update_lms_1() {
        let lms = LeastMeanSquares::new(0.5).unwrap();
        let e_n = OutputSample(2.0);
        let x_n = sample_buffer_from(&[1.0, -1.0]);
        let expected = [1.0, -1.0];
        let mut weights = FilterWeights::zeros(WindowSize::new(2).unwrap());

        lms.update_step(&mut weights, e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn update_lms_2() {
        let lms = LeastMeanSquares::new(1.0).unwrap();
        let e_n = OutputSample(1.0);
        let x_n = sample_buffer_from(&[5.0, 2.0]);
        let expected = [5.0, 2.0];
        let mut weights = FilterWeights::zeros(WindowSize::new(2).unwrap());

        lms.update_step(&mut weights, e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn update_block_lms_1() {
        let lms = LeastMeanSquares::new(0.5).unwrap();
        // Because of the underlying queue implementation, the arrays here are ordered
        // from most to least recent sample
        let e_n = sample_buffer_from(&[5.0, -6.0, 7.0]);
        let x_n = sample_buffer_from(&[1.0, -2.0, 3.0, -4.0]);
        let expected = [19.0, -28.0];
        let mut weights = FilterWeights::zeros(WindowSize::new(2).unwrap());

        lms.update_block(&mut weights, &e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn update_block_lms_2() {
        let lms = LeastMeanSquares::new(1.0).unwrap();
        // Because of the underlying queue implementation, the arrays here are ordered
        // from most to least recent sample
        let e_n = sample_buffer_from(&[1.0, -1.0, 1.5]);
        let x_n = sample_buffer_from(&[1.0, -2.0, 3.0, -4.0, 5.0]);
        let expected = [7.5, -11.0, 14.5];
        let mut weights = FilterWeights::zeros(WindowSize::new(3).unwrap());

        lms.update_block(&mut weights, &e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn mu_range() {
        LeastMeanSquares::new(1.0).unwrap();
        LeastMeanSquares::new(f64::MAX).unwrap();

        assert!(matches!(
            LeastMeanSquares::new(0.0),
            Err(Error::NonPositiveStepSize)
        ));
        assert!(matches!(
            LeastMeanSquares::new(-1.0),
            Err(Error::NonPositiveStepSize)
        ));
    }
}
