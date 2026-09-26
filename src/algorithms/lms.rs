use crate::types::FilterWeights;
use crate::types::buffers::{BlockNoiseBuffer, ErrorBuffer, NoiseBuffer};
use crate::types::signals::OutputSample;
use crate::{Error, Result};

use crate::algorithms::{Algorithm, BlockAlgorithm};

#[derive(Debug, Clone, PartialEq)]
/// Least mean squares algorithm.
pub struct Lms {
    /// Step size for weight updates.
    mu: f64,
}
impl Lms {
    /// # Errors
    ///
    /// Returns an error if mu <= 0.0.
    pub fn new(mu: f64) -> Result<Self> {
        if mu > 0.0 {
            Ok(Lms { mu })
        } else {
            Err(Error::NonPositiveStepSize)
        }
    }
}
impl Algorithm for Lms {
    /// Updates the filter weights using the following equation:
    ///
    /// $w_{n+1} = \mu ``e_n`` ``x_n``$
    /// where $``e_n``$ is the scalar error for the current sample,
    /// and $``x_n``$ is a vector of length `window_size` of the
    /// most recent noise reference samples.
    fn update_step(
        &mut self,
        weights: &mut FilterWeights,
        error: OutputSample,
        noise_ref: &NoiseBuffer,
    ) {
        for (w, x) in weights.iter_mut().zip(noise_ref.iter()) {
            *w += self.mu * (*error) * x;
        }
    }
}
impl BlockAlgorithm for Lms {
    /// Updates the filter weights using the following equation:
    ///
    /// $w_{n+1} = \mu ``X_n``^T ``e_n``$
    /// where $``X_n``$ is a matrix with shape `(block_size, window_size)`,
    /// and $``e_n``$ is a vector of length `block_size`.
    fn update_block(
        &self,
        weights: &mut FilterWeights,
        error: &ErrorBuffer,
        noise_ref: &BlockNoiseBuffer,
    ) {
        for (n, w) in weights.iter_mut().enumerate() {
            let mut acc = 0_f64;

            #[allow(
                clippy::unwrap_used,
                reason = "BlockNoiseBuffer has length `window_size + block_size - 1`.
                This means the highest valid index is `window_size + block_size - 2`.
                The max values for `n` and `b` are `window_size - 1` and `block_size - 1` respectively.
                `(window_size - 1) + (block_size - 1) == window_size + block_size - 2`"
            )]
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
        test_utils::{
            all_approx_equal, block_noise_buffer_from, error_buffer_from, noise_buffer_from,
        },
        types::{FilterWeights, WindowSize},
    };

    #[test]
    fn update_lms_1() {
        let mut lms = Lms::new(0.5).unwrap();
        let e_n = OutputSample(2.0);
        let x_n = noise_buffer_from(&[1.0, -1.0]);
        let expected = [1.0, -1.0];
        let mut weights = FilterWeights::new(WindowSize::new(2).unwrap());

        lms.update_step(&mut weights, e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn update_lms_2() {
        let mut lms = Lms::new(1.0).unwrap();
        let e_n = OutputSample(1.0);
        let x_n = noise_buffer_from(&[5.0, 2.0]);
        let expected = [5.0, 2.0];
        let mut weights = FilterWeights::new(WindowSize::new(2).unwrap());

        lms.update_step(&mut weights, e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn update_block_lms_1() {
        let lms = Lms::new(0.5).unwrap();
        // Because of the underlying queue implementation, the arrays here are ordered
        // from most to least recent sample
        let e_n = error_buffer_from(&[5.0, -6.0, 7.0]);
        let x_n = block_noise_buffer_from(&[1.0, -2.0, 3.0, -4.0]);
        let expected = [19.0, -28.0];
        let mut weights = FilterWeights::new(WindowSize::new(2).unwrap());

        lms.update_block(&mut weights, &e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn update_block_lms_2() {
        let lms = Lms::new(1.0).unwrap();
        // Because of the underlying queue implementation, the arrays here are ordered
        // from most to least recent sample
        let e_n = error_buffer_from(&[1.0, -1.0, 1.5]);
        let x_n = block_noise_buffer_from(&[1.0, -2.0, 3.0, -4.0, 5.0]);
        let expected = [7.5, -11.0, 14.5];
        let mut weights = FilterWeights::new(WindowSize::new(3).unwrap());

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
