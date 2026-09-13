use crate::types::{FilterWeights, OutputSample, SampleBuffer};
use crate::{Error, Result};

use crate::algorithms::Algorithm;

#[derive(Debug, Clone)]
#[allow(clippy::exhaustive_structs, reason = "No more fields have to be added")]
pub struct LeastMeanSquares {
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
    fn update_step(
        &mut self,
        weights: &mut FilterWeights,
        error: OutputSample,
        noise_ref: &SampleBuffer,
    ) {
        for (w, x) in weights.iter_mut().zip(noise_ref.iter()) {
            *w += self.mu * (*error) * x;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;
    use crate::{
        test_utils::{approx_equal, sample_buffer_from},
        types::FilterWeights,
    };
    use std::num::NonZero;

    #[test]
    fn update_lms_1() {
        let mut lms = LeastMeanSquares::new(0.5).unwrap();
        let e_n = OutputSample(2.0);
        let x_n = sample_buffer_from(&[1.0, -1.0]);
        let expected = [1.0, -1.0];
        let mut weights = FilterWeights::zeros(NonZero::new(2).unwrap());

        lms.update_step(&mut weights, e_n, &x_n);

        let output_correct = weights
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| approx_equal(*a, *b, 1e-6));
        assert!(output_correct);
    }

    #[test]
    fn update_lms_2() {
        let mut lms = LeastMeanSquares::new(1.0).unwrap();
        let e_n = OutputSample(1.0);
        let x_n = sample_buffer_from(&[5.0, 2.0]);
        let expected = [5.0, 2.0];
        let mut weights = FilterWeights::zeros(NonZero::new(2).unwrap());

        lms.update_step(&mut weights, e_n, &x_n);

        let output_correct = weights
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| approx_equal(*a, *b, 1e-6));
        assert!(output_correct);
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
