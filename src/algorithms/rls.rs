use crate::types::{FilterWeights, OutputSample, SampleBuffer};
use crate::{Error, Result};

use crate::algorithms::Algorithm;

#[derive(Debug, Clone)]
#[allow(clippy::exhaustive_structs, reason = "No more fields have to be added")]
pub struct RecursiveLeastSquares {
    /// Forgetting factor for weight updates.
    #[allow(
        dead_code,
        reason = "Temporary just for committing and sharing current work"
    )]
    lambda: f64,
    /// Delta scalar value used for initalizing `p_matrix`.
    delta: f64,
    /// Inverse correlation matrix, referred to as P[n], for RLS updates.
    p_matrix: Option<Vec<f64>>,
}

impl RecursiveLeastSquares {
    /// # Errors
    ///
    /// Returns an error if lambda <= 0.0 or > 1.0.
    /// Returns an error if delta <= 0.0.
    pub fn new(lambda: f64, delta: f64) -> Result<Self> {
        if !(lambda > 0.0 && lambda <= 1.0) {
            return Err(Error::IncorrectLambdaRange);
        }

        if delta <= 0.0 {
            return Err(Error::NonPositiveDelta);
        }

        Ok(RecursiveLeastSquares {
            lambda,
            delta,
            p_matrix: None,
        })
    }

    fn initial_p_matrix(&self, n: usize) -> Vec<f64> {
        let mut p = vec![0.0; n * n];

        for i in 0..n {
            if let Some(elem) = p.get_mut(i * n + i) {
                *elem = self.delta;
            }
        }
        p
    }

    // fn calculate_k(&self, noise_ref: &SampleBuffer) -> Vec<f64> {}
}

impl Algorithm for RecursiveLeastSquares {
    /// Updates the filter weights using the following equation:
    ///
    fn update_step(
        &mut self,
        weights: &mut FilterWeights,
        _error: OutputSample,
        _noise_ref: &SampleBuffer,
    ) {
        // Updates p_matrix on first iteration once n is known
        if self.p_matrix.is_none() {
            self.p_matrix = Some(self.initial_p_matrix(weights.len()));
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;
    use crate::{
        test_utils::{all_approx_equal, sample_buffer_from},
        types::FilterWeights,
    };
    use std::num::NonZero;

    #[test]
    fn init_p_matrix_works() {
        let n = 3;
        let expected = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        let rls = RecursiveLeastSquares::new(0.5, 10.0).unwrap();
        let p_matrix = rls.initial_p_matrix(n);
        assert_eq!(p_matrix, expected);
    }

    #[test]
    fn update_rls_1() {
        let mut rls = RecursiveLeastSquares::new(0.5, 1.0).unwrap();
        let e_n = OutputSample(2.0);
        let x_n = sample_buffer_from(&[1.0, -1.0]);
        let expected = [0.8, -0.8];
        let mut weights = FilterWeights::zeros(NonZero::new(2).unwrap());

        rls.update_step(&mut weights, e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn update_rls_2() {
        let mut rls = RecursiveLeastSquares::new(1.0, 1.0).unwrap();
        let e_n = OutputSample(1.0);
        let x_n = sample_buffer_from(&[5.0, 2.0]);
        let expected = [5.0 / 30.0, 2.0 / 30.0];
        let mut weights = FilterWeights::zeros(NonZero::new(2).unwrap());

        rls.update_step(&mut weights, e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn lambda_range() {
        RecursiveLeastSquares::new(0.01, 3.0).unwrap();
        RecursiveLeastSquares::new(1.0, 3.0).unwrap();

        assert!(matches!(
            RecursiveLeastSquares::new(0.0, 1.0),
            Err(Error::IncorrectLambdaRange)
        ));
        assert!(matches!(
            RecursiveLeastSquares::new(-1.0, 1.0),
            Err(Error::IncorrectLambdaRange)
        ));
    }

    #[test]
    fn delta_range() {
        RecursiveLeastSquares::new(0.01, 100.0).unwrap();

        assert!(matches!(
            RecursiveLeastSquares::new(0.5, 0.0),
            Err(Error::NonPositiveDelta)
        ));
    }
}
