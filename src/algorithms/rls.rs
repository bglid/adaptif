use crate::types::buffers::NoiseBuffer;
use crate::types::signals::OutputSample;
use crate::types::{FilterWeights, WindowSize};
use crate::{Error, Result};

use crate::algorithms::Algorithm;

#[derive(Debug, Clone)]
#[allow(clippy::exhaustive_structs, reason = "No more fields have to be added")]
/// Recursive least squares algorithm.
pub struct Rls {
    /// Forgetting factor for weight updates.
    forgetting_factor: f64,
    /// Delta scalar value used for initalizing the inverse correlation `p_matrix`.
    delta: f64,
    /// Inverse correlation matrix, referred to as P[n], for RLS updates.
    p_matrix: Option<Vec<f64>>,
}

impl Rls {
    /// # Errors
    ///
    /// Returns an error if forgetting factor <= 0.0 or > 1.0.
    /// Returns an error if delta <= 0.0.
    pub fn new(forgetting_factor: f64, delta: f64) -> Result<Self> {
        if forgetting_factor <= 0.0 || forgetting_factor > 1.0 {
            return Err(Error::InvalidForgettingFactorRange);
        }

        if delta <= 0.0 {
            return Err(Error::NonPositiveDelta);
        }

        Ok(Rls {
            forgetting_factor,
            delta,
            p_matrix: None,
        })
    }

    fn initial_p_matrix(&self, window_size: WindowSize) -> Vec<f64> {
        let n = *window_size;
        let mut p = vec![0.0; n * n];

        for i in 0..n {
            if let Some(elem) = p.get_mut(i * n + i) {
                *elem = self.delta;
            }
        }
        p
    }

    fn calculate_k(&self, p: &[f64], noise_ref: &NoiseBuffer) -> Vec<f64> {
        let numerator = p
            .chunks_exact(noise_ref.len())
            .map(|row| row.iter().zip(noise_ref.iter()).map(|(px, x)| px * x).sum())
            .collect::<Vec<f64>>();
        let denominator = self.forgetting_factor
            + noise_ref
                .iter()
                .zip(numerator.iter())
                .map(|(noise, num)| noise * num)
                .sum::<f64>();

        numerator.iter().map(|n| n / denominator).collect()
    }

    // I've added comments to try and make this reasonable to read and compare to lit
    fn next_p_matrix(&self, p: &[f64], k_n: &[f64], noise_ref: &NoiseBuffer) -> Vec<f64> {
        // This gets computes the section [x^T_n p_{n-1}]
        let xt_p = (0..noise_ref.len())
            .map(|col| {
                noise_ref
                    .iter()
                    .zip(p.iter().skip(col).step_by(noise_ref.len()))
                    .map(|(x, p)| x * p)
                    .sum::<f64>()
            })
            .collect::<Vec<f64>>();

        // takes result^ and computes lambda^-1 * [p_{n-1} - k(xt_p)]
        p.iter()
            .zip(k_n.iter().flat_map(|k| xt_p.iter().map(move |val| k * val)))
            .map(|(p, update)| (p - update) / self.forgetting_factor)
            .collect::<Vec<f64>>()
    }
}

impl Algorithm for Rls {
    /// Updates the filter weights using the following algorithm.
    ///
    /// The gain vector, ``k_n`` is calculated as:
    ///
    /// $``k_n`` = \frac{P_{n-1} ``x_n``}
    /// {\lambda + ``x_n^T`` P_{n-1} ``x_n``}$.
    ///
    /// The filter weights are updated as:
    ///
    /// $``w_n`` = w_{n-1} + ``k_n`` ``e_n``$.
    ///
    /// The inverse correlation matrix is updated as:
    ///
    /// $``P_n`` = \frac{1}{\lambda}
    /// \left(P_{n-1} - ``k_n`` ``x_n^T`` P_{n-1}\right)$.
    ///
    /// where `e_n` is the scalar error for the current sample,
    /// and $``x_n``$ is a vector of length `window_size` of the
    /// most recent noise reference samples.
    fn update_step(
        &mut self,
        weights: &mut FilterWeights,
        error: OutputSample,
        noise_ref: &NoiseBuffer,
    ) {
        // Updates p_matrix on first iteration once n is known
        #[allow(
            clippy::unwrap_used,
            reason = "weights is initialized from a WindowSize so it can't panic"
        )]
        let window_size = WindowSize::new(weights.len()).unwrap();

        if self.p_matrix.is_none() {
            self.p_matrix = Some(self.initial_p_matrix(window_size));
        }

        let p = match self.p_matrix.as_deref() {
            Some(p) => p,
            None => &self.initial_p_matrix(window_size),
        };

        let k_n = self.calculate_k(p, noise_ref);
        let new_p_matrix = self.next_p_matrix(p, &k_n, noise_ref);
        self.p_matrix = Some(new_p_matrix);

        for (w, k) in weights.iter_mut().zip(k_n.iter()) {
            *w += k * (*error);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;
    use crate::{
        test_utils::{all_approx_equal, noise_buffer_from},
        types::{FilterWeights, WindowSize},
    };

    #[test]
    fn init_p_matrix_works() {
        let rls = Rls::new(0.5, 10.0).unwrap();
        let n = WindowSize::new(3).unwrap();
        let expected = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        let p_matrix = rls.initial_p_matrix(n);
        assert_eq!(p_matrix, expected);
    }

    #[test]
    fn calculate_k_works() {
        let mut rls = Rls::new(1.0, 1.0).unwrap();
        rls.p_matrix = Some(vec![1.0, 0.0, 0.0, 1.0]);
        let p = rls.p_matrix.as_deref().unwrap();
        let x_n = noise_buffer_from(&[1.0, 2.0]);
        let expected = [1.0 / 6.0, 1.0 / 3.0];

        let k_n = rls.calculate_k(p, &x_n);

        assert!(all_approx_equal(k_n.iter(), expected.iter()));
    }

    #[test]
    fn update_p_matrix_works() {
        let mut rls = Rls::new(1.0, 1.0).unwrap();
        rls.p_matrix = Some(vec![1.0, 0.0, 0.0, 1.0]);
        let p = rls.p_matrix.as_deref().unwrap();
        let x_n = noise_buffer_from(&[1.0, 2.0]);
        let k_n = vec![1.0 / 6.0, 1.0 / 3.0];

        let new_p = rls.next_p_matrix(p, &k_n, &x_n);

        let expected = [5.0 / 6.0, -1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0];

        assert!(all_approx_equal(new_p.iter(), expected.iter()));
    }

    #[test]
    fn update_rls_1() {
        let mut rls = Rls::new(0.5, 1.0).unwrap();
        let e_n = OutputSample(2.0);
        let x_n = noise_buffer_from(&[1.0, -1.0]);
        let expected = [0.8, -0.8];
        let mut weights = FilterWeights::new(WindowSize::new(2).unwrap());

        rls.update_step(&mut weights, e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn update_rls_2() {
        let mut rls = Rls::new(1.0, 1.0).unwrap();
        let e_n = OutputSample(1.0);
        let x_n = noise_buffer_from(&[5.0, 2.0]);
        let expected = [5.0 / 30.0, 2.0 / 30.0];
        let mut weights = FilterWeights::new(WindowSize::new(2).unwrap());

        rls.update_step(&mut weights, e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn forgetting_factor_range() {
        Rls::new(0.01, 3.0).unwrap();
        Rls::new(1.0, 3.0).unwrap();

        assert!(matches!(
            Rls::new(0.0, 1.0),
            Err(Error::InvalidForgettingFactorRange)
        ));
        assert!(matches!(
            Rls::new(-1.0, 1.0),
            Err(Error::InvalidForgettingFactorRange)
        ));

        assert!(matches!(
            Rls::new(2.0, 1.0),
            Err(Error::InvalidForgettingFactorRange)
        ));
    }

    #[test]
    fn delta_range() {
        Rls::new(0.01, 100.0).unwrap();

        assert!(matches!(Rls::new(0.5, 0.0), Err(Error::NonPositiveDelta)));
    }
}
