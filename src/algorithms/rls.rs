use std::ops::{Deref, DerefMut};

use crate::types::buffers::NoiseBuffer;
use crate::types::signals::OutputSample;
use crate::types::{FilterWeights, WindowSize};
use crate::{Error, Result};

use crate::algorithms::Algorithm;

#[derive(Debug, Clone, Copy)]
/// Positive scalar of `f64` used in initalizing the RLS inverse correlation matrix.
///
/// `Delta` must be greater than zero.
pub struct Delta(f64);
impl Delta {
    /// # Errors
    ///
    /// Returns an error if delta <= 0.0.
    pub fn new(delta: f64) -> Result<Self> {
        if delta <= 0.0 {
            return Err(Error::NonPositiveDelta);
        }

        Ok(Delta(delta))
    }
}
impl Deref for Delta {
    type Target = f64;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
/// M dimensional vector of Kalman gains.
pub struct KalmanGain(Vec<f64>);
impl KalmanGain {
    pub fn new(noise_ref: &NoiseBuffer) -> Self {
        let mut k = KalmanGain(Vec::with_capacity(noise_ref.len()));
        k.resize(noise_ref.len(), 0.0);
        k
    }
}
impl Deref for KalmanGain {
    type Target = Vec<f64>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for KalmanGain {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug, Clone)]
/// Inverse Correlation Matrix with shape M * M.
pub struct InverseCorrMatrix(Vec<f64>);
impl InverseCorrMatrix {
    pub fn new(window_size: WindowSize, delta: Delta) -> Self {
        let n = *window_size;
        let mut p = Vec::with_capacity(n * n);
        p.resize(n * n, 0.0);

        for i in 0..n {
            if let Some(elem) = p.get_mut(i * n + i) {
                *elem = *delta;
            }
        }
        InverseCorrMatrix(p)
    }
}
impl Deref for InverseCorrMatrix {
    type Target = [f64];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for InverseCorrMatrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::exhaustive_structs, reason = "No more fields have to be added")]
/// Recursive least squares algorithm.
pub struct Rls {
    /// Forgetting factor for weight updates.
    forgetting_factor: f64,
    /// Delta scalar value used for initalizing the inverse correlation `p_matrix`.
    delta: Delta,
    /// Inverse correlation matrix, referred to as P[n], for RLS updates.
    inverse_corr_matrix: Option<InverseCorrMatrix>,
    /// Kalman Gain vector used in updating filter coefficients
    /// Initialized as none because past history is unecessary.Size is based off of `window_size`.
    /// NOTE: Double check with other lit.
    kalman_gain: Option<KalmanGain>,
}

impl Rls {
    /// # Errors
    ///
    /// Returns an error if forgetting factor <= 0.0 or > 1.0.
    pub fn new(forgetting_factor: f64, delta: Delta) -> Result<Self> {
        if forgetting_factor <= 0.0 || forgetting_factor > 1.0 {
            return Err(Error::InvalidForgettingFactorRange);
        }

        Ok(Rls {
            forgetting_factor,
            delta,
            inverse_corr_matrix: None,
            kalman_gain: None,
        })
    }

    fn calculate_k(
        forgetting_factor: f64,
        p: &InverseCorrMatrix,
        kalman: &mut KalmanGain,
        noise_ref: &NoiseBuffer,
    ) {
        let numerator = p.chunks_exact(noise_ref.len()).map(|row| {
            row.iter()
                .zip(noise_ref.iter())
                .map(|(px, x)| px * x)
                .sum::<f64>()
        });
        let denominator = forgetting_factor
            + noise_ref
                .iter()
                .zip(numerator.clone())
                .map(|(noise, num)| noise * num)
                .sum::<f64>();

        for (kalman_i, numberator_i) in (*kalman).iter_mut().zip(numerator) {
            *kalman_i = numberator_i / denominator;
        }
    }

    // I've added comments to try and make this reasonable to read and compare to lit
    fn next_p_matrix(
        forgetting_factor: f64,
        p: &mut InverseCorrMatrix,
        kalman: &mut KalmanGain,
        noise_ref: &NoiseBuffer,
    ) {
        for col in 0..noise_ref.len() {
            // This gets computes the section [x^T_n p_{n-1}]
            let xt_p_col = noise_ref
                .iter()
                .zip(p.iter().skip(col).step_by(noise_ref.len()))
                .map(|(x, p)| x * p)
                .sum::<f64>();

            // takes result^ and computes lambda^-1 * [p_{n-1} - k(xt_p)]
            for (row, k) in kalman.iter().enumerate() {
                // index is into a flat buffer, so row * n gives us the start of each row
                let index = row * noise_ref.len() + col;
                if let Some(p_i) = p.get_mut(index) {
                    *p_i = (*p_i - k * xt_p_col) / forgetting_factor;
                }
            }
        }
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

        if self.inverse_corr_matrix.is_none() {
            self.inverse_corr_matrix = Some(InverseCorrMatrix::new(window_size, self.delta));
            self.kalman_gain = Some(KalmanGain::new(noise_ref));
        }

        let p = match self.inverse_corr_matrix.as_mut() {
            Some(p) => p,
            None => &mut InverseCorrMatrix::new(window_size, self.delta),
        };

        let k = match self.kalman_gain.as_mut() {
            Some(k) => k,
            None => &mut KalmanGain::new(noise_ref),
        };

        Self::calculate_k(self.forgetting_factor, p, k, noise_ref);

        let new_kalman = match self.kalman_gain.as_mut() {
            Some(new_k) => new_k,
            None => &mut KalmanGain::new(noise_ref),
        };

        for (w, new_k) in weights.iter_mut().zip(new_kalman.iter()) {
            *w += new_k * (*error);
        }

        Self::next_p_matrix(self.forgetting_factor, p, new_kalman, noise_ref);
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
        let n = WindowSize::new(3).unwrap();
        let p_matrix = InverseCorrMatrix::new(n, Delta::new(10.0).unwrap());

        let expected = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];

        assert_eq!(*p_matrix, expected);
    }

    #[test]
    fn calculate_k_works() {
        let rls = Rls::new(1.0, Delta::new(1.0).unwrap()).unwrap();
        let p = InverseCorrMatrix(vec![1.0, 0.0, 0.0, 1.0]);
        let x_n = noise_buffer_from(&[1.0, 2.0]);
        let mut k = KalmanGain(vec![0.0; 2]);

        Rls::calculate_k(rls.forgetting_factor, &p, &mut k, &x_n);

        let expected = [1.0 / 6.0, 1.0 / 3.0];

        assert!(all_approx_equal(k.iter(), expected.iter()));
    }

    #[test]
    fn update_p_matrix_works() {
        let rls = Rls::new(1.0, Delta::new(1.0).unwrap()).unwrap();
        let mut p = InverseCorrMatrix(vec![1.0, 0.0, 0.0, 1.0]);
        let x_n = noise_buffer_from(&[1.0, 2.0]);
        let mut k = KalmanGain(vec![1.0 / 6.0, 1.0 / 3.0]);

        Rls::next_p_matrix(rls.forgetting_factor, &mut p, &mut k, &x_n);

        let expected = InverseCorrMatrix(vec![5.0 / 6.0, -1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0]);

        assert!(all_approx_equal(p.iter(), expected.iter()));
    }

    #[test]
    fn update_rls_1() {
        let mut rls = Rls::new(0.5, Delta::new(1.0).unwrap()).unwrap();
        let e_n = OutputSample(2.0);
        let x_n = noise_buffer_from(&[1.0, -1.0]);
        let expected = [0.8, -0.8];
        let mut weights = FilterWeights::new(WindowSize::new(2).unwrap());

        rls.update_step(&mut weights, e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn update_rls_2() {
        let mut rls = Rls::new(1.0, Delta::new(1.0).unwrap()).unwrap();
        let e_n = OutputSample(1.0);
        let x_n = noise_buffer_from(&[5.0, 2.0]);
        let expected = [5.0 / 30.0, 2.0 / 30.0];
        let mut weights = FilterWeights::new(WindowSize::new(2).unwrap());

        rls.update_step(&mut weights, e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn forgetting_factor_range() {
        let delta: Delta = Delta::new(1.0).unwrap();
        Rls::new(0.01, delta).unwrap();
        Rls::new(1.0, delta).unwrap();

        assert!(matches!(
            Rls::new(0.0, delta),
            Err(Error::InvalidForgettingFactorRange)
        ));
        assert!(matches!(
            Rls::new(-1.0, delta),
            Err(Error::InvalidForgettingFactorRange)
        ));

        assert!(matches!(
            Rls::new(2.0, delta),
            Err(Error::InvalidForgettingFactorRange)
        ));
    }

    #[test]
    fn delta_range() {
        let good_delta: Delta = Delta::new(100.0).unwrap();
        Rls::new(0.01, good_delta).unwrap();

        assert!(matches!(Delta::new(0.0), Err(Error::NonPositiveDelta)));
        assert!(matches!(Delta::new(-1.0), Err(Error::NonPositiveDelta)));
    }
}
