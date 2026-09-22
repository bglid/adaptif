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
/// M-dimensional vector of Kalman gains, where M is the filter's window size.
pub struct KalmanGain(Box<[f64]>);
impl KalmanGain {
    // TODO: this should probably be init from WindowSize
    pub fn new(noise_ref: &NoiseBuffer) -> Self {
        KalmanGain(vec![0.0; noise_ref.len()].into_boxed_slice())
    }
}
impl Deref for KalmanGain {
    type Target = [f64];
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
/// Inverse Correlation Matrix with shape M * M, where M is the filter's window size.
pub struct InverseCorrMatrix(Box<[f64]>);
impl InverseCorrMatrix {
    pub fn new(window_size: WindowSize, delta: Delta) -> Self {
        let mut p = vec![0.0; (*window_size) * (*window_size)].into_boxed_slice();

        for i in 0..(*window_size) {
            if let Some(elem) = p.get_mut(i * (*window_size) + i) {
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
    inverse_corr_matrix: InverseCorrMatrix,
    /// Kalman Gain vector used in updating filter coefficients
    /// Initialized as none because past history is unnecessary. Size is determined by the filter's
    /// window size.
    /// NOTE: Double check with other lit.
    kalman_gain: KalmanGain,
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
            inverse_corr_matrix: InverseCorrMatrix(vec![].into_boxed_slice()),
            kalman_gain: KalmanGain(vec![].into_boxed_slice()),
        })
    }

    fn update_kalman_gain(&mut self, noise_ref: &NoiseBuffer) {
        // reusing kalman buffer for getting numerator to avoid clone of numerator
        for (k_i, row) in self
            .kalman_gain
            .iter_mut()
            .zip(self.inverse_corr_matrix.chunks_exact(noise_ref.len()))
        {
            *k_i = row
                .iter()
                .zip(noise_ref.iter())
                .map(|(px, x)| px * x)
                .sum::<f64>();
        }

        let denominator = self.forgetting_factor
            + noise_ref
                .iter()
                .zip(self.kalman_gain.iter())
                .map(|(noise, num)| noise * num)
                .sum::<f64>();

        for k_i in self.kalman_gain.iter_mut() {
            *k_i /= denominator;
        }
    }

    // I've added comments to try and make this reasonable to read and compare to lit
    fn update_p_matrix(&mut self, noise_ref: &NoiseBuffer) {
        for col in 0..noise_ref.len() {
            // This gets computes the section [x^T_n p_{n-1}]
            let xt_p_col = noise_ref
                .iter()
                .zip(
                    self.inverse_corr_matrix
                        .iter()
                        .skip(col)
                        .step_by(noise_ref.len()),
                )
                .map(|(x, p)| x * p)
                .sum::<f64>();

            // takes result^ and computes lambda^-1 * [p_{n-1} - k(xt_p column)]
            for (row, k_i) in self.kalman_gain.iter().enumerate() {
                // index is into a flat buffer, so row * n gives us the start of each row
                let index = row * noise_ref.len() + col;
                if let Some(p_i) = self.inverse_corr_matrix.get_mut(index) {
                    *p_i = (*p_i - k_i * xt_p_col) / self.forgetting_factor;
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
        // TODO: replace with weights.window_size() after merge
        #[allow(
            clippy::unwrap_used,
            reason = "weights is initialized from a WindowSize so it can't panic"
        )]
        let window_size = WindowSize::new(weights.len()).unwrap();

        // Updates p_matrix on first iteration once n is known
        // TODO: remove once proper init is implemented
        if self.inverse_corr_matrix.is_empty() {
            self.inverse_corr_matrix = InverseCorrMatrix::new(window_size, self.delta);
        }

        if self.kalman_gain.is_empty() {
            self.kalman_gain = KalmanGain::new(noise_ref);
        }

        self.update_kalman_gain(noise_ref);

        for (w, k) in weights.iter_mut().zip(self.kalman_gain.iter()) {
            *w += (*k) * (*error);
        }

        self.update_p_matrix(noise_ref);
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
    fn update_kalman_gain_works() {
        let mut rls = Rls::new(1.0, Delta::new(1.0).unwrap()).unwrap();
        rls.inverse_corr_matrix = InverseCorrMatrix(vec![1.0, 0.0, 0.0, 1.0].into_boxed_slice());
        rls.kalman_gain = KalmanGain(vec![0.0; 2].into_boxed_slice());
        let x_n = noise_buffer_from(&[1.0, 2.0]);

        rls.update_kalman_gain(&x_n);

        let expected = [1.0 / 6.0, 1.0 / 3.0];

        assert!(all_approx_equal(rls.kalman_gain.iter(), expected.iter()));
    }

    #[test]
    fn update_p_matrix_works() {
        let mut rls = Rls::new(1.0, Delta::new(1.0).unwrap()).unwrap();
        rls.inverse_corr_matrix = InverseCorrMatrix(vec![1.0, 0.0, 0.0, 1.0].into_boxed_slice());
        rls.kalman_gain = KalmanGain(vec![1.0 / 6.0, 1.0 / 3.0].into_boxed_slice());
        let x_n = noise_buffer_from(&[1.0, 2.0]);

        rls.update_p_matrix(&x_n);

        let expected = InverseCorrMatrix(
            vec![5.0 / 6.0, -1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0].into_boxed_slice(),
        );

        assert!(all_approx_equal(
            rls.inverse_corr_matrix.iter(),
            expected.iter()
        ));
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
