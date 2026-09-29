use std::ops::{Deref, DerefMut};

use crate::types::buffers::NoiseBuffer;
use crate::types::signals::OutputSample;
use crate::types::{FilterWeights, Float, WindowSize};
use crate::{Error, Result};

use crate::algorithms::Algorithm;

#[derive(Debug, Clone, PartialEq)]
/// M-dimensional vector of Kalman gains, where M is the filter's window size.
pub struct KalmanGain<F: Float>(Box<[F]>);
impl<F: Float> KalmanGain<F> {
    pub fn new(window_size: WindowSize) -> Self {
        KalmanGain(vec![F::zero(); *window_size].into_boxed_slice())
    }
}
impl<F: Float> Deref for KalmanGain<F> {
    type Target = [F];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F: Float> DerefMut for KalmanGain<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Inverse Correlation Matrix with shape M * M, where M is the filter's window size.
pub struct InverseCorrMatrix<F: Float>(Box<[F]>);
impl<F: Float> InverseCorrMatrix<F> {
    /// Creates an M x M-dimensional Identity matrix multiplied by a positive scalar `p_init_scale`.
    /// I.e. `p_init_scale` is the resulting value along main diagonal and all other values are initialized
    /// to zero.
    ///
    /// Serves as inverse correlation matrix in RLS algorithm, where M is the filter's window size.
    pub fn new(window_size: WindowSize, p_init_scale: F) -> Self {
        let mut p = vec![F::zero(); (*window_size) * (*window_size)].into_boxed_slice();

        for i in 0..(*window_size) {
            if let Some(elem) = p.get_mut(i * (*window_size) + i) {
                *elem = p_init_scale;
            }
        }
        InverseCorrMatrix(p)
    }
}
impl<F: Float> Deref for InverseCorrMatrix<F> {
    type Target = [F];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F: Float> DerefMut for InverseCorrMatrix<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Recursive least squares algorithm.
pub struct Rls<F: Float> {
    /// Forgetting factor, often referred to as `lambda`, used for weight updates.
    forgetting_factor: F,
    /// Positive scalar value, often referred to as `delta`, used for initalizing the inverse correlation `p_matrix`.
    p_init_scale: F,
    /// Inverse correlation matrix, referred to as P[n], for RLS updates. Size is determined by the filter's
    /// window size. The matrix size is M x M, where M corresponds to the filter's window size.
    inverse_corr_matrix: InverseCorrMatrix<F>,
    /// Kalman Gain vector used in updating filter coefficients
    /// Initialized as none because past history is unnecessary. Size is determined by the filter's
    /// window size.
    kalman_gain: KalmanGain<F>,
}

impl<F: Float> Rls<F> {
    /// # Errors
    ///
    /// Returns an error if forgetting factor <= 0.0 or > 1.0.
    /// Returns an error if init scale <= 0.0.
    pub fn new(forgetting_factor: F, p_init_scale: F) -> Result<Self> {
        if forgetting_factor <= F::zero() || forgetting_factor > F::one() {
            return Err(Error::InvalidForgettingFactorRange);
        }

        if p_init_scale <= F::zero() {
            return Err(Error::NonPositivePInitScale);
        }
        Ok(Rls {
            forgetting_factor,
            p_init_scale,
            inverse_corr_matrix: InverseCorrMatrix(vec![].into_boxed_slice()),
            kalman_gain: KalmanGain(vec![].into_boxed_slice()),
        })
    }

    /// Updates the Kalman gain vector used in the RLS weight update.
    ///
    /// Kalman gain vector update calculated as:
    ///
    /// $``k_n`` = \frac{P_{n-1} ``x_n``}
    /// {\lambda + ``x_n^T`` P_{n-1} ``x_n``}$.
    ///
    /// Existing Kalman gain buffer gets reused to store the numerator
    /// value in update calculation to avoid reallocating a new vector.
    fn update_kalman_gain(&mut self, noise_ref: &NoiseBuffer<F>) {
        // reusing kalman buffer for getting numerator to avoid clone of numerator
        for (k_i, row) in self
            .kalman_gain
            .iter_mut()
            .zip(self.inverse_corr_matrix.chunks_exact(noise_ref.len()))
        {
            *k_i = row
                .iter()
                .copied()
                .zip(noise_ref.iter().copied())
                .map(|(px, x)| px * x)
                .sum::<F>();
        }

        let denominator = self.forgetting_factor
            + noise_ref
                .iter()
                .copied()
                .zip(self.kalman_gain.iter().copied())
                .map(|(noise, num)| noise * num)
                .sum::<F>();

        for k_i in self.kalman_gain.iter_mut() {
            *k_i /= denominator;
        }
    }

    /// Updates the inverse correlation matrix used in the RLS weight update.
    /// Utilizes current Kalman gain ``k_n``.
    ///
    /// Inverse correlation matrix update calculated as:
    ///
    /// $``P_n`` = \frac{1}{\lambda}
    /// \left(P_{n-1} - ``k_n`` ``x_n^T`` P_{n-1}\right)$.
    ///
    /// The matrix is stored as a flat buffer and updated in place.
    fn update_p_matrix(&mut self, noise_ref: &NoiseBuffer<F>) {
        // We calculate [x^T_n p_{n-1}] by computing the dot product of
        // ``noise_ref`` with each columnn in ``inverse_corr_matrix``
        for col in 0..noise_ref.len() {
            let p_col = self
                .inverse_corr_matrix
                .iter()
                .copied()
                .skip(col)
                .step_by(noise_ref.len());
            let xt_p_col = noise_ref
                .iter()
                .copied()
                .zip(p_col)
                .map(|(x, p)| x * p)
                .sum::<F>();

            // takes result^ and computes lambda^-1 * [p_{n-1} - k(xt_p column)]
            for (row, k_i) in self.kalman_gain.iter().copied().enumerate() {
                // index is into a flat buffer, so row * n gives us the start of each row
                let index = row * noise_ref.len() + col;

                #[allow(
                    clippy::expect_used,
                    reason = "Index should always be valid based on the type invariants"
                )]
                let p_i = self
                    .inverse_corr_matrix
                    .get_mut(index)
                    .expect("internal error, index should always be valid");
                *p_i = (*p_i - k_i * xt_p_col) / self.forgetting_factor;
            }
        }
    }
}

impl<F: Float> Algorithm<F> for Rls<F> {
    /// Updates the filter weights using the following algorithm.
    ///
    /// The Kalman gain vector, ``k_n`` is calculated as:
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
        weights: &mut FilterWeights<F>,
        error: OutputSample<F>,
        noise_ref: &NoiseBuffer<F>,
    ) {
        // Updates p_matrix on first iteration once n is known
        // No scenario where one is initialized and the other isn't
        // TODO: remove once proper init is implemented
        if self.inverse_corr_matrix.is_empty() {
            self.inverse_corr_matrix =
                InverseCorrMatrix::new(weights.window_size(), self.p_init_scale);
            self.kalman_gain = KalmanGain::new(weights.window_size());
        }

        self.update_kalman_gain(noise_ref);

        for (w, k) in weights.iter_mut().zip(self.kalman_gain.iter().copied()) {
            *w += k * (*error);
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
        let p_matrix = InverseCorrMatrix::new(n, 10.0);

        let expected = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];

        assert_eq!(*p_matrix, expected);
    }

    #[test]
    fn update_kalman_gain_works() {
        let mut rls = Rls::new(1.0, 1.0).unwrap();
        rls.inverse_corr_matrix = InverseCorrMatrix(vec![1.0, 0.0, 0.0, 1.0].into_boxed_slice());
        rls.kalman_gain = KalmanGain(vec![0.0; 2].into_boxed_slice());
        let x_n = noise_buffer_from(&[1.0, 2.0]);

        rls.update_kalman_gain(&x_n);

        let expected = [1.0 / 6.0, 1.0 / 3.0];

        assert!(all_approx_equal(rls.kalman_gain.iter(), expected.iter()));
    }

    #[test]
    fn update_p_matrix_works() {
        let mut rls = Rls::new(1.0, 1.0).unwrap();
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
        let p_init_scale = 1.0;
        Rls::new(0.01, p_init_scale).unwrap();
        Rls::new(1.0, p_init_scale).unwrap();

        assert!(matches!(
            Rls::new(0.0, p_init_scale),
            Err(Error::InvalidForgettingFactorRange)
        ));
        assert!(matches!(
            Rls::new(-1.0, p_init_scale),
            Err(Error::InvalidForgettingFactorRange)
        ));

        assert!(matches!(
            Rls::new(2.0, p_init_scale),
            Err(Error::InvalidForgettingFactorRange)
        ));
    }

    #[test]
    fn p_init_scale_range() {
        let good_init_scale = 100.0;
        Rls::new(0.01, good_init_scale).unwrap();

        assert!(matches!(
            Rls::new(0.5, 0.0),
            Err(Error::NonPositivePInitScale)
        ));
        assert!(matches!(
            Rls::new(0.5, -1.0),
            Err(Error::NonPositivePInitScale)
        ));
    }
}
