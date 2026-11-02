use std::ops::{Deref, DerefMut};

use crate::types::buffers::NoiseWindow;
use crate::types::signals::OutputSample;
use crate::types::{FilterWeights, Float, WindowSize};
use crate::{Error, Result};

use crate::algorithms::Algorithm;

#[derive(Debug, Clone, PartialEq)]
/// Recursive least squares algorithm.
pub struct Rls<F: Float> {
    /// Forgetting factor, often referred to as `lambda`, used for weight updates.
    forgetting_factor: F,
    /// Positive scalar value, often referred to as `delta`, used for initalizing the inverse correlation `p_matrix`.
    p_init_scale: F,
    /// Window size of the filter.
    /// Determines the size of the inverse correlation matrix and the kalman gain vector.
    window_size: WindowSize,
    /// Inverse correlation matrix, referred to as P[n], for RLS updates.
    /// Its dimensions are (M, M), where M is the filter's window size.
    inverse_corr_matrix: InverseCorrMatrix<F>,
    /// Kalman Gain vector used in updating filter coefficients.
    /// Its length is equal to the filter's window size.
    kalman_gain: KalmanGain<F>,
}

impl<F: Float> Rls<F> {
    /// # Errors
    ///
    /// Returns an error if forgetting factor <= 0.0 or > 1.0.
    /// Returns an error if init scale <= 0.0.
    pub fn new(forgetting_factor: F, p_init_scale: F, window_size: WindowSize) -> Result<Self> {
        if forgetting_factor <= F::zero() || forgetting_factor > F::one() {
            return Err(Error::InvalidForgettingFactorRange);
        }

        if p_init_scale <= F::zero() {
            return Err(Error::NonPositivePInitScale);
        }
        Ok(Rls {
            forgetting_factor,
            p_init_scale,
            window_size,
            inverse_corr_matrix: InverseCorrMatrix::new(window_size, p_init_scale),
            kalman_gain: KalmanGain::new(window_size),
        })
    }
}

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
impl<F: Float> Algorithm<F> for Rls<F> {
    fn update_step(
        &mut self,
        weights: &mut FilterWeights<F>,
        error: OutputSample<F>,
        noise_window: NoiseWindow<F>,
    ) {
        self.kalman_gain.update(
            &self.inverse_corr_matrix,
            &noise_window,
            self.forgetting_factor,
        );

        for (w, k) in weights.iter_mut().zip(self.kalman_gain.iter().copied()) {
            *w += k * (*error);
        }

        self.inverse_corr_matrix
            .update(&self.kalman_gain, &noise_window, self.forgetting_factor);
    }

    fn window_size(&self) -> WindowSize {
        self.window_size
    }
}

#[derive(Debug, Clone, PartialEq)]
/// M-dimensional vector of Kalman gains, where M is the filter's window size.
struct KalmanGain<F: Float> {
    inner: Box<[F]>,
}
impl<F: Float> KalmanGain<F> {
    pub fn new(window_size: WindowSize) -> Self {
        KalmanGain {
            inner: vec![F::zero(); *window_size].into_boxed_slice(),
        }
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
    fn update(
        &mut self,
        inverse_corr_matrix: &InverseCorrMatrix<F>,
        noise_ref: &NoiseWindow<F>,
        lambda: F,
    ) {
        // reusing kalman buffer for getting numerator to avoid clone of numerator
        for (k_i, row) in self
            .iter_mut()
            .zip(inverse_corr_matrix.chunks_exact(noise_ref.len()))
        {
            *k_i = row
                .iter()
                .copied()
                .zip(noise_ref.iter())
                .map(|(px, x)| px * x)
                .sum::<F>();
        }

        let denominator = lambda
            + noise_ref
                .iter()
                .zip(self.iter().copied())
                .map(|(noise, num)| noise * num)
                .sum::<F>();

        for k_i in self.iter_mut() {
            *k_i /= denominator;
        }
    }
}
impl<F: Float> Deref for KalmanGain<F> {
    type Target = [F];
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<F: Float> DerefMut for KalmanGain<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Inverse Correlation Matrix with shape M * M, where M is the filter's window size.
struct InverseCorrMatrix<F: Float> {
    inner: Box<[F]>,
}
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
        InverseCorrMatrix { inner: p }
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
    fn update(&mut self, kalman_gain: &KalmanGain<F>, noise_ref: &NoiseWindow<F>, lambda: F) {
        // We calculate [x^T P] by computing the dot product of
        // noise_ref (x) with each column in inverse_corr_matrix (P),
        // then update the values for that column in place.
        for col in 0..noise_ref.len() {
            // Get the j-th column in P and take its dot product with x.
            let p_col = self.iter().copied().skip(col).step_by(noise_ref.len());
            let xt_p_col = noise_ref.iter().zip(p_col).map(|(x, p)| x * p).sum::<F>();

            // Update the values in that column in place
            for (row, k_i) in kalman_gain.iter().copied().enumerate() {
                // index is into a flat buffer, so row * n gives us the start of each row
                let index = row * noise_ref.len() + col;

                let p_i = self
                    .get_mut(index)
                    // TODO(ben): How is this enforced?
                    .expect("Index should always be valid based on the type invariants");
                *p_i = (*p_i - k_i * xt_p_col) / lambda;
            }
        }
    }
}
impl<F: Float> Deref for InverseCorrMatrix<F> {
    type Target = [F];
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<F: Float> DerefMut for InverseCorrMatrix<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;
    use crate::{
        test_utils::{all_approx_equal, block_noise_buffer_from},
        types::{BlockSize, FilterWeights, WindowSize, buffers::BlockError},
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
        let mut k = KalmanGain {
            inner: vec![0.0; 2].into_boxed_slice(),
        };
        let p = InverseCorrMatrix {
            inner: vec![1.0, 0.0, 0.0, 1.0].into_boxed_slice(),
        };
        let x_n = block_noise_buffer_from(
            &[1.0, 2.0],
            WindowSize::new(2).unwrap(),
            BlockSize::new(1).unwrap(),
        );

        k.update(&p, &x_n.get_window(0).unwrap(), 1.0);

        let expected = [1.0 / 6.0, 1.0 / 3.0];

        assert!(all_approx_equal(k.iter(), expected.iter()));
    }

    #[test]
    fn update_p_matrix_works() {
        let mut p = InverseCorrMatrix {
            inner: vec![1.0, 0.0, 0.0, 1.0].into_boxed_slice(),
        };
        let k = KalmanGain {
            inner: vec![1.0 / 6.0, 1.0 / 3.0].into_boxed_slice(),
        };
        let x_n = block_noise_buffer_from(
            &[1.0, 2.0],
            WindowSize::new(2).unwrap(),
            BlockSize::new(1).unwrap(),
        );

        p.update(&k, &x_n.get_window(0).unwrap(), 1.0);

        let expected = InverseCorrMatrix {
            inner: vec![5.0 / 6.0, -1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0].into_boxed_slice(),
        };

        assert!(all_approx_equal(p.iter(), expected.iter()));
    }

    #[test]
    fn update_rls_1() {
        let window_size = WindowSize::new(2).unwrap();
        let block_size = BlockSize::new(1).unwrap();

        let mut rls = Rls::new(0.5, 1.0, window_size).unwrap();
        let e_n = {
            let mut e_n = BlockError::new(block_size);
            e_n.push(OutputSample(2.0));
            e_n
        };
        let x_n = block_noise_buffer_from(&[1.0, -1.0], window_size, block_size);
        let expected = [0.8, -0.8];
        let mut weights = FilterWeights::new(window_size);

        rls.update_block(&mut weights, &e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn update_rls_2() {
        let window_size = WindowSize::new(2).unwrap();
        let block_size = BlockSize::new(1).unwrap();

        let mut rls = Rls::new(1.0, 1.0, window_size).unwrap();
        let e_n = {
            let mut e_n = BlockError::new(block_size);
            e_n.push(OutputSample(1.0));
            e_n
        };
        let x_n = block_noise_buffer_from(&[5.0, 2.0], window_size, block_size);
        let expected = [5.0 / 30.0, 2.0 / 30.0];
        let mut weights = FilterWeights::new(WindowSize::new(2).unwrap());

        rls.update_block(&mut weights, &e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn forgetting_factor_range() {
        let p_init_scale = 1.0;
        let window_size = WindowSize::new(8).unwrap();
        Rls::new(0.01, p_init_scale, window_size).unwrap();
        Rls::new(1.0, p_init_scale, window_size).unwrap();

        assert!(matches!(
            Rls::new(0.0, p_init_scale, window_size),
            Err(Error::InvalidForgettingFactorRange)
        ));
        assert!(matches!(
            Rls::new(-1.0, p_init_scale, window_size),
            Err(Error::InvalidForgettingFactorRange)
        ));

        assert!(matches!(
            Rls::new(2.0, p_init_scale, window_size),
            Err(Error::InvalidForgettingFactorRange)
        ));
    }

    #[test]
    fn p_init_scale_range() {
        let good_init_scale = 100.0;
        let window_size = WindowSize::new(8).unwrap();
        Rls::new(0.01, good_init_scale, window_size).unwrap();

        assert!(matches!(
            Rls::new(0.5, 0.0, window_size),
            Err(Error::NonPositivePInitScale)
        ));
        assert!(matches!(
            Rls::new(0.5, -1.0, window_size),
            Err(Error::NonPositivePInitScale)
        ));
    }

    #[test]
    fn window_size_works() {
        let window_size = WindowSize::new(16).unwrap();

        let rls = Rls::new(0.01, 1.0, window_size).unwrap();

        assert_eq!(rls.window_size(), window_size);
    }
}
