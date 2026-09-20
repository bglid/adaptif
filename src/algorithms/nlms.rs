use crate::types::FilterWeights;
use crate::types::buffers::NoiseBuffer;
use crate::types::signals::OutputSample;
use crate::{Error, Result};

use crate::algorithms::Algorithm;

#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::exhaustive_structs, reason = "No more fields have to be added")]
// Normalized least mean squares algorithm.
pub struct Nlms {
    /// Step size for weight updates.
    mu: f64,
    /// Regularization term to avoid division by zero.
    eps: f64,
}
impl Nlms {
    /// # Errors
    ///
    /// Returns an error if mu <= 0.0.
    /// Returns an error if eps <= 0.0.
    pub fn new(mu: f64, eps: f64) -> Result<Self> {
        if mu <= 0.0 {
            return Err(Error::NonPositiveStepSize);
        }

        if eps <= 0.0 {
            return Err(Error::NonPositiveEpsilon);
        }

        Ok(Nlms { mu, eps })
    }
}
impl Algorithm for Nlms {
    /// Updates the filter weights using the following equation:
    ///
    /// $w_{n+1} = ``w_n`` + \frac{\mu}{\epsilon + \|``x_n``\|^2} ``e_n`` ``x_n``$
    /// where $``e_n``$ is the scalar error for the current sample,
    /// and $``x_n``$ is a vector of length `window_size` of the
    /// most recent noise reference samples.
    fn update_step(
        &self,
        weights: &mut FilterWeights,
        error: OutputSample,
        noise_ref: &NoiseBuffer,
    ) {
        let norm_squared: f64 = noise_ref.iter().map(|x| x * x).sum();
        let mu_normalized = self.mu / (self.eps + norm_squared);

        for (w, x) in weights.iter_mut().zip(noise_ref.iter()) {
            *w += mu_normalized * (*error) * x;
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
    fn update_nlms_1() {
        let nlms = Nlms::new(0.5, 1e-8).unwrap();
        let e_n = OutputSample(2.0);
        let x_n = noise_buffer_from(&[1.0, -1.0]);
        let expected = [1.0 / (2.0 + nlms.eps), -1.0 / (2.0 + nlms.eps)];
        let mut weights = FilterWeights::new(WindowSize::new(2).unwrap());

        nlms.update_step(&mut weights, e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn update_nlms_2() {
        let nlms = Nlms::new(1.0, 1e-8).unwrap();
        let e_n = OutputSample(1.0);
        let x_n = noise_buffer_from(&[5.0, 2.0]);
        let expected = [(5.0 / (29.0 + nlms.eps)), (2.0 / (29.0 + nlms.eps))];
        let mut weights = FilterWeights::new(WindowSize::new(2).unwrap());

        nlms.update_step(&mut weights, e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn mu_range() {
        Nlms::new(1.0, 1e-8).unwrap();
        Nlms::new(f64::MAX, 1e-8).unwrap();

        assert!(matches!(
            Nlms::new(0.0, 1e-8),
            Err(Error::NonPositiveStepSize)
        ));
        assert!(matches!(
            Nlms::new(-1.0, 1e-8),
            Err(Error::NonPositiveStepSize)
        ));
    }

    #[test]
    fn eps_range() {
        Nlms::new(1.0, 1e-8).unwrap();

        assert!(matches!(
            Nlms::new(1.0, 0.0),
            Err(Error::NonPositiveEpsilon)
        ));
        assert!(matches!(
            Nlms::new(1.0, -1.0),
            Err(Error::NonPositiveEpsilon)
        ));
    }
}
