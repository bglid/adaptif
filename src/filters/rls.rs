use std::ops::{Deref, DerefMut};

use super::FilterBase;

use crate::Result;
use crate::algorithms::Rls;
use crate::filters::ProcessingMode;
use crate::types::{Float, WindowSize};

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct RlsOptions<F: Float> {
    pub forgetting_factor: F,
    pub p_init_scale: F,
    pub window_size: usize,
}
impl<F: Float> Default for RlsOptions<F> {
    fn default() -> Self {
        #[allow(
            clippy::unwrap_used,
            clippy::missing_panics_doc,
            reason = "Trivial conversions"
        )]
        Self {
            forgetting_factor: F::from(0.999).unwrap(),
            p_init_scale: F::from(0.001).unwrap(),
            window_size: 8,
        }
    }
}

#[allow(clippy::doc_paragraphs_missing_punctuation, reason = "maths")]
/// Recursive Least Squares (RLS) adaptive filter.
///
/// Updates the filter weights using the [recursive least squares] algorithm.
/// RLS recursively finds the coefficients that
/// minimize an exponentially weighted sum of
/// squared errors to adapt the filter weights.
///
/// Unlike [least mean squares], which uses stochastic gradient
/// descent, RLS uses a deterministic least-squares approach.
///
/// RLS has fast convergence, at the cost of high computational complexity.
/// For computationally constrained applications,
/// LMS-family algorithms may be preferable.
///
/// ### Example usage
///
///```rust
/// # use adaptif::Error;
/// use adaptif::filters::{RlsFilter, RlsOptions, ProcessingMode};
/// use adaptif::types::{InputSignal, NoiseReference};
///
/// // Initialize the filter
/// let mut rls= RlsFilter::new(
///     RlsOptions {
///         forgetting_factor: 0.999, // How much weight is given to past observations
///         window_size: 8,           // How many samples the filter processes at a time
///         p_init_scale: 0.001,      // Value of initialization scalar for algorithm
///     },
///     ProcessingMode::Sample,
/// )?;
///
/// // Sample inputs
/// let input_signal = InputSignal::new(vec![1.0, -2.5, 3.0])?;
/// let noise_ref = NoiseReference::new(vec![2.0, -1.2, -3.8])?;
///
/// // Adapt the filter to the inputs.
/// let cleaned_signal = rls.adapt(&input_signal, &noise_ref)?;
///
/// # Ok::<(), Error>(())
///```
///
/// # Update Rule
///
/// The filter output is calculated at every step as:
///
/// $$y(n) = \mathbf{w}^{T}(n)\mathbf{x}(n)$$
///
/// The error signal is calculated at every step as:
///
/// $$e(n) = d(n) - y(n)$$
///
/// The weights are updated at every step according to:
///
/// $$\mathbf{w}(n) = \mathbf{w}(n-1) + \mathbf{k}(n)e(n)$$
///
/// The Kalman gain vector is calculated as:
///
/// $$\mathbf{k}(n) = \frac{\mathbf{P}(n-1)\mathbf{x}(n)}{\lambda + \mathbf{x}^{T}(n)\mathbf{P}(n-1)\mathbf{x}(n)}$$
///
/// The inverse correlation matrix is then updated as:
///
/// $$\mathbf{P}(n) = \frac{1}{\lambda}\left(\mathbf{P}(n-1) - \mathbf{k}(n)\mathbf{x}^{T}(n)\mathbf{P}(n-1)\right)$$
///
/// The inverse correlation matrix is initialized as:
///
/// $$\mathbf{P}(0) = \delta\mathbf{I}$$
///
/// Where:
/// - $d(n)$ is the primary input signal.
/// - $y(n)$ is the estimated noise (filter output).
/// - $\mathbf{w}(n)$ is the filter weight vector.
/// - $\mathbf{x}(n)$ is the reference input vector.
/// - $e(n)$ is the scalar error for the current sample.
/// - $\mathbf{k}(n)$ is the Kalman gain vector.
/// - $\mathbf{P}(n)$ is the inverse correlation matrix.
/// - $\lambda$ is the forgetting factor, with $0 < \lambda \leq 1$.
/// - $\delta$ is  a positive initialization scalar.
/// - $\mathbf{I}$ is the identity matrix.
///
/// [recursive least squares]: https://en.wikipedia.org/wiki/Recursive_least_squares_filter
/// [least mean squares]: https://en.wikipedia.org/wiki/Least_mean_squares_filter
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct RlsFilter<F: Float> {
    inner: FilterBase<F, Rls<F>>,
}
impl<F: Float> RlsFilter<F> {
    /// # Errors
    ///
    /// Returns an error if options.forgetting factor <= 0.0 or > 1.0, or if init scale <= 0.0.
    #[allow(clippy::needless_pass_by_value, reason = "All fields are moved")]
    pub fn new(options: RlsOptions<F>, mode: ProcessingMode) -> Result<Self> {
        let window_size = WindowSize::new(options.window_size)?;
        let block_size = mode.block_size()?;

        let rls = Rls::new(options.forgetting_factor, options.p_init_scale, window_size)?;
        let filter = FilterBase::new(rls, block_size);

        Ok(Self { inner: filter })
    }
}
impl<F: Float> Deref for RlsFilter<F> {
    type Target = FilterBase<F, Rls<F>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<F: Float> DerefMut for RlsFilter<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;
    use crate::types::BlockSize;

    #[test]
    fn new_works() {
        let forgetting_factor = 0.999;
        let p_init_scale = 1e-4;
        let window_size = 16;

        let filter = RlsFilter::new(
            RlsOptions {
                forgetting_factor,
                p_init_scale,
                window_size,
            },
            ProcessingMode::Sample,
        )
        .unwrap();
        let expected_inner = FilterBase::new(
            Rls::new(
                forgetting_factor,
                p_init_scale,
                WindowSize::new(window_size).unwrap(),
            )
            .unwrap(),
            BlockSize::new(1).unwrap(),
        );

        assert_eq!(filter.inner, expected_inner);
        assert_eq!(filter.window_size(), window_size);
    }

    #[test]
    fn load_weights_works() {
        let forgetting_factor = 0.999;
        let p_init_scale = 1e-4;
        let weights = vec![1.0, 2.0, 3.0];

        let filter = {
            let mut filter = RlsFilter::new(
                RlsOptions {
                    forgetting_factor,
                    p_init_scale,
                    window_size: weights.len(),
                },
                ProcessingMode::Sample,
            )
            .unwrap();
            filter.load_weights(weights.clone()).unwrap();
            filter
        };

        let expected_inner = {
            let mut expected_inner = FilterBase::new(
                Rls::new(
                    forgetting_factor,
                    p_init_scale,
                    WindowSize::new(weights.len()).unwrap(),
                )
                .unwrap(),
                BlockSize::new(1).unwrap(),
            );
            expected_inner.load_weights(weights.clone()).unwrap();
            expected_inner
        };

        assert_eq!(filter.inner, expected_inner);
        assert_eq!(filter.window_size(), weights.len());
    }
}
