use std::ops::{Deref, DerefMut};

use super::FilterBase;

use crate::Result;
use crate::algorithms::Nlms;
use crate::filters::ProcessingMode;
use crate::types::{Float, WindowSize};

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NlmsOptions<F: Float> {
    pub step_size: F,
    pub eps: F,
    pub window_size: usize,
}
impl<F: Float> Default for NlmsOptions<F> {
    fn default() -> Self {
        #[allow(
            clippy::unwrap_used,
            clippy::missing_panics_doc,
            reason = "Trivial conversions"
        )]
        Self {
            step_size: F::from(0.01).unwrap(),
            eps: F::from(1e-8).unwrap(),
            window_size: 16,
        }
    }
}

#[allow(clippy::doc_paragraphs_missing_punctuation, reason = "maths")]
/// Normalized Least Mean Squares (NLMS) adaptive filter.
///
/// Updates the filter weights using the [normalized least mean squares] algorithm.
/// The NLMS algorithm uses a stochastic gradient descent approach similar to the LMS algorithm.
/// However, with the "pure" LMS algorithm, it can be difficult to choose a step size $\mu$
/// that isn't sensitive to the scaling of input $x(n)$.
///
/// The NLMS algorithm normalizes with the power of the input to solve this problem, providing
/// a more stable step size range.
///
/// TODO: Question, should the examples be the default, or explicit like below?
/// ### Example usage
///
///```rust
/// # use adaptif::Error;
/// use adaptif::filters::{NlmsFilter, NlmsOptions, ProcessingMode};
/// use adaptif::types::{InputSignal, NoiseReference};
///
/// // Initialize the filter
/// let mut nlms = NlmsFilter::new(
///     NlmsOptions {
///         step_size: 0.01, // How fast the filter adapts
///         window_size: 16, // How many samples the filter processes at a time
///         eps: 1e-8,       // Normalization regularization term
///     },
///     ProcessingMode::Sample,
/// )?;
///
/// // Sample inputs
/// let input_signal = InputSignal::new(vec![1.0, -2.5, 3.0])?;
/// let noise_ref = NoiseReference::new(vec![2.0, -1.2, -3.8])?;
///
/// // Adapt the filter to the inputs.
/// let cleaned_signal = nlms.adapt(&input_signal, &noise_ref)?;
///
/// # Ok::<(), Error>(())
///```
///
/// # Update Rule
///
///
/// The weights are updated at every step according to:
///
/// $$
/// \mathbf{w}(n+1) = \mathbf{w}(n) +\mu \mathbf{e}(n)
/// \frac{\mathbf{x}(n)}
/// {\epsilon + \|\|\mathbf{x}(n)\|\|^2}
/// $$
///
/// Where:
/// - $\mathbf{w}(n)$ is the filter weight vector.
/// - $μ$ is the adaptation step size.
/// - $\mathbf{e}(n)$ is the scalar error.
/// - $\mathbf{x}(n)$ is the reference input vector.
/// - $\epsilon$ is the normalization regularization term that prevents division by 0.
/// - $\|\|\mathbf{x}(n)\|\|^2$ is the squared Euclidean norm of the input vector.
///
/// [normalized least mean squares]: https://en.wikipedia.org/wiki/Least_mean_squares_filter#Normalized_least_mean_squares_filter_(NLMS)
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NlmsFilter<F: Float> {
    inner: FilterBase<F, Nlms<F>>,
}
impl<F: Float> NlmsFilter<F> {
    /// # Errors
    ///
    /// Returns an error if `options.step_size <= 0.0` or `options.eps <= 0.0`.
    #[allow(clippy::needless_pass_by_value, reason = "All fields are moved")]
    pub fn new(options: NlmsOptions<F>, mode: ProcessingMode) -> Result<Self> {
        let window_size = WindowSize::new(options.window_size)?;
        let block_size = mode.block_size()?;

        let nlms = Nlms::new(options.step_size, options.eps, window_size)?;
        let filter = FilterBase::new(nlms, block_size);

        Ok(Self { inner: filter })
    }
}
impl<F: Float> Deref for NlmsFilter<F> {
    type Target = FilterBase<F, Nlms<F>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<F: Float> DerefMut for NlmsFilter<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use crate::types::BlockSize;

    use super::*;

    #[test]
    fn new_works() {
        let step_size = 0.1;
        let eps = 1e-8;
        let window_size = 16;

        let filter = NlmsFilter::new(
            NlmsOptions {
                step_size,
                eps,
                window_size,
            },
            ProcessingMode::Sample,
        )
        .unwrap();

        let window_size = WindowSize::new(window_size).unwrap();
        let expected_inner = FilterBase::new(
            Nlms::new(step_size, eps, window_size).unwrap(),
            BlockSize::new(1).unwrap(),
        );

        assert_eq!(filter.inner, expected_inner);
        assert_eq!(filter.window_size(), *window_size);
    }

    #[test]
    fn load_weights_works() {
        let step_size = 0.1;
        let eps = 1e-8;
        let weights = vec![1.0, 2.0, 3.0];
        let window_size = WindowSize::new(weights.len()).unwrap();

        let filter = {
            let mut filter = NlmsFilter::new(
                NlmsOptions {
                    step_size,
                    eps,
                    window_size: *window_size,
                },
                ProcessingMode::Sample,
            )
            .unwrap();
            filter.load_weights(weights.clone()).unwrap();
            filter
        };

        let expected_inner = {
            let mut expected_inner = FilterBase::new(
                Nlms::new(step_size, eps, window_size).unwrap(),
                BlockSize::new(1).unwrap(),
            );
            expected_inner.load_weights(weights.clone()).unwrap();
            expected_inner
        };

        assert_eq!(filter.inner, expected_inner);
        assert_eq!(filter.window_size(), weights.len());
    }
}
