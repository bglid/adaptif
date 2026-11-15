use std::ops::{Deref, DerefMut};

use super::FilterBase;

use crate::Result;
use crate::algorithms::Lms;
use crate::filters::ProcessingMode;
use crate::types::{Float, WindowSize};

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
/// Used to specify the filter and algorithm parameters for [`LmsFilter`].
pub struct LmsOptions<F: Float> {
    /// Step size for weight updates.
    pub step_size: F,
    /// Number of past noise samples processed by the filter at each time step.
    pub window_size: usize,
}
impl<F: Float> Default for LmsOptions<F> {
    fn default() -> Self {
        #[allow(
            clippy::unwrap_used,
            clippy::missing_panics_doc,
            reason = "Trivial conversions"
        )]
        Self {
            // TODO: if we define our own Float trait we can have an unchecked conversion fn
            // for these cases since we only need to impl it for f64 and f32
            step_size: F::from(0.1).unwrap(),
            window_size: 16,
        }
    }
}

#[allow(clippy::doc_paragraphs_missing_punctuation, reason = "maths")]
/// Least Mean Squares (LMS) adaptive filter.
///
/// Updates the filter weights using the [least mean squares] algorithm.
/// The LMS algorithm uses stochastic gradient descent to minimize
/// the mean squared error between the reference input signal
/// and the filter output.
///
/// ### Example usage
///
///```rust
/// # use adaptif::Error;
/// use adaptif::filters::{LmsFilter, LmsOptions, ProcessingMode};
/// use adaptif::types::{InputSignal, NoiseReference};
///
/// // Initialize the filter
/// let mut lms = LmsFilter::new(
///     LmsOptions {
///         step_size: 0.1,  // How fast the filter adapts
///         window_size: 16, // How many samples the filter processes at a time
///     },
///     ProcessingMode::Sample,
/// )?;
///
/// // Sample inputs
/// let input_signal = InputSignal::new(vec![1.0, -2.5, 3.0])?;
/// let noise_ref = NoiseReference::new(vec![2.0, -1.2, -3.8])?;
///
/// // Adapt the filter to the inputs.
/// let cleaned_signal = lms.adapt(&input_signal, &noise_ref)?;
///
/// # Ok::<(), Error>(())
///```
///
/// # Update Rule
///
/// The weights are updated at every step according to:
///
/// $$\mathbf{w}(n+1) = \mathbf{w}(n) + \mu \mathbf{e}(n)\mathbf{x}(n)$$
///
/// Where:
/// - $\mathbf{w}(n)$ is the filter weight vector.
/// - $μ$ is the adaptation step size.
/// - $\mathbf{e}(n)$ is the scalar error.
/// - $\mathbf{x}(n)$ is the reference input vector.
///
/// [least mean squares]: `https://en.wikipedia.org/wiki/Least_mean_squares_filter`
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LmsFilter<F: Float> {
    inner: FilterBase<F, Lms<F>>,
}
impl<F: Float> LmsFilter<F> {
    /// Initializes a new LMS filter.
    ///
    /// # Errors
    ///
    /// Returns an error if `options.step_size <= 0.0` or `options.window_size == 0`.
    #[allow(clippy::needless_pass_by_value, reason = "All fields are moved")]
    pub fn new(options: LmsOptions<F>, mode: ProcessingMode) -> Result<Self> {
        let window_size = WindowSize::new(options.window_size)?;
        let block_size = mode.block_size()?;

        let lms = Lms::new(options.step_size, window_size)?;
        let filter = FilterBase::new(lms, block_size);

        Ok(Self { inner: filter })
    }
}
impl<F: Float> Deref for LmsFilter<F> {
    type Target = FilterBase<F, Lms<F>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<F: Float> DerefMut for LmsFilter<F> {
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
    fn lms_new_works() {
        let step_size = 0.1;
        let window_size = 16;

        let filter = LmsFilter::new(
            LmsOptions {
                step_size,
                window_size,
            },
            ProcessingMode::Sample,
        )
        .unwrap();

        let window_size = WindowSize::new(window_size).unwrap();
        let expected_inner = FilterBase::new(
            Lms::new(step_size, window_size).unwrap(),
            BlockSize::new(1).unwrap(),
        );

        assert_eq!(filter.inner, expected_inner);
        assert_eq!(filter.window_size(), *window_size);
    }

    #[test]
    fn lms_load_weights_works() {
        let step_size = 0.1;
        let weights = vec![1.0, 2.0, 3.0];
        let window_size = WindowSize::new(weights.len()).unwrap();

        let filter = {
            let mut filter = LmsFilter::new(
                LmsOptions {
                    step_size,
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
                Lms::new(step_size, window_size).unwrap(),
                BlockSize::new(1).unwrap(),
            );
            expected_inner.load_weights(weights.clone()).unwrap();
            expected_inner
        };

        assert_eq!(filter.inner, expected_inner);
        assert_eq!(filter.window_size(), weights.len());
        assert_eq!(filter.block_size(), 1);
    }

    #[test]
    fn block_lms_new_works() {
        let step_size = 0.1;
        let window_size = 16;
        let block_size = 8;

        let filter = LmsFilter::new(
            LmsOptions {
                step_size,
                window_size,
            },
            ProcessingMode::Block { block_size },
        )
        .unwrap();

        let window_size = WindowSize::new(window_size).unwrap();
        let expected_inner = FilterBase::new(
            Lms::new(step_size, window_size).unwrap(),
            BlockSize::new(block_size).unwrap(),
        );

        assert_eq!(filter.inner, expected_inner);
        assert_eq!(filter.window_size(), *window_size);
        assert_eq!(filter.block_size(), block_size);
    }
}
