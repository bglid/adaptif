use std::ops::Range;

use crate::algorithms::Algorithm;

use crate::error::{Error, Result};
use crate::types::buffers::{BlockError, BlockNoiseBuffer};
use crate::types::signals::{InputSample, InputSignal, NoiseReference, OutputSample, OutputSignal};
use crate::types::{BlockSize, FilterWeights, Float, NoiseEstimate, WindowSize};

#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ProcessingMode {
    Sample,
    Block { block_size: usize },
}
impl ProcessingMode {
    /// # Errors
    ///
    /// Returns an error for ``ProcessingMode::Block{ block_size: 0 }``.
    pub fn block_size(&self) -> Result<BlockSize> {
        let block_size = match *self {
            ProcessingMode::Sample => 1,
            ProcessingMode::Block { block_size } => block_size,
        };
        BlockSize::new(block_size)
    }
}

/// Underlying, algorithm-agnostic filter implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterBase<F: Float, A: Algorithm<F>> {
    algorithm: A,
    weights: FilterWeights<F>,
    window_size: WindowSize,
    block_size: BlockSize,
}
impl<F: Float, A: Algorithm<F>> FilterBase<F, A> {
    /// Initializes a filter using the provided algorithm configuration, window size,
    /// and block size.
    /// The weights are intialized to zero.
    pub fn new(algorithm: A, window_size: WindowSize, block_size: BlockSize) -> Self {
        let weights = FilterWeights::new(window_size);

        FilterBase {
            algorithm,
            weights,
            window_size,
            block_size,
        }
    }

    /// This method is intended for loading previously adapted weights
    /// or for non-zero weight initialization, e.g. from a sampled distribution.
    ///
    /// # Errors
    ///
    /// Returns an error if `weights.len() != self.window_size`.
    pub fn load_weights(&mut self, weights: Vec<F>) -> Result<()> {
        if weights.len() == *self.window_size {
            self.weights = FilterWeights::try_from(weights)?;
            Ok(())
        } else {
            Err(Error::WeightsDontMatchWindowSize {
                n_weights: weights.len(),
                window_size: *self.window_size,
            })
        }
    }

    /// Returns the filter's window size. This number is equal to the number of weights.
    pub fn window_size(&self) -> usize {
        *self.window_size
    }

    /// Returns the filter's block size.
    pub fn block_size(&self) -> usize {
        *self.block_size
    }

    /// Returns a reference to the filter's weights.
    pub fn weights(&self) -> &[F] {
        // Returning a slice so that FilterWeights doesn't have to part of the public API
        &self.weights
    }

    /// Iteratively adapts the filter to the input signal and noise reference
    /// using the chosen algorithm, and returns the denoised signal.
    ///
    /// Since adaptation is performed "on-the-fly", the output signal will start noisy
    /// and become less so over time. In order to fully denoise a signal, call `adapt()`
    /// to adapt the filter offline, then call `filter()` to denoise the signal with fixed
    /// weights.
    ///
    /// # Errors
    ///
    /// Returns an error if `input_signal.len() > noise_ref.len()`.
    pub fn adapt(
        &mut self,
        input_signal: &InputSignal<F>,
        noise_ref: &NoiseReference<F>,
    ) -> Result<Vec<F>> {
        check_signal_lengths(input_signal, noise_ref)?;

        let mut noise_ref_buffer = BlockNoiseBuffer::new(&self.weights, self.block_size);
        let mut block_error = BlockError::new(self.block_size);
        let mut cleaned_signal = OutputSignal::new(input_signal);

        let n_samples = input_signal.len();

        for block_start in (0..n_samples).step_by(*self.block_size) {
            let block_end = block_start + *self.block_size;

            if block_end <= n_samples {
                process_block(
                    block_start..block_end,
                    &self.weights,
                    input_signal,
                    noise_ref,
                    &mut noise_ref_buffer,
                    &mut block_error,
                    &mut cleaned_signal,
                );

                self.algorithm
                    .update_block(&mut self.weights, &block_error, &noise_ref_buffer);
            } else {
                // last block: finish off remaining samples w/o updating the weights
                process_block(
                    block_start..n_samples,
                    &self.weights,
                    input_signal,
                    noise_ref,
                    &mut noise_ref_buffer,
                    &mut block_error,
                    &mut cleaned_signal,
                );

                // TODO: try to find a solution so that update_step() _can_ be called here
                // We intentionally don't call update_step() here:
                //
                // Using a fixed-size buffer for the noise reference instead of
                // constructing the full noise matrix X_n is efficient, but it also
                // means the dimensions for the approximated matrix are fixed.
                // If the final block is shorter than the rest, we don't have enough
                // samples to fill this matrix without overlapping with the previous block.
                //
                // The simplest solution is to not update the weights.
                // Since there are no blocks after this one, not updating the
                // weights changes doesn't really matter.
                //
                // (The only case where it changes anything is when calling filter()
                // using the adapted weights, but as long as the input signal used for
                // adaptation is long enough the final update step changes very little.)
            }
        }

        Ok(cleaned_signal.into_inner())
    }

    /// Applies the filter to the input signal without updating the filter coefficients.
    /// This method should be called after adapting the filter to the inputs using `adapt()`.
    ///
    /// # Errors
    ///
    /// Returns an error if `input_signal.len() > noise_ref.len()`.
    pub fn filter(
        &self,
        input_signal: &InputSignal<F>,
        noise_ref: &NoiseReference<F>,
    ) -> Result<Vec<F>> {
        filter_impl(&self.weights, input_signal, noise_ref, self.block_size)
    }
}

/// Because of how monomorphization of generics works, the compiler generates a separate
/// version of `FilterBase` and its methods. Since `filter()` is part of an impl block
/// that is generic over Algorithm, but doesn't use the Algorithm object, this method
/// will be copied 1:1 for each version of `FilterBase` (i.e. `FilterBase<Lms>`,
/// `FilterBase<Nlms>`, etc.). We therefore factor out the (algorithm-independent) logic
/// into `filter_impl()` so that just the call to it will be monomorphized.
fn filter_impl<F>(
    weights: &FilterWeights<F>,
    input_signal: &InputSignal<F>,
    noise_ref: &NoiseReference<F>,
    block_size: BlockSize,
) -> Result<Vec<F>>
where
    F: Float,
{
    check_signal_lengths(input_signal, noise_ref)?;

    let mut noise_ref_buffer = BlockNoiseBuffer::new(weights, block_size);
    let mut block_error = BlockError::new(block_size);
    let mut cleaned_signal = OutputSignal::new(input_signal);

    let n_samples = input_signal.len();

    for block_start in (0..n_samples).step_by(*block_size) {
        let block_end = std::cmp::min(block_start + *block_size, n_samples);

        process_block(
            block_start..block_end,
            weights,
            input_signal,
            noise_ref,
            &mut noise_ref_buffer,
            &mut block_error,
            &mut cleaned_signal,
        );
    }

    Ok(cleaned_signal.into_inner())
}

/// Calculates the error for a block of samples.
/// Like with `filter_impl()`, this function defined outside of `FilterBase`'s impl
/// blocks so that it isn't monomorphized.
///
/// # Panics
///
/// Panics if `range` contains indices that are not within
/// the bounds of `input_signal` or `noise_ref`.
fn process_block<F>(
    range: Range<usize>,
    weights: &FilterWeights<F>,
    input_signal: &InputSignal<F>,
    noise_ref: &NoiseReference<F>,
    noise_ref_buffer: &mut BlockNoiseBuffer<F>,
    block_error: &mut BlockError<F>,
    cleaned_signal: &mut OutputSignal<F>,
) where
    F: Float,
{
    for n in range {
        // Note on the `expect()`: This function is only called internally.
        // If an invalid range is (accidentally) provided by a dev, we don't want pass
        // it to the public caller or have it silently fail, so we panic instead.
        let (input_sample, noise_sample) = input_signal
            .get_sample(n)
            .zip(noise_ref.get_sample(n))
            .expect("process_block() called with invalid range");

        noise_ref_buffer.push(*noise_sample);

        let current_window = noise_ref_buffer
            .iter()
            .copied()
            .take(*weights.window_size());

        let noise_estimate = NoiseEstimate(
            weights
                .iter()
                .copied()
                .zip(current_window)
                .map(|(w, x)| w * x)
                .sum(),
        );

        let error = compute_error(input_sample, noise_estimate);

        block_error.push(error);
        cleaned_signal.push(error);
    }
}

pub fn compute_error<F>(
    input_sample: InputSample<F>,
    noise_estimate: NoiseEstimate<F>,
) -> OutputSample<F>
where
    F: Float,
{
    OutputSample(*input_sample - *noise_estimate)
}

/// Compares the lengths of the input signal and the noise reference to ensure
/// that the latter is at least as long as the prior.
///
/// # Errors
///
/// Returns an error if `input_signal.len() > noise_ref.len()`.
pub fn check_signal_lengths<F>(
    input_signal: &InputSignal<F>,
    noise_ref: &NoiseReference<F>,
) -> Result<()>
where
    F: Float,
{
    if noise_ref.len() < input_signal.len() {
        Err(Error::NoiseRefTooShort {
            input_len: input_signal.len(),
            noise_len: noise_ref.len(),
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::algorithms::Lms;
    use crate::error::Error;
    use crate::test_utils::{all_approx_equal, approx_equal};
    use crate::types::buffers::NoiseWindow;
    use crate::types::signals::{InputSample, OutputSample};

    struct UpdateCallCounter {
        call_count: RefCell<usize>,
    }
    impl UpdateCallCounter {
        pub fn new() -> Self {
            UpdateCallCounter {
                call_count: RefCell::new(0),
            }
        }

        pub fn call_count(&self) -> usize {
            *self.call_count.borrow()
        }
    }
    impl<F: Float> Algorithm<F> for UpdateCallCounter {
        fn update_step(
            &mut self,
            _weights: &mut FilterWeights<F>,
            _error: OutputSample<F>,
            _noise_window: NoiseWindow<F>,
        ) {
            *self.call_count.borrow_mut() += 1;
        }
    }

    fn testing_filter() -> FilterBase<f64, Lms<f64>> {
        let window_size = WindowSize::new(3).unwrap();
        let block_size = BlockSize::new(2).unwrap();
        let weights = [1.0, -2.0, 0.5];

        let mut filter = FilterBase::new(Lms::new(1.0).unwrap(), window_size, block_size);

        for (i, val) in weights.iter().enumerate() {
            filter.weights[i] = *val;
        }

        filter
    }

    #[test]
    fn new_works() {
        let window_size = WindowSize::new(3).unwrap();
        let block_size = BlockSize::new(4).unwrap();
        let filter = FilterBase::new(Lms::new(1.0).unwrap(), window_size, block_size);

        assert_eq!(filter.window_size, window_size);
        assert_eq!(filter.block_size, block_size);
        assert_eq!(filter.algorithm, Lms::new(1.0).unwrap());
        assert!(all_approx_equal(filter.weights.iter(), [0.0; 3].iter()));
    }

    #[test]
    fn window_size_works() {
        let filter = testing_filter();

        assert_eq!(filter.window_size(), *filter.window_size);
    }

    #[test]
    fn block_size_works() {
        let filter = testing_filter();

        assert_eq!(filter.block_size(), *filter.block_size);
    }

    #[test]
    fn weights_works() {
        let filter = testing_filter();

        assert!(all_approx_equal(
            filter.weights().iter(),
            filter.weights.iter()
        ));
    }

    #[test]
    fn load_weights_works() {
        let weights = vec![1.0, 2.0, 3.0];

        let mut filter = FilterBase::new(
            Lms::new(0.1).unwrap(),
            WindowSize::new(weights.len()).unwrap(),
            BlockSize::new(2).unwrap(),
        );
        filter.load_weights(weights.clone()).unwrap();

        assert!(all_approx_equal(weights.iter(), filter.weights().iter()));
    }

    #[test]
    fn load_weights_reject_empty() {
        let empty_vec = vec![];

        let mut filter = testing_filter();

        assert!(matches!(
            filter.load_weights(empty_vec),
            Err(Error::WeightsDontMatchWindowSize { .. })
        ));
    }

    #[test]
    fn load_weights_reject_window_size_mismatch() {
        let vec_too_long = vec![1.0, 2.0, 3.0, 4.0];
        let vec_too_short = vec![1.0, 2.0];

        let mut filter = FilterBase::new(
            Lms::new(0.1).unwrap(),
            WindowSize::new(3).unwrap(),
            BlockSize::new(1).unwrap(),
        );

        assert!(matches!(
            filter.load_weights(vec_too_long),
            Err(Error::WeightsDontMatchWindowSize { .. })
        ));

        assert!(matches!(
            filter.load_weights(vec_too_short),
            Err(Error::WeightsDontMatchWindowSize { .. })
        ));
    }

    #[test]
    fn compute_error_works() {
        assert!(approx_equal(
            *compute_error(InputSample(5.0), NoiseEstimate(3.5)),
            1.5,
            1e-6
        ));
    }

    #[test]
    fn adapt_weights_update() {
        let mut filter = testing_filter();

        let weights_before = filter.weights.clone();

        let input = InputSignal::new(vec![5.0, 3.5, 2.6, -8.4]).unwrap();
        let noise = NoiseReference::new(vec![3.0, 2.8, -1.7, 2.24]).unwrap();

        filter.adapt(&input, &noise).unwrap();

        assert!(!all_approx_equal(
            filter.weights.iter(),
            weights_before.iter()
        ));
    }

    #[test]
    fn filter_weights_dont_update() {
        let filter = testing_filter();

        let weights_before = filter.weights.clone();

        let input = InputSignal::new(vec![5.0, 3.5, 2.6, -8.4]).unwrap();
        let noise = NoiseReference::new(vec![3.0, 2.8, -1.7, 2.24]).unwrap();

        filter.filter(&input, &noise).unwrap();

        assert!(all_approx_equal(
            filter.weights.iter(),
            weights_before.iter()
        ));
    }

    #[test]
    fn adapt_weights_len_invariant() {
        let mut filter = testing_filter();

        let before = filter.weights.len();

        let input = InputSignal::new(vec![1.0, 2.0, 3.0]).unwrap();
        let noise = NoiseReference::new(vec![4.0, 5.0, 6.0]).unwrap();

        filter.adapt(&input, &noise).unwrap();
        let after = filter.weights.len();

        assert_eq!(before, after);
    }

    #[test]
    fn reject_shorter_noise_ref() {
        let mut filter = testing_filter();

        let input = InputSignal::new(vec![1.0, 2.0, 3.0]).unwrap();
        let noise = NoiseReference::new(vec![4.0, 5.0]).unwrap();

        assert!(matches!(
            filter.adapt(&input, &noise),
            Err(Error::NoiseRefTooShort {
                input_len: 3,
                noise_len: 2
            })
        ));

        assert!(matches!(
            filter.filter(&input, &noise),
            Err(Error::NoiseRefTooShort {
                input_len: 3,
                noise_len: 2
            })
        ));
    }

    #[test]
    fn allow_longer_noise_ref() {
        let mut filter = testing_filter();

        let input = InputSignal::new(vec![1.0, 2.0]).unwrap();
        let noise = NoiseReference::new(vec![4.0, 5.0, 6.0]).unwrap();

        filter.adapt(&input, &noise).unwrap();
        filter.filter(&input, &noise).unwrap();
    }

    #[test]
    /// checks that if the end of the final block falls on
    /// the final sample, the update step is still called.
    fn n_samples_multiple_of_block_size() {
        let block_size = 2;
        let mut filter = FilterBase::new(
            UpdateCallCounter::new(),
            WindowSize::new(3).unwrap(),
            BlockSize::new(block_size).unwrap(),
        );

        let input = InputSignal::new(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let noise = NoiseReference::new(vec![4.0, 5.0, 6.0, 7.0]).unwrap();

        filter.adapt(&input, &noise).unwrap();
        assert_eq!(filter.algorithm.call_count(), 2 * block_size);

        filter.filter(&input, &noise).unwrap();
    }

    #[test]
    /// checks that if the end of the final block doesn't align
    /// with the final sample, the update step is NOT called.
    fn n_samples_not_multiple_of_block_size() {
        let block_size = 2;
        let mut filter = FilterBase::new(
            UpdateCallCounter::new(),
            WindowSize::new(3).unwrap(),
            BlockSize::new(block_size).unwrap(),
        );

        let input = InputSignal::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let noise = NoiseReference::new(vec![4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        filter.adapt(&input, &noise).unwrap();
        // update not called on final block because the shapes don't match
        assert_eq!(filter.algorithm.call_count(), 2 * block_size);

        filter.filter(&input, &noise).unwrap();
    }
}
