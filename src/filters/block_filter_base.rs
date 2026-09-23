use std::ops::Range;

use crate::algorithms::BlockAlgorithm;

use crate::error::Result;
use crate::filters::AdaptiveFilter;
use crate::filters::common::{check_signal_lengths, compute_error};
use crate::types::buffers::{BlockNoiseBuffer, ErrorBuffer};
use crate::types::signals::{InputSignal, NoiseReference, OutputSignal};
use crate::types::{BlockSize, FilterWeights, NoiseEstimate, WindowSize};

pub struct BlockFilterBase<B: BlockAlgorithm> {
    algorithm: B,
    weights: FilterWeights,
    window_size: WindowSize,
    block_size: BlockSize,
}
impl<B: BlockAlgorithm> BlockFilterBase<B> {
    /// Initializes a filter using the provided algorithm configuration, window size,
    /// and block size.
    /// The weights are intialized to zero.
    ///
    /// # Errors
    ///
    /// Returns an error if `window_size == 0` or `block_size == 0`.
    pub fn new(algorithm: B, window_size: usize, block_size: usize) -> Result<Self> {
        let window_size = WindowSize::new(window_size)?;
        let block_size = BlockSize::new(block_size)?;
        let weights = FilterWeights::new(window_size);

        Ok(BlockFilterBase {
            algorithm,
            weights,
            window_size,
            block_size,
        })
    }

    /// Creates a filter of the specified algorithm with set weights.
    /// Ownership of the weights is transferred to `BlockFilterBase`.
    /// The filter's window size is equal to `weights.len()`.
    ///
    /// This method is intended for loading previously adapted weights
    /// or for non-zero weight initialization, e.g. from a sampled distribution.
    ///
    /// # Errors
    ///
    /// Returns an error if `weights.is_empty()`.
    pub fn from_weights(algorithm: B, weights: Vec<f64>, block_size: usize) -> Result<Self> {
        let weights = FilterWeights::try_from(weights)?;
        let window_size = weights.window_size();
        let block_size = BlockSize::new(block_size)?;

        Ok(BlockFilterBase {
            algorithm,
            weights,
            window_size,
            block_size,
        })
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
    pub fn weights(&self) -> &[f64] {
        // Returning a slice so that FilterWeights doesn't have to part of the public API
        &self.weights
    }

    /// # Panics
    ///
    /// Panics if `range` contains indices that are not within
    /// the bounds of `input_signal` or `noise_ref`.
    fn process_block(
        &self,
        range: Range<usize>,
        input_signal: &InputSignal,
        noise_ref: &NoiseReference,
        noise_ref_buffer: &mut BlockNoiseBuffer,
        block_error: &mut ErrorBuffer,
        cleaned_signal: &mut OutputSignal,
    ) {
        for n in range {
            #[allow(
                clippy::expect_used,
                reason = "This function is only called internally.
                If an invalid range is (accidentally) provided, we don't want pass
                it to the public caller or have it silently fail, so we panic instead."
            )]
            let (input_sample, noise_sample) = input_signal
                .get_sample(n)
                .zip(noise_ref.get_sample(n))
                .expect("process_block() called with invalid range");

            noise_ref_buffer.push(*noise_sample);

            let current_window = noise_ref_buffer.iter().take(*self.window_size);
            let noise_estimate = NoiseEstimate(
                self.weights
                    .iter()
                    .zip(current_window)
                    .map(|(w, x)| w * x)
                    .sum(),
            );

            let error = compute_error(input_sample, noise_estimate);
            block_error.push(error);
            cleaned_signal.push(error);
        }
    }
}

impl<B: BlockAlgorithm> AdaptiveFilter for BlockFilterBase<B> {
    /// # Errors
    ///
    /// Returns an error if `input_signal.len() > noise_ref.len()`.
    fn adapt(
        &mut self,
        input_signal: &InputSignal,
        noise_ref: &NoiseReference,
    ) -> Result<Vec<f64>> {
        check_signal_lengths(input_signal, noise_ref)?;

        let mut noise_ref_buffer = BlockNoiseBuffer::new(&self.weights, self.block_size);
        let mut block_error = ErrorBuffer::new(self.block_size);
        let mut cleaned_signal = OutputSignal::new(input_signal);

        let n_samples = input_signal.len();

        for block_start in (0..n_samples).step_by(*self.block_size) {
            let block_end = block_start + *self.block_size;

            if block_end <= n_samples {
                self.process_block(
                    block_start..block_end,
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
                self.process_block(
                    block_start..n_samples,
                    input_signal,
                    noise_ref,
                    &mut noise_ref_buffer,
                    &mut block_error,
                    &mut cleaned_signal,
                );

                // We intentionally don't call update_block() here:
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

    /// # Errors
    ///
    /// Returns an error if `input_signal.len() > noise_ref.len()`.
    fn filter(&self, input_signal: &InputSignal, noise_ref: &NoiseReference) -> Result<Vec<f64>> {
        check_signal_lengths(input_signal, noise_ref)?;

        let mut noise_ref_buffer = BlockNoiseBuffer::new(&self.weights, self.block_size);
        let mut block_error = ErrorBuffer::new(self.block_size);
        let mut cleaned_signal = OutputSignal::new(input_signal);

        let n_samples = input_signal.len();

        for block_start in (0..n_samples).step_by(*self.block_size) {
            let block_end = std::cmp::min(block_start + *self.block_size, n_samples);

            self.process_block(
                block_start..block_end,
                input_signal,
                noise_ref,
                &mut noise_ref_buffer,
                &mut block_error,
                &mut cleaned_signal,
            );
        }

        Ok(cleaned_signal.into_inner())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::algorithms::Lms;
    use crate::error::Error;
    use crate::test_utils::all_approx_equal;

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
    impl BlockAlgorithm for UpdateCallCounter {
        fn update_block(
            &self,
            _weights: &mut FilterWeights,
            _error: &ErrorBuffer,
            _noise_ref: &BlockNoiseBuffer,
        ) {
            *self.call_count.borrow_mut() += 1;
        }
    }

    fn testing_filter() -> BlockFilterBase<Lms> {
        let window_size = 3;
        let block_size = 2;
        let weights = [1.0, -2.0, 0.5];

        let mut filter =
            BlockFilterBase::<Lms>::new(Lms::new(1.0).unwrap(), window_size, block_size).unwrap();

        for (i, val) in weights.iter().enumerate() {
            filter.weights[i] = *val;
        }

        filter
    }

    #[test]
    fn new_works() {
        let window_size = 3;
        let block_size = 4;
        let filter =
            BlockFilterBase::<Lms>::new(Lms::new(1.0).unwrap(), window_size, block_size).unwrap();

        assert_eq!(filter.window_size, WindowSize::new(window_size).unwrap());
        assert_eq!(filter.block_size, BlockSize::new(block_size).unwrap());
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
    fn from_weights_works() {
        let weights = vec![1.0, 2.0, 3.0];

        let filter =
            BlockFilterBase::<Lms>::from_weights(Lms::new(1.0).unwrap(), weights.clone(), 1024)
                .unwrap();

        assert!(all_approx_equal(weights.iter(), filter.weights().iter()));
    }

    #[test]
    fn from_weights_reject_empty() {
        let empty_vec = vec![];

        assert!(matches!(
            BlockFilterBase::<Lms>::from_weights(Lms::new(1.0).unwrap(), empty_vec, 1024),
            Err(Error::EmptyInputArr)
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
        let mut filter =
            BlockFilterBase::<UpdateCallCounter>::new(UpdateCallCounter::new(), 3, block_size)
                .unwrap();

        let input = InputSignal::new(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let noise = NoiseReference::new(vec![4.0, 5.0, 6.0, 7.0]).unwrap();

        filter.adapt(&input, &noise).unwrap();
        assert_eq!(filter.algorithm.call_count(), 2);

        filter.filter(&input, &noise).unwrap();
    }

    #[test]
    /// checks that if the end of the final block doesn't align
    /// with the final sample, the update step is NOT called.
    fn n_samples_not_multiple_of_block_size() {
        let block_size = 2;
        let mut filter =
            BlockFilterBase::<UpdateCallCounter>::new(UpdateCallCounter::new(), 3, block_size)
                .unwrap();

        let input = InputSignal::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let noise = NoiseReference::new(vec![4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        filter.adapt(&input, &noise).unwrap();
        // update not called on final block because the shapes don't match
        assert_eq!(filter.algorithm.call_count(), 2);

        filter.filter(&input, &noise).unwrap();
    }
}
