use crate::algorithms::Algorithm;
use crate::error::Result;
use crate::types::buffers::NoiseBuffer;
use crate::types::signals::{
    InputSample, InputSignal, NoiseReference, NoiseSample, OutputSample, OutputSignal,
};
use crate::types::{FilterWeights, WindowSize};

use crate::filters::AdaptiveFilter;
use crate::filters::common::{check_signal_lengths, compute_error, estimate_noise};

// TODO: make f64 generic

/// Underlying, algorithm-agnostic filter implementation.
///
/// Typically, it's more convenient to use an alias like `LMSFilter` over its equivalent `FilterBase<Lms>`.
/// As such, `FilterBase` is mainly recommended for use with custom algorithms.
#[derive(Debug, Clone)]
pub struct FilterBase<A: Algorithm> {
    algorithm: A,
    weights: FilterWeights,
    window_size: WindowSize,
}
impl<A: Algorithm> FilterBase<A> {
    /// Initializes a filter using the provided algorithm configuration and window size.
    /// The weights are intialized to zero.
    ///
    /// # Errors
    ///
    /// Returns an error if `window_size == 0`.
    pub fn new(algorithm: A, window_size: usize) -> Result<Self> {
        let window_size = WindowSize::new(window_size)?;
        let weights = FilterWeights::new(window_size);

        Ok(FilterBase {
            algorithm,
            weights,
            window_size,
        })
    }

    /// Creates a filter of the specified algorithm with set weights.
    /// The filter's window size is equal to `weights.len()`.
    ///
    /// This method is intended for loading previously adapted weights
    /// or for non-zero weight initialization, e.g. from a sampled distribution.
    ///
    /// # Errors
    ///
    /// Returns an error if `weights.is_empty()`.
    // TODO: move weights instead of ref
    pub fn from_weights(algorithm: A, weights: &[f64]) -> Result<Self> {
        let weights = FilterWeights::try_from(weights)?;
        let window_size = weights.window_size();

        Ok(FilterBase {
            algorithm,
            weights,
            window_size,
        })
    }

    // TODO: Impl Default

    /// Returns the filter's window size. This number is equal to the number of weights.
    pub fn window_size(&self) -> usize {
        *self.window_size
    }

    /// Returns a reference to the filter's weights.
    pub fn weights(&self) -> &[f64] {
        // Returning a slice so that FilterWeights doesn't have to part of the public API
        &self.weights
    }

    fn process_sample(
        &self,
        noise_ref_buffer: &mut NoiseBuffer,
        input_sample: InputSample,
        noise_sample: NoiseSample,
    ) -> OutputSample {
        noise_ref_buffer.push(*noise_sample);

        let noise_estimate = estimate_noise(&self.weights, noise_ref_buffer);

        compute_error(input_sample, noise_estimate)
    }
}

impl<A: Algorithm> AdaptiveFilter for FilterBase<A> {
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
    fn adapt(
        &mut self,
        input_signal: &InputSignal,
        noise_ref: &NoiseReference,
    ) -> Result<Vec<f64>> {
        check_signal_lengths(input_signal, noise_ref)?;

        let mut noise_ref_buffer = NoiseBuffer::new(&self.weights);
        let mut cleaned_signal = OutputSignal::new(input_signal);

        for n in 0..input_signal.len() {
            // We set n_samples = input_signal.len() and called check_signal_lengths() (putting in comment so fmt doesn't split lines)
            #[allow(clippy::unwrap_used, reason = "Bounds checked")]
            #[allow(clippy::missing_panics_doc, reason = "Bounds checked")]
            let error = self.process_sample(
                &mut noise_ref_buffer,
                input_signal.get_sample(n).unwrap(),
                noise_ref.get_sample(n).unwrap(),
            );

            cleaned_signal.push(error);

            self.algorithm
                .update_step(&mut self.weights, error, &noise_ref_buffer);
        }

        Ok(cleaned_signal.into_inner())
    }

    /// Applies the filter to the input signal without updating the filter coefficients.
    /// This method should be called after adapting the filter to the inputs using `adapt()`.
    ///
    /// # Errors
    ///
    /// Returns an error if `input_signal.len() > noise_ref.len()`.
    fn filter(&self, input_signal: &InputSignal, noise_ref: &NoiseReference) -> Result<Vec<f64>> {
        check_signal_lengths(input_signal, noise_ref)?;

        let mut noise_ref_buffer = NoiseBuffer::new(&self.weights);
        let mut cleaned_signal = OutputSignal::new(input_signal);

        for n in 0..input_signal.len() {
            // We set n_samples = input_signal.len() and called check_signal_lengths()
            #[allow(clippy::unwrap_used, reason = "Bounds checked")]
            #[allow(clippy::missing_panics_doc, reason = "Bounds checked")]
            let error = self.process_sample(
                &mut noise_ref_buffer,
                input_signal.get_sample(n).unwrap(),
                noise_ref.get_sample(n).unwrap(),
            );

            cleaned_signal.push(error);
        }

        Ok(cleaned_signal.into_inner())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;

    use crate::algorithms::Lms;
    use crate::error::Error;
    use crate::test_utils::all_approx_equal;

    fn testing_filter() -> FilterBase<Lms> {
        let window_size = 3;
        let weights = [1.0, -2.0, 0.5];

        let mut filter = FilterBase::<Lms>::new(Lms::new(1.0).unwrap(), window_size).unwrap();
        for (i, val) in weights.iter().enumerate() {
            filter.weights[i] = *val;
        }
        filter
    }

    #[test]
    fn new_works() {
        let window_size = 3;
        let filter = FilterBase::<Lms>::new(Lms::new(1.0).unwrap(), window_size).unwrap();

        assert_eq!(filter.window_size, WindowSize::new(window_size).unwrap());
        assert_eq!(filter.algorithm, Lms::new(1.0).unwrap());
        assert!(all_approx_equal(filter.weights.iter(), [0.0; 3].iter()));
    }

    #[test]
    fn window_size_works() {
        let filter = testing_filter();

        assert_eq!(filter.window_size(), *filter.window_size);
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
    fn adapt_weights_update() {
        let mut filter = testing_filter();

        let weights_before = filter.weights.clone();

        let input = InputSignal::new(&[5.0, 3.5, 2.6, -8.4]).unwrap();
        let noise = NoiseReference::new(&[3.0, 2.8, -1.7, 2.24]).unwrap();

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

        let input = InputSignal::new(&[5.0, 3.5, 2.6, -8.4]).unwrap();
        let noise = NoiseReference::new(&[3.0, 2.8, -1.7, 2.24]).unwrap();

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

        let input = InputSignal::new(&[1.0, 2.0, 3.0]).unwrap();
        let noise = NoiseReference::new(&[4.0, 5.0, 6.0]).unwrap();

        filter.adapt(&input, &noise).unwrap();
        let after = filter.weights.len();

        assert_eq!(before, after);
    }

    #[test]
    fn reject_shorter_noise_ref() {
        let mut filter = testing_filter();

        let input = InputSignal::new(&[1.0, 2.0, 3.0]).unwrap();
        let noise = NoiseReference::new(&[4.0, 5.0]).unwrap();

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

        let input = InputSignal::new(&[1.0, 2.0]).unwrap();
        let noise = NoiseReference::new(&[4.0, 5.0, 6.0]).unwrap();

        filter.adapt(&input, &noise).unwrap();
        filter.filter(&input, &noise).unwrap();
    }

    #[test]
    fn from_weights_vec() {
        let weights = vec![1.0, 2.0, 3.0];

        let filter = FilterBase::<Lms>::from_weights(Lms::new(1.0).unwrap(), &weights).unwrap();

        assert!(all_approx_equal(weights.iter(), filter.weights().iter()));
    }

    #[test]
    fn from_weights_arr() {
        let weights = [1.0, 2.0, 3.0];

        let filter = FilterBase::<Lms>::from_weights(Lms::new(1.0).unwrap(), &weights).unwrap();

        assert!(all_approx_equal(weights.iter(), filter.weights().iter()));
    }

    #[test]
    fn from_weights_reject_empty() {
        let empty_vec = vec![];

        assert!(matches!(
            FilterBase::<Lms>::from_weights(Lms::new(1.0).unwrap(), &empty_vec),
            Err(Error::EmptyInputArr)
        ));
        assert!(matches!(
            FilterBase::<Lms>::from_weights(Lms::new(1.0).unwrap(), &[]),
            Err(Error::EmptyInputArr)
        ));
    }
}
