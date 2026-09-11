use crate::algorithms::Algorithm;
use crate::error::{Error, Result};
use crate::types::buffers::NoiseBuffer;
use crate::types::signals::{
    InputSample, InputSignal, NoiseReference, NoiseSample, OutputSample, OutputSignal,
};
use crate::types::{FilterWeights, NoiseEstimate, WindowSize};

// TODO: make f64 generic

/// Underlying, algorithm-agnostic filter implementation.
///
/// Typically, it's more convenient to use an alias like `LMSFilter` over its equivalent `FilterBase<LeastMeanSquares>`.
/// As such, `FilterBase` is mainly recommended for use with custom algorithms.
#[derive(Debug, Clone)]
pub struct FilterBase<A: Algorithm> {
    algorithm: A,
    weights: FilterWeights,
    window_size: WindowSize,
}
impl<A: Algorithm> FilterBase<A> {
    /// Initializes a filter using the provided algorithm configuration and window size.
    /// The weights are samples from a normal distribution with $\mu = 0.0 and $\sigma$ = 5e-5.
    pub fn new(algorithm: A, window_size: usize) -> Option<Self> {
        let window_size = WindowSize::new(window_size)?;

        let weights = FilterWeights::new(window_size);

        Some(FilterBase {
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

    // TODO: getter fn for weights + loading weights w/ setter (from_weights() or load_weights())

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
            let error = process_sample(
                &self.weights,
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
    pub fn filter(
        &self,
        input_signal: &InputSignal,
        noise_ref: &NoiseReference,
    ) -> Result<Vec<f64>> {
        check_signal_lengths(input_signal, noise_ref)?;

        let mut noise_ref_buffer = NoiseBuffer::new(&self.weights);
        let mut cleaned_signal = OutputSignal::new(input_signal);

        for n in 0..input_signal.len() {
            // We set n_samples = input_signal.len() and called check_signal_lengths()
            #[allow(clippy::unwrap_used, reason = "Bounds checked")]
            #[allow(clippy::missing_panics_doc, reason = "Bounds checked")]
            let error = process_sample(
                &self.weights,
                &mut noise_ref_buffer,
                input_signal.get_sample(n).unwrap(),
                noise_ref.get_sample(n).unwrap(),
            );

            cleaned_signal.push(error);
        }

        Ok(cleaned_signal.into_inner())
    }
}

fn process_sample(
    weights: &FilterWeights,
    noise_ref_buffer: &mut NoiseBuffer,
    input_sample: InputSample,
    noise_sample: NoiseSample,
) -> OutputSample {
    noise_ref_buffer.push(*noise_sample);

    let noise_estimate = estimate_noise(weights, noise_ref_buffer);

    compute_error(input_sample, noise_estimate)
}

fn estimate_noise(weights: &FilterWeights, x_n: &NoiseBuffer) -> NoiseEstimate {
    // NoiseBuffer is initiated with the same length as weights, therefore we don't need to check
    NoiseEstimate(weights.iter().zip(x_n.iter()).map(|(w, x)| w * x).sum())
}

fn compute_error(input_sample: InputSample, noise_estimate: NoiseEstimate) -> OutputSample {
    OutputSample(*input_sample - *noise_estimate)
}

fn check_signal_lengths(input_signal: &InputSignal, noise_ref: &NoiseReference) -> Result<()> {
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
    use super::*;

    use crate::test_utils::{all_approx_equal, approx_equal, noise_buffer_from};

    struct TestAlgorithm;
    impl Algorithm for TestAlgorithm {
        fn update_step(
            &self,
            weights: &mut FilterWeights,
            error: OutputSample,
            noise_ref: &NoiseBuffer,
        ) {
            for (i, w) in weights.iter_mut().enumerate() {
                *w += (*error) * noise_ref.get(i).unwrap();
            }
        }
    }

    fn testing_filter() -> FilterBase<TestAlgorithm> {
        let window_size = 3;
        let weights = [1.0, -2.0, 0.5];

        let mut filter = FilterBase::<TestAlgorithm>::new(TestAlgorithm {}, window_size).unwrap();
        for (i, val) in weights.iter().enumerate() {
            filter.weights[i] = *val;
        }
        filter
    }

    #[test]
    fn estimate_noise_works() {
        let filter = testing_filter();

        let x_n = noise_buffer_from(&[2.0, 3.0, 4.0]);

        let res = estimate_noise(&filter.weights, &x_n);
        assert!(approx_equal(*res, -2.0, 1e-6));
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
}
