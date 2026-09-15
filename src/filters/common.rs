//!  Processing functions used by multiple files in the module that shouldn't be publically exported.
use crate::error::{Error, Result};
use crate::types::buffers::NoiseBuffer;
use crate::types::signals::{InputSample, InputSignal, NoiseReference, OutputSample};
use crate::types::{FilterWeights, NoiseEstimate};

pub fn estimate_noise(weights: &FilterWeights, noise: &NoiseBuffer) -> NoiseEstimate {
    // NoiseBuffer is initiated with the same length as weights, therefore we don't need to check
    NoiseEstimate(weights.iter().zip(noise.iter()).map(|(w, x)| w * x).sum())
}

pub fn compute_error(input_sample: InputSample, noise_estimate: NoiseEstimate) -> OutputSample {
    OutputSample(*input_sample - *noise_estimate)
}

/// Compares the lengths of the input signal and the noise reference to ensure
/// that the latter is at least as long as the prior.
///
/// # Errors
///
/// Returns an error if `input_signal.len() > noise_ref.len()`.
pub fn check_signal_lengths(input_signal: &InputSignal, noise_ref: &NoiseReference) -> Result<()> {
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
    use crate::test_utils::{approx_equal, noise_buffer_from};
    use crate::types::{FilterWeights, WindowSize};

    #[test]
    fn estimate_noise_works() {
        let mut weights = FilterWeights::new(WindowSize::new(3).unwrap());

        for (i, val) in [1.0, -2.0, 0.5].iter().enumerate() {
            weights[i] = *val;
        }

        let x_n = noise_buffer_from(&[2.0, 3.0, 4.0]);

        let res = estimate_noise(&weights, &x_n);
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
}
