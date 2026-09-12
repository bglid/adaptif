use crate::algorithms::BlockAlgorithm;

use crate::error::Result;
use crate::filters::common::{check_signal_lengths, compute_error};
use crate::types::buffers::{BlockNoiseBuffer, ErrorBuffer};
use crate::types::signals::{InputSample, InputSignal, NoiseReference, NoiseSample, OutputSignal};
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

    pub fn window_self(&self) -> usize {
        *self.window_size
    }

    pub fn block_self(&self) -> usize {
        *self.block_size
    }

    /// # Errors
    ///
    /// Returns an error if `input_signal.len() > noise_ref.len()`.
    pub fn adapt(
        &mut self,
        input_signal: &InputSignal,
        noise_ref: &NoiseReference,
    ) -> Result<Vec<f64>> {
        check_signal_lengths(input_signal, noise_ref)?;

        let mut noise_ref_buffer = BlockNoiseBuffer::new(&self.weights, self.block_size);
        let mut block_error = ErrorBuffer::new(self.block_size);
        let mut cleaned_signal = OutputSignal::new(input_signal);

        for block_start in (0..input_signal.len()).step_by(*self.block_size) {
            for n in block_start..block_start + *self.block_size {
                // padding with 0.0 if necessary
                let input_sample = input_signal.get_sample(n).unwrap_or(InputSample(0.0));
                let noise_sample = noise_ref.get_sample(n).unwrap_or(NoiseSample(0.0));

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

            self.algorithm
                .update_block(&mut self.weights, &block_error, &noise_ref_buffer);
        }

        Ok(cleaned_signal.into_inner())
    }
}
