use crate::types::buffers::{BlockError, BlockNoiseBuffer, NoiseWindow};
use crate::types::signals::OutputSample;
use crate::types::{FilterWeights, Float};

/// Trait used for implementing algorithms with block-based processing used in conjuction with
/// `FilterBase`.
pub trait Algorithm<F: Float> {
    /// Updates the weights for the next block based on the algorithm's update rules.
    /// This function is called for every processing block by the filter during adapation.
    /// `error` are the cleaned samples from the current block.
    /// `noise_ref` is the noise reference signal within the current processing window (the $k$ most recent samples).
    ///
    fn update_step(
        &mut self,
        weights: &mut FilterWeights<F>,
        error: OutputSample<F>,
        noise_window: NoiseWindow<F>,
    );

    fn update_block(
        &mut self,
        weights: &mut FilterWeights<F>,
        error: &BlockError<F>,
        noise_ref: &BlockNoiseBuffer<F>,
    ) {
        for (window_start, window_error) in error.iter().copied().enumerate() {
            #[allow(
                clippy::unwrap_used,
                clippy::missing_panics_doc,
                reason = "get_window() can only fail if window_start >= noise_ref.block_size.
                error.len() == block_size, so window_start is always < block_size"
            )]
            let current_window = noise_ref.get_window(window_start).unwrap();

            self.update_step(weights, OutputSample(window_error), current_window);
        }
    }
}
