use crate::types::buffers::NoiseWindowView;
use crate::types::{Float, WindowSize};

/// Trait used for implementing algorithms.
pub trait Algorithm<F: Float> {
    /// Updates the weights for the next block based on the algorithm's update rules.
    /// This function is called for every processing block by the filter during adapation.
    /// `error` are the cleaned samples from the current block.
    /// `noise_ref` is the noise reference signal within the current processing window (the $k$ most recent samples).
    fn update_step(&mut self, weights: &mut [F], error: F, noise_window: NoiseWindowView<F>);

    /// Window size used by the algorithm.
    fn window_size(&self) -> WindowSize;
}
