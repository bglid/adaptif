use crate::types::{FilterWeights, OutputSample, SampleBuffer};

mod lms;
pub use lms::LeastMeanSquares;

/// Trait used for implementing algorithms with sample-based processing used in conjuction with
/// `SampleFilter`.
pub trait Algorithm {
    /// Updates the weights for the next time step based on the algorithm's update rules.
    /// This function is called every processing iteration by the filter during adapation.
    /// `error` is the cleaned sample from the current time step.
    /// `noise_ref` is the noise reference signal within the current processing window (the $k$ most recent samples).
    ///
    fn update_step(
        &self,
        weights: &mut FilterWeights,
        error: OutputSample,
        noise_ref: &SampleBuffer,
    );
}

/// Trait used for implementing algorithms with block-based processing used in conjuction with
/// `BlockFilter`.
pub trait BlockAlgorithm {
    /// Updates the weights for the next block based on the algorithm's update rules.
    /// This function is called for every processing block by the filter during adapation.
    /// `error` are the cleaned samples from the current block.
    /// `noise_ref` is the noise reference signal within the current processing window (the $k$ most recent samples).
    ///
    fn update_block(
        &self,
        weights: &mut FilterWeights,
        error: &SampleBuffer, // TODO: replace with own type
        noise_ref: &SampleBuffer,
    );
}
