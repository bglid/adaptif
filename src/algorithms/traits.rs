use crate::types::buffers::NoiseWindowView;
use crate::types::{Float, WindowSize};

/// Trait used for implementing algorithms.
///
/// # Example
///
/// ``` rust
/// # use adaptif::Error;
/// use adaptif::Algorithm;
/// use adaptif::types::NoiseWindowView;
/// use adaptif::types:: WindowSize;
///
/// // Create a struct to hold any required parameters or state
/// pub struct MyAlgorithm {
///     // Some parameter
///     pub alpha: f64,
///     // The algorithm's window size
///     pub window_size: WindowSize,
/// }
/// // Implement the Algorithm trait so the algorithm can be used with SampleFilterBase
/// impl Algorithm<f64> for MyAlgorithm {
///     // This function is called every iteration during adaptation to update the weights
///     fn update_step(&mut self, weights: &mut [f64], error: f64, noise_window: NoiseWindowView<f64>) {
///         for (w, x) in weights.iter_mut().zip(noise_window.iter()) {
///             *w += self.alpha * error * x;
///         }
///     }
///
///     // Certain algorithms need to know the filter's window size to initialize buffers, etc.
///     // To make sure `FilterBase` has the same window size as the algorithm object, we
///     // have to define this function so that it can be access when constructing `FilterBase`.
///     fn window_size(&self) -> WindowSize {
///         self.window_size
///     }
/// }
///
/// # Ok::<(), Error>(())
/// ```
pub trait Algorithm<F: Float> {
    /// Updates the weights for the next block based on the algorithm's update rules.
    /// This function is called for every processing block by the filter during adapation.
    /// `error` are the cleaned samples from the current block.
    /// `noise_ref` is the noise reference signal within the current processing window (the $k$ most recent samples).
    fn update_step(&mut self, weights: &mut [F], error: F, noise_window: NoiseWindowView<F>);

    /// Window size used by the algorithm.
    fn window_size(&self) -> WindowSize;
}
