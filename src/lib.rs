//! This crate implements several [adaptive filters] for digital signal processing.
//!
//! Unlike a regular filter, which is the same for every input signal, an adaptive
//! filter updates its inner state to the input signal(s) using a cost function,
//! and is thus able to adapt over time.
//!
//! An example application is [Adaptive Noise Cancelling (ANC)], where an adaptive
//! filter is used to remove noise from an audio signal by adapting to a reference
//! noise signal.
//!
//! [adaptive filters]: https://en.wikipedia.org/wiki/Adaptive_filter
//! [Adaptive Noise Cancelling (ANC)]: https://en.wikipedia.org/wiki/Adaptive_noise_cancelling
//!
//! # Overview
//!
//! Currently, the following filters are available:
//! - [`LmsFilter`], which uses a [least mean squares] algorithm.
//! - [`NlmsFilter`], which uses a [normalized least mean squares] algorithm.
//! - [`RlsFilter`], which uses a [recursive least squares] algorithm.
//!
//! [`LmsFilter`]: filters/struct.LmsFilter.html
//! [`NlmsFilter`]: filters/struct.NlmsFilter.html
//! [`RlsFilter`]: filters/struct.RlsFilter.html
//!
//! [least mean squares]: https://en.wikipedia.org/wiki/Least_mean_squares_filter
//! [normalized least mean squares]: https://en.wikipedia.org/wiki/Least_mean_squares_filter#Normalized_least_mean_squares_filter_(NLMS)
//! [recursive least squares]: https://en.wikipedia.org/wiki/Recursive_least_squares_filter
//!
//! It is also possible to create custom algorithms by implementing the [`Algorithm`]
//! trait. An object that implements [`Algorithm`] can be used with [`FilterBase`]
//! to create a working filter.
//!
//! [`FilterBase`]: filters/struct.FilterBase.html
//!
//! There are also Python bindings available for this crate.
//! See the [Python package] of the same name.
//!
//! [Python package]: https://pypi.org/project/adaptif
//!
//! # Usage
//!
//! Add `adaptif` as a project dependency with Cargo:
//!
//! ```sh
//! $ cargo add adaptif
//! ```
//!
//! To use a filter:
//!
//! ``` rust
//! # use adaptif::Error;
//! use adaptif::filters::{LmsFilter, LmsOptions, ProcessingMode};
//! use adaptif::types::{InputSignal, NoiseReference};
//!
//! // Initialize the filter
//! let mut lms = LmsFilter::new(
//!     LmsOptions {
//!         step_size: 0.1,  // How fast the filter adapts
//!         window_size: 16, // How many samples the filter processes at a time
//!     },
//!     ProcessingMode::Sample,
//! )?;
//!
//! // Sample inputs
//! let input_signal = InputSignal::new(vec![1.0, -2.5, 3.0])?;
//! let noise_ref = NoiseReference::new(vec![2.0, -1.2, -3.8])?;
//!
//! // Adapt the filter to the inputs.
//! let cleaned_signal = lms.adapt(&input_signal, &noise_ref)?;
//!
//! // ... or apply the previously adapted filter without updating it.
//! let cleaned_signal = lms.filter(&input_signal, &noise_ref)?;
//!
//! # Ok::<(), Error>(())
//! ```
//!
//! To implement a custom algorithm:
//!
//! ``` rust
//! # use adaptif::Error;
//! use adaptif::Algorithm;
//! use adaptif::filters::FilterBase;
//! use adaptif::types::NoiseWindowView;
//! use adaptif::types::{BlockSize, WindowSize};
//! use adaptif::types::{InputSignal, NoiseReference};
//!
//! // Create a struct to hold any required parameters or state
//! pub struct MyAlgorithm {
//!     // Some parameter
//!     pub alpha: f64,
//!     // The algorithm's window size
//!     pub window_size: WindowSize,
//! }
//! // Implement the Algorithm trait so the algorithm can be used with SampleFilterBase
//! impl Algorithm<f64> for MyAlgorithm {
//!     // This function is called every iteration during adaptation to update the weights
//!     fn update_step(&mut self, weights: &mut [f64], error: f64, noise_window: NoiseWindowView<f64>) {
//!         for (w, x) in weights.iter_mut().zip(noise_window.iter()) {
//!             *w += self.alpha * error * x;
//!         }
//!     }
//!
//!     // Certain algorithms need to know the filter's window size to initialize buffers, etc.
//!     // To make sure `FilterBase` has the same window size as the algorithm object, we
//!     // have to define this function so that it can be access when constructing `FilterBase`.
//!     fn window_size(&self) -> WindowSize {
//!         self.window_size
//!     }
//! }
//!
//! // Define the algorithm parameters
//! let algorithm_cfg = MyAlgorithm {
//!     alpha: 1.0,
//!     window_size: WindowSize::new(16)?,
//! };
//!
//! // Instantiate the filter using FilterBase and our custom algorithm
//! let mut filter = FilterBase::new(algorithm_cfg, BlockSize::new(1)?);
//!
//! // Sample inputs
//! let input_signal = InputSignal::new(vec![1.0, -2.5, 3.0])?;
//! let noise_ref = NoiseReference::new(vec![2.0, -1.2, -3.8])?;
//!
//! // Adapt the filter using our algorithm's update rules
//! let output = filter.adapt(&input_signal, &noise_ref)?;
//!
//! # Ok::<(), Error>(())
//! ```

mod algorithms;
pub use algorithms::Algorithm;

mod error;
pub use error::{Error, Result};

pub mod filters;
pub mod types;

#[cfg(feature = "python")]
pub mod python;

#[cfg(test)]
mod test_utils;
