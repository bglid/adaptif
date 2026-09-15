use crate::Result;
use crate::types::signals::{InputSignal, NoiseReference};

mod filter_base;
pub use filter_base::FilterBase;

use crate::algorithms::{Lms, Nlms, RecursiveLeastSquares};

mod block_filter_base;
pub use block_filter_base::BlockFilterBase;

mod common;

// Define aliases for easier use
pub type LMSFilter = FilterBase<Lms>;
pub type BlockLMSFilter = BlockFilterBase<Lms>;
pub type NLMSFilter = FilterBase<Nlms>;
pub type RLSFilter = FilterBase<RecursiveLeastSquares>;

/// Defines the public API for filter models.
pub trait AdaptiveFilter {
    /// Iteratively adapts the filter to the input signal and noise reference
    /// using the chosen algorithm, and returns the denoised signal.
    ///
    /// # Errors
    ///
    /// Should return an error if `input_signal.len() > noise_ref.len()`.
    fn adapt(&mut self, input_signal: &InputSignal, noise_ref: &NoiseReference)
    -> Result<Vec<f64>>;

    /// Applies the filter to the input signal without updating the filter coefficients.
    ///
    /// # Errors
    ///
    /// Should return an error if `input_signal.len() > noise_ref.len()`.
    fn filter(&self, input_signal: &InputSignal, noise_ref: &NoiseReference) -> Result<Vec<f64>>;
}
