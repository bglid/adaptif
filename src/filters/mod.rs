use crate::Result;
use crate::types::Float;
use crate::types::signals::{InputSignal, NoiseReference};

mod filter_base;
pub use filter_base::FilterBase;

use crate::algorithms::{Lms, Nlms, Rls};

mod block_filter_base;
pub use block_filter_base::BlockFilterBase;

mod common;

// Define aliases for easier use
// TODO: these may work better as structs
pub type LMSFilter<F> = FilterBase<F, Lms<F>>;
pub type BlockLMSFilter<F> = BlockFilterBase<F, Lms<F>>;
pub type NLMSFilter<F> = FilterBase<F, Nlms<F>>;
pub type RLSFilter<F> = FilterBase<F, Rls<F>>;

/// Defines the public API for filter models.
pub trait AdaptiveFilter<F: Float> {
    /// Iteratively adapts the filter to the input signal and noise reference
    /// using the chosen algorithm, and returns the denoised signal.
    ///
    /// # Errors
    ///
    /// Should return an error if `input_signal.len() > noise_ref.len()`.
    fn adapt(
        &mut self,
        input_signal: &InputSignal<F>,
        noise_ref: &NoiseReference<F>,
    ) -> Result<Vec<F>>;

    /// Applies the filter to the input signal without updating the filter coefficients.
    ///
    /// # Errors
    ///
    /// Should return an error if `input_signal.len() > noise_ref.len()`.
    fn filter(
        &self,
        input_signal: &InputSignal<F>,
        noise_ref: &NoiseReference<F>,
    ) -> Result<Vec<F>>;
}
