mod filter_base;
pub use filter_base::FilterBase;

use crate::algorithms::{LeastMeanSquares, NormalizedLeastMeanSquares, RecursiveLeastSquares};

// Define aliases for easier use
pub type LMSFilter = FilterBase<LeastMeanSquares>;
pub type NLMSFilter = FilterBase<NormalizedLeastMeanSquares>;
pub type RLSFilter = FilterBase<RecursiveLeastSquares>;
