mod filter_base;
pub use filter_base::FilterBase;

use crate::algorithms::{LeastMeanSquares, NormalizedLeastMeanSquares};

mod block_filter_base;
pub use block_filter_base::BlockFilterBase;

mod common;

// Define aliases for easier use
pub type LMSFilter = FilterBase<LeastMeanSquares>;
pub type BlcokLMSFilter = BlockFilterBase<LeastMeanSquares>;
pub type NLMSFilter = FilterBase<NormalizedLeastMeanSquares>;
