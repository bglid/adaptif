mod filter_base;
pub use filter_base::FilterBase;

use crate::algorithms::{LeastMeanSquares, NormalizedLeastMeanSquares};

mod block_filter_base;
pub use block_filter_base::BlockFilterBase;

mod common;
pub use common::AdaptiveFilter;

// Define aliases for easier use
pub type LMSFilter = FilterBase<LeastMeanSquares>;
pub type BlockLMSFilter = BlockFilterBase<LeastMeanSquares>;
pub type NLMSFilter = FilterBase<NormalizedLeastMeanSquares>;
