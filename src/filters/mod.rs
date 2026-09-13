mod filter_base;
pub use filter_base::FilterBase;

mod block_filter_base;
pub use block_filter_base::BlockFilterBase;

mod common;

use crate::algorithms::LeastMeanSquares;

// Define aliases for easier use
pub type LMSFilter = FilterBase<LeastMeanSquares>;
