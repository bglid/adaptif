mod filter_base;
pub use filter_base::FilterBase;

use crate::algorithms::{Lms, Nlms};

mod block_filter_base;
pub use block_filter_base::BlockFilterBase;

mod common;
pub use common::AdaptiveFilter;

// Define aliases for easier use
pub type LMSFilter = FilterBase<Lms>;
pub type BlockLMSFilter = BlockFilterBase<Lms>;
pub type NLMSFilter = FilterBase<Nlms>;
