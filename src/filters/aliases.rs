use super::{BlockFilterBase, FilterBase};
use crate::algorithms::{Lms, Nlms, Rls};

// Define aliases for easier use
// TODO: these may work better as structs
pub type LMSFilter<F> = FilterBase<F, Lms<F>>;
pub type BlockLMSFilter<F> = BlockFilterBase<F, Lms<F>>;
pub type NLMSFilter<F> = FilterBase<F, Nlms<F>>;
pub type RLSFilter<F> = FilterBase<F, Rls<F>>;
