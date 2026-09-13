use std::ops::{Deref, DerefMut};

use crate::types::WindowSize;

#[derive(Debug, Clone)]
pub struct FilterWeights {
    weights: Box<[f64]>, // We use a boxed slice instead of a Vec to ensure length doesn't change
}
impl FilterWeights {
    pub fn new(window_size: WindowSize) -> Self {
        FilterWeights {
            weights: std::iter::repeat_n(0.0, *window_size)
                .collect::<Vec<f64>>()
                .into_boxed_slice(),
        }
    }
}
impl Deref for FilterWeights {
    type Target = Box<[f64]>;
    fn deref(&self) -> &Self::Target {
        &self.weights
    }
}
impl DerefMut for FilterWeights {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.weights
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;
    use crate::test_utils::all_approx_equal;

    #[test]
    fn filter_weights_init() {
        const WINDOW_SIZE: usize = 1024;
        let weights = FilterWeights::new(WindowSize::new(WINDOW_SIZE).unwrap());

        assert_eq!(WINDOW_SIZE, weights.len());
        assert!(all_approx_equal(weights.iter(), [0.0; WINDOW_SIZE].iter()));
    }
}
