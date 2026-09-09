use rand_distr::{Distribution as _, Normal};
use std::ops::{Deref, DerefMut};

use crate::types::WindowSize;

#[derive(Debug, Clone)]
pub struct FilterWeights {
    weights: Box<[f64]>, // We use a boxed slice instead of a Vec to ensure length doesn't change
}
impl FilterWeights {
    // TODO: rename to from_normal_dist()? or just use default() with from_distribution()
    pub fn new(window_size: WindowSize, mean: f64, std_dev: f64) -> Option<Self> {
        let mut rng = rand::rng();

        let normal_dist = Normal::new(mean, std_dev).ok()?;

        let weights = normal_dist
            .sample_iter(&mut rng)
            .take(*window_size)
            .collect::<Vec<f64>>()
            .into_boxed_slice();

        Some(FilterWeights { weights })
    }

    // TODO: from_distribution() ?

    // mostly used for testing functions
    pub fn zeros(window_size: WindowSize) -> Self {
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
        let weights = FilterWeights::new(WindowSize::new(WINDOW_SIZE).unwrap(), 0.0, 5e-5).unwrap();

        assert_eq!(WINDOW_SIZE, weights.len());
        assert!(!all_approx_equal(weights.iter(), [0.0; WINDOW_SIZE].iter()));
    }

    #[test]
    fn filter_weights_zero() {
        const WINDOW_SIZE: usize = 1024;
        let weights = FilterWeights::zeros(WindowSize::new(WINDOW_SIZE).unwrap());

        assert_eq!(WINDOW_SIZE, weights.len());
        assert!(all_approx_equal(weights.iter(), [0.0; WINDOW_SIZE].iter()));
    }
}
