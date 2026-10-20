use std::ops::{Deref, DerefMut};

use crate::Error;
use crate::types::{Float, WindowSize};

#[derive(Debug, Clone, PartialEq)]
pub struct FilterWeights<F: Float> {
    weights: Box<[F]>, // We use a boxed slice instead of a Vec to ensure length doesn't change
    window_size: WindowSize,
}
impl<F: Float> FilterWeights<F> {
    pub fn new(window_size: WindowSize) -> Self {
        FilterWeights {
            weights: std::iter::repeat_n(F::zero(), *window_size)
                .collect::<Vec<F>>()
                .into_boxed_slice(),
            window_size,
        }
    }

    pub fn window_size(&self) -> WindowSize {
        self.window_size
    }
}
impl<F: Float> TryFrom<Vec<F>> for FilterWeights<F> {
    type Error = crate::Error;

    fn try_from(value: Vec<F>) -> Result<Self, Self::Error> {
        let window_size = WindowSize::new(value.len()).map_err(|_e| Error::EmptyInputArr)?;
        Ok(FilterWeights {
            weights: value.into_boxed_slice(),
            window_size,
        })
    }
}
impl<F: Float> From<FilterWeights<F>> for Vec<F> {
    fn from(value: FilterWeights<F>) -> Self {
        value.weights.into()
    }
}
impl<F: Float> Deref for FilterWeights<F> {
    type Target = Box<[F]>;
    fn deref(&self) -> &Self::Target {
        &self.weights
    }
}
impl<F: Float> DerefMut for FilterWeights<F> {
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
        assert_eq!(WINDOW_SIZE, *weights.window_size());
        assert!(all_approx_equal(weights.iter(), [0.0; WINDOW_SIZE].iter()));
    }
}
