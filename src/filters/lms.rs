use std::ops::{Deref, DerefMut};

use super::FilterBase;

use crate::Result;
use crate::algorithms::Lms;
use crate::filters::ProcessingMode;
use crate::types::{Float, WindowSize};

#[derive(Debug, Clone)]
pub struct LmsOptions<F: Float> {
    pub mu: F,
    pub window_size: usize,
}

#[derive(Debug, Clone)]
pub struct LmsFilter<F: Float> {
    inner: FilterBase<F, Lms<F>>,
}
impl<F: Float> LmsFilter<F> {
    /// # Errors
    ///
    /// Returns an error if ``mu <= 0.0`` or ``window_size == 0``.
    #[allow(clippy::needless_pass_by_value, reason = "All fields are moved")]
    pub fn new(options: LmsOptions<F>, mode: ProcessingMode) -> Result<Self> {
        let lms = Lms::new(options.mu)?;
        let window_size = WindowSize::new(options.window_size)?;
        let block_size = mode.block_size()?;
        let filter = FilterBase::new(lms, window_size, block_size);
        Ok(Self { inner: filter })
    }

    /// # Errors
    ///
    /// Returns an error if `weights.is_empty()`.
    #[allow(clippy::needless_pass_by_value, reason = "All fields are moved")]
    pub fn from_weights(
        options: LmsOptions<F>,
        weights: Vec<F>,
        mode: ProcessingMode,
    ) -> Result<Self> {
        let lms = Lms::new(options.mu)?;

        if options.window_size != weights.len() {
            // TODO
        }

        let filter = FilterBase::from_weights(lms, weights, mode.block_size()?)?;
        Ok(Self { inner: filter })
    }
}
impl<F: Float> Deref for LmsFilter<F> {
    type Target = FilterBase<F, Lms<F>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<F: Float> DerefMut for LmsFilter<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;
    use crate::types::BlockSize;

    #[test]
    fn lms_new_works() {
        let mu = 0.1;
        let window_size = 16;

        let filter =
            LmsFilter::new(LmsOptions { mu, window_size }, ProcessingMode::Sample).unwrap();
        let expected_inner = FilterBase::new(
            Lms::new(mu).unwrap(),
            WindowSize::new(window_size).unwrap(),
            BlockSize::new(1).unwrap(),
        );

        assert_eq!(filter.inner, expected_inner);
        assert_eq!(filter.window_size(), window_size);
    }

    #[test]
    fn lms_from_weights_works() {
        let mu = 0.1;
        let weights = vec![1.0, 2.0, 3.0];

        let filter = LmsFilter::from_weights(
            LmsOptions {
                mu,
                window_size: weights.len(),
            },
            weights.clone(),
            ProcessingMode::Sample,
        )
        .unwrap();
        let expected_inner = FilterBase::from_weights(
            Lms::new(mu).unwrap(),
            weights.clone(),
            BlockSize::new(1).unwrap(),
        )
        .unwrap();

        assert_eq!(filter.inner, expected_inner);
        assert_eq!(filter.window_size(), weights.len());
    }

    #[test]
    fn block_lms_new_works() {
        let mu = 0.1;
        let window_size = 16;
        let block_size = 8;

        let filter = LmsFilter::new(
            LmsOptions { mu, window_size },
            ProcessingMode::Block { block_size },
        )
        .unwrap();
        let expected_inner = FilterBase::new(
            Lms::new(mu).unwrap(),
            WindowSize::new(window_size).unwrap(),
            BlockSize::new(block_size).unwrap(),
        );

        assert_eq!(filter.inner, expected_inner);
        assert_eq!(filter.window_size(), window_size);
        assert_eq!(filter.block_size(), block_size);
    }

    #[test]
    fn block_lms_from_weights_works() {
        let mu = 0.1;
        let weights = vec![1.0, 2.0, 3.0];
        let block_size = 8;

        let filter = LmsFilter::from_weights(
            LmsOptions {
                mu,
                window_size: weights.len(),
            },
            weights.clone(),
            ProcessingMode::Block { block_size },
        )
        .unwrap();
        let expected_inner = FilterBase::from_weights(
            Lms::new(mu).unwrap(),
            weights.clone(),
            BlockSize::new(block_size).unwrap(),
        )
        .unwrap();

        assert_eq!(filter.inner, expected_inner);
        assert_eq!(filter.window_size(), weights.len());
        assert_eq!(filter.block_size(), block_size);
    }
}
