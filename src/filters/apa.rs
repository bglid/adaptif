use std::ops::{Deref, DerefMut};

use super::FilterBase;

use crate::Result;
use crate::algorithms::Apa;
use crate::filters::ProcessingMode;
use crate::types::{Float, WindowSize};

#[derive(Debug, Clone)]
pub struct ApaOptions<F: Float> {
    pub mu: F,
    pub eps: F,
    pub projection_order: usize,
    pub window_size: usize,
}
impl<F: Float> Default for ApaOptions<F> {
    fn default() -> Self {
        #[allow(
            clippy::unwrap_used,
            clippy::missing_panics_doc,
            reason = "Trivial conversions"
        )]
        // TODO: I don't know if the mu is ideal here, the others I am confident about though
        Self {
            mu: F::from(0.1).unwrap(),
            eps: F::from(1e-8).unwrap(),
            projection_order: 2,
            window_size: 8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ApaFilter<F: Float> {
    inner: FilterBase<F, Apa<F>>,
}
impl<F: Float> ApaFilter<F> {
    /// # Errors
    ///
    /// Returns an error if mu <= 0.0.
    /// Returns an error if eps <= 0.0.
    /// Returns an error if projection order <= 0.
    #[allow(clippy::needless_pass_by_value, reason = "All fields are moved")]
    pub fn new(options: ApaOptions<F>, mode: ProcessingMode) -> Result<Self> {
        let apa = Apa::new(options.mu, options.eps, options.projection_order)?;
        let window_size = WindowSize::new(options.window_size)?;
        let block_size = mode.block_size()?;
        let filter = FilterBase::new(apa, window_size, block_size);
        Ok(Self { inner: filter })
    }
}
impl<F: Float> Deref for ApaFilter<F> {
    type Target = FilterBase<F, Apa<F>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<F: Float> DerefMut for ApaFilter<F> {
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
    fn new_works() {
        let mu = 0.1;
        let eps = 1e-8;
        let projection_order = 2;
        let window_size = 16;

        let filter = ApaFilter::new(
            ApaOptions {
                mu,
                eps,
                projection_order,
                window_size,
            },
            ProcessingMode::Sample,
        )
        .unwrap();
        let expected_inner = FilterBase::new(
            Apa::new(mu, eps, projection_order).unwrap(),
            WindowSize::new(window_size).unwrap(),
            BlockSize::new(1).unwrap(),
        );

        assert_eq!(filter.inner, expected_inner);
        assert_eq!(filter.window_size(), window_size);
    }

    #[test]
    fn from_weights_works() {
        let mu = 0.1;
        let eps = 1e-8;
        let projection_order = 2;
        let weights = vec![1.0, 2.0, 3.0];

        let filter = {
            let mut filter = ApaFilter::new(
                ApaOptions {
                    mu,
                    eps,
                    projection_order,
                    window_size: weights.len(),
                },
                ProcessingMode::Sample,
            )
            .unwrap();
            filter.load_weights(weights.clone()).unwrap();
            filter
        };

        let expected_inner = {
            let mut expected_inner = FilterBase::new(
                Apa::new(mu, eps, projection_order).unwrap(),
                WindowSize::new(weights.len()).unwrap(),
                BlockSize::new(1).unwrap(),
            );
            expected_inner.load_weights(weights.clone()).unwrap();
            expected_inner
        };

        assert_eq!(filter.inner, expected_inner);
        assert_eq!(filter.window_size(), weights.len());
    }
}
