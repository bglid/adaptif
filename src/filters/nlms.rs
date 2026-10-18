use std::ops::{Deref, DerefMut};

use super::FilterBase;

use crate::Result;
use crate::algorithms::Nlms;
use crate::filters::ProcessingMode;
use crate::types::{Float, WindowSize};

#[derive(Debug, Clone)]
pub struct NlmsOptions<F: Float> {
    pub mu: F,
    pub eps: F,
    pub window_size: usize,
}

#[derive(Debug, Clone)]
pub struct NlmsFilter<F: Float> {
    inner: FilterBase<F, Nlms<F>>,
}
impl<F: Float> NlmsFilter<F> {
    /// # Errors
    ///
    /// Returns an error if options.mu or options.eps <= 0.0.
    #[allow(clippy::needless_pass_by_value, reason = "All fields are moved")]
    pub fn new(options: NlmsOptions<F>, mode: ProcessingMode) -> Result<Self> {
        let nlms = Nlms::new(options.mu, options.eps)?;
        let window_size = WindowSize::new(options.window_size)?;
        let block_size = mode.block_size()?;
        let filter = FilterBase::new(nlms, window_size, block_size);
        Ok(Self { inner: filter })
    }
}
impl<F: Float> Deref for NlmsFilter<F> {
    type Target = FilterBase<F, Nlms<F>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<F: Float> DerefMut for NlmsFilter<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use crate::types::BlockSize;

    use super::*;

    #[test]
    fn new_works() {
        let mu = 0.1;
        let eps = 1e-8;
        let window_size = 16;

        let filter = NlmsFilter::new(
            NlmsOptions {
                mu,
                eps,
                window_size,
            },
            ProcessingMode::Sample,
        )
        .unwrap();
        let expected_inner = FilterBase::new(
            Nlms::new(mu, eps).unwrap(),
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
        let weights = vec![1.0, 2.0, 3.0];

        let filter = {
            let mut filter = NlmsFilter::new(
                NlmsOptions {
                    mu,
                    eps,
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
                Nlms::new(mu, eps).unwrap(),
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
