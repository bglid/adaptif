use std::ops::{Deref, DerefMut};

use super::FilterBase;

use crate::Result;
use crate::algorithms::Rls;
use crate::filters::ProcessingMode;
use crate::types::{Float, WindowSize};

#[derive(Debug, Clone)]
pub struct RlsOptions<F: Float> {
    pub forgetting_factor: F,
    pub p_init_scale: F,
    pub window_size: usize,
}

#[derive(Debug, Clone)]
pub struct RlsFilter<F: Float> {
    inner: FilterBase<F, Rls<F>>,
}
impl<F: Float> RlsFilter<F> {
    /// # Errors
    ///
    /// Returns an error if options.forgetting factor <= 0.0 or > 1.0, or if init scale <= 0.0.
    #[allow(clippy::needless_pass_by_value, reason = "All fields are moved")]
    pub fn new(options: RlsOptions<F>, mode: ProcessingMode) -> Result<Self> {
        let rls = Rls::new(options.forgetting_factor, options.p_init_scale)?;
        let window_size = WindowSize::new(options.window_size)?;
        let block_size = mode.block_size()?;
        let filter = FilterBase::new(rls, window_size, block_size);
        Ok(Self { inner: filter })
    }
}
impl<F: Float> Deref for RlsFilter<F> {
    type Target = FilterBase<F, Rls<F>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<F: Float> DerefMut for RlsFilter<F> {
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
        let forgetting_factor = 0.999;
        let p_init_scale = 1e-4;
        let window_size = 16;

        let filter = RlsFilter::new(
            RlsOptions {
                forgetting_factor,
                p_init_scale,
                window_size,
            },
            ProcessingMode::Sample,
        )
        .unwrap();
        let expected_inner = FilterBase::new(
            Rls::new(forgetting_factor, p_init_scale).unwrap(),
            WindowSize::new(window_size).unwrap(),
            BlockSize::new(1).unwrap(),
        );

        assert_eq!(filter.inner, expected_inner);
        assert_eq!(filter.window_size(), window_size);
    }

    #[test]
    fn from_weights_works() {
        let forgetting_factor = 0.999;
        let p_init_scale = 1e-4;
        let weights = vec![1.0, 2.0, 3.0];

        let filter = {
            let mut filter = RlsFilter::new(
                RlsOptions {
                    forgetting_factor,
                    p_init_scale,
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
                Rls::new(forgetting_factor, p_init_scale).unwrap(),
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
