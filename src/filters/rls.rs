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
}

#[derive(Debug, Clone)]
pub struct RlsFilter<F: Float> {
    inner: FilterBase<F, Rls<F>>,
}
impl<F: Float> RlsFilter<F> {
    /// # Errors
    ///
    /// Returns an error if forgetting factor <= 0.0 or > 1.0, if init scale <= 0.0.
    #[allow(clippy::needless_pass_by_value, reason = "All fields are moved")]
    pub fn new(options: RlsOptions<F>, window_size: WindowSize) -> Result<Self> {
        let rls = Rls::new(options.forgetting_factor, options.p_init_scale)?;
        let filter = FilterBase::new(rls, *window_size, ProcessingMode::Sample)?;
        Ok(Self { inner: filter })
    }

    /// # Errors
    ///
    /// Returns an error if `weights.is_empty()`.
    #[allow(clippy::needless_pass_by_value, reason = "All fields are moved")]
    pub fn from_weights(options: RlsOptions<F>, weights: Vec<F>) -> Result<Self> {
        let rls = Rls::new(options.forgetting_factor, options.p_init_scale)?;
        let filter = FilterBase::from_weights(rls, weights, ProcessingMode::Sample)?;
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

    #[test]
    fn new_works() {
        let forgetting_factor = 0.999;
        let p_init_scale = 1e-4;
        let window_size = 16;

        let filter = RlsFilter::new(
            RlsOptions {
                forgetting_factor,
                p_init_scale,
            },
            WindowSize::new(window_size).unwrap(),
        )
        .unwrap();
        let expected_inner = FilterBase::new(
            Rls::new(forgetting_factor, p_init_scale).unwrap(),
            window_size,
            ProcessingMode::Sample,
        )
        .unwrap();

        assert_eq!(filter.inner, expected_inner);
        assert_eq!(filter.window_size(), window_size);
    }

    #[test]
    fn from_weights_works() {
        let forgetting_factor = 0.999;
        let p_init_scale = 1e-4;
        let weights = vec![1.0, 2.0, 3.0];

        let filter = RlsFilter::from_weights(
            RlsOptions {
                forgetting_factor,
                p_init_scale,
            },
            weights.clone(),
        )
        .unwrap();
        let expected_inner = FilterBase::from_weights(
            Rls::new(forgetting_factor, p_init_scale).unwrap(),
            weights.clone(),
            ProcessingMode::Sample,
        )
        .unwrap();

        assert_eq!(filter.inner, expected_inner);
        assert_eq!(filter.window_size(), weights.len());
    }
}
