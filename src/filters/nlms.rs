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
}

// #[derive(Debug, Clone)]
// pub struct NlmsFilter<F: Float> {
//     inner: SampleFilterBase<F, Nlms<F>>,
// }
// impl<F: Float> NlmsFilter<F> {
//     /// # Errors
//     ///
//     /// Returns an error if mu or eps <= 0.0.
//     #[allow(clippy::needless_pass_by_value, reason = "All fields are moved")]
//     pub fn new(options: NlmsOptions<F>, window_size: WindowSize) -> Result<Self> {
//         let nlms = Nlms::new(options.mu, options.eps)?;
//         let filter = SampleFilterBase::new(nlms, *window_size)?;
//         Ok(Self { inner: filter })
//     }
//
//     /// # Errors
//     ///
//     /// Returns an error if `weights.is_empty()`.
//     #[allow(clippy::needless_pass_by_value, reason = "All fields are moved")]
//     pub fn from_weights(options: NlmsOptions<F>, weights: Vec<F>) -> Result<Self> {
//         let nlms = Nlms::new(options.mu, options.eps)?;
//         let filter = SampleFilterBase::from_weights(nlms, weights)?;
//         Ok(Self { inner: filter })
//     }
// }
// impl<F: Float> Deref for NlmsFilter<F> {
//     type Target = SampleFilterBase<F, Nlms<F>>;
//
//     fn deref(&self) -> &Self::Target {
//         &self.inner
//     }
// }
// impl<F: Float> DerefMut for NlmsFilter<F> {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.inner
//     }
// }
#[derive(Debug, Clone)]
pub struct NlmsFilter<F: Float> {
    inner: FilterBase<F, Nlms<F>>,
}
impl<F: Float> NlmsFilter<F> {
    /// # Errors
    ///
    /// Returns an error if mu or eps <= 0.0.
    #[allow(clippy::needless_pass_by_value, reason = "All fields are moved")]
    pub fn new(options: NlmsOptions<F>, window_size: WindowSize) -> Result<Self> {
        let nlms = Nlms::new(options.mu, options.eps)?;
        let filter = FilterBase::new(nlms, *window_size, ProcessingMode::Sample)?;
        Ok(Self { inner: filter })
    }

    /// # Errors
    ///
    /// Returns an error if `weights.is_empty()`.
    #[allow(clippy::needless_pass_by_value, reason = "All fields are moved")]
    pub fn from_weights(options: NlmsOptions<F>, weights: Vec<F>) -> Result<Self> {
        let nlms = Nlms::new(options.mu, options.eps)?;
        let filter = FilterBase::from_weights(nlms, weights, ProcessingMode::Sample)?;
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
    use super::*;

    #[test]
    fn new_works() {
        let mu = 0.1;
        let eps = 1e-8;
        let window_size = 16;

        let filter = NlmsFilter::new(
            NlmsOptions { mu, eps },
            WindowSize::new(window_size).unwrap(),
        )
        .unwrap();
        let expected_inner = FilterBase::new(
            Nlms::new(mu, eps).unwrap(),
            window_size,
            ProcessingMode::Sample,
        )
        .unwrap();

        assert_eq!(filter.inner, expected_inner);
        assert_eq!(filter.window_size(), window_size);
    }

    #[test]
    fn from_weights_works() {
        let mu = 0.1;
        let eps = 1e-8;
        let weights = vec![1.0, 2.0, 3.0];

        let filter = NlmsFilter::from_weights(NlmsOptions { mu, eps }, weights.clone()).unwrap();
        let expected_inner = FilterBase::from_weights(
            Nlms::new(mu, eps).unwrap(),
            weights.clone(),
            ProcessingMode::Sample,
        )
        .unwrap();

        assert_eq!(filter.inner, expected_inner);
        assert_eq!(filter.window_size(), weights.len());
    }
}
