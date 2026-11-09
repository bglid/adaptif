use std::ops::Deref;

use crate::error::{Error, Result};
use crate::types::Float;

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NoiseReference<F: Float> {
    samples: Vec<F>,
}
impl<F: Float> NoiseReference<F> {
    /// # Errors
    /// Returns an error if `noise_ref` is empty.
    pub fn new(noise_ref: Vec<F>) -> Result<Self> {
        if noise_ref.is_empty() {
            Err(Error::EmptyInputArr)
        } else {
            Ok(NoiseReference { samples: noise_ref })
        }
    }

    pub fn get(&self, n: usize) -> Option<NoiseSample<F>> {
        Some(NoiseSample(*self.samples.get(n)?))
    }

    #[allow(
        clippy::len_without_is_empty,
        reason = "NoiseReference can't be empty (see new())"
    )]
    pub fn len(&self) -> usize {
        self.samples.len()
    }
}
impl<F: Float> AsRef<[F]> for NoiseReference<F> {
    fn as_ref(&self) -> &[F] {
        &self.samples
    }
}
impl<F: Float> From<NoiseReference<F>> for Vec<F> {
    fn from(value: NoiseReference<F>) -> Self {
        value.samples
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NoiseSample<F: Float>(pub F);
impl<F: Float> Deref for NoiseSample<F> {
    type Target = F;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_empty_noise() {
        assert!(matches!(
            NoiseReference::<f64>::new(vec![]),
            Err(Error::EmptyInputArr)
        ));
    }
}
