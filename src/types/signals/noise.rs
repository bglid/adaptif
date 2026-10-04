use std::ops::Deref;

use crate::error::{Error, Result};
use crate::types::Float;

#[derive(Debug, Clone)]
pub struct NoiseReference<F: Float>(Vec<F>);
impl<F: Float> Deref for NoiseReference<F> {
    type Target = [F];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<F: Float> NoiseReference<F> {
    /// # Errors
    /// Returns an error if `noise_ref` is empty.
    pub fn new(noise_ref: Vec<F>) -> Result<Self> {
        if noise_ref.is_empty() {
            Err(Error::EmptyInputArr)
        } else {
            Ok(NoiseReference(noise_ref))
        }
    }

    pub fn get_sample(&self, n: usize) -> Option<NoiseSample<F>> {
        Some(NoiseSample(*self.get(n)?))
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
