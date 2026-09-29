use std::ops::Deref;

use crate::error::{Error, Result};
use crate::types::Float;

#[derive(Debug, Clone)]
pub struct InputSignal<F: Float>(Vec<F>);
impl<F: Float> Deref for InputSignal<F> {
    type Target = [F];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<F: Float> InputSignal<F> {
    /// # Errors
    /// Returns an error if `input_signal` is empty.
    pub fn new(input_signal: Vec<F>) -> Result<Self> {
        if input_signal.is_empty() {
            Err(Error::EmptyInputArr)
        } else {
            Ok(InputSignal(input_signal))
        }
    }

    pub fn get_sample(&self, n: usize) -> Option<InputSample<F>> {
        Some(InputSample(*self.get(n)?))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InputSample<F: Float>(pub F);
impl<F: Float> Deref for InputSample<F> {
    type Target = F;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

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

#[derive(Debug, Clone)]
pub struct OutputSignal<F: Float>(Vec<F>);
impl<F: Float> OutputSignal<F> {
    pub fn new(input_signal: &InputSignal<F>) -> Self {
        OutputSignal(Vec::with_capacity(input_signal.len()))
    }

    pub fn push(&mut self, error: OutputSample<F>) {
        self.0.push(*error);
    }

    pub fn into_inner(self) -> Vec<F> {
        self.0
    }
}
impl<F: Float> Deref for OutputSignal<F> {
    type Target = Vec<F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OutputSample<F: Float>(pub F);
impl<F: Float> Deref for OutputSample<F> {
    type Target = F;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    #[test]
    fn reject_empty_signals() {
        assert!(matches!(
            InputSignal::<f64>::new(vec![]),
            Err(Error::EmptyInputArr)
        ));

        assert!(matches!(
            NoiseReference::<f64>::new(vec![]),
            Err(Error::EmptyInputArr)
        ));
    }
}
