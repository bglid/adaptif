#![allow(
    clippy::exhaustive_structs,
    reason = "Simple wrappers for primitives, so adding fields is highly unlikely."
)]

use crate::error::{Error, Result};
use std::ops::Deref;

#[derive(Debug, Clone)]
pub struct InputSignal(Vec<f64>);
impl Deref for InputSignal {
    type Target = [f64];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl InputSignal {
    /// # Errors
    /// Returns an error if `input_signal` is empty.
    pub fn new(input_signal: Vec<f64>) -> Result<Self> {
        if input_signal.is_empty() {
            Err(Error::EmptyInputArr)
        } else {
            Ok(InputSignal(input_signal))
        }
    }

    pub fn get_sample(&self, n: usize) -> Option<InputSample> {
        Some(InputSample(*self.get(n)?))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InputSample(pub f64);
impl Deref for InputSample {
    type Target = f64;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct NoiseReference(Vec<f64>);
impl Deref for NoiseReference {
    type Target = [f64];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl NoiseReference {
    /// # Errors
    /// Returns an error if `noise_ref` is empty.
    pub fn new(noise_ref: Vec<f64>) -> Result<Self> {
        if noise_ref.is_empty() {
            Err(Error::EmptyInputArr)
        } else {
            Ok(NoiseReference(noise_ref))
        }
    }

    pub fn get_sample(&self, n: usize) -> Option<NoiseSample> {
        Some(NoiseSample(*self.get(n)?))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NoiseSample(pub f64);
impl Deref for NoiseSample {
    type Target = f64;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct OutputSignal(Vec<f64>);
impl OutputSignal {
    pub fn new(input_signal: &InputSignal) -> Self {
        OutputSignal(Vec::with_capacity(input_signal.len()))
    }

    pub fn push(&mut self, error: OutputSample) {
        self.0.push(*error);
    }

    pub fn into_inner(self) -> Vec<f64> {
        self.0
    }
}
impl Deref for OutputSignal {
    type Target = Vec<f64>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OutputSample(pub f64);
impl Deref for OutputSample {
    type Target = f64;
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
            InputSignal::new(vec![]),
            Err(Error::EmptyInputArr)
        ));

        assert!(matches!(
            NoiseReference::new(vec![]),
            Err(Error::EmptyInputArr)
        ));
    }
}
