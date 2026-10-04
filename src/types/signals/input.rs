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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_empty_input() {
        assert!(matches!(
            InputSignal::<f64>::new(vec![]),
            Err(Error::EmptyInputArr)
        ));
    }
}
