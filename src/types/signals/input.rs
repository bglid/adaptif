use std::ops::Deref;

use crate::error::{Error, Result};
use crate::types::Float;

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct InputSignal<F: Float> {
    inner: Vec<F>,
}
impl<F: Float> InputSignal<F> {
    /// # Errors
    /// Returns an error if `input_signal` is empty.
    pub fn new(input_signal: Vec<F>) -> Result<Self> {
        if input_signal.is_empty() {
            Err(Error::EmptyInputArr)
        } else {
            Ok(InputSignal {
                inner: input_signal,
            })
        }
    }

    pub fn get(&self, n: usize) -> Option<InputSample<F>> {
        Some(InputSample(*self.inner.get(n)?))
    }

    #[allow(
        clippy::len_without_is_empty,
        reason = "InputSignal can't be empty (see new())"
    )]
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}
impl<F: Float> AsRef<[F]> for InputSignal<F> {
    fn as_ref(&self) -> &[F] {
        &self.inner
    }
}
impl<F: Float> From<InputSignal<F>> for Vec<F> {
    fn from(value: InputSignal<F>) -> Self {
        value.inner
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
