use std::ops::Deref;

use crate::error::{Error, Result};
use crate::types::Float;

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
/// Input signal to the filter.
///
/// In mathematical notation, this signal is commonly represented as a vector `x(n)`.
pub struct InputSignal<F: Float> {
    samples: Vec<F>,
}
impl<F: Float> InputSignal<F> {
    /// Creates a new `InputSignal` from a `Vec`.
    /// The input `Vec` cannot be empty.
    ///
    /// # Errors
    ///
    /// Returns an error if `input_signal` is empty.
    pub fn new(input_signal: Vec<F>) -> Result<Self> {
        if input_signal.is_empty() {
            Err(Error::EmptyInputArr)
        } else {
            Ok(InputSignal {
                samples: input_signal,
            })
        }
    }

    // TODO: should this be `pub` ? that would require making `InputSample` pub too
    pub(crate) fn get(&self, n: usize) -> Option<InputSample<F>> {
        Some(InputSample(*self.samples.get(n)?))
    }

    #[allow(
        clippy::len_without_is_empty,
        reason = "InputSignal can't be empty (see new())"
    )]
    pub fn len(&self) -> usize {
        self.samples.len()
    }
}
impl<F: Float> AsRef<[F]> for InputSignal<F> {
    fn as_ref(&self) -> &[F] {
        &self.samples
    }
}
impl<F: Float> From<InputSignal<F>> for Vec<F> {
    fn from(value: InputSignal<F>) -> Self {
        value.samples
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
