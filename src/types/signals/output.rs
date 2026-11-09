use std::ops::Deref;

use crate::types::Float;

use super::InputSignal;

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct OutputSignal<F: Float> {
    samples: Vec<F>,
}
impl<F: Float> OutputSignal<F> {
    pub fn new(input_signal: &InputSignal<F>) -> Self {
        OutputSignal {
            samples: Vec::with_capacity(input_signal.len()),
        }
    }

    pub fn push(&mut self, error: OutputSample<F>) {
        self.samples.push(*error);
    }
}
impl<F: Float> From<OutputSignal<F>> for Vec<F> {
    fn from(value: OutputSignal<F>) -> Self {
        value.samples
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
