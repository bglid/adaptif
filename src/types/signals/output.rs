use std::ops::Deref;

use crate::types::Float;

use super::InputSignal;

#[derive(Debug, Clone)]
pub struct OutputSignal<F: Float> {
    inner: Vec<F>,
}
impl<F: Float> OutputSignal<F> {
    pub fn new(input_signal: &InputSignal<F>) -> Self {
        OutputSignal {
            inner: Vec::with_capacity(input_signal.len()),
        }
    }

    pub fn push(&mut self, error: OutputSample<F>) {
        self.inner.push(*error);
    }

    pub fn into_inner(self) -> Vec<F> {
        self.inner
    }
}
impl<F: Float> Deref for OutputSignal<F> {
    type Target = Vec<F>;
    fn deref(&self) -> &Self::Target {
        &self.inner
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
