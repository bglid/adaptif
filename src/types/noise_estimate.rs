use std::ops::Deref;

use crate::types::Float;

#[derive(Debug, Clone, Copy)]
pub struct NoiseEstimate<F: Float>(pub F);
impl<F: Float> Deref for NoiseEstimate<F> {
    type Target = F;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
