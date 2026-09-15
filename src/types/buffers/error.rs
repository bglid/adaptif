use std::num::NonZero;
use std::ops::{Deref, DerefMut};

use crate::types::signals::OutputSample;

use super::{BlockSize, SampleBuffer};

pub struct ErrorBuffer(SampleBuffer);
impl ErrorBuffer {
    pub fn new(block_size: BlockSize) -> Self {
        #[allow(
            clippy::unwrap_used,
            clippy::missing_panics_doc,
            reason = "BlockSize type cannot be zero"
        )]
        ErrorBuffer(SampleBuffer::new(NonZero::new(*block_size).unwrap()))
    }

    pub fn push(&mut self, item: OutputSample) {
        self.0.push(*item);
    }
}
impl Deref for ErrorBuffer {
    type Target = SampleBuffer;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for ErrorBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;
    use crate::test_utils::all_approx_equal;

    #[test]
    fn error_buffer_init_to_zero() {
        let buffer = ErrorBuffer::new(BlockSize::new(2).unwrap());
        assert!(all_approx_equal(buffer.iter(), [0_f64; 2].iter()));
    }
}
