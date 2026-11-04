mod filter_weights;
pub use filter_weights::FilterWeights; // pub because used in Algorithm::update_step()

mod window_size;
pub use window_size::WindowSize;

mod block_size;
pub use block_size::BlockSize;

mod noise_estimate;
pub(crate) use noise_estimate::NoiseEstimate;

mod float;
pub(crate) use float::Float;

pub mod buffers;
pub mod signals;

// TODO: use pub(crate) to limit public exports to only the types needed for the public API
