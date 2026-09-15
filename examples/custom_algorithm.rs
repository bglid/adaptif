#![allow(clippy::unwrap_used, clippy::exhaustive_structs, reason = "Examples")]

use adaptif::algorithms::Algorithm;
use adaptif::filters::{AdaptiveFilter as _, FilterBase};
use adaptif::types::FilterWeights;
use adaptif::types::buffers::NoiseBuffer;
use adaptif::types::signals::{InputSignal, NoiseReference, OutputSample};

// Create a struct to hold any required parameters or state
pub struct MyAlgorithm {
    pub alpha: f64,
}
// Implement the Algorithm trait so the algorithm can be used with FilterBase
impl Algorithm for MyAlgorithm {
    // This function is called every iteration during adaptation to update the weights
    fn update_step(
        &mut self,
        weights: &mut FilterWeights,
        error: OutputSample,
        noise_ref: &NoiseBuffer,
    ) {
        for (w, x) in weights.iter_mut().zip(noise_ref.iter()) {
            *w += self.alpha * (*error) * x;
        }
    }
}

fn main() {
    // Sample inputs
    let input_signal = InputSignal::new(&[1.0, -2.5, 3.0]).unwrap();
    let noise_ref = NoiseReference::new(&[2.0, -1.2, -3.8]).unwrap();

    // Define the algorithm parameters
    let algorithm_cfg = MyAlgorithm { alpha: 1.0 };
    let window_size = 1024;

    // Instantiate the filter using FilterBase and our custom algorithm
    let mut filter = FilterBase::<MyAlgorithm>::new(algorithm_cfg, window_size).unwrap();

    // Adapt the filter using our algorithm's update rules
    let _output = filter.adapt(&input_signal, &noise_ref).unwrap();
}
