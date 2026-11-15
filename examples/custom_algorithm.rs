use std::{error::Error, path::PathBuf};

use adaptif::Algorithm;
use adaptif::filters::FilterBase;
use adaptif::types::NoiseWindowView;
use adaptif::types::{BlockSize, WindowSize};
use adaptif::types::{InputSignal, NoiseReference};

mod common;
use common::load_signals::load_signals;

// Create a struct to hold any required parameters or state
pub struct MyAlgorithm {
    pub alpha: f64,
    pub window_size: WindowSize,
}
// Implement the Algorithm trait so the algorithm can be used with SampleFilterBase
impl Algorithm<f64> for MyAlgorithm {
    // This function is called every iteration during adaptation to update the weights
    fn update_step(&mut self, weights: &mut [f64], error: f64, noise_window: NoiseWindowView<f64>) {
        for (w, x) in weights.iter_mut().zip(noise_window.iter()) {
            *w += self.alpha * error * x;
        }
    }

    // Certain algorithms need to know the filter's window size to initialize buffers, etc.
    // To make sure `FilterBase` has the same window size as the algorithm object, we
    // have to define this function so that it can be access when constructing `FilterBase`.
    fn window_size(&self) -> WindowSize {
        self.window_size
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // getting path for wav fixtures
    let clean_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("integration")
        .join("fixtures");
    let wav_path = std::fs::read_dir(clean_dir)?
        .next()
        .ok_or("No WAV fixtures")??
        .path();

    // loading the wav files into Vec<f64> and injecting white noise
    let (noisy_signal, noise) = load_signals(wav_path)?;
    let input_signal = InputSignal::new(noisy_signal)?;
    let noise_ref = NoiseReference::new(noise)?;

    // Define the algorithm parameters
    let algorithm_cfg = MyAlgorithm {
        alpha: 1.0,
        window_size: WindowSize::new(16)?,
    };
    let block_size = BlockSize::new(1)?;

    // Instantiate the filter using SampleFilterBase and our custom algorithm
    let mut filter = FilterBase::new(algorithm_cfg, block_size);

    // Adapt the filter using our algorithm's update rules
    let _output = filter.adapt(&input_signal, &noise_ref)?;

    Ok(())
}
