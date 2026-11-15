use std::{error::Error, path::PathBuf};

use adaptif::Algorithm;
use adaptif::filters::FilterBase;
use adaptif::types::NoiseWindowView;
use adaptif::types::{BlockSize, WindowSize};
use adaptif::types::{InputSignal, NoiseReference};

use rand::{self, RngExt as _, SeedableRng as _, rngs::StdRng};
use rand_distr::StandardNormal;

// Create a struct to hold any required parameters or state
pub struct MyAlgorithm {
    pub alpha: f64,
    pub window_size: WindowSize,
}
// Implement the Algorithm trait so the algorithm can be used with FilterBase
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
    // Define the algorithm parameters
    let algorithm_cfg = MyAlgorithm {
        alpha: 1.0,
        window_size: WindowSize::new(16)?,
    };
    let block_size = BlockSize::new(1)?;

    // Instantiate the filter using FilterBase and our custom algorithm
    let mut filter = FilterBase::new(algorithm_cfg, block_size);

    // Load some noisy audio signal and a corresponding noise reference.
    let (noisy_signal, noise) = load_signals()?;

    let input_signal = InputSignal::new(noisy_signal)?;
    let noise_ref = NoiseReference::new(noise)?;

    // Adapt the filter using our algorithm's update rules
    let _output = filter.adapt(&input_signal, &noise_ref)?;

    Ok(())
}

/// Loads example audio signals from WAV files.
///
/// Here we inject noise for demonstrational purposes. In a real application,
/// it would already be part of the signal.
#[allow(clippy::missing_errors_doc, reason = "Example")]
pub fn load_signals() -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    // Get the path to the audio file
    let clean_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("integration")
        .join("fixtures");
    let wav_path = std::fs::read_dir(clean_dir)?
        .next()
        .ok_or("No WAV fixtures")??
        .path();

    // Using hound to load the WAV file
    let mut reader = hound::WavReader::open(wav_path)?;
    let clean_signal: Vec<f64> = reader
        .samples::<i16>()
        .map(|s| Ok(f64::from(s?) / f64::from(i16::MAX)))
        .collect::<Result<_, hound::Error>>()?;

    // Injecting white noise into the signal (again, demonstrational purposes only)
    let mut rng = StdRng::seed_from_u64(42);
    let noise: Vec<f64> = std::iter::repeat_with(|| rng.sample(StandardNormal))
        .take(clean_signal.len())
        .collect();
    let noisy_signal: Vec<f64> = clean_signal
        .iter()
        .zip(&noise)
        .map(|(c, n)| c + n)
        .collect();

    Ok((noisy_signal, noise))
}
