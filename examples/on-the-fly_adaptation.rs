use std::{error::Error, path::PathBuf};

use adaptif::filters::{LmsFilter, LmsOptions, ProcessingMode};
use adaptif::types::{InputSignal, NoiseReference};

mod common;
use common::load_signals::load_signals;

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

    // Initialize the filter
    let mut lms = LmsFilter::new(
        LmsOptions {
            step_size: 0.1,  // How fast the filter adapts
            window_size: 16, // How many samples the filter processes at a time
        },
        ProcessingMode::Sample,
    )?;

    // Adapt the filter to the inputs and get the iteratively cleaned signal.
    let _cleaned_signal = lms.adapt(&input_signal, &noise_ref)?;

    Ok(())
}
