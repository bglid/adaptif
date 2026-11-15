use std::{error::Error, path::PathBuf};

use adaptif::filters::{LmsFilter, LmsOptions, ProcessingMode};
use adaptif::types::{InputSignal, NoiseReference};

use rand::{self, RngExt as _, SeedableRng as _, rngs::StdRng};
use rand_distr::StandardNormal;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the filter
    let mut lms = LmsFilter::new(
        LmsOptions {
            step_size: 0.1,  // How fast the filter adapts
            window_size: 16, // How many samples the filter processes at a time
        },
        ProcessingMode::Sample,
    )?;

    // Load some noisy audio signal and a corresponding noise reference.
    let (noisy_signal, noise) = load_signals()?;

    let input_signal = InputSignal::new(noisy_signal)?;
    let noise_ref = NoiseReference::new(noise)?;

    // Adapt the filter to the inputs and get the iteratively cleaned signal.
    let _cleaned_signal = lms.adapt(&input_signal, &noise_ref)?;

    Ok(())
}

/// Loads example audio signals from WAV files.
///
/// Here we inject noise for demonstrational purposes. In a real application,
/// it would already be part of the signal.
#[allow(clippy::missing_errors_doc, reason = "Example")]
pub fn load_signals() -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
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
