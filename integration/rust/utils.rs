#![allow(clippy::unwrap_used, reason = "tests utils")]
use adaptif::{
    Result,
    types::signals::{InputSignal, NoiseReference},
};
use core::f64;
use hound;
use num_traits::cast::ToPrimitive as _;
use std::path::PathBuf;

use rand::{self, RngExt as _, SeedableRng as _, rngs::StdRng};
use rand_distr::StandardNormal;

pub struct TestSignalBatch {
    pub clean_signal: Vec<f64>,
    pub noisy_signal: InputSignal<f64>,
    pub noise_ref: NoiseReference<f64>,
}
impl TestSignalBatch {
    pub fn new(path: PathBuf, seed: u64) -> Self {
        let mut reader = hound::WavReader::open(path).unwrap();
        let clean_signal: Vec<f64> = reader
            .samples::<i16>()
            .map(|s| f64::from(s.unwrap()) / f64::from(i16::MAX))
            .collect();

        let (noisy_signal, noise_ref) = add_white_noise(&clean_signal, seed, 0.0).unwrap();

        TestSignalBatch {
            clean_signal,
            noisy_signal,
            noise_ref,
        }
    }
}

/// Generates test signals for integration tests.
pub fn load_test_signals() -> Vec<TestSignalBatch> {
    // Doing this in this fashion so it can be run from anywhere
    let clean_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("integration")
        .join("fixtures");

    std::fs::read_dir(clean_dir)
        .unwrap()
        .enumerate()
        .map(|(i, path)| {
            // 42 + i is the seed for white noise generation
            // It ensures different noise across signals, but the same noise per-signal across tests
            let seed = u64::try_from(42 + i).unwrap();
            TestSignalBatch::new(path.unwrap().path(), seed)
        })
        .collect()
}

/// Injects white noise into a copy of the clean signal.
/// Returns a tuple of the new noisy signal and the white noise reference.
fn add_white_noise(
    clean_signal: &[f64],
    seed: u64,
    snr_db: f64,
) -> Result<(InputSignal<f64>, NoiseReference<f64>)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut noise: Vec<f64> = std::iter::repeat_with(|| rng.sample(StandardNormal))
        .take(clean_signal.len())
        .collect();

    // Getting noise scale to update noise vec
    let signal_power = get_power(clean_signal);
    let noise_power = get_power(&noise);
    let target_noise_power = signal_power / (10.0_f64.powf(snr_db / 10.0));
    let scale = (target_noise_power / noise_power).sqrt();

    for sample in &mut noise {
        *sample *= scale;
    }

    let noisy_signal = clean_signal
        .iter()
        .zip(&noise)
        .map(|(c, n)| c + n)
        .collect();

    Ok((InputSignal::new(noisy_signal)?, NoiseReference::new(noise)?))
}

/// Calculates the signal power for converting to dB.
fn get_power(signal: &[f64]) -> f64 {
    signal.iter().map(|sample| sample.powi(2)).sum::<f64>()
}

/// Calculates mean square error (MSE) of original signal and newly "cleaned" signal.
pub fn mse(original_signal: &[f64], input_signal: &[f64]) -> f64 {
    original_signal
        .iter()
        .zip(input_signal)
        .map(|(d, i)| (d - i).powi(2))
        .sum::<f64>()
        / original_signal.len().to_f64().unwrap()
}

/// Calculates signal-to-noise ratio (SNR) of original signal and newly "cleaned" signal.
pub fn snr(original_signal: &[f64], input_signal: &[f64]) -> f64 {
    let signal_power = original_signal.iter().map(|d| d.powi(2)).sum::<f64>();
    let noise_power = original_signal
        .iter()
        .zip(input_signal)
        .map(|(d, i)| (d - i).powi(2))
        .sum::<f64>();

    10.0 * (signal_power / (noise_power + 1e-8)).log10()
}
