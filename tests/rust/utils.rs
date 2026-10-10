#![allow(clippy::unwrap_used, reason = "tests utils")]
use core::f64;
use hound;
use num_traits::cast::ToPrimitive as _;
use std::path::PathBuf;

use rand::{self, RngExt as _, SeedableRng as _, rngs::StdRng};
use rand_distr::StandardNormal;

fn add_white_noise(clean_signal: &[f64], seed: u64, snr_db: f64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut noise: Vec<f64> = std::iter::repeat_with(|| rng.sample(StandardNormal))
        .take(clean_signal.len())
        .collect();

    // For getting the scale of the noise correct in dB
    let signal_power =
        clean_signal.iter().map(|c| c.powi(2)).sum::<f64>() / clean_signal.len().to_f64().unwrap();
    let noise_power = noise.iter().map(|n| n.powi(2)).sum::<f64>() / noise.len().to_f64().unwrap();
    let target_noise_power = signal_power / (10.0_f64.powf(snr_db / 10.0_f64));
    for sample in &mut noise {
        *sample *= (target_noise_power / noise_power).sqrt();
    }

    let noisy_signal = clean_signal
        .iter()
        .zip(&noise)
        .map(|(clean, noise)| clean + noise)
        .collect();

    (noisy_signal, noise)
}

pub fn test_signals() -> Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    // Doing this in this fashion so it can be run from anywhere
    let clean_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("CleanSpeech");

    let clean_paths: Vec<PathBuf> = std::fs::read_dir(clean_dir)
        .unwrap()
        .map(|path| path.unwrap().path())
        .collect();

    let mut signals = Vec::with_capacity(clean_paths.len());

    for (i, clean_path) in clean_paths.iter().enumerate() {
        let mut reader = hound::WavReader::open(clean_path).unwrap();
        let clean_samples: Vec<f64> = reader
            .samples::<i16>()
            .map(|s| f64::from(s.unwrap()) / f64::from(i16::MAX))
            .collect();

        #[allow(
            clippy::as_conversions,
            reason = "Test generating new seed. Precision isn't needed"
        )]
        let (noisy_signal, noise) = add_white_noise(&clean_samples, 42 + i as u64, 0.0);
        signals.push((clean_samples, noisy_signal, noise));
    }
    signals
}

pub fn mse(original_signal: &[f64], input_signal: &[f64]) -> f64 {
    original_signal
        .iter()
        .zip(input_signal)
        .map(|(d, i)| (d - i).powi(2))
        .sum::<f64>()
        / original_signal.len().to_f64().unwrap()
}

pub fn snr(original_signal: &[f64], input_signal: &[f64]) -> f64 {
    let signal_power = original_signal.iter().map(|d| d.powi(2)).sum::<f64>();
    let noise_power = original_signal
        .iter()
        .zip(input_signal)
        .map(|(d, i)| (d - i).powi(2))
        .sum::<f64>()
        + 1e-8;

    10.0 * (signal_power / noise_power).log10()
}
