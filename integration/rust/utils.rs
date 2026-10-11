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
    let power = |signal: &mut dyn Iterator<Item = f64>| signal.map(|x| x.powi(2)).sum::<f64>();

    // Getting noise scale to update noise vec
    let signal_power = power(&mut clean_signal.iter().copied());
    let noise_power = power(&mut noise.iter().copied());
    let target_noise_power = signal_power / (10.0_f64.powf(snr_db / 10.0_f64));
    let scale = (target_noise_power / noise_power).sqrt();

    for sample in &mut noise {
        *sample *= scale;
    }

    let noisy_signal = clean_signal
        .iter()
        .zip(&noise)
        .map(|(c, n)| c + n)
        .collect();

    (noisy_signal, noise)
}

pub fn test_signals() -> Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    // Doing this in this fashion so it can be run from anywhere
    let clean_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("integration")
        .join("fixtures");

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

        let (noisy_signal, noise) =
            add_white_noise(&clean_samples, u64::try_from(42 + i).unwrap(), 0.0);
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
        .sum::<f64>();

    10.0 * (signal_power / (noise_power + 1e-8)).log10()
}
