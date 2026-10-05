use num_traits::cast::ToPrimitive as _;
use std::f64::consts::PI;

// NOTE: PLEASE READ / TODO
// This one function is slop generated and is going to be replaced
// -> by a real .wav loader. Demo purposes. Will be removed
pub fn test_signals() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let sample_rate = 16_000.0;
    let num_samples = 4_000;

    let clean: Vec<f64> = (0..num_samples)
        .map(|n| {
            let t = f64::from(n) / sample_rate;
            (2.0 * PI * 440.0 * t).sin()
        })
        .collect();

    let noise_ref: Vec<f64> = (0..num_samples)
        .map(|n| {
            let t = f64::from(n) / sample_rate;
            0.2 * (2.0 * PI * 1000.0 * t).sin()
        })
        .collect();

    let noisy: Vec<f64> = clean
        .iter()
        .zip(&noise_ref)
        .map(|(clean, noise)| clean + noise)
        .collect();

    (clean, noisy, noise_ref)
}

#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
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
