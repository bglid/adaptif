use rand::{self, RngExt as _, SeedableRng as _, rngs::StdRng};
use rand_distr::StandardNormal;
use std::{error::Error, path::PathBuf};
pub fn load_signals(path: PathBuf) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    // Using hound to load the WAV file
    let mut reader = hound::WavReader::open(path)?;
    let clean_signal: Vec<f64> = reader
        .samples::<i16>()
        .map(|s| Ok(f64::from(s?) / f64::from(i16::MAX)))
        .collect::<Result<_, hound::Error>>()?;

    // artificially injectiing white noise to the signal
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
