use std::error::Error;

use adaptif::filters::{LmsFilter, LmsOptions};
use adaptif::types::signals::{InputSignal, NoiseReference};

fn main() -> Result<(), Box<dyn Error>> {
    // Sample inputs
    let input_signal = InputSignal::new(vec![1.0, -2.5, 3.0])?;
    let noise_ref = NoiseReference::new(vec![2.0, -1.2, -3.8])?;

    // Initialize the filter
    let mut lms = LmsFilter::new(LmsOptions {
        mu: 0.1,         // The filter's step size
        window_size: 16, // How many samples we process at a time
    })?;

    // Adapt the filter to the inputs and get the iteratively cleaned signal.
    let _cleaned_signal = lms.adapt(&input_signal, &noise_ref)?;

    Ok(())
}
