#![allow(clippy::shadow_unrelated, reason = "Examples")]

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

    // Adapt the filter to the inputs -- because we call `filter()` later, we can discard the output signal.
    let _ = lms.adapt(&input_signal, &noise_ref)?;

    // Apply the learned filter without updating it
    let _cleaned_signal = lms.filter(&input_signal, &noise_ref)?;

    // `filter()` doesn't require the filter to be mutable, so we can also do this:
    let lms_adapted = {
        let mut lms = LmsFilter::new(LmsOptions {
            mu: 0.1,
            window_size: 16,
        })?;
        lms.adapt(&input_signal, &noise_ref)?;
        lms
    };

    // ... then call `filter()` on the immutable filter
    let _cleaned_signal = lms_adapted.filter(&input_signal, &noise_ref)?;

    Ok(())
}
