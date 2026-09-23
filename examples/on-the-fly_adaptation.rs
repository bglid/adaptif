use std::error::Error;

use adaptif::algorithms::Lms;
use adaptif::filters::{AdaptiveFilter as _, LMSFilter};
use adaptif::types::signals::{InputSignal, NoiseReference};

fn main() -> Result<(), Box<dyn Error>> {
    // Sample inputs
    let input_signal = InputSignal::new(vec![1.0, -2.5, 3.0])?;
    let noise_ref = NoiseReference::new(vec![2.0, -1.2, -3.8])?;

    let lms_config = Lms::new(1.0)?; // The parameters used by the LMS filter
    let window_size = 1024; // How many samples we process at a time
    // Initialize the filter
    let mut lms = LMSFilter::new(lms_config, window_size)?;

    // Adapt the filter to the inputs and get the iteratively cleaned signal.
    let _cleaned_signal = lms.adapt(&input_signal, &noise_ref)?;

    Ok(())
}
