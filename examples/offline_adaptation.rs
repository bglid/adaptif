#![allow(clippy::unwrap_used, clippy::shadow_unrelated, reason = "Examples")]

use adaptif::algorithms::Lms;
use adaptif::filters::{AdaptiveFilter as _, LMSFilter};
use adaptif::types::signals::{InputSignal, NoiseReference};

fn main() {
    // Sample inputs
    let input_signal = InputSignal::new(&[1.0, -2.5, 3.0]).unwrap();
    let noise_ref = NoiseReference::new(&[2.0, -1.2, -3.8]).unwrap();

    // The parameters used by the LMS filter
    let lms_config = Lms::new(1.0).unwrap();
    // How many samples we process at a time
    let window_size = 1024;
    // Initialize the filter
    let mut lms = LMSFilter::new(lms_config, window_size).unwrap();

    // Adapt the filter to the inputs -- because we call `filter()` later, we can discard the output signal.
    let _ = lms.adapt(&input_signal, &noise_ref).unwrap();

    // Apply the learned filter without updating it
    let _cleaned_signal = lms.filter(&input_signal, &noise_ref).unwrap();

    // `filter()` doesn't require the filter to be mutable, so we can also do this:
    let lms_adapted = {
        let mut lms = LMSFilter::new(Lms::new(1.0).unwrap(), window_size).unwrap();
        lms.adapt(&input_signal, &noise_ref).unwrap();
        lms
    };

    // ... then call `filter()` on the immutable filter
    let _cleaned_signal = lms_adapted.filter(&input_signal, &noise_ref).unwrap();
}
