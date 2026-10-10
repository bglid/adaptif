#![allow(
    clippy::tests_outside_test_module,
    reason = "This crate is exclusively an integration test target"
)]

use adaptif::algorithms::Lms;
use adaptif::filters::{AdaptiveFilter as _, FilterBase};
use adaptif::types::signals::{InputSignal, NoiseReference};

use crate::rust::utils::{mse, snr, test_signals};

#[test]
#[allow(clippy::unwrap_used, reason = "Integration test")]
fn filter_adapts_signal() {
    let signals = test_signals();
    for (original_signal, noisy, noise_ref) in signals {
        let noisy_signal = InputSignal::new(noisy.clone()).unwrap();
        let noise_reference = NoiseReference::new(noise_ref).unwrap();
        let lms_config = Lms::new(0.01).unwrap();
        let window_size = 16;
        let mut lms = FilterBase::new(lms_config, window_size).unwrap();

        let cleaned_signal = lms.adapt(&noisy_signal, &noise_reference).unwrap();

        assert_eq!(cleaned_signal.len(), noisy.len());
        assert!(cleaned_signal.iter().all(|sample| sample.is_finite()));

        let before_mse = mse(&original_signal, &noisy_signal);
        let after_mse = mse(&original_signal, &cleaned_signal);
        assert!(after_mse < before_mse);

        let before_snr = snr(&original_signal, &noisy_signal);
        let after_snr = snr(&original_signal, &cleaned_signal);
        assert!(after_snr > before_snr);
    }
}
