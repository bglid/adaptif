#![allow(
    clippy::tests_outside_test_module,
    reason = "This crate is exclusively an integration test target"
)]

use adaptif::algorithms::{Lms, Nlms, Rls};
use adaptif::filters::{AdaptiveFilter as _, FilterBase};
use adaptif::types::signals::{InputSignal, NoiseReference};

use super::macros::generate_integration_filter_tests;
use crate::rust::utils::{mse, snr, test_signals};

// TODO: Later Issue, turn this into a proc macro that generates a diff test per algo
// Reason: Currently this just shows up as one test. Still tests each filter, but
// -- viz as to what's happening is not great
#[test]
#[allow(clippy::unwrap_used, reason = "Integration test")]
fn filter_adapts_signal() {
    let signals = test_signals();
    for (original_signal, noisy, noise_ref) in signals {
        let noisy_signal = InputSignal::new(noisy.clone()).unwrap();
        let noise_reference = NoiseReference::new(noise_ref).unwrap();
        let window_size = 16;

        let lms_config = Lms::new(0.1).unwrap();
        let mut lms = FilterBase::new(lms_config, window_size).unwrap();
        generate_integration_filter_tests!(lms, original_signal, noisy_signal, noise_reference);

        let nlms_config = Nlms::new(0.01, 1e-6).unwrap();
        let mut nlms = FilterBase::new(nlms_config, window_size).unwrap();
        generate_integration_filter_tests!(nlms, original_signal, noisy_signal, noise_reference);

        let rls_config = Rls::new(0.999, 0.001).unwrap();
        let mut rls = FilterBase::new(rls_config, window_size).unwrap();
        generate_integration_filter_tests!(rls, original_signal, noisy_signal, noise_reference);
    }
}
