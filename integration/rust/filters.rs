#![allow(
    clippy::tests_outside_test_module,
    reason = "This crate is exclusively an integration test target"
)]
#![allow(clippy::unwrap_used, reason = "Integration test")]

use adaptif::algorithms::{Lms, Nlms, Rls};
use adaptif::filters::{AdaptiveFilter as _, FilterBase};
use adaptif::types::signals::{InputSignal, NoiseReference};

use super::macros::generate_integration_filter_tests;
use crate::rust::utils::{mse, snr, test_signals};

// TODO: Later Issue, turn this into a proc macro that generates a diff test per algo
// Reason: Currently this just shows up as one test. Still tests each filter, but
// -- viz as to what's happening is not great
#[test]
fn lms_adapts_signal() {
    let signals = test_signals();
    let lms_config = Lms::new(0.1).unwrap();
    generate_integration_filter_tests!(lms_config, signals);
}

#[test]
fn nlms_adapts_signal() {
    let signals = test_signals();
    let nlms_config = Nlms::new(0.01, 1e-6).unwrap();
    generate_integration_filter_tests!(nlms_config, signals);
}

#[test]
fn rls_adapts_signal() {
    let signals = test_signals();
    let rls_config = Rls::new(0.999, 0.001).unwrap();
    generate_integration_filter_tests!(rls_config, signals);
}
