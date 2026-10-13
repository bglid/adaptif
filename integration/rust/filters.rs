#![allow(
    clippy::tests_outside_test_module,
    reason = "This crate is exclusively an integration test target"
)]
#![allow(clippy::unwrap_used, reason = "Integration test")]

use adaptif::filters::{LmsFilter, LmsOptions, NlmsFilter, NlmsOptions, RlsFilter, RlsOptions};
use adaptif::types::WindowSize;
use adaptif::types::signals::{InputSignal, NoiseReference};

use super::macros::generate_integration_filter_tests;
use crate::rust::utils::{mse, snr, test_signals};

// TODO: Later Issue, turn this into a proc macro that generates a diff test per algo
// Reason: Currently this just shows up as one test. Still tests each filter, but
// -- viz as to what's happening is not great
#[test]
fn lms_adapts_signal() {
    let mut filter = LmsFilter::new(LmsOptions { mu: 0.1 }, WindowSize::new(16).unwrap()).unwrap();
    generate_integration_filter_tests!(filter, test_signals());
}

#[test]
fn nlms_adapts_signal() {
    let mut filter = NlmsFilter::new(
        NlmsOptions {
            mu: 0.01,
            eps: 1e-6,
        },
        WindowSize::new(16).unwrap(),
    )
    .unwrap();
    generate_integration_filter_tests!(filter, test_signals());
}

#[test]
fn rls_adapts_signal() {
    let mut filter = RlsFilter::new(
        RlsOptions {
            forgetting_factor: 0.999,
            p_init_scale: 0.001,
        },
        WindowSize::new(16).unwrap(),
    )
    .unwrap();
    generate_integration_filter_tests!(filter, test_signals());
}
