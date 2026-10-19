#![allow(
    clippy::tests_outside_test_module,
    reason = "This crate is exclusively an integration test target"
)]
#![allow(clippy::unwrap_used, reason = "Integration test")]

use adaptif::filters::{
    LmsFilter, LmsOptions, NlmsFilter, NlmsOptions, ProcessingMode, RlsFilter, RlsOptions,
};
use adaptif::types::signals::{InputSignal, NoiseReference};

use super::macros::generate_integration_filter_tests;
use crate::rust::utils::{mse, snr, test_signals};

#[test]
fn lms_adapts_signal() {
    let mut filter = LmsFilter::new(LmsOptions::default(), ProcessingMode::Sample).unwrap();
    generate_integration_filter_tests!(filter, test_signals());
}

#[test]
fn nlms_adapts_signal() {
    let mut filter = NlmsFilter::new(NlmsOptions::default(), ProcessingMode::Sample).unwrap();
    generate_integration_filter_tests!(filter, test_signals());
}

#[test]
fn rls_adapts_signal() {
    let mut filter = RlsFilter::new(RlsOptions::default(), ProcessingMode::Sample).unwrap();
    generate_integration_filter_tests!(filter, test_signals());
}
