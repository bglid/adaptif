#![allow(
    clippy::tests_outside_test_module,
    reason = "This crate is exclusively an integration test target"
)]
use adaptif::{algorithms::Lms, filters};

// Just getting this setup
#[test]
fn lms_filters_signal() {
    #[allow(clippy::unwrap_used, reason = "Tests")]
    let lms_config = Lms::new(1.0).unwrap();
    let window_size = 1024;
    let mut _lms = filters::LMSFilter::new(lms_config, window_size);
}
