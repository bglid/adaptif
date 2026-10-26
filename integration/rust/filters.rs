#![allow(
    clippy::tests_outside_test_module,
    reason = "This crate is exclusively an integration test target"
)]
#![allow(clippy::unwrap_used, reason = "Integration test")]

use adaptif::algorithms::Algorithm;
use adaptif::filters::{
    ApaFilter, ApaOptions, FilterBase, LmsFilter, LmsOptions, NlmsFilter, NlmsOptions,
    ProcessingMode, RlsFilter, RlsOptions,
};

use super::utils::{TestSignalBatch, load_test_signals, mse, snr};

use std::sync::OnceLock;

static TEST_SIGNALS: OnceLock<Vec<TestSignalBatch>> = OnceLock::new();

fn test_filter<A>(filter: &mut FilterBase<f64, A>)
where
    A: Algorithm<f64>,
{
    let test_signals = TEST_SIGNALS.get_or_init(load_test_signals);

    for batch in test_signals {
        let cleaned_signal = filter.adapt(&batch.noisy_signal, &batch.noise_ref).unwrap();

        assert_eq!(cleaned_signal.len(), batch.noisy_signal.len());
        assert!(cleaned_signal.iter().all(|sample| sample.is_finite()));

        let before_mse = mse(&batch.clean_signal, &batch.noisy_signal);
        let after_mse = mse(&batch.clean_signal, &cleaned_signal);
        assert!(after_mse < before_mse);

        let before_snr = snr(&batch.clean_signal, &batch.noisy_signal);
        let after_snr = snr(&batch.clean_signal, &cleaned_signal);
        assert!(after_snr > before_snr);
    }
}

#[test]
fn lms_adapts_signal() {
    let mut filter = LmsFilter::new(LmsOptions::default(), ProcessingMode::Sample).unwrap();
    test_filter(&mut filter);
}

#[test]
fn nlms_adapts_signal() {
    let mut filter = NlmsFilter::new(NlmsOptions::default(), ProcessingMode::Sample).unwrap();
    test_filter(&mut filter);
}

#[test]
fn rls_adapts_signal() {
    let mut filter = RlsFilter::new(RlsOptions::default(), ProcessingMode::Sample).unwrap();

    test_filter(&mut filter);
}

#[test]
fn apa_adapts_signal() {
    let mut filter = ApaFilter::new(ApaOptions::default(), ProcessingMode::Sample).unwrap();

    test_filter(&mut filter);
}

#[test]
fn block_lms_adapts_signal() {
    let mut filter = LmsFilter::new(
        LmsOptions::default(),
        ProcessingMode::Block { block_size: 4 },
    )
    .unwrap();
    test_filter(&mut filter);
}

#[test]
fn block_nlms_adapts_signal() {
    let mut filter = NlmsFilter::new(
        NlmsOptions::default(),
        ProcessingMode::Block { block_size: 4 },
    )
    .unwrap();
    test_filter(&mut filter);
}

#[test]
fn block_rls_adapts_signal() {
    let mut filter = RlsFilter::new(
        RlsOptions::default(),
        ProcessingMode::Block { block_size: 4 },
    )
    .unwrap();

    test_filter(&mut filter);
}

#[test]
fn block_apa_adapts_signal() {
    let mut filter = ApaFilter::new(
        ApaOptions::default(),
        ProcessingMode::Block { block_size: 4 },
    )
    .unwrap();

    test_filter(&mut filter);
}
