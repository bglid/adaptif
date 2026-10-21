/// Declarative macro for parameterizing filter algorithms for integration tests.
macro_rules! generate_integration_filter_tests {
    ($filter: expr, $signals: expr) => {
        for batch in $signals {
            let cleaned_signal = $filter
                .adapt(&batch.noisy_signal, &batch.noise_ref)
                .unwrap();

            assert_eq!(cleaned_signal.len(), batch.noisy_signal.len());
            assert!(cleaned_signal.iter().all(|sample| sample.is_finite()));

            let before_mse = mse(&batch.clean_signal, &batch.noisy_signal);
            let after_mse = mse(&batch.clean_signal, &cleaned_signal);
            assert!(after_mse < before_mse);

            let before_snr = snr(&batch.clean_signal, &batch.noisy_signal);
            let after_snr = snr(&batch.clean_signal, &cleaned_signal);
            assert!(after_snr > before_snr);
        }
    };
}

pub(crate) use generate_integration_filter_tests;
