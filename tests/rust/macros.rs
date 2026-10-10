/// Declarative macro for parameterizing filter algorithms for integration tests.
macro_rules! generate_integration_filter_tests {
    ($filter_config: ident, $original_signal: expr, $noisy_signal: expr, $noise_reference: expr) => {
        let window_size = 16;
        let mut filter = FilterBase::new($filter_config, window_size).unwrap();
        let cleaned_signal = filter.adapt(&$noisy_signal, &$noise_reference).unwrap();

        assert_eq!(cleaned_signal.len(), $noisy_signal.len());
        assert!(cleaned_signal.iter().all(|sample| sample.is_finite()));

        let before_mse = mse(&$original_signal, &$noisy_signal);
        let after_mse = mse(&$original_signal, &cleaned_signal);
        assert!(after_mse < before_mse);

        let before_snr = snr(&$original_signal, &$noisy_signal);
        let after_snr = snr(&$original_signal, &cleaned_signal);
        assert!(after_snr > before_snr);
    };
}

pub(crate) use generate_integration_filter_tests;
