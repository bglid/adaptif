/// Declarative macro for parameterizing filter algorithms for integration tests.
macro_rules! generate_integration_filter_tests {
    ($filter_config: ident, $signals: expr) => {
        // TODO: Move logic for following two lines with filter init update
        let window_size = 16;
        let mut filter = FilterBase::new($filter_config, window_size).unwrap();
        for (original_signal, noisy, noise_ref) in $signals {
            let noisy_signal = InputSignal::new(noisy.clone()).unwrap();
            let noise_reference = NoiseReference::new(noise_ref).unwrap();
            let cleaned_signal = filter.adapt(&noisy_signal, &noise_reference).unwrap();

            assert_eq!(cleaned_signal.len(), noisy_signal.len());
            assert!(cleaned_signal.iter().all(|sample| sample.is_finite()));

            let before_mse = mse(&original_signal, &noisy_signal);
            let after_mse = mse(&original_signal, &cleaned_signal);
            assert!(after_mse < before_mse);

            let before_snr = snr(&original_signal, &noisy_signal);
            let after_snr = snr(&original_signal, &cleaned_signal);
            assert!(after_snr > before_snr);
        }
    };
}

pub(crate) use generate_integration_filter_tests;
