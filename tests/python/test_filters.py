import numpy as np
from adaptif import LMSFilter

from .utils import load_test_signals, mse, snr


# TODO: MAKE as a fixture to test across each filter
def test_lms_process_audio() -> None:
    _, noisy_signal, noise_reference = load_test_signals()

    lms = LMSFilter(mu=0.01, window_size=32)
    cleaned_signal = lms.adapt(input_signal=noisy_signal, noise_ref=noise_reference)

    assert cleaned_signal.shape == noisy_signal.shape
    assert np.all(np.isfinite(cleaned_signal))


# TODO: MAKE as a fixture to test across each filter
def test_lms_adapts_signal() -> None:
    original_signal, noisy_signal, noise_reference = load_test_signals()

    lms = LMSFilter(mu=0.01, window_size=32)
    cleaned_signal = lms.adapt(input_signal=noisy_signal, noise_ref=noise_reference)

    before_snr = snr(desired_signal=original_signal, noisy_signal=noisy_signal)
    after_snr = snr(desired_signal=original_signal, noisy_signal=cleaned_signal)
    assert after_snr > before_snr

    before_mse = mse(desired_signal=original_signal, input_signal=noisy_signal)
    after_mse = mse(desired_signal=original_signal, input_signal=cleaned_signal)
    assert after_mse < before_mse
