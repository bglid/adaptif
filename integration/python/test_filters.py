import numpy as np
import pytest
from adaptif import LMSFilter, NLMSFilter, RLSFilter

from .utils import load_test_signals, mse, snr


@pytest.fixture(params=load_test_signals(), ids=lambda signal: signal[0])
def signals(request):
    name, clean, noisy, noise = request.param
    return name, clean, noisy, noise


@pytest.mark.parametrize(
    "filter_class",
    [LMSFilter, NLMSFilter, RLSFilter],
)
@pytest.mark.parametrize("block_size", [1, 4])
def test_filter_adapts_signal(filter_class, block_size, signals) -> None:
    filter = filter_class(block_size=block_size)
    _, original_signal, noisy_signal, noise_reference = signals

    cleaned_signal = filter.adapt(input_signal=noisy_signal, noise_ref=noise_reference)

    assert cleaned_signal.shape == noisy_signal.shape
    assert np.all(np.isfinite(cleaned_signal))

    before_snr = snr(original_signal=original_signal, noisy_signal=noisy_signal)
    after_snr = snr(original_signal=original_signal, noisy_signal=cleaned_signal)
    assert after_snr > before_snr

    # NOTE: point of this is that it doesn't only remove noise, but recovers the signal
    before_mse = mse(original_signal=original_signal, input_signal=noisy_signal)
    after_mse = mse(original_signal=original_signal, input_signal=cleaned_signal)
    assert after_mse < before_mse
