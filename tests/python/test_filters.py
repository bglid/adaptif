import numpy as np
import pytest
from adaptif import LMSFilter, NLMSFilter, RLSFilter

from .utils import load_test_signals, mse, snr


# placing in values that would actually be used
@pytest.fixture(
    params=[
        (LMSFilter, {"mu": 0.01, "window_size": 16}),
        (NLMSFilter, {"mu": 0.001, "eps": 1e-6, "window_size": 16}),
        (
            RLSFilter,
            {"forgetting_factor": 0.999, "p_init_scale": 0.001, "window_size": 16},
        ),
    ],
    ids=[
        "lms",
        "nlms",
        "rls",
    ],
)
def filter(request):
    filter_class, kwargs = request.param
    return filter_class(**kwargs)


def test_filter_process_audio(filter) -> None:
    _, noisy_signal, noise_reference = load_test_signals()

    cleaned_signal = filter.adapt(input_signal=noisy_signal, noise_ref=noise_reference)

    assert cleaned_signal.shape == noisy_signal.shape
    assert np.all(np.isfinite(cleaned_signal))


def test_filter_adapts_signal(filter) -> None:
    original_signal, noisy_signal, noise_reference = load_test_signals()

    cleaned_signal = filter.adapt(input_signal=noisy_signal, noise_ref=noise_reference)

    before_snr = snr(desired_signal=original_signal, noisy_signal=noisy_signal)
    after_snr = snr(desired_signal=original_signal, noisy_signal=cleaned_signal)
    assert after_snr > before_snr

    before_mse = mse(desired_signal=original_signal, input_signal=noisy_signal)
    after_mse = mse(desired_signal=original_signal, input_signal=cleaned_signal)
    assert after_mse < before_mse
