import numpy as np
import pytest
from adaptif import BlockLMSFilter, LMSFilter, NLMSFilter, RLSFilter

from .utils import load_test_signals, mse, snr


# placing in realistic values that would actually be used
@pytest.fixture(
    params=[
        (LMSFilter, {"mu": 0.01, "window_size": 16}),
        (NLMSFilter, {"mu": 0.001, "eps": 1e-6, "window_size": 16}),
        (
            RLSFilter,
            {"forgetting_factor": 0.999, "p_init_scale": 0.001, "window_size": 16},
        ),
        (BlockLMSFilter, {"mu": 1e-4, "window_size": 32, "block_size": 32}),
    ],
    ids=["lms", "nlms", "rls", "block-lms"],
)
def filter(request):
    filter_class, kwargs = request.param
    return filter_class(**kwargs)


def test_filter_adapts_signal(filter) -> None:
    original_signal, noisy_signal, noise_reference = load_test_signals()

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
