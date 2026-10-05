from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

DATA = Path(__file__).parents[1] / "data" / "MS-SNSD"


def load_test_signals() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads the test data for integration tests.

    Returns:
        tuple(np.ndarray, np.ndarray, np.ndarray): Tuple of the clean signal, noisy signal, and noise.

    """
    clean_path = next((DATA / "CleanSpeech_training").glob("*.wav"))
    noisy_path = next((DATA / "NoisySpeech_training").glob("*.wav"))

    clean_signal, clean_sr = sf.read(clean_path)
    noisy_signal, noisy_sr = sf.read(noisy_path)

    assert clean_sr == noisy_sr

    n = min(len(clean_signal), len(noisy_signal))
    clean_signal = clean_signal[:n]
    noisy_signal = noisy_signal[:n]

    noise_reference = noisy_signal - clean_signal

    return clean_signal, noisy_signal, noise_reference


def mse(
    original_signal: NDArray[np.float64], input_signal: NDArray[np.float64]
) -> np.float64:
    """Calculates the Mean Squared Error = 1/n * sum(y - y_hat)**2

    Args:
        original_signal (NDArray[np.float64]): Original signal/value
        input_signal (NDArray[np.float64]): Prediction of signal/value

    Returns:
        np.float64: Mean squared error
    """
    return np.mean((original_signal - input_signal) ** 2)


def snr(
    original_signal: NDArray[np.float64], noisy_signal: NDArray[np.float64]
) -> np.float64:
    """Calculates the Signal to Noise Ratio in dB: SNR = (Power of Signal)/(Power of Noise).

    Args:
        original_signal (NDArray[np.float64]): Original clean signal
        noisy_signal (NDArray[np.float64]): Input Noisy signal

    Returns:
        np.float64: SNR in dB
    """
    signal_power = np.sum(original_signal**2)
    noise_power = np.sum((original_signal - noisy_signal) ** 2) + 1e-8
    snr = signal_power / noise_power
    return 10 * np.log10(snr)
