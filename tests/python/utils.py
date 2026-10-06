from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

DATA = Path(__file__).parents[1] / "fixtures"
CLEAN = DATA / "CleanSpeech"
NOISY = DATA / "NoisySpeech"


def load_test_signals() -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Loads the test data for integration tests.

    Returns:
        list(tuple(np.ndarray, np.ndarray, np.ndarray)): Tuple of the clean signal, noisy signal, and noise.

    """
    clean_paths = sorted((CLEAN).glob("*.wav"))
    noisy_paths = sorted((NOISY).glob("*.wav"))

    print("CLEAN WAVS", "\n")
    print(clean_paths)

    print("NOISY WAVS", "\n")
    print(clean_paths)

    signals = []

    for clean_path, noisy_path in zip(clean_paths, noisy_paths, strict=True):
        file_name = noisy_path.name
        clean_signal, clean_sr = sf.read(clean_path)
        noisy_signal, noisy_sr = sf.read(noisy_path)
        assert clean_sr == noisy_sr

        n = min(len(clean_signal), len(noisy_signal))
        clean_signal = clean_signal[:n]
        noisy_signal = noisy_signal[:n]

        noise_reference = noisy_signal - clean_signal

        signals.append((file_name, clean_signal, noisy_signal, noise_reference))

    return signals


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
