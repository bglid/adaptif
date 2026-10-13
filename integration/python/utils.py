from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

CLEAN = Path(__file__).parents[1] / "fixtures"


def add_white_noise(
    clean: np.ndarray, seed: int, snr_db: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Adds white noise to a clean signal and returns noisy signal & noise. Based on seed and snr_db.

    Args:
        clean (np.ndarray): Clean signal.
        seed (int): Random seed for distribution.
        snr_db (float): Target SNR level for noisy signal.

    Returns:
        (tuple(np.ndarray, np.ndarray)): Tuple of noisy signal + white noise, at target SNR in DB.
    """
    rng = np.random.default_rng(seed)
    raw_noise = rng.standard_normal(clean.shape)

    signal_power = np.mean(clean**2)
    noise_power = np.mean(raw_noise**2)

    target_noise_power = signal_power / (10 ** (snr_db / 10))
    raw_noise *= np.sqrt(target_noise_power / noise_power)
    noisy_signal = clean + raw_noise
    return noisy_signal, raw_noise


def load_test_signals() -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Loads the test data for integration tests.

    Returns:
        list(tuple(str, np.ndarray, np.ndarray, np.ndarray)): Tuple of the clean signal, noisy signal, and noise.

    """
    clean_paths = sorted((CLEAN).glob("*.wav"))
    signals = []

    for i, clean_path in enumerate(clean_paths):
        file_name = clean_path.name
        clean_signal, _ = sf.read(clean_path, dtype="float64")
        noisy_signal, noise_reference = add_white_noise(
            clean=clean_signal, seed=(42 + i), snr_db=0.0
        )

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
    noise_power = np.sum((original_signal - noisy_signal) ** 2)
    snr = signal_power / (noise_power + 1e-8)
    return 10 * np.log10(snr)
