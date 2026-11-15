from numpy.typing import NDArray

class LMSFilter:
    """Least Mean Squares (LMS) adaptive filter.

    Args:
        step_size: Step size. Must be greater than 0.0.
            Defaults to 0.1.
        window_size: Number of filter weights. Must be greater than 0.
            Defaults to 16.
        block_size: Number of samples processed per block. Must be greater than 0.
            Defaults to 1.
    """
    def __init__(self, step_size: float, window_size: int, block_size: int) -> None: ...
    @property
    def window_size(self) -> int:
        """Return the window size of a filter. This is also the number of filter weights."""
    @property
    def block_size(self) -> int:
        """Return the block size of a filter."""
    @property
    def weights(self) -> NDArray:
        """Return the weights of a filter."""
    def load_weights(self, weights: NDArray) -> None:
        """Replace the filter's weights with previously learned weights.

        Args:
            weights: One-dimensional float64 NumPy array
                containing the previously learned weights. Length must match window_size.

        Raises:
            ValueError: If the number of weights doesn't match the window_size.
        """
    def adapt(self, input_signal: NDArray, noise_ref: NDArray) -> NDArray:
        """Adapt the filter weights using an input and reference signal.
        Return error signal. Updates the filter weights while processing the signals.

        Args:
            input_signal: One-dimensional float64 NumPy array
                containing the signal to process.
            noise_ref: One-dimensional float64 NumPy array
                containing the reference noise signal.

        Returns:
            A one-dimensional NumPy array containing the error signal.

        Raises:
            ValueError: If either signal is empty or noise_ref is shorter than input_signal.
        """
    def filter(self, input_signal: NDArray, noise_ref: NDArray) -> NDArray:
        """Filter the input signal without updating filter weights.

        Uses the current filter weights to process the input
        and reference signals. Unlike `adapt()`, this method
        does not perform weight adaptation.

        Args:
            input_signal: One-dimensional float64 NumPy array
                containing the signal to process.
            noise_ref: One-dimensional float64 NumPy array
                containing the reference noise signal.

        Returns:
            A one-dimensional NumPy array containing the error signal.

        Raises:
            ValueError: If either signal is empty or noise_ref is shorter than input_signal.
        """

class NLMSFilter:
    """Normalized Least Mean Squares (NLMS) adaptive filter.

    Args:
        step_size: Step size. Must be greater than 0.0.
            Defaults to 0.1.
        eps: Normalization epsilon. Must be greater than 0.0
            Defaults to 1e-8.
        window_size: Number of filter weights. Must be greater than 0.
            Defaults to 16.
        block_size: Number of samples processed per block. Must be greater than 0.
            Defaults to 1.
    """
    def __init__(
        self, step_size: float, eps: float, window_size: int, block_size: int
    ) -> None: ...
    @property
    def window_size(self) -> int:
        """Return the window size of a filter. This is also the number of filter weights."""
    @property
    def block_size(self) -> int:
        """Return the block size of a filter."""
    @property
    def weights(self) -> NDArray:
        """Return the weights of a filter."""
    def load_weights(self, weights: NDArray) -> None:
        """Replace the filter's weights with previously learned weights.

        Args:
            weights: One-dimensional float64 NumPy array
                containing the previously learned weights. Length must match window_size.

        Raises:
            ValueError: If the number of weights doesn't match the window_size.
        """
    def adapt(self, input_signal: NDArray, noise_ref: NDArray) -> NDArray:
        """Adapt the filter weights using an input and reference signal.
        Return error signal. Updates the filter weights while processing the signals.

        Args:
            input_signal: One-dimensional float64 NumPy array
                containing the signal to process.
            noise_ref: One-dimensional float64 NumPy array
                containing the reference noise signal.

        Returns:
            A one-dimensional NumPy array containing the error signal.

        Raises:
            ValueError: If either signal is empty or noise_ref is shorter than input_signal.
        """
    def filter(self, input_signal: NDArray, noise_ref: NDArray) -> NDArray:
        """Filter the input signal without updating filter weights.

        Uses the current filter weights to process the input
        and reference signals. Unlike `adapt()`, this method
        does not perform weight adaptation.

        Args:
            input_signal: One-dimensional float64 NumPy array
                containing the signal to process.
            noise_ref: One-dimensional float64 NumPy array
                containing the reference noise signal.

        Returns:
            A one-dimensional NumPy array containing the error signal.

        Raises:
            ValueError: If either signal is empty or noise_ref is shorter than input_signal.
        """

class RLSFilter:
    """Recursive Least Squares (RLS) adaptive filter.

    Args:
        forgetting_factor: RLS Forgetting factor. Must be greater than 0.0 and less-than or equal to 1.0.
            Defaults to 0.999.
        p_init_scale: Scalar value used in initializing the inverse correlation matrix. Must be greater than 0.0.
            Defaults to 0.001.
        window_size: Number of filter weights. Must be greater than 0.
            Defaults to 8.
        block_size: Number of samples processed per block. Must be greater than 0.
            Defaults to 1.
    """
    def __init__(
        self,
        forgetting_factor: float,
        p_init_scale: float,
        window_size: int,
        block_size: int,
    ) -> None: ...
    @property
    def window_size(self) -> int:
        """Return the window size of a filter. This is also the number of filter weights."""
    @property
    def block_size(self) -> int:
        """Return the block size of a filter."""
    @property
    def weights(self) -> NDArray:
        """Return the weights of a filter."""
    def load_weights(self, weights: NDArray) -> None:
        """Replace the filter's weights with previously learned weights.

        Args:
            weights: One-dimensional float64 NumPy array
                containing the previously learned weights. Length must match window_size.

        Raises:
            ValueError: If the number of weights doesn't match the window_size.
        """
    def adapt(self, input_signal: NDArray, noise_ref: NDArray) -> NDArray:
        """Adapt the filter weights using an input and reference signal.
        Return error signal. Updates the filter weights while processing the signals.

        Args:
            input_signal: One-dimensional float64 NumPy array
                containing the signal to process.
            noise_ref: One-dimensional float64 NumPy array
                containing the reference noise signal.

        Returns:
            A one-dimensional NumPy array containing the error signal.

        Raises:
            ValueError: If either signal is empty or noise_ref is shorter than input_signal.
        """
    def filter(self, input_signal: NDArray, noise_ref: NDArray) -> NDArray:
        """Filter the input signal without updating filter weights.

        Uses the current filter weights to process the input
        and reference signals. Unlike `adapt()`, this method
        does not perform weight adaptation.

        Args:
            input_signal: One-dimensional float64 NumPy array
                containing the signal to process.
            noise_ref: One-dimensional float64 NumPy array
                containing the reference noise signal.

        Returns:
            A one-dimensional NumPy array containing the error signal.

        Raises:
            ValueError: If either signal is empty or noise_ref is shorter than input_signal.
        """
