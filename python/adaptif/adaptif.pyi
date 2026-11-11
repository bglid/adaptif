from numpy.typing import NDArray

class LMSFilter:
    """Least Mean Squares (LMS) adaptive filter.

    Args:
        ``mu``: Step size. Must be greater than 0.0.
            Defaults to 0.1.
        ``window_size``: Number of filter coefficients. Must be greater than 0.
            Defaults to 16.
        ``block_size``: Number of samples processed per block. Must be greater than 0.
            Defaults to 1.
    """
    def __init__(self, mu: float, window_size: int, block_size: int) -> None: ...
    @property
    def window_size(self) -> int: ...
    @property
    def block_size(self) -> int: ...
    @property
    def weights(self) -> NDArray: ...
    def load_weights(weights: NDArray) -> LMSFilter: ...
    def adapt(self, input_signal: NDArray, noise_ref: NDArray) -> NDArray: ...
    def filter(self, input_signal: NDArray, noise_ref: NDArray) -> NDArray: ...

class NLMSFilter:
    """Normalized Least Mean Squares (NLMS) adaptive filter.

    Args:
        ``mu``: Step size. Must be greater than 0.0.
            Defaults to 0.1.
        ``eps``: Normalization epsilon. Must be greater than 0.0
            Defaults to 1e-8.
        ``window_size``: Number of filter coefficients. Must be greater than 0.
            Defaults to 16.
        ``block_size``: Number of samples processed per block. Must be greater than 0.
            Defaults to 1.
    """
    def __init__(
        self, mu: float, eps: float, window_size: int, block_size: int
    ) -> None: ...
    @property
    def window_size(self) -> int: ...
    @property
    def block_size(self) -> int: ...
    @property
    def weights(self) -> NDArray: ...
    def load_weights(weights: NDArray) -> NLMSFilter: ...
    def adapt(self, input_signal: NDArray, noise_ref: NDArray) -> NDArray: ...
    def filter(self, input_signal: NDArray, noise_ref: NDArray) -> NDArray: ...

class RLSFilter:
    """Recursive Least Squares (RLS) adaptive filter.

    Args:
        ``forgetting_factor``: RLS Forgetting factor. Must be greter than 0.0 and less-than or equal to 1.0.
            Defaults to 0.999.
        ``p_init_scale``: Scalar value used in initializing the inverse correlation matrix. Must be greater than 0.0.
            Defaults to 0.001.
        ``window_size``: Number of filter coefficients. Must be greater than 0.
            Defaults to 8.
        ``block_size``: Number of samples processed per block. Must be greater than 0.
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
    def window_size(self) -> int: ...
    @property
    def block_size(self) -> int: ...
    @property
    def weights(self) -> NDArray: ...
    def load_weights(weights: NDArray) -> RLSFilter: ...
    def adapt(self, input_signal: NDArray, noise_ref: NDArray) -> NDArray: ...
    def filter(self, input_signal: NDArray, noise_ref: NDArray) -> NDArray: ...
