import numpy as np
from numpy.typing import NDArray


class FrequencyDomainAF:
    def __init__(
        self,
        mu: float,
        filter_order: int,
        block_size: int,
    ) -> None:
        self.mu = mu
        self.N = filter_order
        self.block_size = block_size
        self.hop_size = block_size // 2

        # Overriding with FD specs
        self.half_bins = self.block_size // 2 + 1
        self.eps = 1e-8
        self.algorithm = "FDAF"
        self.W = np.zeros(self.half_bins, dtype=np.float64)

    def noise_estimate(self, x_n: NDArray[np.float64]) -> np.float64:
        """Predict the noise estimate, given vector X[n], noise reference. Uses formula W^T[n]X[n].

        Args:
            x_n (NDArray[np.float64]): vector[n] of array X, the noise estimate

        Returns:
            np.float64: Predicted noise estimate output of the FIR filter and the noise reference
        """
        return np.dot(self.W, x_n)

    def error(self, d_n: float, noise_estimate: float) -> float:
        """Calculate the error, e[n] = d[n] - y[n], y[n] is output of W^T[n]X[n].

        Args:
            d_n (float): Desired sample at point n of array D, noisy input
            noise_estimate (float): The noise estimate product (y[n])

        Returns:
            float: error of (noisy) desired input[n] - noise estimate. Ideally, this should be the clean signal
        """
        return d_n - noise_estimate

    # Setting the update step to include conj
    def update_step(
        self, e_n: NDArray[np.complex128], x_n: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        """Update for FDAF.

        Args:
            e_n (NDArray[np.complex128]): Block error in the frequency domain.
            x_n (NDArray[np.complex128]): Block noise estimate in the frequency domain.

        Returns:
            NDArray[np.complex128]: The frequency domain weight update vec.
        """
        return self.mu * np.multiply(np.conj(x_n), e_n)

    # FD specs
    def filter(
        self,
        d: NDArray[np.float64],
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Iterate Adaptive filter alorithm and updates for length of input signal X.

        Args:
            d (NDArray[np.float64]):
                "Desired Signal", which in the ANC use-case is the noisy input signal.
            x (NDArray[np.float64]):
                Input reference matrix X, which in the ANC case is the noise reference.

        Returns:
            NDArray[np.float64]: "Clean output" The error signal of d - y.

        Raises:
            ValueError: If Signal dims are not compatible (1D)
        """
        # turning D and X into np arrays, if not already
        d = np.asarray(d).ravel()
        if d.ndim != 1:
            raise ValueError(f"Expected desired signal to be 1D, got shape: {d.shape}")

        x = np.asarray(x).ravel()
        if x.ndim != 1:
            raise ValueError(f"Expected input signal to be 1D, got shape: {x.shape}")

        if d.shape[0] < x.shape[0]:
            x = x[: d.shape[0]]

        # pad amount
        P = 1 << int(np.ceil(np.log2(self.N + self.hop_size - 1)))
        # padding the impulse response
        self.H = np.fft.rfft(np.pad(self.W, (0, P - self.N)), n=P)

        # getting the number of samples from x len
        num_samples = len(x)

        # initializing the arrays to hold error and noise estimate
        noise_estimate = np.zeros(num_samples)
        error = np.zeros(num_samples)

        # handling odd or '1' sample leftovers
        for sample in range(0, num_samples - self.hop_size + 1, self.hop_size):
            block = x[sample : sample + P]
            if len(block) < P:
                block = np.pad(block, (0, P - len(block)))

            # converting our block of input to the F domain
            x_f = np.fft.rfft(block)

            # getting our output via "convolution" in F domain
            y_f = self.H * x_f
            y_time = np.fft.irfft(y_f, n=P)

            # getting the overlapping
            valid = y_time[self.N - 1 : self.N - 1 + self.hop_size]
            noise_estimate[sample : sample + self.hop_size] = np.real(valid)

            # getting the overlapping error
            error[sample : sample + self.hop_size] = (
                d[sample : sample + self.hop_size]
                - noise_estimate[sample : sample + self.hop_size]
            )

            # now update H
            e_f = np.fft.rfft(
                np.pad(error[sample : sample + self.hop_size], (0, P - self.hop_size)),
                n=P,
            )

            self.H += self.update_step(e_f, x_f)

            # checking for leftover
            leftover_sample = (num_samples // self.hop_size) * self.hop_size
            if leftover_sample < num_samples:
                block = x[leftover_sample : leftover_sample + P]
                block = np.pad(block, (0, P - len(block)))

                # converting our block of input to the F domain
                x_f = np.fft.rfft(block)

                # getting our output via "convolution" in F domain
                y_f = self.H * x_f
                y_time = np.fft.irfft(y_f, n=P)

                # getting the overlapping
                valid = y_time[
                    self.N - 1 : self.N - 1 + (num_samples - leftover_sample)
                ]
                noise_estimate[leftover_sample:] = np.real(valid)

                # getting the overlapping error
                error[leftover_sample:] = d[leftover_sample:] - valid

                # now update H
                e_f = np.fft.rfft(
                    np.pad(valid, (0, P - len(valid))),
                    n=P,
                )
                self.H += self.update_step(e_f, x_f)

        return error
