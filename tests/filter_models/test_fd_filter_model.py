import numpy as np
import pytest

from adaptive_filter.filter_models.fd_filter_model import FrequencyDomainAF


@pytest.fixture
def sample_model():
    # creating a sample model
    filter_model = FrequencyDomainAF(mu=0.1, filter_order=3, block_size=2)
    # setting the weights manually
    filter_model.W = np.array([1.0, -2.0, 0.5])
    return filter_model


def test_noise_estimate(sample_model):
    x_n = np.array([2.0, 3.0, 4.0])
    assert sample_model.noise_estimate(x_n) == pytest.approx(-2.0)


def test_error(sample_model):
    d_n = 5.0
    noise_estimate = 3.5
    assert sample_model.error(d_n, noise_estimate) == pytest.approx(1.5)


def test_update_step(sample_model):
    e_n = np.array([5.0, 5.0, 5.0], dtype=np.complex128)
    x_n = np.array([2.0, 3.0, 4.0], dtype=np.complex128)

    # using is close for complex vals
    expected = np.array([1.0, 1.5, 2.0], dtype=np.complex128)

    output = sample_model.update_step(e_n, x_n)
    assert isinstance(output, np.ndarray)
    assert output.shape == x_n.shape
    assert np.allclose(output, expected)


def test_filter_returns_array(monkeypatch: pytest.MonkeyPatch):
    model = FrequencyDomainAF(mu=0.1, filter_order=1, block_size=2)

    # basically this replaces update_step with a lambda that returns
    # -> an array of 0.0, to test that filter() returns the array correctly.
    # Doesn't update algo, checks ndarray and shape
    monkeypatch.setattr(
        model,
        "update_step",
        lambda e_n, x_n: np.array([0.0], dtype=np.complex128),
    )

    d = np.linspace(1, 5, 5)
    x = np.linspace(0.5, 2.5, 5)

    result = model.filter(d, x)

    assert isinstance(result, np.ndarray)
    assert result.shape == d.shape
