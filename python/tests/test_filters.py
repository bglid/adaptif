import numpy as np
import pytest
from adaptif import LMSFilter, NLMSFilter, RLSFilter


@pytest.fixture(
    params=[
        (LMSFilter, {"mu": 0.1, "window_size": 16}),
        (NLMSFilter, {"mu": 0.01, "eps": 1e-8, "window_size": 16}),
        (
            RLSFilter,
            {"forgetting_factor": 0.999, "p_init_scale": 0.01, "window_size": 8},
        ),
    ]
)
def filter(request):
    filter_class, kwargs = request.param
    return filter_class(**kwargs)


@pytest.mark.parametrize(
    ["filter_class", "kwargs"],
    [
        (LMSFilter, {"mu": 0.1}),
        (NLMSFilter, {"mu": 0.01, "eps": 1e-8}),
        (RLSFilter, {"forgetting_factor": 0.999, "p_init_scale": 0.01}),
    ],
)
@pytest.mark.parametrize("block_size", [1, 4])
def test_window_size(filter_class, kwargs, block_size):
    kwargs["block_size"] = block_size
    kwargs["window_size"] = 1
    filter = filter_class(**kwargs)
    assert filter.window_size == 1

    kwargs["window_size"] = -1
    with pytest.raises(OverflowError):
        filter_class(**kwargs)

    kwargs["window_size"] = 0
    with pytest.raises(ValueError):
        filter_class(**kwargs)


@pytest.mark.parametrize(
    ["filter_class", "kwargs"],
    [
        (LMSFilter, {"mu": 0.1}),
        (NLMSFilter, {"mu": 0.01, "eps": 1e-8}),
        (RLSFilter, {"forgetting_factor": 0.999, "p_init_scale": 0.01}),
    ],
)
def test_block_size(filter_class, kwargs):
    kwargs["window_size"] = 16

    kwargs["block_size"] = 1
    filter = filter_class(**kwargs)
    assert filter.block_size == 1

    kwargs["block_size"] = 2
    filter = filter_class(**kwargs)
    assert filter.block_size == 2

    kwargs["block_size"] = -1
    with pytest.raises(OverflowError):
        filter_class(**kwargs)

    kwargs["block_size"] = 0
    with pytest.raises(ValueError):
        filter_class(**kwargs)


def test_weights(filter):
    assert isinstance(filter.weights, np.ndarray)
    assert np.allclose(filter.weights, np.zeros(filter.window_size))


def test_load_weights(filter):
    weights = np.ones([filter.window_size])
    filter.load_weights(weights)
    assert filter.window_size == len(weights)
    assert np.allclose(filter.weights, weights)

    for w in [
        np.array([]),
        np.ones(filter.window_size - 1),
        np.ones(filter.window_size + 1),
    ]:
        with pytest.raises(ValueError):
            filter.load_weights(w)


def test_adapt(filter):
    input_signal = np.linspace(1, 5, 32)
    noise_ref = np.linspace(0.5, 2.5, 32)

    output = filter.adapt(input_signal, noise_ref)

    assert isinstance(output, np.ndarray)
    assert output.shape == input_signal.shape
    assert not np.allclose(filter.weights, np.zeros(filter.window_size))


def test_filter(filter):
    input_signal = np.linspace(1, 5, 32)
    noise_ref = np.linspace(0.5, 2.5, 32)

    output = filter.filter(input_signal, noise_ref)

    assert isinstance(output, np.ndarray)
    assert output.shape == input_signal.shape
    assert np.allclose(filter.weights, np.zeros(filter.window_size))


@pytest.mark.parametrize(
    "input, noise",
    [
        (np.linspace(1, 5, 5), np.array([])),
        (np.array([]), np.linspace(0.5, 2.5, 5)),
        (np.array([]), np.array([])),
    ],
)
def test_empty_input(filter, input, noise):
    with pytest.raises(ValueError):
        filter.adapt(input, noise)
    with pytest.raises(ValueError):
        filter.filter(input, noise)


@pytest.mark.parametrize("fn_name", ["adapt", "filter"])
def test_signal_lengths(filter, fn_name):
    filter_fn = getattr(filter, fn_name)

    short_input = np.linspace(1, 5, 4)
    long_input = np.linspace(1, 5, 6)
    noise_ref = np.linspace(0.5, 2.5, 5)

    short_output = filter_fn(short_input, noise_ref)
    assert isinstance(short_output, np.ndarray)
    assert short_output.shape == short_input.shape

    with pytest.raises(ValueError):
        filter_fn(long_input, noise_ref)


# NOTE: this test is only needed if input signals are passed by reference
# If they are copied instead, the can be non-contiguous.
# @pytest.mark.parametrize("fn_name", ["adapt", "filter"])
# def test_input_contiguous(filter, fn_name):
#     filter_fn = getattr(filter, fn_name)
#
#     input_signal = np.linspace(1, 5, 10)
#     noise_ref = np.linspace(0.5, 2.5, 10)
#
#     # regular slices are contiguous -> allowed
#     filter_fn(input_signal[:5], noise_ref[:5])
#
#     # strided slices are not contiguous -> not allowed
#     with pytest.raises(ValueError):
#         filter_fn(input_signal[::2], noise_ref[:5])
#     with pytest.raises(ValueError):
#         filter_fn(input_signal[:5], noise_ref[::2])
#     with pytest.raises(ValueError):
#         filter_fn(input_signal[::2], noise_ref[::2])
#
#     # calling np.ascontiguousarray makes it possible to use non-contiguous arrays as input
#     filter_fn(
#         np.ascontiguousarray(input_signal[::2]), np.ascontiguousarray(noise_ref[::2])
#     )
