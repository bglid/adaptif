import pytest
from adaptif import BlockLMSFilter, LMSFilter, NLMSFilter, RLSFilter


# TODO: update kwargs once default args are added (+ in other tests too)
@pytest.mark.parametrize(
    ["filter_class", "kwargs"],
    [
        (LMSFilter, {"mu": 1.0, "window_size": 1024}),
        (NLMSFilter, {"mu": 1.0, "eps": 1e-8, "window_size": 1024}),
        (
            RLSFilter,
            {"forgetting_factor": 0.5, "p_init_scale": 1.0, "window_size": 1024},
        ),
    ],
)
def test_filter_bindings(filter_class, kwargs):
    filter = filter_class(**kwargs)
    assert hasattr(filter_class, "from_weights")
    assert hasattr(filter, "window_size")
    assert hasattr(filter, "weights")
    assert hasattr(filter, "adapt")
    assert hasattr(filter, "filter")
    assert not hasattr(filter, "block_size")


@pytest.mark.parametrize(
    ["filter_class", "kwargs"],
    [
        (BlockLMSFilter, {"mu": 1.0, "window_size": 1024, "block_size": 1024}),
    ],
)
def test_block_filter_bindings(filter_class, kwargs):
    filter = filter_class(**kwargs)
    assert hasattr(filter_class, "from_weights")
    assert hasattr(filter, "window_size")
    assert hasattr(filter, "block_size")
    assert hasattr(filter, "weights")
    assert hasattr(filter, "adapt")
    assert hasattr(filter, "filter")
