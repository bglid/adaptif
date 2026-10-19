import pytest
from adaptif import LMSFilter, NLMSFilter, RLSFilter


# TODO: update kwargs once default args are added (+ in other tests too)
@pytest.mark.parametrize(
    ["filter_class", "kwargs"],
    [
        (LMSFilter, {"mu": 0.1, "window_size": 16}),
        (NLMSFilter, {"mu": 0.01, "eps": 1e-8, "window_size": 16}),
        (
            RLSFilter,
            {
                "forgetting_factor": 0.999,
                "p_init_scale": 0.01,
                "window_size": 16,
            },
        ),
    ],
)
def test_filter_bindings(filter_class, kwargs):
    filter = filter_class(**kwargs)
    assert hasattr(filter, "window_size")
    assert hasattr(filter, "weights")
    assert hasattr(filter, "load_weights")
    assert hasattr(filter, "adapt")
    assert hasattr(filter, "filter")
    assert hasattr(filter, "block_size")
