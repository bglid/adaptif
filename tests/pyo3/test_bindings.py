import pytest
from adaptif import BlockLMSFilter, LMSFilter, NLMSFilter


# TODO: update kwargs once default args are added (+ in other tests too)
@pytest.mark.parametrize(
    ["filter_class", "kwargs"],
    [
        (LMSFilter, {"mu": 1.0, "window_size": 1024}),
        (NLMSFilter, {"mu": 1.0, "eps": 1e-8, "window_size": 1024}),
    ],
)
def test_filter_bindings(filter_class, kwargs):
    filter = filter_class(**kwargs)
    assert hasattr(filter, "window_size")
    assert hasattr(filter, "adapt")
    assert hasattr(filter, "filter")
    assert not hasattr(filter, "block_size")


def test_block_filter_bindings():
    filter = BlockLMSFilter(1.0, 1024, 1024)
    assert hasattr(filter, "window_size")
    assert hasattr(filter, "block_size")
    assert hasattr(filter, "adapt")
    assert hasattr(filter, "filter")
