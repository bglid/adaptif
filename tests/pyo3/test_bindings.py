import pytest

from adaptif import BlockLMSFilter, LMSFilter, NLMSFilter


@pytest.mark.parametrize("filter_algo", [LMSFilter, NLMSFilter])
def test_filter_bindings(filter_algo):
    filter = filter_algo(1.0, 1024)
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
