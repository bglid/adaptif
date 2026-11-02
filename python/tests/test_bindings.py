import pytest
from adaptif import APAFilter, LMSFilter, NLMSFilter, RLSFilter


@pytest.mark.parametrize("filter_class", [LMSFilter, NLMSFilter, RLSFilter, APAFilter])
def test_filter_bindings(filter_class):
    filter = filter_class()
    assert hasattr(filter, "window_size")
    assert hasattr(filter, "weights")
    assert hasattr(filter, "load_weights")
    assert hasattr(filter, "adapt")
    assert hasattr(filter, "filter")
    assert hasattr(filter, "block_size")
