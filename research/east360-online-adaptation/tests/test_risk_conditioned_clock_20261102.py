import numpy as np
import pytest

from experiments.risk_aligned_online.risk_conditioned_feedback_20261102.collect import (
    maturity_index,
)


def test_condition_is_observed_before_the_affected_query():
    assert maturity_index(160) == 159
    assert maturity_index(np.int64(200)) == 199


@pytest.mark.parametrize("query", [True, 159.0, 0, 159])
def test_invalid_or_unsupported_query_fails(query):
    with pytest.raises(ValueError):
        maturity_index(query)
