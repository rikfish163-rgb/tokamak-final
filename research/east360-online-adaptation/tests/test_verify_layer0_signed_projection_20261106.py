import numpy as np
import pytest

from verify_layer0_signed_projection_20261106 import FIRST, audit_anchors


def example():
    n = FIRST + 2
    a = {
        k: np.full(n, 0.5, dtype=np.float32)
        for k in ("static", "intra", "signed", "blind_signed")
    }
    a.update(
        anchor_supported=np.array([False]),
        anchor_positive=np.array([True]),
        anchor_zero=np.array([False]),
        anchor_native_logit=np.array([0.0]),
        anchor_initial_logit=np.array([0.5]),
        anchor_projected_logit=np.array([0.0]),
        anchor_norms=np.array([[0.5, 0.4, 0.4, 0.4, 0.4]]),
        anchor_later_up=np.array([True]),
    )
    a["intra"][FIRST] = 1 / (1 + np.exp(-0.5))
    a["signed"][FIRST + 1] += 0.01
    return a


def test_anchor_can_satisfy_constraint_while_later_endpoint_rises():
    result, error = audit_anchors(example())
    assert result["guarded_positive"] == result["later_upshift"] == 1
    assert result["guarded_zero"] == 0
    assert error < 1e-7


@pytest.mark.parametrize(
    "key", ["anchor_later_up", "anchor_projected_logit", "anchor_norms"]
)
def test_corrupt_or_infeasible_receipt_fails(key):
    a = example()
    a[key].flat[-1] = 0.1 if a[key].dtype.kind != "b" else False
    with pytest.raises(ValueError):
        audit_anchors(a)
