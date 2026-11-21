"""Check that masked episodes intervene on availability only as declared."""

import numpy as np
import pytest

import run_patch_utility_target_views_20261106 as views


def test_mask_is_exact_causal_nonmutating_and_does_not_reveal_unobserved():
    v = np.arange(115, dtype=np.float32).reshape(5, 23)
    a = np.ones(v.shape, dtype=bool)
    a[:, 5] = False
    v[:, 5] = 0.0
    raw = dict(values=v, available=a, times=np.arange(5), shot=42)
    for name, missing in views.VIEWS.items():
        changed = views.mask(raw, name)
        other = np.setdiff1d(np.arange(23), missing)
        assert not changed["available"][:, missing].any()
        assert not changed["values"][:, missing].any()
        assert np.array_equal(changed["values"][:, other], v[:, other])
        assert np.array_equal(changed["available"][:, other], a[:, other])
        assert np.array_equal(changed["times"], raw["times"])
        assert raw["available"][:, missing].all()
        pre = views.mask({k: (z[:3] if isinstance(z, np.ndarray) else z)
                          for k, z in raw.items()}, name)
        assert np.array_equal(pre["values"], changed["values"][:3])
        assert np.array_equal(pre["available"], changed["available"][:3])
    with pytest.raises(KeyError):
        views.mask(raw, "unregistered")
