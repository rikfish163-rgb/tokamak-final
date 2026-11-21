import numpy as np
import pytest

from experiments.risk_aligned_online.normal_reference_alignment_20261031.core import (
    fit_coordinates,
    make_transport,
    transport,
    covariance_matrix,
    ended_groups,
)


def rows(seed=3, shift=0.0, stretch=1.0):
    rng = np.random.default_rng(seed)
    result = []
    for sid, n in enumerate((31, 47, 59)):
        x = (shift + stretch * rng.normal(size=(n, 23))).astype(np.float32)
        result.append(dict(shot=sid, values=x, available=np.ones_like(x, bool)))
    return result


def test_equal_shot_statistics_not_duration_weighted():
    r = rows()
    r[0]["values"][:] = 0
    r[1]["values"][:] = 2
    r[2]["values"][:] = 4
    s = fit_coordinates(r, np.zeros(23), np.ones(23))
    assert np.allclose(s["mean"], np.mean(np.arcsinh([0.0, 2.0, 4.0])))
    assert s["shot_ids"] == [0, 1, 2]


def test_target_to_source_covariance_direction():
    source = np.array([[2.0, 0.5], [0.5, 1.0]])
    target = np.array([[1.0, -0.4], [-0.4, 3.0]])
    a = covariance_matrix(source, target)
    assert np.allclose(a.T @ (target + np.eye(2)) @ a, source + np.eye(2), atol=1e-12)
    assert not np.allclose(a.T @ (source + np.eye(2)) @ a, target + np.eye(2))


@pytest.mark.parametrize("mode", ["diagonal", "correlation"])
def test_identical_statistics_identity_and_masks(mode):
    r = rows()
    s = fit_coordinates(r, np.zeros(23), np.ones(23))
    fit = make_transport(s, s, mode)
    x = r[0]["values"].copy()
    m = r[0]["available"].copy()
    m[:5, 1] = False
    x[:5, 1] = -10
    out = transport(dict(values=x, available=m), fit, np.zeros(23), np.ones(23))
    assert np.allclose(out["values"], x, rtol=1e-6, atol=2e-7)
    assert np.array_equal(out["available"], m)
    assert np.array_equal(out["values"][~m], x[~m])
    assert np.array_equal(r[0]["values"], x) is False


def test_exclusions_and_constants_unchanged():
    src, tar = rows(), rows(4, 3, 2)
    for r in src + tar:
        r["values"][:, 5] = 4
    a = fit_coordinates(src, np.zeros(23), np.ones(23))
    b = fit_coordinates(tar, np.zeros(23), np.ones(23))
    fit = make_transport(a, b, "correlation")
    assert not np.asarray(fit["active"])[[2, 5, 10]].any()
    signal = {k: tar[0][k] for k in ("values", "available")}
    out = transport(signal, fit, np.zeros(23), np.ones(23))
    assert np.array_equal(out["values"][:, [2, 5, 10]], signal["values"][:, [2, 5, 10]])


def test_no_future_fit_and_pointwise_application():
    s = fit_coordinates(rows(), np.zeros(23), np.ones(23))
    t = fit_coordinates(rows(4, 2), np.zeros(23), np.ones(23))
    fit = make_transport(s, t, "correlation")
    r = rows(9)[0]
    r.pop("shot")
    r["available"][4:7, 1] = False
    all_ = transport(r, fit, np.zeros(23), np.ones(23))["values"]
    for end in (1, 5, 17):
        prefix = {k: v[:end] for k, v in r.items()}
        assert np.array_equal(
            transport(prefix, fit, np.zeros(23), np.ones(23))["values"], all_[:end]
        )


def test_insufficient_shots_not_invented_as_scale():
    r = rows()
    r[1]["available"][:, 1] = False
    r[2]["available"][:, 1] = False
    a = fit_coordinates(r, np.zeros(23), np.ones(23))
    fit = make_transport(a, a, "diagonal")
    assert not fit["active"][1]
    assert fit["status"][1] == "fewer_than_three_observed_shots"


def test_rejects_labels_duplicates_nonfinite():
    r = rows()
    r[0]["label"] = 1
    with pytest.raises(ValueError):
        fit_coordinates(r, np.zeros(23), np.ones(23))
    r = rows()
    r[1]["shot"] = r[0]["shot"]
    with pytest.raises(ValueError):
        fit_coordinates(r, np.zeros(23), np.ones(23))
    r = rows()
    r[0]["values"][0, 0] = np.inf
    with pytest.raises(ValueError):
        fit_coordinates(r, np.zeros(23), np.ones(23))


def test_masked_values_cannot_affect_other_coordinates():
    a = fit_coordinates(rows(), np.zeros(23), np.ones(23))
    b = fit_coordinates(rows(4, 3), np.zeros(23), np.ones(23))
    fit = make_transport(a, b, "correlation")
    s = rows(5)[0]
    s.pop("shot")
    s["available"][:, 1] = False
    out = transport(s, fit, np.zeros(23), np.ones(23))
    s["values"][:, 1] = 1e20
    changed = transport(s, fit, np.zeros(23), np.ones(23))
    assert np.array_equal(
        out["values"][:, np.arange(23) != 1], changed["values"][:, np.arange(23) != 1]
    )


def test_ended_prefix_only_and_explicit_seals():
    r = rows()
    flags = {0: False, 1: True, 2: False}
    seals = {i: "a" * 64 for i in range(3)}
    groups = ended_groups(r, flags, order=[0, 1, 2, 3], count=3, seals=seals)
    assert [x["shot"] for x in groups["safe"]] == [0, 2]
    with pytest.raises(ValueError):
        ended_groups(r, flags, order=[1, 0, 2, 3], count=3, seals=seals)
    with pytest.raises(ValueError):
        ended_groups(r, {**flags, 3: False}, order=[0, 1, 2, 3], count=3, seals=seals)
    with pytest.raises(ValueError):
        ended_groups(r, flags, order=[0, 1, 2, 3], count=3, seals={0: "a" * 64})


def test_overflow_is_failure_not_silent_clipping():
    s = fit_coordinates(rows(), np.zeros(23), np.ones(23))
    fit = make_transport(s, s, "diagonal")
    fit["target"] = {**s, "std": [1e-100] * 23}
    r = rows(8)[0]
    r.pop("shot")
    with pytest.raises(FloatingPointError):
        transport(r, fit, np.zeros(23), np.ones(23))
