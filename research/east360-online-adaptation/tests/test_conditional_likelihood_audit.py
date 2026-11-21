"""CPU-only regression tests for the independent cold20 artifact audit."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "experiments/risk_aligned_online/conditional_likelihood_20261029/audit_v2.py"
spec = importlib.util.spec_from_file_location("conditional_likelihood_independent_audit", MODULE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_future_groups_and_terminal_boundaries_are_pointwise():
    times = np.array([[0.874, 0.875, 0.969, 0.970, 1.000, 1.001]])
    labels, groups = audit.future_groups(times, {"disruptive": True, "down": 1.0})
    np.testing.assert_array_equal(labels, [[0, 1, 1, 1, 1, 0]])
    np.testing.assert_array_equal(groups["early_normal"], [[1, 0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(groups["danger_timely"], [[0, 1, 1, 0, 0, 0]])
    np.testing.assert_array_equal(groups["danger_late"], [[0, 0, 0, 1, 1, 0]])
    safe_y, safe_groups = audit.future_groups(times, {"disruptive": False, "down": None})
    assert not safe_y.any()
    assert safe_groups["safe"].all()


def test_original_endpoint_rule_applies_to_origin_and_arrival_not_future():
    outcome = {"disruptive": True, "down": 1.0}
    origins = np.array([0.874, 0.875, 0.950, 0.975])
    arrivals = np.array([0.899, 0.900, 0.975, 1.000])
    eligible = audit.origin_timely_eligible(origins, arrivals, outcome)
    np.testing.assert_array_equal(eligible, [False, True, False, False])
    labels, groups = audit.future_groups(np.array([[0.901]]), outcome)
    assert labels[0, 0] and groups["danger_timely"][0, 0]
    assert not eligible[0]  # The later point is timely; its origin is 126 ms away.


def test_binary_loss_is_stable_at_extreme_logits_and_validates_shapes():
    loss = audit.binary_cross_entropy(np.array([-1000.0, 1000.0, 0.0]), np.array([0, 0, 1]))
    np.testing.assert_allclose(loss, [0.0, 1000.0, np.log(2.0)])
    with pytest.raises(AssertionError, match="shape"):
        audit.binary_cross_entropy(np.zeros((1, 2)), np.zeros(2))


def test_v1_safe_shape_failure_is_reproduced_without_original_evaluator():
    # The first held safe shot has 141 episodes and 25 future logits each;
    # the v1 safe-label branch returned only a length-141 vector.
    with pytest.raises(ValueError, match="broadcast"):
        _ = np.zeros(141) * np.zeros((141, 25))


def test_summary_is_equal_shot_weighted_and_gate_checks_control():
    rows = [
        {"arm": arm, "group": group, "shot": shot, "windows": windows, "timepoints": windows,
         "delta_bce": value}
        for arm, group, shot, windows, value in [
            ("real", "safe", 1, 100, -0.1), ("real", "safe", 2, 1, -0.3),
            ("misplaced", "safe", 1, 100, -0.05), ("misplaced", "safe", 2, 1, -0.15),
            ("real", "danger_timely", 1, 100, -0.1), ("real", "danger_timely", 2, 1, -0.3),
            ("misplaced", "danger_timely", 1, 100, -0.05), ("misplaced", "danger_timely", 2, 1, -0.15),
        ]
    ]
    groups = audit.aggregate_rows(rows)
    assert groups["real"]["safe"]["mean_delta_bce"] == pytest.approx(-0.2)
    assert groups["real"]["safe"]["windows"] == 101
    assert audit.direction_gate(groups) == (True, [])
    groups["misplaced"]["safe"]["mean_delta_bce"] = -0.21
    assert audit.direction_gate(groups) == (False, ["safe"])


def test_fold_validation_requires_disjoint_fit_and_single_held_coverage():
    ids = list(range(20))
    folds = [{"fold": f, "fit": [s for s in ids if s % 5 != f],
              "held": [s for s in ids if s % 5 == f]} for f in range(5)]
    audit.validate_folds(folds, ids)
    bad = [{k: (v.copy() if isinstance(v, list) else v) for k, v in f.items()} for f in folds]
    bad[0]["fit"][0] = bad[0]["held"][0]
    with pytest.raises(AssertionError, match="overlap|duplicate"):
        audit.validate_folds(bad, ids)
    bad = [{k: (v.copy() if isinstance(v, list) else v) for k, v in f.items()} for f in folds]
    bad[1]["held"][0] = bad[0]["held"][0]
    with pytest.raises(AssertionError):
        audit.validate_folds(bad, ids)


def test_identical_control_rejects_any_change_to_prediction_or_feedback():
    sealed = {"identical_logs": np.array([[3.0, 3.0]]), "identical_counts": np.array([1]),
              "real_counts": np.array([1])}
    neural = {"baseline_logits": np.array([[2.0, -1.0]]),
              "identical_logits": np.array([[2.0, -1.0]]),
              "identical_state_norm": np.array([0.0]),
              "identical_evidence_gradient": np.array([0.0]),
              "identical_changed": np.array([False])}
    audit.validate_identical_control(sealed, neural)
    neural["identical_logits"] = np.array([[2.0, -1.0001]])
    with pytest.raises(AssertionError, match="identical"):
        audit.validate_identical_control(sealed, neural)


def test_norm_checks_count_nonzero_and_reject_out_of_bound_or_bad_flags():
    neural = {}
    for arm in ("real", "misplaced", "identical"):
        neural[arm + "_state_norm"] = np.array([0.0, 0.04])
        neural[arm + "_changed"] = np.array([False, True])
    counts, maxima = audit.validate_norms(neural)
    assert counts == {"real": 1, "misplaced": 1, "identical": 1}
    assert maxima["real"] == 0.04
    neural["real_state_norm"][1] = 0.051
    with pytest.raises(AssertionError, match="norm"):
        audit.validate_norms(neural)
    neural["real_state_norm"][1] = 0.04
    neural["real_changed"][1] = False
    with pytest.raises(AssertionError, match="changed"):
        audit.validate_norms(neural)


def test_episode_alignment_requires_maturity_before_every_future_time():
    sealed = {"ids": np.array([1, 1]), "ends": np.array([127, 152]),
              "origin_time": np.array([0.127, 0.152]), "arrival_time": np.array([0.152, 0.177])}
    neural = {"ids": np.array([1]), "ends": np.array([127]),
              "future_times": np.arange(153, 178, dtype=np.float64)[None, :] / 1000.0}
    audit.validate_episode_alignment(sealed, neural)
    neural["future_times"][0, 0] = 0.151
    with pytest.raises(AssertionError, match="maturity|future"):
        audit.validate_episode_alignment(sealed, neural)


def test_actual_archives_recompute_published_neural_claims():
    report = audit.audit(ROOT / "probe_outputs/conditional_likelihood_cold20_v1",
                         ROOT / "probe_outputs/conditional_likelihood_cold20_v2")
    assert report["arrays"]["all_v1_v2_equal"]
    assert report["folds"]["unique_held"] == 20
    assert report["counts"]["nonzero_state_norm"] == {"real": 2707, "misplaced": 2707, "identical": 0}
    real = report["groups"]["real"]
    assert (real["danger_timely"]["shots"], real["danger_timely"]["worsened"]) == (17, 17)
    assert (real["safe"]["shots"], real["safe"]["improved"]) == (3, 3)
    assert report["gate"]["eligible"] is False
    assert report["gate"]["failed_groups"] == ["danger_timely"]
    assert report["comparisons"]["published_neural_matches"]
    assert report["origin_boundary"]["future_timely_timepoints"] == 1615
    boundary = report["origin_boundary"]
    assert boundary["future_timely_points_origin_over_125_ms"] == 638
    assert boundary["future_timely_episodes_origin_over_125_ms"] == 34
    assert boundary["future_timely_points_origin_eligible"] == 977
    assert boundary["restricted_groups"]["real"]["danger_timely"]["worsened"] == 17
    assert boundary["restricted_gate"]["eligible"] is False
    assert report["counts"]["origin_logits_exact"] == 984
    assert report["counts"]["future_logit_changes"] == {"real": 67675, "misplaced": 67675, "identical": 0}
    assert report["bounds"]["max_origin_logit_abs_difference"] == pytest.approx(4.76837158203125e-6)
