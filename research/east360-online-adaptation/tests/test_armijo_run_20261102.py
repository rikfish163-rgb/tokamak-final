"""CPU contracts for the complete-stream, matched Armijo comparison."""

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.armijo_feedback_20261102 import run


def scores(tp=126, fp=32):
    result = {
        a: {"full360": dict(tp=125, fp=35, n_disrupt=180, n_safe=180)} for a in run.ARMS
    }
    result["meta_real"]["full360"].update(tp=tp, fp=fp)
    return result


def test_operating_gain_is_not_matched_mechanism_separation():
    data = scores()
    data["meta_hold"]["full360"].update(tp=126, fp=32)
    result = run.score_gates(data)
    assert result["operational_gate"] and not result["control_separation_gate"]
    assert result["control_comparisons"]["meta_hold"] is False


def test_fpr_tradeoff_allowed_but_integer_tie_not_called_gain():
    assert run.score_gates(scores(127, 36))["operational_gate"]
    assert not run.score_gates(scores(126, 36))["operational_gate"]
    assert not run.score_gates(scores(128, 37))["operational_gate"]
    assert not run.score_gates(scores(124, 30))["operational_gate"]


def test_heads_are_matched_modes_not_old_fixed_step_recipe():
    meta, forecast = torch.ones(25, 23), torch.zeros(25, 23)
    heads = run.build_heads({"meta": meta, "forecast": forecast})
    assert set(heads) == set(run.FRESH) and len(heads) == 6
    for arm, (table, mode) in heads.items():
        prefix, suffix = arm.split("_")
        assert mode == suffix and table is {"meta": meta, "forecast": forecast}[prefix]
    with pytest.raises(ValueError):
        run.build_heads({"meta": meta, "forecast": torch.zeros(24, 23)})


def test_archived_references_first160_and_warmup_are_exact():
    times = np.arange(200, dtype=np.float64) / 1000
    reference = dict(
        times=times,
        cold=np.full(200, 0.2, np.float32),
        full_real=np.full(200, 0.2, np.float32),
    )
    reference["full_real"][180:] = 0.3
    curves = {
        a: reference[run.REUSED[a]].copy()
        if a in run.REUSED
        else reference["cold"].copy()
        for a in run.ARMS
    }
    run.validate_curves(times, curves, reference, warmup=False)
    bad = {k: v.copy() for k, v in curves.items()}
    bad["meta_real"][159] += 0.01
    with pytest.raises(ValueError):
        run.validate_curves(times, bad, reference, warmup=False)
    with pytest.raises(ValueError):
        run.validate_curves(times, curves, reference, warmup=True)
    warmup = {a: reference["cold"].copy() for a in run.ARMS}
    run.validate_curves(times, warmup, reference, warmup=True)


def test_gate_rejects_changed_denominator():
    data = scores()
    data["cold"]["full360"]["n_safe"] = 179
    with pytest.raises(ValueError):
        run.score_gates(data)
