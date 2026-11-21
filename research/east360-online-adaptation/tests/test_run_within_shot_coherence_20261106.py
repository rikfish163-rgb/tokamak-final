import numpy as np
import pytest

from run_within_shot_coherence_20261106 import ARMS, CASES, REGIONS, gate, score_again


def row(shot, *, good=True, primary=True):
    cases = {}
    for case in CASES:
        if case in ("mir", "sxr", "xuv"):
            v = dict.fromkeys(ARMS, 1.0)
            v.update(coherent=0.7 if good else 1.0, oracle=0.2)
        else:
            v = dict.fromkeys(ARMS, 0.0)
            v["single"] = 0.01
        cases[case] = {k: v.copy() for k in REGIONS}
    return dict(rank=21 + shot if primary else shot, shot=shot, cases=cases)


def test_gate_requires_all_primary_controls_and_preservation():
    good = [row(i) for i in range(1, 4)]
    assert gate(good)["passed"]
    for r in good:
        for c in ("mir", "sxr", "xuv"):
            r["cases"][c]["nonanchor"]["scalar"] = 0.6
    assert not gate(good)["passed"]
    good = [row(i) for i in range(1, 4)]
    for r in good:
        r["cases"]["common"]["mature"]["coherent"] = 0.1
    assert not gate(good)["passed"]


def test_gate_cannot_use_an_absent_case_or_a_nonprimary_shot():
    rows = [row(1, good=False), row(2, primary=False)]
    assert not gate(rows)["passed"]
    rows = [row(1)]
    rows[0]["cases"]["sxr"] = None
    assert not gate(rows)["passed"]


def test_recount_keeps_original_and_corrupted_references_separate():
    t = np.arange(640)
    clean = np.zeros(640)
    curves = dict.fromkeys(ARMS, np.ones(640))
    curves["coherent"] = np.full(640, 0.25)
    single = score_again(t, clean, curves, "mir")
    common = score_again(t, clean, curves, "common")
    assert single["nonanchor"]["coherent"] == pytest.approx(0.25)
    assert common["nonanchor"]["coherent"] == pytest.approx(0.75)
    assert single["counts"] == dict(mature=192, nonanchor=186, pre=256, recovery=96)
