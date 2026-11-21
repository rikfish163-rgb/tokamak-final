import hashlib
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

import verify_patch_utility_selection_20261106 as check


def shot(native, *, sid=1, disruptive=False, down=None, blocks=None, times=None):
    native = np.asarray(native, dtype=np.float32)
    if times is None:
        times = np.arange(len(native), dtype=np.float64) * 0.001
    return SimpleNamespace(
        sid=sid, disruptive=disruptive, down=down, times=np.asarray(times),
        native=native, blocks=blocks, teacher={},
    )


def menu(native, clock, actions=2):
    curves = np.zeros((len(clock), actions, 32), dtype=np.float32)
    for b, c in enumerate(clock):
        tail = native[c + 1:c + 33]
        curves[b, :, :len(tail)] = tail
    return curves, np.ones((len(clock), actions), dtype=bool)


def test_first_alarm_uses_float64_threshold_rounding_and_hdf_tolerance():
    x = np.float32(0.1)
    threshold = np.nextafter(np.float64(x), np.inf)
    assert check.first_alarm([0.0], [x], False, None, threshold)["first"] is None
    times = [-0.020, 0.0, np.nextafter(1e-9, 0), 1.001e-9]
    eligible = check.first_alarm(times, [0, 0, 1, 1], True, 0.0, 0.5)
    assert eligible == dict(first=2, lead_ms=-0.000001, timely=False, false_alarm=False)
    assert check.first_alarm(times, [0, 0, 0, 1], True, 0.0, 0.5)["first"] is None
    assert not check.first_alarm([0.0], [1], True, 0.01000000000049, 0.5)["timely"]
    assert check.first_alarm([0.0], [1], True, 0.0100000000016, 0.5)["timely"]


def test_initial_crossing_cannot_be_overwritten_and_tail_padding_is_ignored():
    q = shot(np.zeros(73))
    q.native[3] = 1
    curves, allowed = menu(q.native, [63])
    curves[0, 1, :] = 1
    analysis = check.menu_analysis(q, [63], curves, allowed, 0.5)
    assert analysis["reachable_first"] == [3]
    assert not analysis["hard"].any()
    assert check.first_alarm(q.times, check.replay(q.native, [63], curves, allowed, [1]),
                             False, None, 0.5)["first"] == 3

    q.native[3] = 0
    curves, allowed = menu(q.native, [63])
    curves[0, 1, 9:] = 1  # outside the nine real post-release samples
    analysis = check.menu_analysis(q, [63], curves, allowed, 0.5)
    assert analysis["reachable_first"] == [73]
    assert not analysis["hard"].any()
    assert check.replay(q.native, [63], curves, allowed, [1]).max() == 0


def test_multiwrite_removal_is_reachable_without_any_positive_single_edit():
    q = shot(np.zeros(136))
    q.native[[70, 102]] = 1
    curves, allowed = menu(q.native, [63, 95])
    curves[0, 1, 6] = 0
    curves[1, 1, 6] = 0
    analysis = check.menu_analysis(q, [63, 95], curves, allowed, 0.5)
    assert analysis["baseline"]["first"] == 70
    assert not analysis["hard"].any()
    assert analysis["reachable_first"] == [70, 102, 136]
    assert analysis["oracle_no_false_alarm"]
    assert check.first_alarm(q.times, check.replay(q.native, [63, 95], curves, allowed, [1, 1]),
                             False, None, 0.5)["first"] is None


def test_menu_threshold_is_float64_and_unaffected_gaps_stop_later_choices():
    value = np.float32(0.1)
    threshold = np.nextafter(np.float64(value), np.inf)
    q = shot(np.zeros(140))
    q.native[68] = value
    curves, allowed = menu(q.native, [63, 103])
    assert check.menu_analysis(q, [63, 103], curves, allowed, threshold)["reachable_first"] == [140]
    q.native[98] = 1  # an uneditable gap before the second block
    curves, allowed = menu(q.native, [63, 103])
    curves[1, 1, 2] = 1
    assert check.menu_analysis(q, [63, 103], curves, allowed, threshold)["reachable_first"] == [98]


def test_exact_menu_and_single_edits_match_exhaustive_small_menus():
    import patch_utility_targets_20261106 as target

    rng = np.random.default_rng(125)
    for i in range(70):
        n, clock = 110, np.array([63, 95])
        native = (rng.random(n) < 0.015).astype(np.float32)
        flag = bool(i % 2)
        q = shot(native, disruptive=flag, down=0.105 if flag else None)
        curves, allowed = menu(native, clock, actions=3)
        curves[:, 1:] = (rng.random((2, 2, 32)) < 0.03)
        allowed[:, 1:] = rng.random((2, 2)) > 0.25
        got = check.menu_analysis(q, clock, curves, allowed, 0.5)
        reference = target.outcomes(native, q.times, clock, curves, allowed, tau=0.5,
                                    disruptive=flag, down=q.down, horizon=25)
        np.testing.assert_array_equal(got["hard"], reference["hard"])
        for key in ("baseline", "reachable_first", "oracle_timely", "oracle_no_false_alarm"):
            assert got[key] == reference[key]
        base = check.first_alarm(q.times, native, flag, q.down, 0.5)
        assert got["baseline"] == base
        reachable = set()
        for choice in itertools.product(*[np.flatnonzero(row) for row in allowed]):
            value = check.first_alarm(q.times, check.replay(native, clock, curves, allowed, choice),
                                      flag, q.down, 0.5)
            reachable.add(n if value["first"] is None else value["first"])
        assert got["reachable_first"] == sorted(reachable)
        for b in range(len(clock)):
            for a in range(3):
                if not allowed[b, a]:
                    assert got["hard"][b, a] == 0
                    continue
                choice = np.zeros(2, dtype=int)
                choice[b] = a
                value = check.first_alarm(q.times, check.replay(native, clock, curves, allowed, choice),
                                          flag, q.down, 0.5)
                expected = (int(value["timely"]) - int(base["timely"]) if flag else
                            int(base["false_alarm"]) - int(value["false_alarm"]))
                assert got["hard"][b, a] == expected
        assert got["oracle_timely"] == (flag and any(
            k < n and round(float((q.down - q.times[k]) * 1000), 9) > 10
            for k in reachable))
        assert got["oracle_no_false_alarm"] == (not flag and n in reachable)


def test_scalar_replay_uses_same_selected_action_and_float32_addition():
    q = shot(np.zeros(70))
    q.native[64:70] = np.float32(0.1)
    curves, allowed = menu(q.native, [63], actions=3)
    curves[0, 2, :6] = 0.9
    effect = np.zeros((1, 3, 5), np.float32)
    effect[0, 2, 2] = np.float32(0.2)
    actual = check.replay(q.native, [63], curves, allowed, [2], effect=effect)
    np.testing.assert_array_equal(actual[64:70], q.native[64:70] + effect[0, 2, 2])
    assert actual.dtype == np.float32


def test_fold_hash_is_order_invariant_and_prediction_uses_strict_margin():
    ids = [991, 4, 33, 2, 18, 7, 8]
    order = sorted(ids, key=lambda i: (hashlib.sha256(str(i).encode()).digest(), i))
    expected = {sid: rank % 5 for rank, sid in enumerate(order)}
    assert dict(zip(ids, check.deterministic_folds(ids))) == expected
    reverse = ids[::-1]
    assert dict(zip(reverse, check.deterministic_folds(reverse))) == expected
    model = dict(mean=[1.0], scale=[2.0], coefficient=[1.0], intercept=0.0)
    allowed = np.array([[True, True, True], [True, False, True]])
    p = check.model_predict(np.array([[[1], [2], [2]], [[1], [100], [3]]]), allowed, model)
    np.testing.assert_array_equal(check.choose(p, allowed, 0.5), [0, 2])
    np.testing.assert_array_equal(check.choose(p, allowed, 0.49), [1, 2])
    np.testing.assert_array_equal(check.choose(p, allowed, np.inf), [0, 0])


def test_waveform_features_reconstruct_action_interactions_and_motion():
    _, data, *_ = fit_fixture()
    data["risk"][0] = np.arange(10, dtype=np.float32) / 10
    data["effect"][0, 1, 2] = np.float32(0.25)
    data["ordered"][0, 0] = np.arange(27, dtype=np.float32)
    data["ordered"][0, 1] = -np.arange(27, dtype=np.float32)
    data["states"][0, 1, 0] = 2
    data["g"][0, 0, :, 0] = 3
    data["g"][0, 1, :, 0] = -3
    x = check.features(data, np.array([0]), "true", 0.5)[0, 1]
    risk = data["risk"][0]
    expected = np.concatenate((risk, [1], data["effect"][0, 1], [risk[7] - np.float32(0.5)],
        np.float32(0.25) * risk[[0, 7, 8, 9]], data["ordered"][0, 0],
        np.full(9, 2), np.float32(0.25) * data["ordered"][0, 0, 9:18]))
    np.testing.assert_array_equal(x, expected)
    reverse = check.features(data, np.array([0]), "reverse", 0.5)[0, 1]
    null = check.features(data, np.array([0]), "risk", 0.5)[0, 1]
    np.testing.assert_array_equal(x + reverse, 2 * null)


def fit_fixture():
    ids = [11, 12, 13, 14, 15]
    folds = check.deterministic_folds(ids)
    qs = [shot(np.zeros(96), sid=sid, blocks=slice(i, i + 1)) for i, sid in enumerate(ids)]
    count, actions = len(qs), 2
    curves = np.zeros((count, actions, 32), np.float32)
    data = dict(clock=np.full(count, 63), curves=curves,
                allowed=np.ones((count, actions), bool),
                risk=np.zeros((count, 10), np.float32),
                effect=np.zeros((count, actions, 5), np.float32),
                states=np.zeros((count, actions, 64), np.float32),
                ordered=np.zeros((count, 2, 27), np.float32),
                g=np.zeros((count, 2, 9, 64), np.float32))
    width = check.features(data, slice(0, 1), "risk", 0.5).shape[-1]
    model = dict(mean=[0.] * width, scale=[1.] * width,
                 coefficient=[0.] * width, intercept=0.1)
    prediction = np.broadcast_to(np.array([0, 0.1], np.float32), (3, count, actions)).copy()
    choices = np.tile([1, 0, 1, 0, 1, 0], (count, 1))
    summary = dict(timely=0, fp=0, recovered=[], lost=[], fp_added=[], fp_removed=[],
                   first_changed=0)
    evidence = {}
    for name in ("risk", "true", "reverse"):
        records = []
        for k in range(5):
            records.append(dict(fit_fold=k, calibrate_fold=(k + 1) % 5,
                held_shots=[sid for sid, f in zip(ids, folds) if f == k],
                calibrate_shots=[sid for sid, f in zip(ids, folds) if f == (k + 1) % 5],
                fit_shots=[sid for sid, f in zip(ids, folds) if f not in (k, (k + 1) % 5)],
                margin=None, model=model,
                calibration=[dict(margin=None if np.isinf(m) else m, eligible=True, **summary)
                             for m in check.MARGINS]))
        evidence[name] = records
    return qs, data, choices, prediction, np.arange(count), folds, evidence


def test_fit_receipts_reconstruct_all_choices_and_detect_tampering():
    qs, data, choices, prediction, owner, folds, evidence = fit_fixture()
    result = check.verify_fits(qs, data, choices, prediction, owner, folds, evidence, 0.5)
    assert result["held_choices_checked"] == 30
    assert result["calibration_rules_checked"] == 120
    choices[0, 1] = 1
    with pytest.raises(ValueError, match="choice"):
        check.verify_fits(qs, data, choices, prediction, owner, folds, evidence, 0.5)
    choices[0, 1] = 0
    evidence["risk"][0]["fit_shots"].append(evidence["risk"][0]["held_shots"][0])
    with pytest.raises(ValueError, match="fold"):
        check.verify_fits(qs, data, choices, prediction, owner, folds, evidence, 0.5)


def test_fit_receipts_reject_tampered_prediction_and_calibration():
    qs, data, choices, prediction, owner, folds, evidence = fit_fixture()
    prediction[0, 0, 1] += 0.01
    with pytest.raises(ValueError, match="utility"):
        check.verify_fits(qs, data, choices, prediction, owner, folds, evidence, 0.5)
    prediction[0, 0, 1] = 0.1
    evidence["true"][0]["calibration"][0]["fp_added"] = [999]
    with pytest.raises(ValueError, match="calibration"):
        check.verify_fits(qs, data, choices, prediction, owner, folds, evidence, 0.5)


def replay_fixture():
    qs, data, _, _, _, folds, _ = fit_fixture()
    for i, q in enumerate(qs):
        q.disruptive = i in (2, 3)
        q.down = 0.110 if q.disruptive else None
        q.native[70] = float(i in (0, 3))
        curves, _ = menu(q.native, [63])
        curves[0, 1, 6] = float(i in (1, 2))
        data["curves"][i] = curves[0]
    choices = np.tile([1, 0, 0, 0, 0, 0], (len(qs), 1))
    hard = []
    for q in qs:
        y = check.menu_analysis(q, data["clock"][q.blocks], data["curves"][q.blocks],
                                data["allowed"][q.blocks], 0.5)
        hard.extend(y.pop("hard"))
        q.teacher = y
    hard = np.asarray(hard)
    records = {}
    for col, (view, mode) in enumerate(check.ORDER):
        label = f"{view}_{mode}"
        records[label] = []
        for q in qs:
            sl = q.blocks
            z = check.replay(q.native, data["clock"][sl], data["curves"][sl],
                             data["allowed"][sl], choices[sl, col])
            records[label].append([q.sid, q.teacher["baseline"],
                check.first_alarm(q.times, z, q.disruptive, q.down, 0.5)])
    for mode in ("open", "guarded"):
        records[f"true_{mode}_scalar"] = [[q.sid, q.teacher["baseline"], q.teacher["baseline"]]
                                           for q in qs]
    scores = {name: check.summarize(rows) | {"writes": int(np.count_nonzero(
        choices[:, check.ORDER.index((name.split("_")[0], name.split("_")[1]))]))}
              for name, rows in records.items()}
    baseline = check.summarize([[q.sid, q.teacher["baseline"], q.teacher["baseline"]] for q in qs])
    result = dict(baseline=baseline, scores=scores, per_shot=records,
                  n_shots=5, disruptive=2, blocks=5, distinct_true_benefit=[], promoted=False,
                  capacity=dict(available_actions=5, positive_actions=2, negative_actions=2,
                                single_edit_timely=[13], single_edit_fp_removed=[11],
                                oracle_timely=[13], oracle_fp_removed=[11]))
    return qs, data, hard, choices, folds, result


def test_every_arm_per_shot_and_capacity_are_checked_not_just_totals():
    qs, data, hard, choices, folds, result = replay_fixture()
    evidence = check.verify_replays(qs, data, hard, choices, folds, result, 0.5)
    assert evidence["arms"]["risk_open"]["recovered"] == [13]
    assert evidence["arms"]["risk_open"]["fp_added"] == [12]
    assert evidence["arms"]["risk_open"]["fp_removed"] == [11]
    assert evidence["arms"]["risk_open"]["lost"] == [14]
    assert sum(len(f["positive_shots"]) for f in evidence["hard_support_by_fold"].values()) == 2
    result["per_shot"]["risk_open"][0][2] = dict(result["per_shot"]["risk_open"][1][2])
    with pytest.raises(ValueError, match="per-shot"):
        check.verify_replays(qs, data, hard, choices, folds, result, 0.5)


def test_capacity_and_single_edit_labels_detect_tampering():
    qs, data, hard, choices, folds, result = replay_fixture()
    result["capacity"]["oracle_fp_removed"] = []
    with pytest.raises(ValueError, match="capacity"):
        check.verify_replays(qs, data, hard, choices, folds, result, 0.5)
    result["capacity"]["oracle_fp_removed"] = [11]
    hard[0, 1] = 0
    with pytest.raises(ValueError, match="hard"):
        check.verify_replays(qs, data, hard, choices, folds, result, 0.5)


def test_fold_evidence_identifies_multiwrite_only_false_alarm_removal():
    q = shot(np.zeros(136), sid=31, blocks=slice(0, 2))
    q.native[[70, 102]] = 1
    curves, allowed = menu(q.native, [63, 95])
    curves[0, 1, 6] = curves[1, 1, 6] = 0
    y = check.menu_analysis(q, [63, 95], curves, allowed, 0.5)
    hard = y.pop("hard")
    q.teacher = y
    data = dict(clock=np.array([63, 95]), curves=curves, allowed=allowed,
                effect=np.zeros((2, 2, 5), np.float32))
    choices = np.zeros((2, 6), dtype=np.int64)
    records = [[q.sid, y["baseline"], y["baseline"]]]
    summary = check.summarize(records)
    labels = [f"{view}_{mode}" for view, mode in check.ORDER] + ["true_open_scalar", "true_guarded_scalar"]
    result = dict(per_shot={label: records for label in labels},
                  scores={label: summary | {"writes": 0} for label in labels},
                  baseline=summary, n_shots=1, disruptive=0, blocks=2,
                  capacity=dict(available_actions=2, positive_actions=0, negative_actions=0,
                      single_edit_timely=[], single_edit_fp_removed=[], oracle_timely=[], oracle_fp_removed=[31]),
                  distinct_true_benefit=[], promoted=False)
    evidence = check.verify_replays([q], data, hard, choices, np.array([2]), result, 0.5)
    fold = evidence["hard_support_by_fold"]["2"]
    assert fold["positive_shots"] == []
    assert fold["multiwrite_only_fp_removed"] == [31]
    assert fold["counts"]["multiwrite_only_fp_removed"] == 1
