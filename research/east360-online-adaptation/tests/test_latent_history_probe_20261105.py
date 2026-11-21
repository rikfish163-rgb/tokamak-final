import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.special import expit, logit


SCRIPT = Path(__file__).resolve().parents[1] / "latent_history_probe_20261105.py"
SPEC = importlib.util.spec_from_file_location("latent_history_probe", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def make_bank():
    rng = np.random.default_rng(27)
    ranks = [1, 2, 19, 20, 21, 37, 65, 100, 101, 150, 220, 360]
    rows = []
    for rank in ranks:
        labels = [False, True, True] + ([True] if rank == 101 else [])
        for index, positive in enumerate(labels):
            state = rng.normal(size=66).astype(np.float32)
            state[1] = np.log1p((1000 + index * 100) / 1000)
            state[2] += float(positive)
            history = rng.normal(size=128).astype(np.float32)
            history[0] += float(positive)
            native_probability = [0.8, 0.2, 0.6, 0.4][index]
            rows.append((
                10000 + rank, rank, 1000 + index * 100, positive,
                (30.001 if index == 1 else 125.0) if positive else np.nan,
                np.float32(logit(native_probability)), state, history,
            ))
    names = ("shot", "rank", "query", "positive", "lead_ms", "native_logit", "state", "history")
    dtypes = (np.int64, np.int64, np.int64, bool, np.float64, np.float32, np.float32, np.float32)
    return {name: np.asarray([row[i] for row in rows], dtype=dtypes[i])
            for i, name in enumerate(names)}


def copy_bank(bank):
    return {name: value.copy() for name, value in bank.items()}


def test_class_then_shot_then_row_weights_include_mixed_shots():
    shots = np.array([11, 11, 12, 13, 13, 13, 11])
    labels = np.array([1, 1, 1, 0, 0, 0, 0], dtype=bool)
    weights = probe.class_shot_weights(shots, labels)
    np.testing.assert_allclose(weights, [1/8, 1/8, 1/4, 1/12, 1/12, 1/12, 1/4])
    assert weights[labels].sum() == pytest.approx(0.5)
    assert weights[~labels].sum() == pytest.approx(0.5)
    with pytest.raises(ValueError, match="both classes"):
        probe.class_shot_weights(shots[:3], labels[:3])


def test_weighted_auc_handles_ties_and_equal_class_shot_weights():
    shots = np.array([1, 1, 2, 3, 4, 4])
    labels = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    scores = np.array([0.9, 0.7, 0.2, 0.7, 0.4, 0.8])
    assert probe.shot_auc(shots, labels, scores) == pytest.approx(0.375)
    assert probe.shot_auc(shots[:3], labels[:3], scores[:3]) is None


def test_hard_slices_use_query_native_probability_and_fixed_boundaries():
    threshold = probe.THRESHOLD
    probabilities = np.array([threshold - 0.251, threshold - 0.249,
                              threshold - 0.001, threshold + 0.001])
    positive = np.array([True, True, False, False])
    masks = probe.slice_masks(positive, logit(probabilities))
    np.testing.assert_array_equal(masks["hard_positive"], [True, False, False, False])
    np.testing.assert_array_equal(masks["hard_safe"], [False, False, False, True])
    np.testing.assert_array_equal(masks["hard"], [True, False, False, True])


def test_hard_slice_boundaries_are_inclusive(monkeypatch):
    probabilities = np.array([probe.THRESHOLD - probe.HARD_GAP, probe.THRESHOLD])
    monkeypatch.setattr(probe, "expit", lambda _: probabilities)
    masks = probe.slice_masks(np.array([True, False]), np.zeros(2))
    np.testing.assert_array_equal(masks["hard_positive"], [True, False])
    np.testing.assert_array_equal(masks["hard_safe"], [False, True])


def test_fit_minimizes_the_declared_weighted_logistic_objective():
    bank = make_bank()
    train = bank["rank"] <= 20
    x, y, shots = bank["state"][train, :2].astype(float), bank["positive"][train], bank["shot"][train]
    fit = probe.fit_probe(x, y, shots)
    standardized = (x - fit["mean"]) / fit["scale"]
    coefficient = np.asarray(fit["coefficient"])
    logits = standardized @ coefficient + fit["intercept"]
    weights = probe.class_shot_weights(shots, y)
    objective = weights @ (np.logaddexp(0, logits) - y * logits) + probe.L2 * (coefficient @ coefficient) / 2
    residual = weights * (expit(logits) - y)
    gradient = np.r_[standardized.T @ residual + probe.L2 * coefficient, residual.sum()]
    assert fit["objective"] == pytest.approx(objective)
    assert np.max(np.abs(gradient)) < 1e-5


def test_empty_evaluation_slices_and_one_class_auc_are_explicitly_null():
    shots = np.array([1, 1, 2])
    positive = np.ones(3, dtype=bool)
    result = probe.evaluate(shots, positive, np.full(3, 10.0), np.zeros(3))
    assert result["metrics"]["overall"]["auc"] is None
    for name in ("safe", "hard_positive", "hard_safe", "hard"):
        assert result["metrics"][name] == {"n_rows": 0, "n_shots": 0, "bce": None}


def test_splits_training_only_scaling_and_suffix_cannot_change_fit():
    bank = make_bank()
    baseline = probe.diagnose(bank, "synthetic")
    altered = copy_bank(bank)
    suffix = altered["rank"] > 100
    altered["state"][suffix] += 100
    altered["history"][suffix] -= 100
    altered["native_logit"][suffix] += 10
    altered["positive"][suffix] = ~altered["positive"][suffix]
    altered["lead_ms"][suffix] = np.where(altered["positive"][suffix], 75.0, np.nan)
    modified = probe.diagnose(altered, "synthetic")
    for budget in ("train_le_20", "train_le_100"):
        for family in ("clock_logit", "state", "state_history"):
            assert baseline["models"][budget][family]["fit"] == modified["models"][budget][family]["fit"]
    train = bank["rank"] <= 20
    weights = probe.class_shot_weights(bank["shot"][train], bank["positive"][train])
    mean = weights @ bank["state"][train, :2]
    np.testing.assert_allclose(baseline["models"]["train_le_20"]["clock_logit"]["fit"]["mean"], mean)
    assert baseline["splits"]["train_le_20"]["n_shots"] == 4
    assert baseline["splits"]["train_le_100"]["n_shots"] == 8
    assert baseline["splits"]["test_gt_100"]["n_shots"] == 4
    assert baseline["splits"]["test_gt_100"]["n_positive_shots"] == 4
    assert baseline["splits"]["test_gt_100"]["n_safe_shots"] == 4


def test_budgets_are_independent_and_query_native_logit_is_not_a_feature():
    bank = make_bank()
    baseline = probe.diagnose(bank, "synthetic")
    altered = copy_bank(bank)
    middle = (altered["rank"] > 20) & (altered["rank"] <= 100)
    altered["state"][middle] += 15
    altered["history"][middle] -= 15
    changed = probe.diagnose(altered, "synthetic")
    for family in ("clock_logit", "state", "state_history"):
        assert baseline["models"]["train_le_20"][family]["fit"] == changed["models"]["train_le_20"][family]["fit"]
        assert baseline["models"]["train_le_100"][family]["fit"] != changed["models"]["train_le_100"][family]["fit"]
    altered = copy_bank(bank)
    altered["native_logit"] += 4
    changed = probe.diagnose(altered, "synthetic")
    for budget in baseline["models"]:
        for family in baseline["models"][budget]:
            assert baseline["models"][budget][family]["fit"] == changed["models"][budget][family]["fit"]


def test_metrics_are_equal_shot_and_predictions_align_to_test_rows():
    report = probe.diagnose(make_bank(), "synthetic")
    rows = report["test_rows"]
    result = report["models"]["train_le_20"]["state_history"]
    predictions = result["predictions"]
    assert len(rows) == len(predictions) == report["splits"]["test_gt_100"]["n_rows"]
    assert all(row["rank"] > 100 for row in rows)
    for row, prediction in zip(rows, predictions):
        assert prediction["probability"] == pytest.approx(expit(prediction["logit"]))
        assert prediction["bce"] == pytest.approx(np.logaddexp(0, prediction["logit"]) - row["positive"] * prediction["logit"])
    per_shot = [np.mean([prediction["bce"] for row, prediction in zip(rows, predictions) if row["shot"] == shot])
                for shot in sorted({row["shot"] for row in rows})]
    assert result["metrics"]["overall"]["bce"] == pytest.approx(np.mean(per_shot))
    assert [item["bce"] for item in result["per_shot"]] == pytest.approx(per_shot)
    for key in ("positive", "safe", "hard_positive", "hard_safe", "hard"):
        assert result["metrics"][key]["n_rows"] > 0
        assert result["metrics"][key]["n_shots"] > 0


@pytest.mark.parametrize("case", ["wrong_width", "wrong_dtype", "nonfinite", "bad_lead", "safe_lead", "bad_rank", "rank_collision", "no_test", "one_class"])
def test_malformed_or_unfittable_banks_are_rejected(case):
    bank = make_bank()
    if case == "wrong_width":
        bank["history"] = bank["history"][:, :127]
    elif case == "wrong_dtype":
        bank["positive"] = bank["positive"].astype(np.int64)
    elif case == "nonfinite":
        bank["state"][0, 2] = np.nan
    elif case == "bad_lead":
        bank["lead_ms"][np.flatnonzero(bank["positive"])[0]] = 30
    elif case == "safe_lead":
        bank["lead_ms"][np.flatnonzero(~bank["positive"])[0]] = 126
    elif case == "bad_rank":
        bank["rank"][0] = 0
    elif case == "rank_collision":
        bank["rank"][bank["rank"] == 2] = 1
    elif case == "no_test":
        bank = {key: value[bank["rank"] <= 100] for key, value in bank.items()}
    else:
        train = bank["rank"] <= 20
        bank["positive"][train] = False
        bank["lead_ms"][train] = np.nan
    with pytest.raises(ValueError):
        probe.diagnose(bank, "synthetic")


def test_cli_hashes_the_exact_synthetic_bank_and_will_not_overwrite(tmp_path):
    bank_path = tmp_path / "synthetic.npz"
    output_path = tmp_path / "report.json"
    np.savez(bank_path, **make_bank())
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    command = [sys.executable, str(SCRIPT), "--bank", str(bank_path), "--output", str(output_path)]
    subprocess.run(command, check=True, env=env, capture_output=True, text=True)
    report = json.loads(output_path.read_text())
    assert report["bank_sha256"] == hashlib.sha256(bank_path.read_bytes()).hexdigest()
    assert "producer" in report["settings"]["selection_contract"]
    assert "predictions[i]" in report["settings"]["prediction_alignment"]
    assert report["splits"]["test_gt_100"]["n_rows"] == len(report["test_rows"])
    original = output_path.read_bytes()
    repeated = subprocess.run(command, env=env, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert output_path.read_bytes() == original
