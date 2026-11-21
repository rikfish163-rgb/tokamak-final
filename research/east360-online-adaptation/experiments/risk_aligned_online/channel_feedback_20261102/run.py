"""Conditional channel-weight LOSO screen; no neural or alarm update."""

import argparse
from pathlib import Path
import time
import zipfile

import numpy as np
import scipy

from ..history_gradient_metric_20261102.core import group_summary
from ..history_gradient_metric_20261102.run import ROOT, read, sha, write
from .metric import evaluate, fit

SOURCE = ROOT / "probe_outputs/channel_feedback_gradients20_v1"
SOURCE_SHA = "91b5793fae75961dd07bc62cac67866426c1d3e04a096eca1b0f8282a80d73d6"
PLAN = ROOT / "CHANNEL_FEEDBACK_PLAN_20261102.md"


def leave_one_shot_out(gradients, risk, rows, observations, reverse):
    g, r, obs, rev = (np.asarray(v) for v in (gradients, risk, observations, reverse))
    if len(rows) != len(g) or rev.shape != g.shape:
        raise ValueError("matching training and control arrays required")
    ids = np.asarray([row["shot"] for row in rows])
    folds, actual, control = [], [], []
    for sid in sorted(set(ids.tolist())):
        train, held = np.flatnonzero(ids != sid), np.flatnonzero(ids == sid)
        fitted = fit(g[train], r[train], [rows[i] for i in train], obs[train])
        assert sid not in fitted["training_ids"]
        active = fitted["active_indices"]
        unsupported = sorted(set(np.flatnonzero(obs[held].sum(0) > 0)) - set(active))
        folds.append(
            dict(
                held_shot=sid,
                fit=fitted,
                held_queries=[rows[i]["query"] for i in held],
                unsupported_held_channels=[int(c) for c in unsupported],
            )
        )
        for i in held:
            actual.append(evaluate(g[i], r[i], rows[i], fitted["weights"]))
            # No independently optimized reverse-control rule.
            control.append(evaluate(rev[i], r[i], rows[i], fitted["weights"]))
    return folds, actual, control


def main(root):
    if root.exists():
        raise FileExistsError(root)
    started = time.monotonic()
    assert sha(SOURCE / "completed.json") == SOURCE_SHA
    done, parent = read(SOURCE / "completed.json"), read(SOURCE / "protocol.json")
    assert sha(SOURCE / "gradients.npz") == done["gradients_sha256"]
    assert sha(SOURCE / "records.json") == done["records_sha256"]
    rows = read(SOURCE / "records.json")
    ids = parent["ids"]
    assert len(rows) == 100 and len(ids) == len(set(ids)) == 20
    assert {row["shot"] for row in rows} == set(ids)
    with np.load(SOURCE / "gradients.npz", allow_pickle=False) as stored:
        g, rev, r, obs = [
            stored[k] for k in ("real", "reverse", "task", "observations")
        ]
    assert g.shape == rev.shape == (100, 23, 256)
    assert r.shape == (100, 256) and obs.shape == (100, 23)
    assert np.isfinite(rev).all()
    old = Path(parent["source_probe"])
    assert sha(old / "completed.json") == parent["source_completed_sha256"]
    assert sha(old / "samples.json") == parent["samples_sha256"]
    old_rows = {(row["shot"], row["query"]): row for row in read(old / "samples.json")}
    max_difference = dict(real=0.0, reverse=0.0)
    for i, row in enumerate(rows):
        previous = old_rows[(row["shot"], row["query"])]
        assert all(row[k] == previous[k] for k in ("label", "disruptive"))
        np.testing.assert_array_equal(
            r[i], np.asarray(previous["task_gradient256"], np.float32)
        )
        for name, values in (("real", g), ("reverse", rev)):
            total = values[i].astype(np.float64).sum(0)
            reference = previous["results"]["difference_" + name]["receipt"][
                "gradient256"
            ]
            np.testing.assert_allclose(total, reference, atol=2e-7, rtol=2e-4)
            max_difference[name] = max(
                max_difference[name], float(np.max(np.abs(total - reference)))
            )
    files = set(Path(__file__).parent.glob("*.py")) | {PLAN}
    files |= {
        ROOT / "experiments/risk_aligned_online/history_gradient_metric_20261102" / f
        for f in ("core.py", "run.py")
    }
    root.mkdir(parents=True)
    protocol = dict(
        scope="CONDITIONAL_CHANNEL_LOSO_GEOMETRY_NOT_ONLINE",
        source_collection=str(SOURCE),
        source_completed_sha256=SOURCE_SHA,
        gradients_sha256=done["gradients_sha256"],
        records_sha256=done["records_sha256"],
        source_model_sha256=parent["source_model_sha256"],
        cold_state_sha256=parent["cold_state_sha256"],
        input_map_sha256=parent["input_map_sha256"],
        ids=ids,
        feature_names=parent["feature_names"],
        n_samples=100,
        n_channels=23,
        dim=256,
        regularizer=1.0,
        weight_bounds=[0.0, 4.0],
        active_weight_mean=1.0,
        active_channels_from_training_observations_only=True,
        reverse_control_reuses_real_fit=True,
        solver=dict(name="SLSQP", ftol=1e-12, maxiter=500),
        projected_gradient_residual_limit=1e-5,
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        cold_parent_previously_saw_ended20=True,
        labels="unchanged_HDF5_EAST125ms",
        raw_signal_reads=0,
        raw_outcome_reads=0,
        suffix_raw_reads=0,
        plan_sha256=sha(PLAN),
        code_sha256={str(f.relative_to(ROOT)): sha(f) for f in files},
    )
    write(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(ROOT)))
    folds, held, reverse = leave_one_shot_out(g, r, rows, obs, rev)
    assert len(folds) == 20 and len(held) == len(reverse) == 100
    write(root / "folds.json", folds)
    write(root / "held_out_real.json", held)
    write(root / "held_out_reverse.json", reverse)
    write(root / "full_fit.json", fit(g, r, rows, obs))
    actual_summary, control_summary = group_summary(held), group_summary(reverse)
    result = dict(
        status="COMPLETED_CHANNEL_GEOMETRIC_SCREEN",
        real=actual_summary,
        reverse_same_weights=control_summary,
        folds=20,
        held_rows=100,
        unsupported_held_channels={
            str(f["held_shot"]): f["unsupported_held_channels"]
            for f in folds
            if f["unsupported_held_channels"]
        },
        max_channel_sum_difference=max_difference,
        full_fit_used_for_validation=False,
        ready_for_neural_probe=actual_summary["gates"]["passed"],
        method_promoted=False,
        online_gain_measured=False,
        independent_future_confirmation=False,
        scientific_goal_achieved=False,
        artifact_sha256={
            name: sha(root / name)
            for name in (
                "folds.json",
                "held_out_real.json",
                "held_out_reverse.json",
                "full_fit.json",
            )
        },
        seconds=time.monotonic() - started,
    )
    write(root / "completed.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
