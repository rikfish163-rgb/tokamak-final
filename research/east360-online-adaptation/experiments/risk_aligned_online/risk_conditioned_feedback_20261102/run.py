"""Ended-shot LOSO test of a causal condition versus an identical constant gate."""

import argparse
from pathlib import Path
import time
import zipfile

import numpy as np
import scipy

from ..channel_feedback_20261102.metric import evaluate
from ..history_gradient_metric_20261102.core import group_summary
from ..history_gradient_metric_20261102.run import ROOT, read, sha, write
from .core import effective_weights, fit

SOURCE = ROOT / "probe_outputs/channel_feedback_gradients20_v1"
SOURCE_SHA = "91b5793fae75961dd07bc62cac67866426c1d3e04a096eca1b0f8282a80d73d6"
CONTEXT = ROOT / "probe_outputs/risk_conditioned_context20_v1"
PLAN = ROOT / "RISK_CONDITIONED_FEEDBACK_PLAN_20261102.md"


def leave_one_shot_out(gradients, risk, rows, observations, condition, reverse):
    g, r, obs, u, rev = (
        np.asarray(v) for v in (gradients, risk, observations, condition, reverse)
    )
    if len(rows) != len(g) or g.shape != rev.shape or u.shape != (len(rows),):
        raise ValueError("matching arrays required")
    ids = np.array([row["shot"] for row in rows])
    folds, actual, control, constant = [], [], [], []
    for sid in sorted(set(ids.tolist())):
        train, held = np.flatnonzero(ids != sid), np.flatnonzero(ids == sid)
        training = [rows[i] for i in train]
        fitted = fit(g[train], r[train], training, obs[train], u[train])
        fixed = fit(g[train], r[train], training, obs[train], np.full(len(train), 0.5))
        assert sid not in fitted["training_ids"] and sid not in fixed["training_ids"]
        unsupported = sorted(
            set(np.flatnonzero(obs[held].sum(0) > 0)) - set(fitted["active_indices"])
        )
        folds.append(
            dict(
                held_shot=sid,
                fit=fitted,
                constant_fit=fixed,
                held_queries=[rows[i]["query"] for i in held],
                unsupported_held_channels=[int(c) for c in unsupported],
            )
        )
        for i in held:
            w = effective_weights(fitted, u[i])
            common = dict(condition=float(u[i]), weights=w.tolist())
            actual.append(dict(**evaluate(g[i], r[i], rows[i], w), **common))
            control.append(dict(**evaluate(rev[i], r[i], rows[i], w), **common))
            constant.append(
                dict(
                    **evaluate(g[i], r[i], rows[i], effective_weights(fixed, 0.5)),
                    condition=0.5,
                )
            )
    return folds, actual, control, constant


def main(root):
    if root.exists():
        raise FileExistsError(root)
    started = time.monotonic()
    assert sha(SOURCE / "completed.json") == SOURCE_SHA
    parent, done = read(SOURCE / "protocol.json"), read(SOURCE / "completed.json")
    assert sha(SOURCE / "gradients.npz") == done["gradients_sha256"]
    assert sha(SOURCE / "records.json") == done["records_sha256"]
    context, context_done = (
        read(CONTEXT / "protocol.json"),
        read(CONTEXT / "completed.json"),
    )
    assert context_done["status"] == "COMPLETED_CONDITION_COLLECTION"
    assert sha(CONTEXT / "conditions.json") == context_done["conditions_sha256"]
    assert context["records_sha256"] == done["records_sha256"]
    assert context["source_completed_sha256"] == SOURCE_SHA
    assert context["plan_sha256"] == sha(PLAN)
    rows, conditions = read(SOURCE / "records.json"), read(CONTEXT / "conditions.json")
    assert len(rows) == len(conditions) == 100
    assert [(r["shot"], r["query"]) for r in rows] == [
        (r["shot"], r["query"]) for r in conditions
    ]
    assert all(r["maturity"] + 1 == r["query"] for r in conditions)
    u = np.array([r["condition"] for r in conditions])
    assert np.isfinite(u).all() and ((u >= 0) & (u <= 1)).all()
    with np.load(SOURCE / "gradients.npz", allow_pickle=False) as stored:
        g, r, obs, reverse = [
            stored[k] for k in ("real", "task", "observations", "reverse")
        ]
    assert g.shape == reverse.shape == (100, 23, 256)
    assert r.shape == (100, 256) and obs.shape == (100, 23)
    files = set(Path(__file__).parent.glob("*.py")) | {PLAN}
    files |= {
        ROOT / "experiments/risk_aligned_online" / folder / file
        for folder, file in [
            ("channel_feedback_20261102", "metric.py"),
            ("history_gradient_metric_20261102", "core.py"),
            ("history_gradient_metric_20261102", "run.py"),
        ]
    }
    root.mkdir(parents=True)
    protocol = dict(
        scope="CONDITIONAL_CONTROLLER_LOSO_NOT_NEURAL_UPDATE",
        source_collection=str(SOURCE),
        source_completed_sha256=SOURCE_SHA,
        source_gradients_sha256=done["gradients_sha256"],
        context_collection=str(CONTEXT),
        context_completed_sha256=sha(CONTEXT / "completed.json"),
        conditions_sha256=context_done["conditions_sha256"],
        ids=parent["ids"],
        source_model_sha256=parent["source_model_sha256"],
        cold_state_sha256=parent["cold_state_sha256"],
        input_map_sha256=parent["input_map_sha256"],
        condition="fixed_parent_zero_fast_risk_at_query_minus_1",
        feature_names=parent["feature_names"],
        regularizer=1.0,
        weight_bounds=[0.0, 4.0],
        solver=dict(name="SLSQP", ftol=1e-12, maxiter=500),
        projected_gradient_limit=1e-5,
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
    folds, actual, control, constant = leave_one_shot_out(g, r, rows, obs, u, reverse)
    assert len(folds) == 20 and len(actual) == len(control) == len(constant) == 100
    for name, value in [
        ("folds", folds),
        ("held_real", actual),
        ("held_reverse", control),
        ("held_constant", constant),
        ("full_fit", fit(g, r, rows, obs, u)),
    ]:
        write(root / (name + ".json"), value)
    summary, control_summary, fixed_summary = map(
        group_summary, (actual, control, constant)
    )
    deltas = {
        name: summary["groups"][name]["after_mean_cosine"]
        - fixed_summary["groups"][name]["after_mean_cosine"]
        for name in ("class1", "safe")
    }
    conditioning_gate = all(v >= 0 for v in deltas.values()) and any(
        v > 1e-12 for v in deltas.values()
    )
    result = dict(
        status="COMPLETED_RISK_CONDITIONED_GEOMETRIC_SCREEN",
        real=summary,
        reverse_same_weights=control_summary,
        constant_condition=fixed_summary,
        real_minus_constant_after=deltas,
        conditioning_gate=conditioning_gate,
        condition_quantiles=np.quantile(u, [0, 0.25, 0.5, 0.75, 1]).tolist(),
        ready_for_neural_probe=bool(summary["gates"]["passed"] and conditioning_gate),
        method_promoted=False,
        online_gain_measured=False,
        independent_future_confirmation=False,
        full_fit_used_for_validation=False,
        scientific_goal_achieved=False,
        artifact_sha256={
            name: sha(root / name)
            for name in (
                "folds.json",
                "held_real.json",
                "held_reverse.json",
                "held_constant.json",
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
