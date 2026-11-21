"""CPU controller-only folds; linearized risk-direction scores, not alarm scores."""

import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch

from .. import io23
from ..history_gradient_metric_20261102.run import read, sha, write
from ..risk_bridge_forecaster_20261102.prepare import folds
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from .prepare import TEACHER, PLAN

PREPARED = io23.ROOT / "probe_outputs/conditional_channel_context20_v1"
ARMS = ("uniform", "global", "conditional_true", "conditional_shuffled")
GROUPS = ("safe", "danger30_125", "disruptive_early")


def summarize(rows):
    result = {}
    for group in GROUPS:
        local = [r for r in rows if r["group"] == group]
        per_shot = defaultdict(list)
        for r in local:
            per_shot[r["shot"]].append(r)
        if not per_shot:
            raise ValueError("all three evaluation groups required")
        result[group] = dict(
            episodes=len(local),
            shots=len(per_shot),
            mean_delta=float(
                np.mean([np.mean([r["delta"] for r in rs]) for rs in per_shot.values()])
            ),
            mean_proxy_slope=float(
                np.mean(
                    [
                        np.mean([r["proxy_slope"] for r in rs])
                        for rs in per_shot.values()
                    ]
                )
            ),
        )
    result["balanced_proxy_slope"] = (
        result["safe"]["mean_proxy_slope"] + result["danger30_125"]["mean_proxy_slope"]
    ) / 2
    return result


def gates(summary):
    candidate = summary["conditional_true_real"]
    g = dict(
        safe_relative_improved=candidate["safe"]["mean_delta"] < -1e-9,
        danger_relative_improved=candidate["danger30_125"]["mean_delta"] < -1e-9,
        early_not_worsened=candidate["disruptive_early"]["mean_delta"] <= 1e-9,
    )
    for name in (
        "global_real",
        "conditional_shuffled_real",
        "conditional_true_reverse",
        "conditional_true_hold",
    ):
        g["better_than_" + name] = (
            candidate["balanced_proxy_slope"]
            < summary[name]["balanced_proxy_slope"] - 1e-9
        )
    g["passed"] = all(g.values())
    return g


def context_x(features, selected, arm):
    if arm in ("uniform", "global"):
        return np.empty((len(selected), 0), np.float64)
    key = "true" if arm == "conditional_true" else "shuffled"
    return np.asarray(
        [features[r["shot"], r["query"]]["context"][key] for r in selected], np.float64
    )


def main(root):
    from .core import fit, predict

    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    torch.set_num_threads(4)
    started = time.monotonic()
    previous, prepared = (
        read(TEACHER / "completed.json"),
        read(PREPARED / "completed.json"),
    )
    for directory, done in ((TEACHER, previous), (PREPARED, prepared)):
        for name, digest in done["artifact_sha256"].items():
            if sha(directory / name) != digest:
                raise ValueError("fixed preparation or teacher changed")
    p = read(PREPARED / "protocol.json")
    if p["teacher_completed_sha256"] != sha(TEACHER / "completed.json"):
        raise ValueError("teacher linkage mismatch")
    feature_rows = read(PREPARED / "features.json")
    features = {(r["shot"], r["query"]): r for r in feature_rows}
    teachers = read(TEACHER / "records.json")
    if len(features) != 100 or len(teachers) != 300:
        raise ValueError("fixed100 episodes with matched3 feedbacks required")
    reference = {(r["shot"], r["query"], r["mode"]): r for r in teachers}
    if len(reference) != 300:
        raise ValueError("duplicate teacher")
    real = [r for r in teachers if r["mode"] == "real"]
    split = folds(real, p["ids"])
    root.mkdir()
    bound = imported_code_hashes()
    bound[str(PLAN)] = sha(PLAN)
    write(
        root / "protocol.json",
        dict(
            scope="CONDITIONAL_CONTROLLER_HELDOUT_LINEAR_DIRECTION_ONLY",
            ids=p["ids"],
            folds=split,
            source_sha256=p["source_sha256"],
            teacher_completed_sha256=sha(TEACHER / "completed.json"),
            prepared_completed_sha256=sha(PREPARED / "completed.json"),
            bound_code_sha256=bound,
            arms=list(ARMS),
            steps=256,
            optimizer="Adam CPUfloat64 lr.01 fullbatch",
            suffix_raw_reads=0,
            controller_holdout_only=True,
            parent_and_C_previously_saw_all20=True,
            independent_confirmation=False,
            linearized_fixed_uniform_cap_not_finite_neural_step=True,
        ),
    )
    records = []
    for fold in split:
        training = [
            r for r in real if r["shot"] in fold["fit"] and r["group"] in GROUPS[:2]
        ]
        held = [r for r in real if r["shot"] in fold["held"]]
        if {r["shot"] for r in training} != set(fold["fit"]):
            raise ValueError("training identities drift")
        models = {}
        for arm in ARMS[1:]:
            models[arm] = fit(
                context_x(features, training, arm),
                np.asarray([r["contributions23"] for r in training]),
                [r["shot"] for r in training],
                [r["group"] for r in training],
                steps=256,
            )
            if set(models[arm]["fit_ids"]) != set(fold["fit"]):
                raise ValueError("controller fit used wrong shots")
        write(root / f"fold{fold['fold']}_models.json", models)
        for arm in ARMS:
            weights = (
                np.ones((len(held), 23))
                if arm == "uniform"
                else predict(context_x(features, held, arm), models[arm])
            )
            if (
                weights.shape != (len(held), 23)
                or not np.isfinite(weights).all()
                or not np.allclose(weights.mean(1), 1, atol=1e-12, rtol=0)
            ):
                raise ValueError("invalid controller weights")
            for index, r in enumerate(held):
                key = r["shot"], r["query"]
                for mode in ("real", "reverse", "hold"):
                    teacher = reference[*key, mode]
                    a = np.asarray(teacher["contributions23"], np.float64)
                    w = weights[index]
                    records.append(
                        dict(
                            fold=fold["fold"],
                            arm=arm,
                            mode=mode,
                            **{
                                k: r[k]
                                for k in (
                                    "shot",
                                    "query",
                                    "label",
                                    "disruptive",
                                    "actual_lead_ms",
                                    "group",
                                )
                            },
                            weights23=w.tolist(),
                            context=context_x(features, [r], arm)[0].tolist(),
                            uniform_proxy_slope=float(a.sum()),
                            proxy_slope=float(a @ w),
                            delta=float(a @ (w - 1)),
                        )
                    )
        print(
            dict(
                fold=fold["fold"],
                records=len(records),
                seconds=time.monotonic() - started,
            ),
            flush=True,
        )
    summary = {
        arm + "_" + mode: summarize(
            [r for r in records if r["arm"] == arm and r["mode"] == mode]
        )
        for arm in ARMS
        for mode in ("real", "reverse", "hold")
    }
    write(root / "records.json", records)
    write(root / "summary.json", summary)
    if any(sha(Path(path)) != digest for path, digest in bound.items()):
        raise AssertionError("frozen code changed")
    done = dict(
        status="COMPLETED_LINEAR_DIRECTION_SCREEN",
        records=len(records),
        gate=gates(summary),
        seconds=time.monotonic() - started,
        source_parameters_updated=0,
        suffix_raw_reads=0,
        neural_updates_recomputed=0,
        online_gain_measured=False,
        method_promoted=False,
        artifact_sha256={f.name: sha(f) for f in root.iterdir() if f.is_file()},
    )
    write(root / "completed.json", done)
    print(done, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
