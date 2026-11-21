"""Post-run verification of neural gain, timing, and actually issued forecasts.

This is never imported by the producer and never changes a running policy.
An incomplete flow is rejected rather than summarized as a complete result.
"""

import argparse
from pathlib import Path

import numpy as np

from .history_gradient_metric_20261102.run import read, sha, write
from .hdf5_score_recheck_20261103 import score_curve
from .joint_lora_20261104.run import ARMS, paired
from .source_anticipation_20261104.static_stream import timing_decomposition
from . import io23


def forecast_error(prediction, target, mask, released):
    prediction, target, mask, released = map(
        np.asarray, (prediction, target, mask, released)
    )
    if (
        prediction.ndim != 2
        or prediction.shape != target.shape
        or mask.shape != target.shape
        or mask.dtype != bool
        or released.shape != (len(mask),)
    ):
        raise ValueError(
            "matching two-dimensional forecast/target/mask and maturity required"
        )
    if (
        (mask[released < 0]).any()
        or not np.isfinite(prediction).all()
        or not np.isfinite(target[mask]).all()
    ):
        raise ValueError(
            "pending forecasts cannot have observations; supported values must be finite"
        )
    supported = mask.sum(1)
    n = int(mask.sum())
    p = prediction[mask].astype(np.float64)
    y = target[mask].astype(np.float64)
    return dict(
        observed_elements=n,
        pending=int((released < 0).sum()),
        unsupported_matured=int(((released >= 0) & (supported == 0)).sum()),
        mse=float(np.mean((p - y) ** 2)) if n else None,
        hold_mse=float(np.mean(y**2)) if n else None,
    )


def capped_metric(rows):
    nd = sum(r["disruptive"] for r in rows)
    ns = len(rows) - nd
    if not nd or not ns:
        raise ValueError("both hard-outcome classes required")
    tp = sum(
        r["disruptive"] and r["lead_ms"] is not None and 30 < r["lead_ms"] <= 3000
        for r in rows
    )
    fp = sum(not r["disruptive"] and r["alarm"] for r in rows)
    return dict(tp=tp, fp=fp, tpr=tp / nd, fpr=fp / ns, J=tp / nd - fp / ns)


def analyze(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    done = read(run / "completed.json")
    if done["status"] != "COMPLETED_JOINT_LORA_PAIRED_DEVELOPMENT":
        raise ValueError("a terminal complete paired run is required")
    for filename, digest in done["artifact_sha256"].items():
        if sha(run / filename) != digest:
            raise ValueError("sealed artifact hash changed")
    protocol, per, summary = (
        read(run / name) for name in ("protocol.json", "per_shot.json", "summary.json")
    )
    ids = [r["shot"] for r in protocol["order"]]
    if len(ids) != 360 or len(set(ids)) != 360 or set(per) != set(ARMS):
        raise ValueError("all seven arms on the entire locked360 required")
    inventory = io23.ROOT / "probe_outputs/hdf5_outcome_authority_20261104_v4"
    inventory_done = read(inventory / "completed.json")
    assert inventory_done["status"] == "PASS"
    for name, digest in inventory_done["artifact_sha256"].items():
        assert sha(inventory / name) == digest
    truth = {r["shot"]: r for r in read(inventory / "hdf5_metadata.json")["EAST"]}
    forecast_rows = {arm: [] for arm in ARMS[1:]}
    updates = {
        arm: dict(
            events=0,
            updated=0,
            nonzero_gradient_events=0,
            max_displacement=0.0,
            max_issued_recomputed_gap=0.0,
        )
        for arm in ARMS[1:]
    }
    early_locked_fp = {"intra_vs_static": [], "both_vs_inter": []}
    for index, sid in enumerate(ids, 1):
        t = truth[sid]
        prefix = f"{index:04d}_{sid}"
        with np.load(run / "shots" / (prefix + ".npz"), allow_pickle=False) as curves:
            expected = t["start"] + np.arange(t["samples"]) / t["rate"]
            assert np.array_equal(curves["times"], expected)
            for arm in ARMS:
                row = per[arm][index - 1]
                scored = dict(
                    shot=sid,
                    disruptive=t["disruptive"],
                    down_s=t["down"],
                    **score_curve(
                        expected,
                        curves[arm],
                        t["disruptive"],
                        t["down"],
                        protocol["threshold"],
                    ),
                )
                assert row == scored, (sid, arm)
            if index <= 20:
                assert all(
                    np.array_equal(curves[arm], curves["native"]) for arm in ARMS
                )
            else:
                for candidate, reference in (
                    ("intra", "static"),
                    ("hold", "static"),
                    ("reverse", "static"),
                    ("both", "inter"),
                ):
                    assert np.array_equal(
                        curves[candidate][:135], curves[reference][:135]
                    ), (sid, candidate)
                    if candidate in ("intra", "both") and not t["disruptive"]:
                        hit = np.flatnonzero(curves[reference] >= protocol["threshold"])
                        if len(hit) and hit[0] < 135:
                            early_locked_fp[candidate + "_vs_" + reference].append(sid)
        if index <= 20:
            continue
        history = read(run / "shots" / (prefix + "_updates.json"))
        with np.load(
            run / "shots" / (prefix + "_forecasts.npz"), allow_pickle=False
        ) as f:
            canonical = {
                k: f["static__" + k]
                for k in ("origin", "released_at", "target", "mask")
            }
            for arm in ARMS[1:]:
                data = {
                    k: f[arm + "__" + k]
                    for k in ("origin", "released_at", "prediction", "target", "mask")
                }
                for key in canonical:
                    assert np.array_equal(data[key], canonical[key], equal_nan=True), (
                        sid,
                        arm,
                        key,
                    )
                mature = data["released_at"] >= 0
                assert np.array_equal(
                    data["released_at"][mature], data["origin"][mature] + 8
                )
                assert len(history[arm]) == int(mature.sum())
                stats = forecast_error(
                    data["prediction"],
                    data["target"],
                    data["mask"],
                    data["released_at"],
                )
                forecast_rows[arm].append(dict(shot=sid, **stats))
                for h in history[arm]:
                    assert h["arrival"] == h["origin"] + 8
                    assert h["parameter_displacement"] <= 0.5
                    u = updates[arm]
                    u["events"] += 1
                    u["updated"] += h["updated"]
                    u["nonzero_gradient_events"] += (
                        h["gradient_norm"] is not None and h["gradient_norm"] > 0
                    )
                    u["max_displacement"] = max(
                        u["max_displacement"], h["parameter_displacement"]
                    )
                    if h["issued_vs_recomputed_max_abs"] is not None:
                        u["max_issued_recomputed_gap"] = max(
                            u["max_issued_recomputed_gap"],
                            h["issued_vs_recomputed_max_abs"],
                        )
                if arm in ("static", "inter"):
                    assert updates[arm]["updated"] == 0
    re_summary = {
        arm: {
            name: timing_decomposition(rows, deadline_ms=30)
            for name, rows in (
                ("full360", per[arm]),
                ("startup20", per[arm][:20]),
                ("suffix340", per[arm][20:]),
            )
        }
        for arm in ARMS
    }
    assert summary == re_summary
    forecast_summary = {}
    for arm, rows in forecast_rows.items():
        supported = [r for r in rows if r["mse"] is not None]
        denom = sum(r["observed_elements"] for r in supported)
        forecast_summary[arm] = dict(
            supported_shots=len(supported),
            pending=sum(r["pending"] for r in rows),
            unsupported_matured=sum(r["unsupported_matured"] for r in rows),
            shot_equal_mse=float(np.mean([r["mse"] for r in supported]))
            if supported
            else None,
            element_weighted_mse=sum(
                r["mse"] * r["observed_elements"] for r in supported
            )
            / denom
            if denom
            else None,
            shot_equal_hold_mse=float(np.mean([r["hold_mse"] for r in supported]))
            if supported
            else None,
        )
    output.mkdir(parents=True)
    report = dict(
        status="VERIFIED_SAVED_COMPLETE_FLOW_NOT_INDEPENDENT_VALIDATION",
        run_completion_sha256=sha(run / "completed.json"),
        inventory_completion_sha256=sha(inventory / "completed.json"),
        full360={arm: summary[arm]["full360"] for arm in ARMS},
        capped30_3000_sensitivity={
            arm: capped_metric(rows) for arm, rows in per.items()
        },
        paired=paired(per),
        forecast_summary=forecast_summary,
        updates=updates,
        false_alarms_already_sealed_before_first_intra_update=early_locked_fp,
        analysis_sha256=sha(Path(__file__)),
        training_performed=False,
        independent_validation=False,
    )
    write(output / "analysis.json", report)
    write(output / "forecast_per_shot.json", forecast_rows)
    print(dict(status=report["status"], full360=report["full360"]), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    analyze(a.run, a.output)
