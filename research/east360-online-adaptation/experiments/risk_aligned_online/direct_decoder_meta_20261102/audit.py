"""CPU checks of saved direct-decoder records, not an independent neural fit."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import io23
from ..group_protected_meta_20261102.audit import compatible
from ..source_event_tasks_20261102.audit import (
    compare,
    read,
    require,
    sha,
    summarize_rows,
    write,
)

TASKS = ("native", "no_MNM", "no_MNM_CIII")
HEADS = ("forecast_only", "mean_meta", "worst_meta")
ARMS = tuple(a + "_" + m for a in HEADS for m in ("real", "reverse", "hold"))
PREVIOUS = io23.ROOT / "probe_outputs/source_event_masked_tasks_v1"


def check_numeric(r):
    z = np.asarray([r["before_logit"], r["after_logit"]], np.float64)
    y = r["label"]
    require(np.isfinite(z).all() and y in (0.0, 1.0), "invalid risk values")
    losses = np.logaddexp(0.0, z) - y * z
    expected = (*losses, losses[1] - losses[0])
    error = max(
        abs(r[k] - v)
        for k, v in zip(("before_bce", "after_bce", "delta_bce"), expected)
    )
    require(np.isfinite(error) and error <= 2e-6, "double BCE discrepancy")
    require(
        np.isfinite(r["absolute_diagnostic_before"])
        and r["absolute_diagnostic_before"] >= 0,
        "invalid absolute diagnostic error",
    )
    require(
        0 <= r["query_rms"] <= 0.25 + 1e-6
        and np.isfinite(r["gradient_norm"])
        and np.isfinite(r["step_norm"])
        and r["gradient_norm"] >= 0
        and r["step_norm"] >= 0
        and type(r["accepted"]) is bool,
        "invalid update norms",
    )
    old, new = r["loss_before"], r["loss_proposal"]
    if old is None or new is None:
        require(
            old is None
            and new is None
            and not r["accepted"]
            and r["gradient_norm"] == r["step_norm"] == 0,
            "invalid no-target receipt",
        )
    else:
        require(
            np.isfinite([old, new]).all() and r["accepted"] == (new < old),
            "invalid strict acceptance",
        )
    if not r["accepted"]:
        require(
            r["step_norm"] == 0 and r["before_logit"] == r["after_logit"],
            "rejected update changed prediction",
        )
    return error


def gate(s):
    primary, candidate = s["no_MNM_CIII"], s["no_MNM_CIII"]["worst_meta_real"]
    g = {
        name + "_improved": candidate[name]["mean_delta_bce"] is not None
        and candidate[name]["mean_delta_bce"] < -1e-8
        for name in ("safe", "danger", "compatible_danger")
    }
    g["native_protected"] = all(
        s["native"]["worst_meta_real"][name]["mean_delta_bce"] <= 1e-6
        for name in ("safe", "danger")
    )
    for name in (
        "forecast_only_real",
        "mean_meta_real",
        "worst_meta_reverse",
        "worst_meta_hold",
    ):
        g["better_than_" + name] = (
            candidate["balanced"] < primary[name]["balanced"] - 1e-8
        )
    g["passed"] = all(g.values())
    return g


def summary(rows):
    return dict(**summarize_rows(rows), compatible_danger=compatible(rows))


def check_rows(rows, refs, tasks, arms, old_rows=None):
    lookup = {(r["shot"], r["origin"]): r for r in refs}
    require(len(rows) == len(refs) * len(tasks) * len(arms), "incomplete comparisons")
    seen, buckets, baselines, physical = set(), defaultdict(list), {}, {}
    error = 0.0
    for r in rows:
        key = (r["shot"], r["origin"])
        require(
            key in lookup and all(r.get(k) == v for k, v in lookup[key].items()),
            "fixed source metadata changed",
        )
        require(r["task"] in tasks and r["arm"] in arms, "unknown comparison")
        unique = (*key, r["task"], r["arm"])
        require(unique not in seen, "duplicate comparison")
        seen.add(unique)
        error = max(error, check_numeric(r))
        base_key = (key, r["task"])
        require(
            baselines.setdefault(base_key, r["before_logit"]) == r["before_logit"],
            "baseline not paired",
        )
        # All feedback variants use the SAME issued physical forecast error on REAL targets.
        head = r["arm"].rsplit("_", 1)[0]
        physical_key = (key, r["task"], head)
        require(
            physical.setdefault(physical_key, r["absolute_diagnostic_before"])
            == r["absolute_diagnostic_before"],
            "diagnostic baseline not paired",
        )
        if old_rows is not None:
            require(
                abs(old_rows[r["task"], *key]["before_logit"] - r["before_logit"])
                <= 2e-6,
                "previous unadapted risk drift",
            )
        buckets[r["task"], r["arm"]].append(r)
    return {t: {a: summary(buckets[t, a]) for a in arms} for t in tasks}, error


def main(root):
    root = Path(root)
    done, p = read(root / "completed.json"), read(root / "protocol.json")
    require(
        done["status"] == "COMPLETED" and done["source_unchanged"],
        "incomplete source experiment",
    )
    for name, digest in done["artifact_sha256"].items():
        require(
            Path(name).name == name and sha(root / name) == digest,
            "artifact hash mismatch",
        )
    require(
        sha(PREVIOUS / "completed.json") == p["previous_completed_sha256"],
        "previous completed binding mismatch",
    )
    require(
        not p["risk_bridge"]
        and p["head_shape"] == [1088, 575]
        and p["head_parameters"] == 626175
        and p["arms"] == list(HEADS)
        and p["steps"] == 512
        and p["lr"] == 1e-4
        and p["absolute_maintenance_weight"] == 0.20
        and p["risk_scale"] == 0.1
        and p["source_parameters_updated"]
        == p["east_reads"]
        == p["test_signals_read"]
        == 0,
        "direct source protocol mismatch",
    )
    require(
        sha(root / "schedule.json")
        == sha(PREVIOUS / "schedule.json")
        == p["schedule_sha256"],
        "schedule not paired",
    )
    train, val = read(root / "train_records.json"), read(root / "val_records.json")
    require(
        train == read(PREVIOUS / "train_records_selected.json")
        and val == read(PREVIOUS / "val_records.json"),
        "episodes changed",
    )
    roles = io23.source_roles()
    require(
        len(train) == 435
        and len(val) == 946
        and all(r["role"] == "train" and r["shot"] in roles["train"] for r in train)
        and all(r["role"] == "val" and r["shot"] in roles["val"] for r in val),
        "source roles changed",
    )
    schedule, history = read(root / "schedule.json"), read(root / "training.json")
    require(len(schedule) == len(history) == 512, "fit budget mismatch")
    for step, (s, h) in enumerate(zip(schedule, history), 1):
        require(
            s["step"] == step
            and s["task"] == TASKS[(step - 1) % 3]
            and [train[i]["group"] for i in s["indices"]] == ["safe", "danger"]
            and all(h[k] == s[k] for k in ("step", "task", "indices")),
            "fit pair mismatch",
        )
        require(set(h["arms"]) == set(HEADS), "training arms incomplete")
        for arm, item in h["arms"].items():
            d = np.asarray(item["group_risk_changes"], np.float64)
            require(
                d.shape == (2,)
                and np.isfinite(d).all()
                and 0 <= item["accepted"] <= 2
                and item["absolute_diagnostic_loss"] >= 0
                and item["anchor"] >= 0,
                "invalid fit statistic",
            )
            expected = 0.2 * item["absolute_diagnostic_loss"] + 1e-4 * item["anchor"]
            if arm != "forecast_only":
                expected += (d.mean() if arm == "mean_meta" else d.max()) / 0.1
            require(
                abs(item["objective"] - expected) <= 2e-5,
                "outer objective recomputation mismatch",
            )
    old = {
        (r["task"], r["shot"], r["origin"]): r
        for r in read(PREVIOUS / "validation_records.json")
        if r["arm"] == "true_real"
    }
    rows = read(root / "validation_records.json")
    calculated, error = check_rows(rows, val, TASKS, ARMS, old)
    compare(read(root / "validation_summary.json"), calculated)
    tr = read(root / "training_diagnostic_records.json")
    train_calculated, train_error = check_rows(
        tr, train, ("no_MNM_CIII",), tuple(a + "_real" for a in HEADS)
    )
    compare(read(root / "training_diagnostic_summary.json"), train_calculated)
    computed_gate = gate(calculated)
    compare(done["gate"], computed_gate)
    require(
        len(rows) == done["validation_rows"] == 25542
        and done["validation_episodes"] == 946
        and not done["full_east_stream_tested"],
        "completion scope mismatch",
    )
    result = dict(
        status="PASS",
        completed_sha256=sha(root / "completed.json"),
        val_rows=len(rows),
        train_rows=len(tr),
        max_double_bce_error=max(error, train_error),
        gate=computed_gate,
        raw_inputs_read=0,
        neural_replayed=False,
        training_repeated=False,
        summary=calculated,
        train_summary=train_calculated,
    )
    write(root / "audit_cpu.json", result)
    print({k: v for k, v in result.items() if k not in ("summary", "train_summary")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
