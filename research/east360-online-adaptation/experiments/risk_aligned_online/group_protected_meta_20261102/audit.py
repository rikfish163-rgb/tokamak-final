"""Independent saved-record CPU audit; no torch/model/runner import or HDF read."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import io23
from ..source_event_tasks_20261102.audit import (
    check_numeric,
    compare,
    read,
    require,
    sha,
    summarize_rows,
    write,
)

TASKS = ("native", "no_MNM", "no_MNM_CIII")
HEADS = ("mean_risk", "worst_risk")
ARMS = tuple(a + "_" + m for a in HEADS for m in ("real", "reverse", "hold"))
PREVIOUS = io23.ROOT / "probe_outputs/source_event_masked_tasks_v1"
PREVIOUS_SHA = "ecd4825f7d228ebf005ede5d42cdc757b143b780be33c09cfd5a9c6ce6f6cbcb"


def compatible(rows):
    local = [r for r in rows if r["group"] == "danger" and 30 < r["lead_ms"] <= 125]
    per_shot = defaultdict(list)
    for r in local:
        per_shot[r["shot"]].append(r["delta_bce"])
    values = [np.mean(per_shot[sid]) for sid in sorted(per_shot)]
    return dict(
        n_shots=len(per_shot),
        n_points=len(local),
        mean_delta_bce=float(np.mean(values)) if values else None,
    )


def summary(rows):
    return dict(**summarize_rows(rows), compatible_danger=compatible(rows))


def gate(s, old):
    primary, candidate = s["no_MNM_CIII"], s["no_MNM_CIII"]["worst_risk_real"]
    result = {
        g + "_improved": candidate[g]["mean_delta_bce"] is not None
        and candidate[g]["mean_delta_bce"] < -1e-8
        for g in ("safe", "danger", "compatible_danger")
    }
    result["native_protected"] = all(
        s["native"]["worst_risk_real"][g]["mean_delta_bce"] <= 1e-6
        for g in ("safe", "danger")
    )
    for control in ("mean_risk_real", "worst_risk_reverse", "worst_risk_hold"):
        result["better_than_" + control] = (
            candidate["balanced"] < primary[control]["balanced"] - 1e-8
        )
    result["better_than_old_mean_aux"] = (
        candidate["balanced"] < old["no_MNM_CIII"]["true_real"]["balanced"] - 1e-8
    )
    result["passed"] = all(result.values())
    return result


def check_rows(rows, refs, tasks, arms, old_rows=None):
    lookup = {(r["shot"], r["origin"]): r for r in refs}
    require(len(lookup) == len(refs), "duplicate fixed episodes")
    require(len(rows) == len(refs) * len(tasks) * len(arms), "row population mismatch")
    seen, buckets, qmap, baseline = set(), defaultdict(list), {}, {}
    error = 0.0
    for r in rows:
        key = (r["shot"], r["origin"])
        require(
            key in lookup and r["task"] in tasks and r["arm"] in arms,
            "unknown comparison",
        )
        require(
            all(r.get(k) == v for k, v in lookup[key].items()),
            "fixed metadata mismatch",
        )
        unique = (key, r["task"], r["arm"])
        require(unique not in seen, "duplicate comparison")
        seen.add(unique)
        error = max(error, check_numeric(r))
        pair = (key, r["task"])
        require(qmap.setdefault(pair, r["context"]) == r["context"], "q not paired")
        require(
            baseline.setdefault(pair, r["before_logit"]) == r["before_logit"],
            "baseline not paired",
        )
        if old_rows is not None:
            ref = old_rows[r["task"], *key]
            require(
                abs(ref["before_logit"] - r["before_logit"]) <= 2e-6
                and np.allclose(ref["context"], r["context"], atol=1e-6, rtol=0),
                "previous baseline/q drift",
            )
        buckets[r["task"], r["arm"]].append(r)
    return {t: {a: summary(buckets[t, a]) for a in arms} for t in tasks}, error


def main(root):
    root = Path(root)
    done, p = read(root / "completed.json"), read(root / "protocol.json")
    require(
        done["status"] == "COMPLETED" and done["source_unchanged"],
        "incomplete source run",
    )
    for name, digest in done["artifact_sha256"].items():
        require(
            Path(name).name == name and sha(root / name) == digest,
            "artifact hash mismatch",
        )
    require(
        sha(PREVIOUS / "completed.json")
        == p["previous_completed_sha256"]
        == PREVIOUS_SHA,
        "previous source binding mismatch",
    )
    require(
        p["steps"] == 512
        and p["batch_size"] == 2
        and p["new_auxiliary_weight"] == 0.0
        and p["arms"] == list(HEADS)
        and p["modes"] == ["real", "reverse", "hold"]
        and p["source_slow_zero"]
        and p["source_parameter_updates"]
        == p["east_reads"]
        == p["test_signals_read"]
        == 0,
        "fixed source-only protocol mismatch",
    )
    require(
        sha(root / "schedule.json")
        == sha(PREVIOUS / "schedule.json")
        == p["original_schedule_sha256"],
        "actual paired schedule changed",
    )
    train, val = read(root / "train_records.json"), read(root / "val_records.json")
    require(
        train == read(PREVIOUS / "train_records_selected.json")
        and val == read(PREVIOUS / "val_records.json"),
        "fixed episodes changed",
    )
    require(
        len(val) == done["validation_episodes"] == 946
        and len(train) == done["train_episodes"] == 435,
        "episode counts changed",
    )
    roles = io23.source_roles()
    require(
        all(r["role"] == "train" and r["shot"] in roles["train"] for r in train)
        and all(r["role"] == "val" and r["shot"] in roles["val"] for r in val),
        "role mixing",
    )
    schedule, history = read(root / "schedule.json"), read(root / "training.json")
    require(len(schedule) == len(history) == 512, "fit budget changed")
    for step, (s, h) in enumerate(zip(schedule, history), 1):
        require(
            s["step"] == step
            and s["task"] == TASKS[(step - 1) % 3]
            and [train[i]["group"] for i in s["indices"]] == ["safe", "danger"]
            and all(h[k] == s[k] for k in ("step", "task", "indices")),
            "fit pairing changed",
        )
        require(set(h["arms"]) == set(HEADS), "training arms mismatch")
        for arm, item in h["arms"].items():
            delta = np.asarray(item["group_risk_changes"], np.float64)
            require(
                delta.shape == (2,)
                and np.isfinite(delta).all()
                and 0 <= item["accepted"] <= 2
                and np.isfinite(item["gradient_norm"])
                and item["gradient_norm"] >= 0,
                "invalid fit statistic",
            )
            risk = (delta.mean() if arm == "mean_risk" else delta.max()) / 0.1
            require(
                np.isfinite(item["objective"]) and item["objective"] >= risk - 2e-6,
                "objective contradicts risk change plus nonnegative regularizer",
            )
    previous_rows = {
        (r["task"], r["shot"], r["origin"]): r
        for r in read(PREVIOUS / "validation_records.json")
        if r["arm"] == "true_real"
    }
    rows = read(root / "validation_records.json")
    calculated, error = check_rows(rows, val, TASKS, ARMS, previous_rows)
    compare(read(root / "validation_summary.json"), calculated)
    train_rows = read(root / "training_diagnostic_records.json")
    train_calculated, train_error = check_rows(
        train_rows, train, ("no_MNM_CIII",), tuple(a + "_real" for a in HEADS)
    )
    compare(read(root / "training_diagnostic_summary.json"), train_calculated)
    calculated_gate = gate(calculated, read(PREVIOUS / "validation_summary.json"))
    compare(done["gate"], calculated_gate)
    require(
        done["validation_rows"] == len(rows)
        and not done["new_east_performance_measured"]
        and done["east_reads"] == done["test_signals_read"] == 0,
        "completion scope mismatch",
    )
    result = dict(
        status="PASS",
        completed_sha256=sha(root / "completed.json"),
        val_rows=len(rows),
        train_diagnostic_rows=len(train_rows),
        max_double_bce_error=max(error, train_error),
        gate=calculated_gate,
        neural_replayed=False,
        optimizer_training_repeated=False,
        regularizer_per_step_recomputed=False,
        raw_signals_read=0,
        east_reads=0,
        summary=calculated,
        train_summary=train_calculated,
    )
    write(root / "audit_cpu.json", result)
    print({k: v for k, v in result.items() if k not in ("summary", "train_summary")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
