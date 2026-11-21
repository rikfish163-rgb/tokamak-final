"""CPU saved-record audit: no trial/model/torch imports and no HDF access.

verify(root) is read-only; main(root) writes audit_cpu.json exclusively AFTER
verification. BCE is independently checked with double NumPy softplus (absolute
tolerance 2e-6). Shot-macro summaries/gates use those validated serialized deltas,
preserving the experiment's float32 zero/sign semantics. This is NOT neural
replay, proof of recorded hidden RMS, raw-input rehashing, or EAST performance.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from numbers import Real
from pathlib import Path

import numpy as np

from .data import SOURCE_SHA, load_records

TASKS = ("native", "no_MNM", "no_MNM_CIII")
HEADS = ("shared", "true", "shuffled")
ARMS = tuple(a + "_" + m for a in HEADS for m in ("real", "reverse", "hold"))
GROUPS = ("safe", "danger", "early")
TOL = 2e-6
REQUIRED = {"protocol.json", "val_records.json", "train_records_selected.json",
            "schedule.json", "training.json", "validation_records.json",
            "validation_summary.json", "preflight.json", "census.json",
            "tables.pt", "code_snapshot.zip"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def number(value):
    require(isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(value),
            "finite numeric value required")
    return float(value)


def check_numeric(r):
    """Validate exported losses/q/receipt and return largest double BCE error."""
    before, after, label = [number(r[k]) for k in ("before_logit", "after_logit", "label")]
    require(label in (0., 1.), "binary label required")
    logits = np.array([before, after], dtype=np.float64)
    losses = np.logaddexp(0., logits) - label * logits
    expected = (*losses, losses[1] - losses[0])
    error = max(abs(number(r[k]) - v) for k, v in
                zip(("before_bce", "after_bce", "delta_bce"), expected))
    require(error <= TOL, "double softplus BCE/delta mismatch")
    q = np.asarray(r["context"], dtype=np.float64)
    require(q.shape == (6,) and np.isfinite(q).all() and (q >= 0).all()
            and (q <= 1).all() and abs(q.sum() - 1.) <= 1e-6, "invalid context q")
    require(0 <= number(r["query_rms"]) <= .25 + 1e-6, "query RMS cap exceeded")
    gradient, step = number(r["gradient_norm"]), number(r["step_norm"])
    require(gradient >= 0 and step >= 0 and type(r["accepted"]) is bool,
            "invalid receipt norms/acceptance")
    old, proposed = r["loss_before"], r["loss_proposal"]
    if old is None or proposed is None:
        require(old is None and proposed is None and not r["accepted"]
                and gradient == step == 0., "invalid no-target placeholder")
    else:
        require(r["accepted"] == (number(proposed) < number(old)),
                "strict proposal acceptance mismatch")
    if not r["accepted"]:
        require(step == 0. and before == after, "rejected update changed state/readout")
    return float(error)


def summarize_rows(rows):
    """Independently average records WITHIN shot, then equally ACROSS shots."""
    result = {}
    for group in GROUPS:
        local = [r for r in rows if r["group"] == group]
        shots = defaultdict(list)
        for r in local:
            shots[r["shot"]].append(r["delta_bce"])
        result[group] = dict(n_shots=len(shots), n_points=len(local),
            mean_delta_bce=float(np.mean([np.mean(shots[s]) for s in sorted(shots)])) if shots else None,
            helpful_points=sum(r["delta_bce"] < 0 for r in local),
            harmful_points=sum(r["delta_bce"] > 0 for r in local))
    require(all(result[g]["n_shots"] for g in GROUPS[:2]), "safe and danger validation required")
    result["balanced"] = sum(result[g]["mean_delta_bce"] for g in GROUPS[:2]) / 2
    return result


def compute_gate(summary):
    selected = summary["no_MNM_CIII"]["true_real"]
    gate = {g + "_improved": selected[g]["mean_delta_bce"] < -1e-8 for g in GROUPS[:2]}
    for control in ("shared_real", "shuffled_real", "true_reverse", "true_hold"):
        gate["better_than_" + control] = (
            selected["balanced"] < summary["no_MNM_CIII"][control]["balanced"] - 1e-8)
    gate["native_protected"] = all(
        summary["native"]["true_real"][g]["mean_delta_bce"] <= 1e-6 for g in GROUPS[:2])
    gate["passed"] = all(gate.values())
    return gate


def compare(actual, expected):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), "summary keys mismatch")
        for key in expected:
            compare(actual[key], expected[key])
    elif isinstance(expected, float):
        require(abs(number(actual) - expected) <= TOL, "summary numeric mismatch")
    else:
        require(type(actual) is type(expected) and actual == expected, "summary count/value mismatch")


def identity(r):
    return r["shot"], r["origin"]


def record_matches(r, reference):
    require(all(r.get(k) == v for k, v in reference.items()), "archived record/role mismatch")
    require(all(type(r.get(k)) is int and -1 <= r[k] <= 5
                for k in ("origin_family", "query_family")), "invalid nominal family")


def check_training(selected, schedule, history, reference, ids):
    lookup = {identity(r): r for r in reference}
    require(selected and len({identity(r) for r in selected}) == len(selected), "duplicate/empty training set")
    for r in selected:
        require(r["shot"] in ids and r["group"] in GROUPS[:2]
                and identity(r) in lookup, "non-TRAIN/early training record")
        record_matches(r, lookup[identity(r)])
    require(len(schedule) == len(history) == 512, "training step budget mismatch")
    used = set()
    for i, (s, h) in enumerate(zip(schedule, history)):
        require(s["step"] == i + 1 and s["task"] == TASKS[i % 3], "training task/step cycle mismatch")
        take = s["indices"]
        require(len(take) == 2 and all(type(v) is int and 0 <= v < len(selected) for v in take)
                and [selected[v]["group"] for v in take] == ["safe", "danger"],
                "training requires one safe and one danger")
        require(all(h.get(k) == s[k] for k in ("step", "task", "indices")), "unpaired training history")
        for key in ("objectives", "gradient_norms", "accepted"):
            require(set(h[key]) == set(HEADS), "training arm coverage mismatch")
            for value in h[key].values():
                require((type(value) is int and 0 <= value <= 2) if key == "accepted"
                        else (number(value) >= 0 if key == "gradient_norms" else np.isfinite(number(value))),
                        "invalid training statistic")
        used.update(take)
    require(used == set(range(len(selected))), "unused selected training records")


def verify(root):
    """Read-only result check; malformed/tampered/incomplete artifacts raise ValueError."""
    root = Path(root)
    done = read(root / "completed.json")
    require(done.get("status") == "COMPLETED" and done.get("source_unchanged") is True,
            "incomplete or changed-source experiment")
    hashes = done["artifact_sha256"]
    require(REQUIRED <= hashes.keys(), "missing required artifact hashes")
    for name, expected in hashes.items():
        require(Path(name).name == name and name not in ("completed.json", "audit_cpu.json"),
                "invalid artifact name")
        require(sha(root / name) == expected, "artifact hash mismatch: " + name)
    require(done["schedule_sha256"] == hashes["schedule.json"], "schedule hash mismatch")
    p = read(root / "protocol.json")
    fixed = dict(scope="SOURCE_ONLY_OFFLINE_EVENT_MASKED_TASK_TEACHING", source_sha256=SOURCE_SHA,
        qualified_source_parameters_updated=0, source_slow_zero=True, east_reads=0, test_signals_read=0,
        train_role="train", val_role="val", steps=512, batch_size=2, seed=131, lr=.001,
        optimizer="Adam", gradient_clip=1., selection="fixed last iterate",
        source_labels="max(25ms,AdvancedTime), strict lead>10ms", arms=list(HEADS),
        tasks=dict(native=[], no_MNM=["MNM"], no_MNM_CIII=["MNM", "CIII"]),
        local_episode_not_continuous_stream=True)
    require(all(p.get(k) == v for k, v in fixed.items()), "fixed source protocol mismatch")
    require(all(done.get(k) == v for k, v in dict(source_parameter_updates=0, east_reads=0,
        test_signals_read=0, full_east_stream_tested=False).items()), "completion scope mismatch")
    train, train_ids, tp = load_records("train")
    val, val_ids, vp = load_records("val")
    require(not set(train_ids) & set(val_ids) and len(val) == 946
            and all(r["shot"] in val_ids for r in val), "canonical VAL population mismatch")
    require(p["train_provenance"] == dict(tp, raw_inputs_rehashed=True)
            and p["val_provenance"] == dict(vp, raw_inputs_rehashed=True), "archive provenance mismatch")
    archived = read(root / "val_records.json")
    require(len(archived) == 946, "exactly 946 validation episodes required")
    for r, ref in zip(archived, val):
        record_matches(r, ref)
    selected = read(root / "train_records_selected.json")
    check_training(selected, read(root / "schedule.json"), read(root / "training.json"), train, train_ids)
    require(read(root / "preflight.json").get("status") == "PASS", "preflight did not pass")
    require(done.get("source_train_episodes_selected") == len(selected)
            and done.get("validation_episodes") == 946, "completion episode counts mismatch")
    lookup = {identity(r): r for r in archived}
    require(len(lookup) == 946, "duplicate archived validation episode")
    rows = read(root / "validation_records.json")
    require(len(rows) == done.get("validation_records") == 946 * 3 * 9, "validation record count mismatch")
    seen, buckets, contexts, baselines = set(), defaultdict(list), {}, {}
    max_error, accepted, no_target = 0., 0, 0
    for r in rows:
        key, task, arm = identity(r), r["task"], r["arm"]
        require(key in lookup and task in TASKS and arm in ARMS, "unknown validation episode/task/arm")
        record_matches(r, lookup[key])
        unique = (key, task, arm)
        require(unique not in seen, "duplicate validation combination")
        seen.add(unique)
        max_error = max(max_error, check_numeric(r))
        base_key, qkey = (key, task), (key, task, arm.split("_")[0])
        baseline = (r["before_logit"], r["before_bce"])
        require(baselines.setdefault(base_key, baseline) == baseline, "before baseline mismatch")
        require(contexts.setdefault(qkey, r["context"]) == r["context"], "context differs across feedback controls")
        buckets[task, arm].append(r)
        accepted += int(r["accepted"])
        no_target += int(r["loss_before"] is None)
    for key, task, head in contexts:
        if head == "shared":
            require(contexts[key, task, head] == contexts[key, task, "true"], "shared/true context mismatch")
    summary = {task: {arm: summarize_rows(buckets[task, arm]) for arm in ARMS} for task in TASKS}
    compare(read(root / "validation_summary.json"), summary)
    gate = compute_gate(summary)
    compare(done["gate"], gate)
    return dict(status="PASS", completed_sha256=sha(root / "completed.json"), artifact_sha256=hashes,
        validation_episodes=946, validation_records=len(rows), tasks=3, arms=9, training_steps=512,
        accepted=accepted, rejected=len(rows) - accepted, no_target=no_target,
        max_bce_absolute_error=max_error, tolerance=TOL, summary=summary, gate=gate,
        raw_inputs_read=0, neural_replayed=False, actual_hidden_rms_verified=False,
        source_weights_rehashed=False, full_east_stream_tested=False)


def main(root):
    result = verify(root)
    write(Path(root) / "audit_cpu.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
