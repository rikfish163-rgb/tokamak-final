"""Fit three matched SOURCE-only auxiliary heads; never retrain the base.

This stage deliberately does not claim validation or online alarm improvement.
"""

import argparse
from copy import deepcopy
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..dual_history_ttt_20261101.stream import targets
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..mask_separated_intra_20261103.run import state_digest
from ..native_token_probe_20261104.core import forward as native_forward
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent, PARENT_SHA256
from ..source_event_tasks_20261102.context_pair import imported_code_hashes
from ..source_event_tasks_20261102.data import load_records
from . import meta, utility
from .design import ARMS, TASKS, validate

PLAN = io23.ROOT / "SOURCE_ANTICIPATION_TRAIN_PLAN_20261104.md"
UTILITY_PLAN = io23.ROOT / "SOURCE_UPDATE_UTILITY_PLAN_20261104.md"


def teaching_objective(name):
    """Select a declared outer loss, without changing the inner update budget."""
    if name == "cosine":
        return meta.objective, PLAN
    if name == "risk_utility":
        return utility.objective, UTILITY_PLAN
    raise ValueError("teaching objective must be cosine or risk_utility")


def validate_design_content(records, schedule, targets_by_arm, protocol):
    validate(records)
    if protocol.get("exposure_matched") is not True or set(targets_by_arm) != set(ARMS):
        raise ValueError(
            "matched exposure protocol and exactly three teachers required"
        )
    labels = {name: np.asarray(targets_by_arm[name], dtype=np.float32) for name in ARMS}
    if any(
        y.shape != (len(records),) or not np.isin(y, [0, 1]).all()
        for y in labels.values()
    ):
        raise ValueError("aligned finite binary teacher target arrays required")
    if not np.array_equal(
        labels["point_contract"], [r["original_source_point_label"] for r in records]
    ) or not np.array_equal(
        labels["anticipation125"], [r["future_disruption_125ms"] for r in records]
    ):
        raise ValueError(
            "hard-derived original and auxiliary targets cannot be overwritten"
        )
    counts = np.zeros(len(records), np.int64)
    for step in schedule:
        idx = step["indices"]
        if (
            step["task"] not in TASKS
            or len(idx) != 4
            or len(set(idx)) != 4
            or any(type(i) is not int or not 0 <= i < len(records) for i in idx)
        ):
            raise ValueError("same four legal coordinates and task per step required")
        np.add.at(counts, idx, 1)
    for sid in {r["shot"] for r in records}:
        used = np.array([r["shot"] == sid for r in records])
        actual = labels["anticipation125"][used]
        control = labels["time_shuffled125"][used]
        if (
            actual.sum() != control.sum()
            or counts[used] @ actual != counts[used] @ control
        ):
            raise ValueError(
                "control must preserve corpus and actual per-shot training prevalence"
            )
    return labels


def load_windows(records, schedule, device, *, role="train"):
    if role not in ("train", "val"):
        raise ValueError("only source TRAIN or VAL signal windows are permitted")
    selected = sorted({i for step in schedule for i in step["indices"]})
    _, allowed, provenance = load_records(role)
    ids = sorted({records[i]["shot"] for i in selected})
    if not set(ids) <= set(allowed) or any(
        records[i]["role"] != role for i in selected
    ):
        raise ValueError("requested source role signal identities only")
    location = {old: new for new, old in enumerate(selected)}
    shape = (len(selected), 128, 23)
    arrays = {
        k: np.empty(shape, dtype=np.float32 if k.startswith("x") else bool)
        for k in ("x0", "a0", "xq", "aq")
    }
    arrays["truth"] = np.empty((len(selected), 25, 23), np.float32)
    arrays["mask"] = np.empty((len(selected), 25, 23), bool)
    hashes = {}
    reader = io23.read_signals

    def checked(device_name, sid, *args, **kwargs):
        if device_name != "J-TEXT" or sid not in ids:
            raise ValueError(
                "only scheduled signals in the declared source role allowed"
            )
        return reader(device_name, sid, *args, **kwargs)

    with (
        patch.object(io23, "read_signals", side_effect=checked),
        patch.object(
            io23,
            "read_outcome",
            side_effect=RuntimeError(
                "outcomes supplied by already verified source task packet"
            ),
        ),
    ):
        for n, sid in enumerate(ids):
            path = io23.path_for("J-TEXT", sid)
            stat = path.stat()
            digest = sha(path)
            if digest != provenance["input_hashes"][str(sid)]:
                raise ValueError("source raw file differs from locked episode archive")
            raw = apply_input_contract(simple_data.read_shot("J-TEXT", sid))
            signal = {k: raw[k] for k in ("values", "available")}
            for old in [i for i in selected if records[i]["shot"] == sid]:
                r = records[old]
                j = location[old]
                o, q = r["origin"], r["query"]
                if q >= len(raw["times"]) or raw["times"][q] != r["hdf5_query_time"]:
                    raise ValueError(
                        "source query clock differs from verified HDF metadata"
                    )
                arrays["x0"][j], arrays["a0"][j] = simple_data.window(
                    {k: v[: o + 1] for k, v in signal.items()}, o, 128
                )
                arrays["xq"][j], arrays["aq"][j] = simple_data.window(
                    {k: v[: q + 1] for k, v in signal.items()}, q, 128
                )
                yy, mm = targets({k: v[: o + 26] for k, v in signal.items()}, o, o + 25)
                arrays["truth"][j], arrays["mask"][j] = yy["real"][0], mm[0]
            if (path.stat().st_size, path.stat().st_mtime_ns) != (
                stat.st_size,
                stat.st_mtime_ns,
            ):
                raise RuntimeError("raw file changed during source window preparation")
            hashes[str(sid)] = digest
            if (n + 1) % 32 == 0 or n + 1 == len(ids):
                print(
                    dict(stage="source_windows", shots=n + 1, total=len(ids)),
                    flush=True,
                )
    return (
        {k: torch.as_tensor(v, device=device) for k, v in arrays.items()},
        location,
        hashes,
    )


def batch_view(pack, indices, task):
    batch = {k: v[indices] for k, v in pack.items()}
    columns = {"native": (), "no_MNM": (2,), "no_MNM_CIII": (1, 2)}[task]
    for key in ("a0", "aq", "mask"):
        batch[key] = batch[key].clone()
        for column in columns:
            batch[key][:, :, column] = False
    return batch


def main(design, output, *, objective="cosine"):
    loss_function, objective_plan = teaching_objective(objective)
    design, output = Path(design), Path(output)
    if output.exists():
        raise FileExistsError(output)
    configure()
    started = time.monotonic()
    done = read(design / "completed.json")
    if done["status"] != "COMPLETED" or (design / "superseded.json").exists():
        raise ValueError("current, completed matched source design required")
    for name, digest in done["artifact_sha256"].items():
        if sha(design / name) != digest:
            raise ValueError("source task packet artifact changed")
    rows, schedule, protocol = [
        read(design / name)
        for name in ("train_records.json", "schedule.json", "protocol.json")
    ]
    labels = validate_design_content(
        rows, schedule, read(design / "training_targets.json"), protocol
    )
    if len(schedule) != 128:
        raise ValueError("fixed128-step teaching budget required")
    base, parent = load_parent(DEFAULT_PARENT)
    base.cuda()
    pack, location, raw_hashes = load_windows(rows, schedule, base.center.device)
    source_before = state_digest(base)
    first = [location[i] for i in schedule[0]["indices"]]
    with torch.no_grad():
        before = native_forward(
            base, pack["xq"][first], pack["aq"][first], base.center.new_zeros(4, 128)
        )["logit"].clone()
    heads = {
        name: deepcopy(base.forecast_head).eval().requires_grad_(True) for name in ARMS
    }
    initial = {
        k: v.detach().clone() for k, v in base.forecast_head.state_dict().items()
    }
    optimizers = {
        name: torch.optim.Adam(head.parameters(), lr=1e-3)
        for name, head in heads.items()
    }
    code = imported_code_hashes()
    output.mkdir(parents=True)
    plans = tuple(dict.fromkeys((PLAN, objective_plan)))
    bound = {
        str(p): sha(p) for p in (*plans, DEFAULT_PARENT, design / "completed.json")
    }
    write(
        output / "protocol.json",
        dict(
            scope="SOURCE_ONLY_MATCHED_AUXILIARY_HEAD_FIT_NOT_BASE_RETRAINING_OR_JTEXT_ONLINE",
            source_sha256=PARENT_SHA256,
            design=str(design.resolve()),
            input_sha256=bound,
            code_sha256=code,
            arms=list(ARMS),
            teaching_objective=objective,
            threshold=float(parent["threshold"]),
            steps=128,
            batch_size=4,
            optimizer="Adam",
            lr=1e-3,
            weight_decay=0,
            clip_grad_norm=1.0,
            shared_schedule=schedule,
            selected_coordinate_count=len(location),
            raw_source_hashes=raw_hashes,
            label_fields="original_source_point_label / future_disruption_125ms / within-shot frequency-matched time control",
            source_parameter_updates=0,
            qualified_base_unchanged=True,
            test_signal_reads=0,
            east_signal_reads=0,
            val_signal_reads=0,
            fixed_final_iterate=True,
            validation_required=True,
        ),
    )
    with zipfile.ZipFile(output / "code.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in code:
            archive.write(path, str(Path(path).relative_to(io23.ROOT)))
        for plan in plans:
            archive.write(plan, plan.name)
    history = []
    for step in schedule:
        idx = step["indices"]
        batch = batch_view(pack, [location[i] for i in idx], step["task"])
        record = dict(step=step["step"], task=step["task"], source_indices=idx, arms={})
        for name, head in heads.items():
            optimizers[name].zero_grad(set_to_none=True)
            yy = base.center.new_tensor(labels[name][idx])
            objective_args = (
                {"threshold": float(parent["threshold"])}
                if objective == "risk_utility"
                else {}
            )
            loss, info = loss_function(
                base,
                head,
                batch["x0"],
                batch["a0"],
                batch["xq"],
                batch["aq"],
                batch["truth"],
                batch["mask"],
                yy,
                initial_head_state=initial,
                **objective_args,
            )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                head.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizers[name].step()
            record["arms"][name] = dict(
                objective=float(loss.detach()),
                alignment=float(info["alignment"].detach()),
                proxy=float(info["absolute_proxy"].detach()),
                gradient_norm=float(norm),
                observed_rows=info["observed_rows"],
                zero_proxy_rows=info["zero_proxy_rows"],
                zero_desired_rows=info["zero_desired_rows"],
                labels=yy.tolist(),
            )
            if objective == "risk_utility":
                record["arms"][name].update(
                    risk_utility=float(info["risk_utility"].detach()),
                    risk_delta_mean=float(info["perrowrisk_delta"].detach().mean()),
                    risk_utility_rows=info["risk_utility_rows"],
                )
        history.append(record)
        if step["step"] % 16 == 0:
            print(
                dict(
                    stage="fit",
                    step=step["step"],
                    seconds=time.monotonic() - started,
                    arms=record["arms"],
                ),
                flush=True,
            )
    for name, head in heads.items():
        optimizers[name].zero_grad(set_to_none=True)
        head.requires_grad_(False)
    if state_digest(base) != source_before or any(
        p.grad is not None for p in base.parameters()
    ):
        raise AssertionError("source classifier/native head changed")
    with torch.no_grad():
        after = native_forward(
            base, pack["xq"][first], pack["aq"][first], base.center.new_zeros(4, 128)
        )["logit"]
    torch.testing.assert_close(after, before, atol=0, rtol=0)
    torch.save(
        dict(
            heads={
                name: {k: v.detach().cpu() for k, v in head.state_dict().items()}
                for name, head in heads.items()
            },
            source_sha256=PARENT_SHA256,
            source_label="unchanged max(25ms,AdvancedTime)",
            trained_not_validated=True,
            teaching_objective=objective,
        ),
        output / "heads.pt",
    )
    write(output / "training.json", history)
    if any(sha(Path(p)) != h for p, h in {**bound, **code}.items()):
        raise AssertionError("bound source/design/code changed during fit")
    result = dict(
        status="TRAINED_NOT_VALIDATED",
        seconds=time.monotonic() - started,
        steps_per_head=128,
        teaching_objective=objective,
        source_and_native_head_unchanged=True,
        static_query_check_exact=True,
        source_signal_shots=len(raw_hashes),
        val_signal_reads=0,
        test_signal_reads=0,
        east_signal_reads=0,
        operational_gain_measured=False,
        artifact_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
    )
    write(output / "completed.json", result)
    print(
        {
            k: result[k]
            for k in ("status", "seconds", "steps_per_head", "source_signal_shots")
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective", choices=("cosine", "risk_utility"), default="cosine")
    args = parser.parse_args()
    main(args.design, args.output, objective=args.objective)
