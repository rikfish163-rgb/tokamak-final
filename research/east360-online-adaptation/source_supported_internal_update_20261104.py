"""Expand internal-update teaching using TRAIN only; no source risk retraining."""

from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import internal_gradient_transport_20261104 as original
from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.direct_fast_memory_20261104.model import levels
from experiments.risk_aligned_online.event_prefix_probe import (
    assign_events,
    DEFAULT_LABELS,
)
from experiments.risk_aligned_online.source_event_tasks_20261102.data import (
    load_records,
    load_events,
)

data, motion = original.data, original.motion
ROOT = data.ROOT
OUT = ROOT / "probe_outputs/source_supported_internal_update_20261104_v1"
PLAN = ROOT / "SOURCE_SUPPORTED_INTERNAL_UPDATE_PLAN_20261104.md"
OLD = original.OUT
LEARNERS = ("east20", "source_only", "mixed_uniform", "mixed_event", "mixed_shuffled")
MODES = ("real", "hold", "reverse")


def select_records(rows, allowed):
    if any(r["shot"] not in allowed for r in rows):
        raise ValueError("only source TRAIN records are allowed")
    groups = defaultdict(list)
    for row in rows:
        if row["query"] >= 159:
            groups[row["shot"], row["group"]].append(row)
    return [
        sorted(group, key=lambda r: r["query"])[len(group) // 2].copy()
        for _, group in sorted(groups.items())
    ]


def family_weights(rows, *, use_family):
    counts = Counter(
        (r["group"], r["family"] if use_family else 0, r["shot"]) for r in rows
    )
    families = defaultdict(set)
    shots = defaultdict(set)
    for group, family, sid in counts:
        families[group].add(family)
        shots[group, family].add(sid)
    if set(families) != {"safe", "early", "danger"}:
        raise ValueError("all three source groups are required")
    weights = []
    for row in rows:
        g, sid = row["group"], row["shot"]
        f = row["family"] if use_family else 0
        if not -1 <= row["family"] <= 5:
            raise ValueError("nominal event family must be unknown or A–F")
        weights.append(
            1 / (3 * len(families[g]) * len(shots[g, f]) * counts[g, f, sid])
        )
    if not np.isclose(sum(weights), 1, rtol=0, atol=1e-12):
        raise ValueError("source balanced weights must sum to one")
    return np.asarray(weights, float)


def feature_at(base, signal, maturity, sid):
    x, a = original.window_batch(signal, [maturity], device="cpu")
    with torch.no_grad():
        z = float(base(x, a)["logit"])
        level = levels(base, x, a)[0, -32:, list(motion.CHANNEL_INDICES)].numpy()
    mask = a[0, -32:, list(motion.CHANNEL_INDICES)].numpy()
    trend = motion.motion_features(level, mask, motion.bin_permutation(sid, maturity))
    return np.r_[
        np.where(mask[-1], level[-1], 0).astype(float),
        mask[-1].astype(float),
        trend["common"].astype(float),
        z,
        trend["long"],
    ]


def reset(params, initial):
    with torch.no_grad():
        for p, value in zip(params, initial, strict=True):
            p.copy_(value)
            p.grad = None


def verify_saved(folder):
    receipt = data.read_json(folder / "completed.json")
    for name, digest in receipt["artifact_sha256"].items():
        if data.sha(folder / name) != digest:
            raise ValueError(f"upstream artifact changed: {name}")
    return receipt


def evaluate(
    base, model, params, initial, learners, features, meta, ids, mapping, hashes
):
    all_directions = {}
    for learner, fitted in learners.items():
        for mode in MODES:
            view = features.copy()
            if mode == "hold":
                view[:, 64:] = 0
            elif mode == "reverse":
                view[:, 64:] *= -1
            all_directions[learner + "__" + mode] = original.predict_transport(
                fitted, view
            )
    results, receipts, k = [], [], 0
    for index, sid in enumerate(ids[20:100], 21):
        signal, cases, receipt = data._read_sealed(index, sid, mapping, hashes)
        receipts.append(receipt)
        for case in cases:
            expected = meta[k]
            if (sid, case["origin"], case["label"]) != (
                expected["shot"],
                expected["origin"],
                expected["label"],
            ):
                raise ValueError("fixed held80 cases changed")
            x, a = original.window_batch(signal, [case["future"]], device="cpu")
            arms = {}
            for arm in ("none", *all_directions):
                reset(params, initial)
                if arm != "none":
                    original.apply_direction(params, all_directions[arm][k])
                with torch.no_grad():
                    z = model(x, a)["logit"]
                    bce = F.binary_cross_entropy_with_logits(
                        z, torch.full_like(z, float(case["label"]))
                    )
                    arms[arm] = dict(
                        bce=float(bce), logit=float(z), probability=float(z.sigmoid())
                    )
            results.append(dict(shot=sid, **case, arms=arms))
            k += 1
        if index % 10 == 0:
            print(dict(evaluated_shots=index - 20, cases=k), flush=True)
    reset(params, initial)
    if k != 240:
        raise ValueError("exact 240 held cases required")
    return results, receipts


def summarise(rows):
    output = {}
    for learner in LEARNERS:
        cast = []
        for row in rows:
            cast.append(
                dict(
                    row,
                    arms=dict(
                        none=row["arms"]["none"],
                        state=row["arms"]["east20__real"],
                        **{m: row["arms"][learner + "__" + m] for m in MODES},
                    ),
                )
            )
        output[learner] = original.summaries(cast)
    return output


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    verify_saved(OLD)
    verify_saved(original.FEATURES)
    rows, roles, info = load_records("train")
    chosen = select_records(rows, set(roles))
    events, _ = load_events()
    events = {sid: e for sid, e in events.items() if sid in set(roles)}
    advanced = io23.advanced_times()
    oldprotocol = data.read_json(data.RUN / "protocol.json")
    ids = [r["shot"] for r in oldprotocol["order"]]
    mapping, hashes = data.read_json(data.MAP), data.read_json(data.RAW_HASHES)
    source_before, bound = (
        None,
        {
            str(p): data.sha(p)
            for p in (
                Path(__file__),
                Path(original.__file__),
                Path(data.__file__),
                Path(motion.__file__),
                PLAN,
                data.DEFAULT_PARENT,
                DEFAULT_LABELS,
                io23.JTEXT_INDEX,
                io23.JTEXT_ADVANCED_TIME_PATH,
                OLD / "completed.json",
                OLD / "fit.npz",
                OLD / "result.json",
                original.FEATURES / "completed.json",
                original.FEATURES / "features_and_logits.npz",
                original.FEATURES / "cases.json",
                data.MAP,
                data.RAW_HASHES,
                data.RUN / "protocol.json",
            )
        },
    )
    OUT.mkdir()
    data.write_json(
        OUT / "protocol.json",
        dict(
            scope="FIXED_SOURCE_TRAIN_SUPPORT_INTERNAL_DIRECTION_HELD_EAST80",
            source_train_ids=roles,
            selected_records=chosen,
            label="JTEXT max(25ms,AdvancedTime); EAST125ms, outcomes HDF only",
            source_view="MNM and CIII explicitly missing, not imputed",
            source_val_or_test_signal_access=False,
            source_model_trained=False,
            csv_usage="nominal source training strata only, no ordinal stage or outcome replacement",
            ridge=0.1,
            mixed_source_mass=0.5,
            step=0.05,
            learner_names=LEARNERS,
            input_sha256=bound,
        ),
    )
    base, parent = data.load_parent(data.DEFAULT_PARENT)
    base.eval().requires_grad_(False)
    source_before = data.state_digest(base)
    if (
        source_before != oldprotocol["source_state_sha256"]
        or parent["threshold"] != oldprotocol["threshold"]
    ):
        raise ValueError("fixed source parent changed")
    model = original.make_adapter(base, "norm")
    params = tuple(p for p in model.parameters() if p.requires_grad)
    initial = tuple(p.detach().clone() for p in params)
    if sum(p.numel() for p in params) != 256:
        raise ValueError("fixed 256-parameter internal scope required")
    grouped = defaultdict(list)
    for row in chosen:
        grouped[row["shot"]].append(row)
    sx, sg, records, source_receipts = [], [], [], []
    for n, (sid, items) in enumerate(sorted(grouped.items()), 1):
        raw_path = io23.path_for("J-TEXT", sid)
        actual_hash = data.sha(raw_path)
        if actual_hash != info["input_hashes"][str(sid)]:
            raise ValueError(f"source TRAIN raw SHA changed: {sid}")
        signal = simple_data.read_shot("J-TEXT", sid)
        disruptive, down = io23.read_outcome("J-TEXT", sid)
        label, valid = io23.labels_at(
            signal["times"],
            disruptive,
            down,
            io23.effective_horizon_ms(sid, advanced)[0],
        )
        signal = dict(
            values=signal["values"].copy(), available=signal["available"].copy()
        )
        signal["available"][:, [1, 2]] = False
        times = io23_start_times(sid, len(signal["values"]))
        assigned, _ = assign_events(times, events.get(sid, []))
        for record in items:
            q = record["query"]
            if (
                not valid[q]
                or bool(record["disruptive"]) != disruptive
                or float(label[q]) != record["label"]
            ):
                raise ValueError("source archive disagrees with fresh HDF point labels")
            if disruptive and round(float((down - times[q]) * 1000), 9) <= 10:
                raise ValueError(
                    "source query must have strictly greater than10ms lead"
                )
            maturity = q - 24
            sx.append(feature_at(base, signal, maturity, sid))
            x, a = original.window_batch(signal, [q], device="cpu")
            z = model(x, a)["logit"]
            loss = F.binary_cross_entropy_with_logits(
                z, torch.full_like(z, float(label[q]))
            )
            grads = torch.autograd.grad(loss, params)
            vector = torch.cat([g.flatten() for g in grads]).detach().double()
            norm = vector.norm()
            sg.append((vector / norm if norm else vector).numpy())
            records.append(
                dict(
                    record,
                    family=int(assigned[maturity]),
                    maturity=maturity,
                    actual_hdf_disruptive=disruptive,
                    gradient_norm=float(norm),
                )
            )
        if data.sha(raw_path) != actual_hash:
            raise ValueError("raw source changed during its read")
        source_receipts.append(dict(shot=sid, raw_sha256=actual_hash, cases=len(items)))
        if n % 100 == 0:
            print(dict(source_train_shots=n, source_cases=len(records)), flush=True)
    sx, sg = np.stack(sx), np.stack(sg)
    uniform = family_weights(records, use_family=False)
    event_weight = family_weights(records, use_family=True)
    shuffled = [r.copy() for r in records]
    rng = np.random.default_rng(131)
    for group in ("safe", "early", "danger"):
        select = [i for i, r in enumerate(shuffled) if r["group"] == group]
        values = rng.permutation([shuffled[i]["family"] for i in select])
        for i, value in zip(select, values, strict=True):
            shuffled[i]["family"] = int(value)
    shuffled_weight = family_weights(shuffled, use_family=True)
    with np.load(OLD / "fit.npz", allow_pickle=False) as values:
        east_teacher = values["teachers"].copy()
        east_weight = values["weights"].copy()
        old_fit = {
            key: values["real_" + key].copy() for key in ("mean", "scale", "coef")
        }
    with np.load(
        original.FEATURES / "features_and_logits.npz", allow_pickle=False
    ) as values:

        def matrix(stage):
            return np.concatenate(
                [
                    values[stage + "_endpoint"],
                    values[stage + "_endpoint_mask"],
                    values[stage + "_common"],
                    values[stage + "_native_logit"][:, None],
                    values[stage + "_long"],
                ],
                axis=1,
            ).astype(float)

        ex, hx = matrix("fit"), matrix("held")
    learners = dict(east20=old_fit, source_only=original.fit_transport(sx, sg, uniform))
    x, y = np.r_[sx, ex], np.r_[sg, east_teacher]
    for name, weight in (
        ("mixed_uniform", uniform),
        ("mixed_event", event_weight),
        ("mixed_shuffled", shuffled_weight),
    ):
        learners[name] = original.fit_transport(
            x, y, np.r_[0.5 * weight, 0.5 * east_weight]
        )
    data.write_json(OUT / "source_records.json", records)
    data.write_json(OUT / "source_inputs.json", source_receipts)
    with (OUT / "fit.npz").open("xb") as f:
        np.savez_compressed(
            f,
            source_features=sx,
            source_gradients=sg,
            source_weights=uniform,
            event_weights=event_weight,
            shuffled_weights=shuffled_weight,
            **{name + "_" + k: v for name, m in learners.items() for k, v in m.items()},
        )
    if data.state_digest(base) != source_before:
        raise ValueError("source was changed while fitting")
    print("source-supported rules fixed; evaluating EAST held80 only", flush=True)
    held_meta = [
        r
        for r in data.read_json(original.FEATURES / "cases.json")
        if r["stage"] == "held"
    ]
    result, receipts = evaluate(
        base, model, params, initial, learners, hx, held_meta, ids, mapping, hashes
    )
    if data.state_digest(base) != source_before or any(
        data.sha(Path(p)) != v for p, v in bound.items()
    ):
        raise ValueError("parent or bound input changed")
    previous = data.read_json(OLD / "result.json")
    if any(
        a["arms"]["east20__real"] != b["arms"]["real"]
        or a["arms"]["east20__hold"] != b["arms"]["hold"]
        or a["arms"]["east20__reverse"] != b["arms"]["reverse"]
        or a["arms"]["none"] != b["arms"]["none"]
        for a, b in zip(result, previous, strict=True)
    ):
        raise ValueError("old E20 control must replay exactly")
    summary = summarise(result)
    for name, value in (
        ("results.json", result),
        ("summary.json", summary),
        ("held_inputs.json", receipts),
    ):
        data.write_json(OUT / name, value)
    data.write_json(
        OUT / "completed.json",
        dict(
            status="COMPLETE_SOURCE_SUPPORTED_INTERNAL_DIRECTION_SCREEN",
            seconds=time.monotonic() - started,
            source_shots=len(grouped),
            source_cases=len(records),
            source_state_unchanged=True,
            east20_control_exact=True,
            artifact_sha256={p.name: data.sha(p) for p in OUT.iterdir() if p.is_file()},
        ),
    )
    print(
        {
            name: {g: v["pairs"]["real_vs_none"]["delta_bce"] for g, v in rows.items()}
            for name, rows in summary.items()
        },
        flush=True,
    )


def io23_start_times(sid, length):
    # Read raw HDF sampling clock directly; never align event boundaries to
    # disruption-relative zero or assume a newly resampled one-kHz timeline.
    import h5py

    with h5py.File(io23.path_for("J-TEXT", sid), "r") as f:
        ds = f["data/ip"]
        rate = float(np.asarray(ds.attrs["SampleRate"]).item())
        start = float(np.asarray(ds.attrs["StartTime"]).item())
        if len(ds) != length:
            raise ValueError("source clock length mismatch")
    return start + np.arange(length) / rate


if __name__ == "__main__":
    run()
