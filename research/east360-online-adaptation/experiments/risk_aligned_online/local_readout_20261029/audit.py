"""Independent raw-data/NumPy recount; production-forward replay is separate.

Outcomes are read retrospectively here, never to choose an online update.
"""

import argparse
from collections import deque
import hashlib
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23
from ..attention_basis_20261029.learning import MatrixAttention
from ..context_memory_20261028.checkpoint import read
from ..gradient_split_20261029.audit import count
from ..qk_forecast_stream_20261029.audit import decision, paired, signal
from ..qk_matched_head_20261029.run import snapshot
from ..readout_identifiability_20261029.audit import numpy_fit
from ..readout_identifiability_20261029.run import collect
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import ARMS, MODES, load_head, predict


def auxiliary(z, sig, valid, center, scale, active):
    """Score sealed predictions independently, with explicit target-time masks."""
    rows = []
    for j, clock in enumerate(z["aux_clock"]):
        c = int(clock)
        if c < 256 or c + 25 > len(valid):
            raise ValueError("incomplete next-window sample")
        allowed = sig["available"][c - 1] & np.isfinite(sig["values"][c - 1]) & active
        if not np.array_equal(allowed, z["aux_available"][j]):
            raise ValueError("last-observation or supported-channel mask differs")
        raw = sig["values"][c : c + 25].astype(np.float64)
        mask = (
            sig["available"][c : c + 25]
            & np.isfinite(raw)
            & allowed
            & valid[c : c + 25, None]
        )
        truth = np.zeros((25, 23), dtype=np.float64)
        normalized = np.arcsinh((raw - center) / scale)
        truth[mask] = normalized[mask]
        losses = {}
        methods = {
            k: z[f"aux_{k}"][j]
            for k in [arm + part for arm in MODES for part in ("_before", "_after")]
        }
        methods["hold"] = np.broadcast_to(z["aux_last"][j], (25, 23))
        for method, prediction in methods.items():
            error = np.zeros_like(truth)
            error[mask] = prediction[mask].astype(np.float64) - truth[mask]
            if not np.isfinite(error).all():
                raise ValueError("nonfinite observed auxiliary prediction")
            magnitude = np.abs(error)
            loss = np.minimum(magnitude, 1.0) ** 2 / 2 + np.maximum(
                magnitude - 1.0, 0.0
            )
            losses[method] = dict(
                sum=float(loss.sum()), channel_sums=loss.sum(0).tolist()
            )
        rows.append(
            dict(
                clock=c,
                elements=int(mask.sum()),
                channel_counts=mask.sum(0).tolist(),
                losses=losses,
            )
        )
    return rows


def check_feedback(updates, sig, active):
    history = deque(maxlen=16)
    clocks = list(range(256, len(sig["values"]), 128))
    records = updates["feedback"]
    if len(records) != len(clocks):
        raise ValueError("feedback clock count differs")
    observed = eligible = 0
    for c, record in zip(clocks, records):
        origins = [c - v for v in (101, 76, 51, 26)]
        masks = []
        for origin in origins:
            mask = (
                sig["available"][origin + 1 : origin + 26]
                & np.isfinite(sig["values"][origin + 1 : origin + 26])
            ).all(0)
            mask &= (
                sig["available"][origin] & np.isfinite(sig["values"][origin]) & active
            )
            masks.append(mask)
        history.extend(masks[:2])
        expected = dict(
            clock=c,
            origins=origins,
            calibration_last_target=c - 51,
            gradient_first_target=c - 50,
            gradient_last_target=c - 1,
            history_fit_rows=len(history),
            fit_counts_max=int(np.stack(history).sum(0).max()),
        )
        history.extend(masks[2:])
        expected["history_rows_after"] = len(history)
        if any(record[k] != v for k, v in expected.items()):
            raise ValueError("fit/gradient time, history, or mask receipt differs")
        if (
            not np.isfinite(record["readout_delta_norm"])
            or record["readout_delta_norm"] < 0
        ):
            raise ValueError("invalid readout change")
        here = sum(int(m.sum()) * 25 for m in masks[2:])
        observed += here
        eligible += int(here > 0)
    for arm in MODES:
        r = updates[arm]
        if (
            r["opportunities"] != len(clocks)
            or r["observed"] != observed
            or not 0 <= r["updates"] <= eligible
            or not 0 <= r["decreased"] <= eligible
            or not 0 <= r["max_norm"] <= 0.050001
            or not np.isfinite([r["loss_before_sum"], r["loss_after_sum"]]).all()
        ):
            raise ValueError("gradient observations/count/radius differs")


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-9)


def summarize(rows, recorded):
    result = {}
    for method, declaration in recorded.items():
        values = []
        for row in rows:
            eligible = [s for s in row["samples"] if s["elements"]]
            if eligible:
                values.append(
                    dict(
                        shot=row["shot"],
                        windows=len(eligible),
                        loss=float(
                            np.mean(
                                [
                                    s["losses"][method]["sum"] / s["elements"]
                                    for s in eligible
                                ]
                            )
                        ),
                    )
                )
        if [r["shot"] for r in values] != [r["shot"] for r in declaration["per_shot"]]:
            raise ValueError("auxiliary shot set differs")
        v = np.array([r["loss"] for r in values])
        close(v, [r["loss"] for r in declaration["per_shot"]])
        close(v.mean(), declaration["mean"])
        close(v.std(ddof=1), declaration["shot_sd"])
        if (
            len(values) != declaration["shots"]
            or sum(r["windows"] for r in values) != declaration["windows"]
        ):
            raise ValueError("auxiliary sample count differs")
        result[method] = dict(
            shots=len(values),
            windows=sum(r["windows"] for r in values),
            mean=float(v.mean()),
            shot_sd=float(v.std(ddof=1)),
            per_shot=values,
        )
    hold = np.array([r["loss"] for r in result["hold"]["per_shot"]])
    for arm in MODES:
        before = np.array([r["loss"] for r in result[arm + "_before"]["per_shot"]])
        after = np.array([r["loss"] for r in result[arm + "_after"]["per_shot"]])
        result[arm + "_after"].update(
            beats_hold_shots=int((after < hold).sum()),
            improved_over_before_shots=int((after < before).sum()),
            after_minus_before_mean=float((after - before).mean()),
        )
    return result


def reproduce_prior(net, protocol, mapping, slow, head):
    cold, receipt = collect(net, protocol["order"][:20], "EAST", {}, mapping, slow)
    if receipt["read_ids"] != protocol["order"][:20] or receipt["outcome_reads"]:
        raise ValueError("prior fit data access differs")
    path = Path(protocol["readout"]) / "cold_readouts.npz"
    error = 0.0
    with np.load(path, allow_pickle=False) as saved:
        for c in np.flatnonzero(head.active.cpu().numpy()):
            take = cold["mask"][:, c]
            now = cold["current"][take, c : c + 1]
            future = cold["future"][take, :, c]
            target = np.concatenate([now, future, future - now], axis=1)
            fit = numpy_fit(cold["last_token"][take], target, cold["ids"][take])
            for key, value in fit.items():
                close(value, saved[f"last_token_{c}_{key}"])
            coef = np.vstack([fit["coef"][:, 26:], fit["target_mean"][26:]])
            stored = head.coef[c].cpu().numpy()
            close(coef, stored)
            error = max(error, float(np.abs(coef - stored).max()))
    return dict(
        read_ids=receipt["read_ids"],
        outcome_reads=0,
        windows=len(cold["ids"]),
        max_coefficient_error=error,
        scope="independent NumPy cold20 ridge; production feature extraction reused",
    )


def replay(root, protocol, changes, mapping):
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base, _ = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda())
    before = snapshot(net)
    slow = torch.load(
        Path(protocol["reference"]) / "models/cold.pt",
        map_location="cuda",
        weights_only=True,
    )["slow"]
    head = load_head(Path(protocol["readout"]) / "cold_readouts.npz", "cuda")
    prior = reproduce_prior(net, protocol, mapping, slow, head)
    chosen = {protocol["order"][i] for i in (20, 180, 359)}
    chosen.update(
        sid for item in changes.values() for group in item.values() for sid in group
    )
    rows = []

    def deny(*args, **kwargs):
        raise AssertionError("outcome accessed during replay prediction")

    for sid in sorted(chosen, key=protocol["order"].index):
        with patch.object(io23, "read_outcome", side_effect=deny):
            curves, info, aux = predict(net, signal(sid, mapping), slow, head)
        i = protocol["order"].index(sid)
        name = f"{i + 1:04d}_{sid}"
        if info != read(root / "shots", name + "_seal.json")["updates"]:
            raise ValueError("replayed feedback differs")
        with np.load(root / "shots" / f"{name}.npz", allow_pickle=False) as saved:
            error = max(float(np.abs(saved[k] - curves[k]).max()) for k in ARMS)
            if error > 2e-6 or any(
                not np.array_equal(
                    saved[k] >= protocol["threshold"],
                    curves[k] >= protocol["threshold"],
                )
                for k in ARMS
            ):
                raise ValueError("replayed risk/crossings differ")
            aux_error = 0.0
            for key in aux[0] if aux else []:
                value = np.stack([row[key] for row in aux])
                old = saved["aux_" + key]
                close(value, old)
                if value.dtype.kind == "f":
                    aux_error = max(aux_error, float(np.abs(value - old).max()))
        rows.append(dict(shot=sid, max_risk_error=error, max_auxiliary_error=aux_error))
        print(
            dict(replayed=sid, risk_error=error, auxiliary_error=aux_error), flush=True
        )
    if any(not torch.equal(v, before[k]) for k, v in snapshot(net).items()):
        raise ValueError("replay changed source classifier")
    return dict(
        prior=prior,
        shots=rows,
        scope="all changed shots plus stream indices21/181/360; not all360 neural replay",
    )


def run(root):
    started = time.monotonic()
    if (root / "audit.json").exists():
        raise FileExistsError(root / "audit.json")
    p, done = read(root, "protocol.json"), read(root, "complete.json")
    if not done["complete"] or tuple(p["arms"]) != ARMS or len(set(p["order"])) != 360:
        raise ValueError("completed declared360 required")
    if sha(DEFAULT_PARENT) != PARENT_SHA256 or p["source_sha256"] != PARENT_SHA256:
        raise ValueError("source changed")
    for name in (
        "protocol",
        "per_shot",
        "receipts",
        "auxiliary_summary",
        "auxiliary_per_shot",
    ):
        if sha(root / (name + ".json")) != done[name + "_sha256"]:
            raise ValueError("changed artifact " + name)
    plan = "LOCAL_READOUT_STREAM_PLAN.md"
    with zipfile.ZipFile(root / "code_snapshot.zip") as archive:
        for name, digest in dict(p["code_sha256"], **{plan: p["plan_sha256"]}).items():
            if (
                hashlib.sha256(archive.read(name)).hexdigest() != digest
                or sha(io23.ROOT / name) != digest
            ):
                raise ValueError("code/plan snapshot changed")
    ref, previous, readout = [Path(p[k]) for k in ("reference", "previous", "readout")]
    for name, digest in p["reference_hashes"].items():
        if sha(ref / name) != digest:
            raise ValueError("cold reference changed")
    if (
        sha(previous / "complete.json") != p["previous_complete_sha256"]
        or sha(readout / "cold_readouts.npz") != p["readout_sha256"]
        or sha(readout / "audit_with_fold_gap.json") != p["readout_audit_sha256"]
    ):
        raise ValueError("parent results changed")
    mapping, hashes = read(ref, "input_map.json"), read(ref, "input_hashes.json")
    base, _ = load_parent(DEFAULT_PARENT)
    center, scale = base.center.numpy(), base.scale.numpy()
    active = load_head(readout / "cold_readouts.npz", "cpu").active.numpy()
    declared, receipts = read(root, "per_shot.json"), read(root, "receipts.json")
    recorded_aux = {r["shot"]: r for r in read(root, "auxiliary_per_shot.json")}
    actual = {arm: [] for arm in ARMS}
    auxiliary_rows = []
    totals = {
        arm: dict(updates=0, decreased=0, opportunities=0, observed=0) for arm in MODES
    }
    for i, sid in enumerate(p["order"]):
        if sha(io23.path_for("EAST", sid)) != hashes[str(sid)]:
            raise ValueError("raw input changed")
        name = f"{i + 1:04d}_{sid}"
        seal = read(root / "shots", name + "_seal.json")
        if (
            seal != receipts[i]
            or seal["sha256"] != sha(root / "shots" / f"{name}.npz")
            or seal["outcome_used"]
            or seal["adaptation_active"] != (i >= 20)
        ):
            raise ValueError("invalid prediction seal")
        if read(previous / "shots", name + "_seal.json")["sha256"] != sha(
            previous / "shots" / f"{name}.npz"
        ):
            raise ValueError("previous seal changed")
        flag, down = io23.read_outcome("EAST", sid)
        with (
            np.load(root / "shots" / f"{name}.npz", allow_pickle=False) as z,
            np.load(previous / "shots" / f"{name}.npz", allow_pickle=False) as old,
        ):
            if (
                not np.array_equal(z["times"], old["times"])
                or np.abs(z["cold"] - old["cold"]).max() > 2e-6
                or not np.array_equal(
                    z["cold"] >= p["threshold"], old["cold"] >= p["threshold"]
                )
            ):
                raise ValueError("cold prediction differs")
            for arm in ARMS:
                d = decision(sid, z["times"], z[arm], flag, down, p["threshold"])
                if d != declared[arm][i]:
                    raise ValueError("per-shot decision differs")
                actual[arm].append(d)
                end = len(z["times"]) if i < 20 else 256
                if np.abs(z[arm][:end] - z["cold"][:end]).max() > 2e-6:
                    raise ValueError("past or cold-prefix risk changed")
            if i < 20:
                if seal["updates"] or seal["aux_samples"]:
                    raise ValueError("cold prefix adaptation active")
                continue
            sig = signal(sid, mapping)
            check_feedback(seal["updates"], sig, active)
            for arm in MODES:
                for k in totals[arm]:
                    totals[arm][k] += seal["updates"][arm][k]
            valid = (
                np.round((down - z["times"]) * 1000, 9) >= -1e-6
                if flag
                else np.ones(len(z["times"]), bool)
            )
            eligible = np.arange(256, len(z["times"]) - 24, 128)
            clocks = (
                eligible[
                    np.linspace(0, len(eligible) - 1, min(4, len(eligible))).astype(int)
                ]
                if len(eligible)
                else []
            )
            if not np.array_equal(
                z["aux_clock"] if len(clocks) else [], clocks
            ) or seal["aux_samples"] != len(clocks):
                raise ValueError("auxiliary sampling clocks differ")
            if len(clocks):
                rows = auxiliary(z, sig, valid, center, scale, active)
                old_rows = recorded_aux[sid]["samples"]
                for r, q in zip(rows, old_rows):
                    if any(
                        r[k] != q[k] for k in ("clock", "elements", "channel_counts")
                    ):
                        raise ValueError("auxiliary target masks differ")
                    for method in r["losses"]:
                        close(
                            r["losses"][method]["channel_sums"],
                            q["losses"][method]["channel_sums"],
                        )
                        close(r["losses"][method]["sum"], q["losses"][method]["sum"])
                auxiliary_rows.append(dict(shot=sid, samples=rows))
    metrics = {
        arm: {
            label: count(sub)
            for label, sub in (
                ("full360", rows),
                ("cold20", rows[:20]),
                ("suffix340", rows[20:]),
            )
        }
        for arm, rows in actual.items()
    }
    for arm, sections in metrics.items():
        for name, values in sections.items():
            for key, value in values.items():
                reported = done["results"][arm][name][key]
                if value is None:
                    assert reported is None
                else:
                    close(value, reported)
    changes = {arm: paired(actual["cold"], actual[arm]) for arm in MODES}
    if (
        changes != done["comparisons"]
        or {k: v["updates"] for k, v in totals.items()} != done["update_counts"]
    ):
        raise ValueError("paired decisions or update counts differ")
    aux = summarize(auxiliary_rows, read(root, "auxiliary_summary.json"))
    replayed = replay(root, p, changes, mapping)
    result = dict(
        complete=True,
        complete_sha256=sha(root / "complete.json"),
        code_sha256=sha(Path(__file__)),
        seconds=time.monotonic() - started,
        metrics=metrics,
        comparisons=changes,
        feedback_totals=totals,
        auxiliary=aux,
        replay=replayed,
        independent_confirmation=False,
        scope="all360 sealed five-arm decisions, raw-signal auxiliary masks/losses, independent cold20 NumPy fit, changed-shot neural replay",
    )
    _durable_json(root / "audit.json", result)
    print(
        dict(
            metrics={k: v["full360"] for k, v in metrics.items()},
            comparisons=changes,
            auxiliary={
                k: {j: v for j, v in row.items() if j != "per_shot"}
                for k, row in aux.items()
            },
            replay=replayed,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root)
