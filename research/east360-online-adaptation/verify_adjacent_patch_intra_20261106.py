"""Independently recount sealed curves and replay selected actual patch states."""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

import run_adjacent_patch_capacity_20261106 as cap
import run_adjacent_patch_intra_20261106 as study
from verify_confirmation_raw_replay_20261105 import Hashes


def outcome_and_clock(path, length):
    with h5py.File(path, "r") as f:
        y = np.asarray(f["meta/IsDisrupt"][()]).reshape(-1)
        d = np.asarray(f["meta/DownTime"][()]).reshape(-1)
        dset = f["data/ip"]
        rate = float(np.asarray(dset.attrs["SampleRate"]).reshape(-1)[0])
        start = float(np.asarray(dset.attrs["StartTime"]).reshape(-1)[0])
        if y.size != 1 or y[0] not in (0, 1) or len(dset) != length:
            raise ValueError("HDF5 shot metadata does not match the sealed row")
    if y[0] and (d.size != 1 or not np.isfinite(d[0])):
        raise ValueError("missing HDF5 disruption time")
    return bool(y[0]), float(d[0]) if y[0] else None, start + np.arange(length) / rate


def first_alarm(t, z, flag, down, tau):
    ok = (down - t) * 1000 >= -1e-6 if flag else np.ones(len(t), dtype=bool)
    ids = np.flatnonzero((z.astype(np.float64) >= tau) & ok)
    k = int(ids[0]) if len(ids) else None
    lead = round(float((down - t[k]) * 1000), 9) if flag and k is not None else None
    return dict(
        first=k,
        lead_ms=lead,
        timely=lead is not None and lead > 10,
        false_alarm=not flag and k is not None,
    )


@torch.no_grad()
def direct(base, x, a, packed, state, mode):
    """Original Simple23.forward with only its embedding output changed."""
    delta = study.op.delta(packed, state, mode)
    calls = []

    def add(_, __, value):
        calls.append(1)
        return value + delta

    handle = base.embedding.register_forward_hook(add)
    try:
        z = base(x, a)["logit"].cpu().numpy()
    finally:
        handle.remove()
    if calls != [1]:
        raise ValueError("original embedding hook did not execute once")
    return z


def recount(root, cfg, seal, report, h, capacity):
    names = [x["shot"] for x in seal["shots"]]
    if names != cfg["roles"] or len(names) != 260 or len(set(names)) != len(names):
        raise ValueError("sealed TRAIN role cohort changed")
    with np.load(root / "curves.npz", allow_pickle=False) as tape:
        data = {k: tape[k] for k in tape.files}
    n = p = 0
    before, examples, hdf = [], {}, {}
    for note, scored in zip(seal["shots"], report["shots"], strict=True):
        sid = note["shot"]
        t = slice(n, n + note["length"])
        b = slice(p, p + note["blocks"])
        n, p = t.stop, b.stop
        path = study.r.io23.path_for("J-TEXT", sid)
        if h(path) != note["sha256"]:
            raise ValueError("raw source hash changed")
        flag, down, times = outcome_and_clock(path, note["length"])
        z, clock = data["curve"][t], data["clock"][b]
        if not np.array_equal(times, data["times"][t]) or not np.array_equal(
            clock, np.arange(63, len(times) - 1, 32)
        ):
            raise ValueError("sample clock or release order differs")
        scored_again = [first_alarm(times, col, flag, down, cfg["tau"]) for col in z.T]
        if (
            scored_again != scored["scores"]
            or scored["shot"] != sid
            or scored["disruptive"] != flag
        ):
            raise ValueError("independent first alarm differs from the reported one")
        before.append(dict(shot=sid, disruptive=flag, scores=scored_again))
        active = data["active"][b]
        support = np.zeros(len(times), dtype=bool)
        for j, m in enumerate(clock):
            ix = slice(m + 1, min(m + 33, len(times)))
            if active[j]:
                support[ix] = True
                offsets = data["offset"][b][j]
                if not np.array_equal(
                    z[ix, 5], z[ix, 0] + offsets[0]
                ) or not np.array_equal(z[ix, 6], z[ix, 0] + offsets[2]):
                    raise ValueError("same-anchor scalar control changed")
        if not np.array_equal(
            z[~support], np.broadcast_to(z[~support, :1], z[~support].shape)
        ):
            raise ValueError(
                "a prediction was overwritten outside an active future block"
            )
        if "prefix_check" in note:
            examples[sid] = (
                z,
                clock,
                data["states"][b],
                active,
                note["prefix_check"]["role"],
            )
        if sid in capacity:
            hdf[sid] = (z[:, 0], times, down)
    if (
        n != len(data["curve"])
        or p != len(data["clock"])
        or not np.isfinite(data["curve"]).all()
    ):
        raise ValueError("truncated or unbound output")
    if np.linalg.norm(data["states"], axis=-1).max() > cfg["radius"] + 5e-7:
        raise ValueError("archived temporary state exceeded the norm budget")
    if (
        data["guard"].max() > cfg["guard"]
        or np.ptp(data["edit_rms"], axis=1).max() > 1e-6
    ):
        raise ValueError("archived guard or edit match failed")
    if study.summarize(before, report["windows"]) != report["summary"]:
        raise ValueError("summary differed on independently recomputed score")
    return examples, hdf


def original_model_witnesses(base, projection, examples, hdf, cap_rows):
    result = []
    for sid in sorted(set(examples) | set(hdf)):
        raw = study.r.simple_data.read_shot("J-TEXT", sid)
        x, a = study.bank.prior.views(base, raw)
        p = study.op.encode(base, x, a, projection)
        worst, checked, guard = 0.0, 0, 0.0
        if sid in examples:
            z, clock, states, active, role = examples[sid]
            for j in np.flatnonzero(active)[:2]:
                m = clock[j]
                ix = np.arange(m + 1, min(m + 33, len(z)))
                for k, mode in enumerate(study.MODES):
                    u = x.new_tensor(states[j, k])
                    seen = direct(
                        base, x[ix], a[ix], p.take(ix), u.expand(len(ix), -1), mode
                    )
                    error = float(np.abs(seen - z[ix, k + 1]).max())
                    if error > 5e-5 or not np.array_equal(
                        seen >= study.TAU, z[ix, k + 1] >= study.TAU
                    ):
                        raise ValueError(
                            "original hook path disagrees with stored adapted curve"
                        )
                    worst, checked = max(worst, error), checked + 1
            result.append(
                dict(shot=sid, stage=role, max_curve_error=worst, checked=checked)
            )
        if sid not in hdf:
            continue
        checked = 0
        z, times, down = hdf[sid]
        for name, mode in zip(cap.ARMS, cap.MODES, strict=True):
            choices = []
            for rec in (q for q in cap_rows if q["shot"] == sid):
                for attempt in rec["arms"][name]:
                    choices.extend(
                        (q["peak"], rec["maturity"], q)
                        for q in (
                            attempt["initial"],
                            attempt["raw"]["witness"],
                            attempt["shape"]["witness"],
                        )
                        if q is not None
                    )
            if not choices:
                continue
            _, m, best = max(choices, key=lambda v: v[0])
            ix = np.arange(m + 1, min(m + 33, len(z)))
            reached = np.array([m - 16, m - 8, m])
            u = x.new_tensor(best["state"])
            edited = z.copy()
            edited[ix] = direct(
                base, x[ix], a[ix], p.take(ix), u.expand(len(ix), -1), mode
            )
            q, v = x[reached], a[reached]
            history = [
                direct(base, q, v, p.take(reached), sign * u.expand(3, -1), mode)
                for sign in (1.0, -1.0)
            ]
            drift = float(np.abs(np.stack(history) - z[reached]).max())
            if (
                drift > study.pair.LIMIT
                or first_alarm(times, edited, True, down, study.TAU)
                != best["full_first"]
            ):
                raise ValueError("independent capacity witness did not replay")
            guard, checked = max(guard, drift), checked + 1
        result.append(
            dict(
                shot=sid, stage="late-capacity", max_trust_drift=guard, checked=checked
            )
        )
    return result


def run(root, capacity_path, out):
    if out.exists():
        raise FileExistsError(out)
    h = Hashes()
    pp, sp = root / "protocol.json", root / "predictions_seal.json"
    cfg, seal, report = (study.r.read(x) for x in (pp, sp, root / "result.json"))
    capacity = study.r.read(capacity_path)
    if (
        cfg["status"] != "LOCKED_ADJACENT_PATCH_TRAIN"
        or report["status"] != "SCORED_TRAIN_DEVELOPMENT_ONLY"
        or capacity["status"] != "MEASURED_FINITE_TRAIN_CAPACITY_NOT_ONLINE"
        or h(pp) != seal["protocol_sha256"]
        or h(sp) != report["seal_sha256"]
        or h(root / "curves.npz") != seal["curves_sha256"]
        or not all(
            h(p) == sha
            for p, sha in {**cfg["inputs"], **capacity["input_sha256"]}.items()
        )
    ):
        raise ValueError("expected stable paired experiment receipts")
    cap_rows = capacity["rows"]
    reach, stats = cap.summarize(cap_rows)
    if reach != capacity["reachable"] or any(
        {**v, "best": {str(s): q for s, q in v["best"].items()}}
        != capacity["statistics"][k]
        for k, v in stats.items()
    ):
        raise ValueError("late summary differs from the sealed witnesses")
    examples, hdf = recount(root, cfg, seal, report, h, {q["shot"] for q in cap_rows})
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    base, _ = study.r.load_parent(study.r.DEFAULT_PARENT)
    if study.r.state_digest(base) != cfg["source_state_sha256"]:
        raise ValueError("frozen source hash changed")
    projection = study.op.basis(base)
    witness = original_model_witnesses(base, projection, examples, hdf, cap_rows)
    h.check()
    old_reach = study.r.read(study.PREVIOUS / "late_capacity.json")["reachable"]
    old_ids = sorted({i for arm in old_reach.values() for seed in arm for i in seed})
    decision = cap.gate(report["summary"]["windows"], reach, old_ids)
    study.r.write_new(
        out,
        dict(
            status="RECOUNTED_ORIGINAL_HDF5_AND_MODEL",
            n_shots=len(seal["shots"]),
            witnesses=witness,
            gate=decision,
            source_result_sha256=h(root / "result.json"),
            capacity_sha256=h(capacity_path),
            new_val_test_east_reads=0,
        ),
    )
    print(dict(witnesses=witness, gate=decision), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("capacity", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    run(a.root, a.capacity, a.out)
