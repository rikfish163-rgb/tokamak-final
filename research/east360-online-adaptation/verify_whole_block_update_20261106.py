"""Verify every completed seal, then independently reveal the held HDF5 outcomes."""

import argparse
from pathlib import Path
import time

import numpy as np
from scipy.special import expit

import audit_whole_block_update_20261106 as audit
import moving_peak_replay_20261105 as r
import run_whole_block_update_20261106 as run
from verify_confirmation_raw_replay_20261105 import Hashes


FIELDS = ("time", "logits", "active", "maturity", "common", "norms", "cos", "offsets")


def preflight(root, h):
    pp, ff = root / "protocol.json", root / "fit_completed.json"
    cp = root / "val" / "completed.json"
    protocol, fit, seal, done = (
        r.read(p) for p in (pp, ff, cp, root / "source_done.json")
    )
    prior, original_train = run.source_inputs(h)
    inv = root / "train_inventory.json"
    train = r.read(inv)["rows"]
    sha, f_sha = h(pp), h(ff)
    if (
        protocol["status"] != "LOCKED_WHOLE_BLOCK_PROTOCOL"
        or fit["status"] != "WHOLE_BLOCK_TRAIN_ONLY_FIT"
        or done["status"] != "COMPLETE_WHOLE_BLOCK_SOURCE"
        or seal["status"] != "SEALED_WHOLE_BLOCK_VAL"
        or seal["held_outcome_reads"] != 0
        or protocol["source_test_reads"] != 0
        or done["east_performance_replay"]
        or tuple(protocol["arms"]) != audit.ARMS
        or audit.ARMS != run.ARMS
        or tuple(protocol["seeds"]) != audit.SEEDS
        or audit.SEEDS != run.SEEDS
        or protocol["threshold"] != prior["threshold"]
        or protocol["roles"] != prior["roles"]
        or protocol["state_sha256"] != prior["source_state_sha256"]
        or set(protocol["roles"]["train"]) & set(protocol["roles"]["val"])
        or set(protocol["roles"]["train"]) & set(protocol["roles"]["test"])
        or set(protocol["roles"]["val"]) & set(protocol["roles"]["test"])
        or fit["fit_steps"] != run.STEPS
        or not all(p["protocol_sha256"] == sha for p in (fit, done, seal))
        or not all(p["fit_sha256"] == f_sha for p in (done, seal))
        or done["seal_sha256"] != h(cp)
        or not all(h(p) == v for p, v in protocol["input_sha256"].items())
        or not all(
            h(root / f"fit_{s}.npz") == fit["fits"][str(s)]["sha256"]
            for s in audit.SEEDS
        )
        or fit["train_inventory_sha256"] != h(inv)
        or fit["train_feasibility_sha256"] != h(root / "train_feasibility.json")
        or [q["shot"] for q in train] != protocol["roles"]["train"]
        or not all(
            q["raw_sha256"] == old["sha256"] == h(r.io23.path_for("J-TEXT", q["shot"]))
            for q, old in zip(train, original_train, strict=True)
        )
        or [q["shot"] for q in seal["rows"]] != protocol["roles"]["val"]
        or not any(
            q["active"] > 0 and q["error"] <= run.PREFIX_ATOL for q in seal["prefix"]
        )
    ):
        raise ValueError("unbound source fit, split, protocol or prediction seal")
    ref = r.read(run.PREVIOUS / "val" / "completed.json")
    if [q["shot"] for q in ref["rows"]] != protocol["roles"]["val"]:
        raise ValueError("legacy comparison has a different VAL cohort")
    for rank, (q, b) in enumerate(zip(seal["rows"], ref["rows"], strict=True), 1):
        sid = q["shot"]
        if (
            (q["rank"], q["file"]) != (rank, f"{rank:04d}_{sid}.npz")
            or h(root / "val" / q["file"]) != q["sha256"]
            or h(run.PREVIOUS / "val" / b["file"]) != b["sha256"]
            or h(r.io23.path_for("J-TEXT", sid)) != q["raw_sha256"]
            or q["raw_sha256"] != b["raw_sha256"]
        ):
            raise ValueError(f"archive provenance failed before reveal: {sid}")
    h.check()
    return protocol, seal, ref


def verify_arrays(a, original, times, available):
    z = a["logits"]
    n = len(times)
    maturity = np.arange(63, n, 32, dtype=np.int64)
    m, s = len(maturity), len(audit.SEEDS)
    common, active = a["common"], a["active"]
    if (
        not np.array_equal(a["time"], times)
        or not np.array_equal(maturity, a["maturity"])
        or z.shape != (n, s, len(audit.ARMS))
        or z.dtype != np.float32
        or common.shape != (m,)
        or active.shape != (n,)
        or common.dtype != bool
        or active.dtype != bool
        or a["norms"].shape != (m, s, 3)
        or a["cos"].shape != (m, s)
        or a["offsets"].shape != (m, s)
        or not all(
            np.isfinite(a[k]).all() for k in ("logits", "norms", "cos", "offsets")
        )
    ):
        raise ValueError("invalid sealed scores, clock, or state shapes")
    names = [3, 4, 5, 14, 15, 16, 17, 19, 20, 21, 22]
    k = (n // 32) * 32
    count = available[:k, names].reshape(-1, 32, 11).sum(1)
    valid = (count[:-1] >= 16) & (count[1:] >= 16)
    support = (
        np.stack(
            [valid[:, lo:hi].any(1) for lo, hi in ((0, 3), (3, 7), (7, 11))], 1
        ).sum(1)
        >= 2
    )
    slots = np.arange(n) // 32 - 2
    expected = np.zeros(n, dtype=bool)
    expected[slots >= 0] = common[slots[slots >= 0]]
    native = z[:, 0, 0]
    old_diff = float(np.max(np.abs(z[:, :, :2] - original["logits"][:, :, :2])))
    if (
        (common & ~support).any()
        or not np.array_equal(common, original["common"])
        or not np.array_equal(active, expected)
        or not np.array_equal(active, original["active"])
        or not np.array_equal(z[:, :, 0], np.broadcast_to(native[:, None], (n, s)))
        or not np.array_equal(
            z[~active], np.broadcast_to(native[~active, None, None], z[~active].shape)
        )
        or old_diff > 8e-5
        or (a["norms"] < 0).any()
        or (a["norms"] > 0.5 + 2e-6).any()
        or (a["norms"][~common] != 0).any()
        or (a["offsets"][~common] != 0).any()
        or (np.abs(a["cos"]) > 1 + 2e-6).any()
        or not np.allclose(
            z[active, :, 5],
            native[active, None] + a["offsets"][slots[active]],
            atol=8e-7,
            rtol=0,
        )
    ):
        raise ValueError(
            "independent clock, support, legacy arm or score control differs"
        )
    return old_diff


def verify(root, out):
    h, start = Hashes(), time.monotonic()
    if out.exists():
        raise FileExistsError(out)
    scorer_sha, auditor_sha = h(Path(audit.__file__)), h(Path(__file__))
    protocol, seal, prior = preflight(root, h)
    arrays, max_legacy = [], 0.0
    for new, old in zip(seal["rows"], prior["rows"], strict=True):
        raw = r.simple_data.read_shot("J-TEXT", new["shot"])
        with (
            np.load(root / "val" / new["file"], allow_pickle=False) as a,
            np.load(run.PREVIOUS / "val" / old["file"], allow_pickle=False) as b,
        ):
            diff = verify_arrays(a, b, raw["times"], raw["available"])
            max_legacy = max(max_legacy, diff)
            arrays.append({name: a[name] for name in FIELDS})
    h.check()
    advance, rows = r.simple_data.advanced_times(), []
    auxiliaries, altered = [], []
    for pred, receipt in zip(arrays, seal["rows"], strict=True):
        sid = receipt["shot"]
        flag, down = r.io23.read_outcome("J-TEXT", sid)
        horizon = r.io23.effective_horizon_ms(sid, advance)[0]
        times, z = pred["time"], pred["logits"]
        row = audit.per_shot(
            times, z, pred["active"], flag, down, horizon, protocol["threshold"], 10.0
        )
        before = (
            (down - times) * 1000 >= -1e-6 if flag else np.ones(len(times), dtype=bool)
        )
        for i, seed in enumerate(audit.SEEDS):
            for j, arm in enumerate(audit.ARMS):
                x = row["seeds"][str(seed)][arm]
                hits = np.flatnonzero(
                    before
                    & (expit(z[:, i, j].astype(np.float64)) >= protocol["threshold"])
                )
                first = int(hits[0]) if len(hits) else None
                if (
                    first != x["first_index"]
                    or x["fp"] != (not flag and first is not None)
                    or x["raw_tp"]
                    != (
                        flag
                        and first is not None
                        and round(float((down - times[first]) * 1000), 9) > 10
                    )
                ):
                    raise ValueError(f"independent actual first alarm disagrees: {sid}")
        row["shot"] = sid
        rows.append(row)
        aux = audit.prediction_diagnostics(
            pred["maturity"],
            pred["common"],
            pred["norms"],
            pred["cos"],
            pred["offsets"],
        )
        auxiliaries.append(aux)
        counts = {
            seed: {
                arm: (
                    value["first_index"],
                    value["first_lead_ms"],
                    value["raw_tp"],
                    value["fp"],
                )
                for arm, value in group.items()
            }
            for seed, group in row["seeds"].items()
        }
        if any(
            count[arm][2:] != count["static"][2:]
            for count in counts.values()
            for arm in audit.ARMS[1:]
        ):
            altered.append(dict(shot=sid, disruptive=flag, arms=counts))
    summary = audit.summarize(rows)
    h.check()
    r.write_new(
        out,
        dict(
            status="INDEPENDENT_HDF5_VAL_RECOUNT",
            seal_sha256=h(root / "val" / "completed.json"),
            scorer_sha256=scorer_sha,
            auditor_sha256=auditor_sha,
            aggregate=summary,
            altered=altered,
            prefix=seal["prefix"],
            max_legacy_logit_error=max_legacy,
            n_common=sum(q["n_common"] for q in auxiliaries),
            n_active=sum(int(q["active"].sum()) for q in arrays),
            seconds=time.monotonic() - start,
        ),
    )
    print(summary["source_gate"], "legacy max difference", max_legacy, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    verify(a.root, a.output or a.root / "val_audit.json")
