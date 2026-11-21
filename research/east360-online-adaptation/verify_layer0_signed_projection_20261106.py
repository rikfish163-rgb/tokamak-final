"""Recount the sealed six-arm replay using independent HDF5 truth and clocks."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from verify_confirmation_raw_replay_20261105 import (
    Hashes,
    RAW_ROOT,
    counts,
    need,
    pair,
    raw_truth,
    read,
    score,
)
from verify_moving_peak_replay_20261105 import ATOL, check_curve, parity


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "probe_outputs"
REFS = {
    "fixed": (ARCHIVE / "layer_interaction_masks360_20261105_v1", 360, 21, "layer0"),
    "conditional": (ARCHIVE / "layer0_conditional835_20261106_v1", 835, 1, "intra"),
}
ARMS = ("static", "intra", "signed", "shrink_match", "lagged_match", "blind_signed")
RECEIPTS = (
    "anchor_supported",
    "anchor_positive",
    "anchor_zero",
    "anchor_later_up",
    "anchor_native_logit",
    "anchor_initial_logit",
    "anchor_projected_logit",
    "anchor_norms",
)
SOURCE = "99affd49f71f41129b87b03624201ef94ef6dc44e652eaef411b2924f8186dc8"
FIT = "eaf4c97f7aa4a829db8058ea879ff0677c092435e3cd30d38a1ab22127639eac"
MAP = "43c37ad981261e059ac8bf8d11d89324bad375dd7405decc194c51e53614aff2"
FIRST, STRIDE, EPS, TAU = 135, 32, 1e-5, 0.7258946895599365


def audit_anchors(a):
    native, intra, signed, blind = (
        a[k] for k in ("static", "intra", "signed", "blind_signed")
    )
    idx = np.arange(FIRST, len(native), STRIDE)
    n = len(idx)
    allow, positive, zero, later = (a[k] for k in RECEIPTS[:4])
    base, start, final, norms = (a[k].astype(np.float64) for k in RECEIPTS[4:])
    need(
        all(
            v.shape == (n,) and v.dtype.kind == "b"
            for v in (allow, positive, zero, later)
        )
        and all(v.shape == (n,) and np.isfinite(v).all() for v in (base, start, final))
        and norms.shape == (n, 5)
        and np.isfinite(norms).all(),
        "invalid anchor receipt shapes or values",
    )
    guarded = positive & ~allow
    untouched = ~guarded
    need(
        np.array_equal(positive, start - base > EPS)
        and np.array_equal(start[~positive], final[~positive])
        and np.all(final[positive] - base[positive] <= EPS)
        and np.array_equal(zero, positive & (norms[:, 4] == 0))
        and np.array_equal(final[zero], base[zero]),
        "finite signed constraint, nonpositive preservation, or known zero failed",
    )
    need(
        ((norms >= 0) & (norms <= 2 + 1e-6)).all()
        and (norms[:, 1:] <= norms[:, :1] + 1e-6).all()
        and (np.abs(norms[:, 2:4] - norms[:, 1:2]) <= 1e-6).all()
        and np.array_equal(norms[untouched, 1], norms[untouched, 0])
        and np.array_equal(norms[~positive, 4], norms[~positive, 0])
        and np.array_equal(norms[guarded, 1], norms[guarded, 4]),
        "incorrect directional controls or changed norm at untouched anchor",
    )

    def sigmoid(z):
        return np.exp(-np.logaddexp(0, -z))

    error = max(
        float(np.max(np.abs(p[idx] - sigmoid(z)), initial=0))
        for p, z in (
            (native, base),
            (intra, start),
            (signed, np.where(guarded, final, start)),
            (blind, final),
        )
    )
    need(error <= ATOL, "anchor logit does not reproduce sealed probability")
    replayed_later = np.array(
        [
            bool(
                guarded[i]
                and np.any(
                    signed[j + 1 : j + STRIDE] > native[j + 1 : j + STRIDE] + 1e-6
                )
            )
            for i, j in enumerate(idx)
        ],
        dtype=bool,
    )
    need(
        np.array_equal(later, replayed_later),
        "subsequent upshift receipt inconsistent with curve",
    )
    return dict(
        n_anchors=n,
        supported=int(allow.sum()),
        positive=int(positive.sum()),
        supported_positive=int((positive & allow).sum()),
        guarded_positive=int(guarded.sum()),
        guarded_zero=int((guarded & zero).sum()),
        blind_zero=int(zero.sum()),
        later_upshift=int(later.sum()),
    ), error


def shot_path(sid):
    return RAW_ROOT / str(sid // 100 * 100) / f"{sid}.hdf5"


def check_bank(h, bank, cfg, cold_raw):
    receipt = read(bank / "completed.json")
    roles = receipt["original_roles"]
    need(
        receipt["status"] == "COMPLETE_COLD20_SIGNED_PHASE_BANK"
        and receipt["earliest_fixed_rank"] == 21
        and receipt["n_pairs"] == 8
        and roles == cfg["bank_roles"]
        and len(roles) == len(set(roles)) == 20
        and (
            receipt["source_state_sha256"],
            receipt["fit_sha256"],
            receipt["map_sha256"],
        )
        == (SOURCE, FIT, MAP)
        and receipt["bank_sha256"] == h(bank / "bank.npz")
        and cfg["bank_receipt_sha256"] == h(bank / "completed.json")
        and all(h(p) == sha for p, sha in receipt["input_sha256"].items()),
        "bank identity or declared label budget is invalid",
    )
    with np.load(bank / "bank.npz", allow_pickle=False) as a:
        keys = {f"{k}_{slot}" for slot in range(8) for k in ("p", "e")}
        need(set(a.files) == keys, "unrecognized phase banks")
        labels = []
        for rank, sid in enumerate(roles, 1):
            path = shot_path(sid)
            digest = h(path)
            truth, clock = raw_truth(path, sid)
            need(
                digest == cold_raw[str(sid)],
                "cold HDF5 file no longer matches fixed stream",
            )
            if truth["disruptive"]:
                j = len(labels)
                grid = np.arange(FIRST, len(clock), STRIDE)
                lead = np.round((truth["down"] - clock[grid]) * 1000.0, 9)
                p = grid[(lead > 30) & (lead <= 125)]
                early = grid[lead >= 500]
                need(0 < len(p) <= len(early), "invalid release bank time partition")
                e = early[np.linspace(0, len(early) - 1, len(p), dtype=int)]
                labels.append(
                    dict(
                        rank=rank,
                        shot=sid,
                        down=truth["down"],
                        raw_sha256=digest,
                        precursor=p.tolist(),
                        early=e.tolist(),
                        slot=j,
                    )
                )
                need(
                    all(
                        a[f"{k}_{j}"].shape == (len(p), 192)
                        and np.isfinite(a[f"{k}_{j}"]).all()
                        for k in ("p", "e")
                    ),
                    "bank keys do not cover corresponding release windows",
                )
    need(
        labels == receipt["historical_disruptive_rows"],
        "bank differs from authoritative history",
    )
    return len(labels)


def comparisons(rows):
    rivals = [("intra", "static"), *(("signed", k) for k in ARMS if k != "signed")]
    return {
        f"{a}_vs_{b}": dict(
            strict=pair(rows[a], rows[b]), capped=pair(rows[a], rows[b], capped=True)
        )
        for a, b in rivals
    }


def verify(run, bank, output):
    if output.exists():
        raise FileExistsError(output)
    h = Hashes()
    cfg, done = [read(run / name) for name in ("protocol.json", "completed.json")]
    mode, order = cfg["mode"], cfg["order"]
    ref, n, active_rank, old_arm = REFS[mode]
    fixed_cfg = read(REFS["fixed"][0] / "protocol.json")
    ref_cfg, old = [read(ref / name) for name in ("protocol.json", "completed.json")]
    deps = cfg["input_sha256"]
    required = [
        ref / "protocol.json",
        ref / "completed.json",
        bank / "completed.json",
        bank / "bank.npz",
    ]
    need(
        len(order) == len(set(order)) == n
        and order == ref_cfg["order"]
        and cfg["raw_sha256"] == ref_cfg["raw_sha256"]
        and (mode == "fixed" or not set(order) & set(fixed_cfg["order"]))
        and cfg["bank_roles"] == fixed_cfg["order"][:20]
        and tuple(cfg["arms"]) == ARMS
        and cfg["bank_active_rank"] == active_rank
        and (cfg["first"], cfg["stride"], cfg["eps_logit"], cfg["threshold"])
        == (FIRST, STRIDE, EPS, TAU)
        and cfg["threshold"] == ref_cfg["threshold"]
        and cfg["source_state_sha256"] == ref_cfg["source_state_sha256"] == SOURCE
        and cfg["source_fit_sha256"] == FIT
        and FIT in deps.values()
        and MAP in deps.values()
        and cfg["development_only"] is True
        and cfg["new_fit_labels"]
        == cfg["inter_updates"]
        == cfg["outcome_reads_during_prediction"]
        == 0
        and all(str(p.resolve()) in deps for p in required)
        and all(h(p) == sha for p, sha in deps.items()),
        "wrong cohort, causal clock, source, or input binding",
    )
    need(
        done["status"] == "COMPLETE_LAYER0_SIGNED_PROJECTION"
        and done["mode"] == mode
        and done["protocol_sha256"] == h(run / "protocol.json")
        and done["n_shots"] == len(done["sealed"]) == len(old["sealed"]) == n
        and done["n_curves"] == n * len(ARMS),
        "full completion required before any scoring",
    )
    producer_sha = h(run / "completed.json")
    n_history = check_bank(h, bank, cfg, fixed_cfg["raw_sha256"])
    rows = {k: [] for k in ARMS}
    max_error = dict(static=0.0, intra=0.0, anchor=0.0)
    anchor_counts = Counter()
    immutable_fp = []
    previous, last = None, cfg["initial_ns"]
    for rank, sid in enumerate(order, 1):
        stem = f"{rank:04d}_{sid}"
        path, seal = [run / "shots" / (stem + ext) for ext in (".npz", "_seal.json")]
        archived = ref / "shots" / path.name
        raw = shot_path(sid)
        note = read(seal)
        active = rank >= active_rank
        need(
            done["sealed"][rank - 1] == dict(note, seal_sha256=h(seal))
            and (note["rank"], note["shot"]) == (rank, sid)
            and note["raw_sha256"] == h(raw) == cfg["raw_sha256"][str(sid)]
            and note["curve_sha256"] == h(path)
            and h(archived) == old["sealed"][rank - 1]["curve_sha256"]
            and note["previous_seal_sha256"] == previous
            and last <= note["seal_ns"] <= done["completed_ns"]
            and note["bank_enabled"] is active,
            f"invalid seal, bank access or HDF5 binding for {sid}",
        )
        previous, last = h(seal), note["seal_ns"]
        truth, clock = raw_truth(raw, sid)
        with (
            np.load(path, allow_pickle=False) as a,
            np.load(archived, allow_pickle=False) as b,
        ):
            expected_keys = {"times", *ARMS, *(RECEIPTS if active else ())}
            need(
                np.array_equal(a["times"], clock)
                and np.array_equal(b["times"], clock)
                and set(a.files) == expected_keys,
                f"invalid saved clocks or fields for {sid}",
            )
            curves = {k: a[k] for k in ARMS}
            for k, old_k in (("static", "static"), ("intra", old_arm)):
                max_error[k] = max(
                    max_error[k], parity(curves[k], b[old_k], TAU, f"{sid}/{k}")
                )
            if active:
                try:
                    receipt, error = audit_anchors(
                        {**curves, **{k: a[k] for k in RECEIPTS}}
                    )
                except ValueError as exc:
                    raise ValueError(f"{sid}: {exc}") from exc
                max_error["anchor"] = max(max_error["anchor"], error)
                anchor_counts.update(receipt)
            end = FIRST if active else len(clock)
            for k, p in curves.items():
                check_curve(p, clock, k)
                need(
                    np.array_equal(p[:end], curves["static"][:end]),
                    f"future write at {sid}/{k}",
                )
                rows[k].append(score(clock, p, truth, TAU))
        baseline = rows["static"][-1]
        if baseline["status"] == "fp" and baseline["first_index"] < FIRST:
            need(
                all(
                    rows[k][-1]["first_index"] == baseline["first_index"] for k in ARMS
                ),
                "immutable FP changed",
            )
            immutable_fp.append(sid)
    expected = (180, 180) if mode == "fixed" else (510, 325)
    full = {k: counts(v) for k, v in rows.items()}
    need(
        (full["static"]["n_disruptive"], full["static"]["n_safe"]) == expected,
        "wrong raw population",
    )
    h.check()
    payload = dict(
        status="PASS_SIGNED_PROJECTION_HDF5_RECOUNT",
        producer_sha256=producer_sha,
        mode=mode,
        development_only=True,
        n_shots=n,
        n_curves=n * len(ARMS),
        bank_n_disruptive=n_history,
        bank_labels=20,
        new_fit_labels=0,
        anchor_counts=dict(anchor_counts),
        max_error=max_error,
        full=full,
        active={k: counts(v[active_rank - 1 :]) for k, v in rows.items()},
        pairs=comparisons(rows),
        immutable_prewrite_fp=immutable_fp,
        per_shot=rows,
    )
    with output.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print(f"verified {mode}: {n} shots / {n * len(ARMS)} curves", flush=True)
    for k, v in full.items():
        print(
            f"  {k}: {v['tp']} strict / {v['tp_30_to_3000']} capped / {v['fp']} FP",
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("bank", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    verify(a.run, a.bank, a.output)
