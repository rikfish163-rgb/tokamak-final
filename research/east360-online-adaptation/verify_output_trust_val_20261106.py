"""Independent checks of sealed whole-source forecasts before HDF5 scoring."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import run_output_trust_val_20261106 as replay
from verify_confirmation_raw_replay_20261105 import Hashes


fit, bank, r, SEEDS, ARMS = replay.fit, replay.bank, replay.r, replay.SEEDS, replay.ARMS
OLD = r.ROOT / "probe_outputs/whole_block_update_20261106_v2/val"


def preflight(root, h):
    fit.checked_fit(root, h)
    out = root / "val"
    pp, cp = out / "protocol.json", out / "completed.json"
    cfg, seal = r.read(pp), r.read(cp)
    previous = r.read(OLD / "completed.json")
    prior_rows = previous["rows"]
    if (
        cfg["status"] != "LOCKED_BLIND_OUTPUT_TRUST_VAL"
        or seal["status"] != "SEALED_OUTPUT_TRUST_VAL"
        or any(q["held_outcome_reads"] for q in (cfg, seal))
        or h(pp) != seal["protocol_sha256"]
        or h(root / "fit_completed.json") != cfg["fit_sha256"]
        or h(root / "protocol.json") != cfg["source_protocol_sha256"]
        or tuple(cfg["arms"]) != ARMS
        or tuple(cfg["seeds"]) != SEEDS
        or cfg["threshold"] != r.THRESHOLD
        or [q["shot"] for q in seal["rows"]] != cfg["val"]
        or [q["shot"] for q in prior_rows] != cfg["val"]
        or r.simple_data.source_roles()["val"] != cfg["val"]
        or len(cfg["val"]) != 160
        or not all(h(p) == v for p, v in cfg["input_sha256"].items())
    ):
        raise ValueError("unchanged blind validation cohort, fit and protocol required")
    for rank, (p, old) in enumerate(zip(seal["rows"], prior_rows, strict=True), 1):
        sid = p["shot"]
        if (
            (p["rank"], p["file"]) != (rank, f"{rank:04d}_{sid}.npz")
            or p["raw_sha256"] != old["raw_sha256"]
            or h(r.io23.path_for("J-TEXT", sid)) != p["raw_sha256"]
            or h(out / p["file"]) != p["sha256"]
            or h(OLD / old["file"]) != old["sha256"]
        ):
            raise ValueError("source validation evidence changed before scoring")
    return cfg, seal, prior_rows


def check_trace(a, previous):
    t, z, m, choice, norms, offset, past, stale = (
        a[k]
        for k in (
            "times",
            "logits",
            "maturity",
            "action",
            "norms",
            "anchor",
            "past",
            "stale_supported",
        )
    )
    n, s, k = len(t), len(SEEDS), len(ARMS)
    dims = (len(m), s, k - 1)
    if (
        not np.array_equal(t, previous["time"])
        or z.shape != (n, s, k)
        or any(a.shape != dims for a in (choice, norms, offset, past))
        or stale.shape != dims[:2]
        or stale.dtype != bool
        or choice.dtype.kind not in "iu"
        or not all(np.isfinite(a).all() for a in (z, norms, offset, past))
        or ((choice < -1) | (choice >= len(bank.ARMS))).any()
        or ((m < 63) | (m >= n - 1) | ((m - 63) % 32 != 0)).any()
        or (np.diff(m) <= 0).any()
        or (past[:, :, 1:] > bank.tr.LIMIT + 1e-7).any()
        or (norms > bank.SITE_NORM + 2e-6).any()
        or (norms < 0).any()
        or not np.array_equal(z[:, :, 0], previous["logits"][:, :, 0])
        or not np.array_equal(choice[:, :, 3], choice[:, :, -1])
        or not np.array_equal(
            choice[:, :, 5], np.where(choice[:, :, 3] >= 0, choice[:, :, 3] ^ 1, -1)
        )
        or not np.array_equal(choice[:, :, 3][~stale], choice[:, :, 4][~stale])
        or not np.allclose(norms[:, :, 3], norms[:, :, 5], atol=1e-7, rtol=0)
        or (norms[choice < 0] != 0).any()
        or (offset[choice < 0] != 0).any()
        or (norms[:, :, -1] != 0).any()
    ):
        raise ValueError("inconsistent time, action, trust, stale, or frozen control")
    allowed = np.zeros((n, s, k - 1), dtype=bool)
    for row, v in enumerate(m):
        q = slice(v + 1, v + 33)
        allowed[q] = choice[row] >= 0
        if not np.allclose(
            z[q, :, -1], z[q, :, 0] + offset[row, :, -1], atol=8e-7, rtol=0
        ):
            raise ValueError("equal-score control used another action or clock")
    pure = np.broadcast_to(z[:, :, :1], z[:, :, 1:].shape)
    if not np.array_equal(z[:, :, 1:][~allowed], pure[~allowed]):
        raise ValueError("changed frozen scores before causal parameter release")
    return t, z


def gate(agg):
    rows = agg["by_seed"]
    main, static = ([v[k] for v in rows.values()] for k in ("guarded", "static"))
    pair = list(zip(main, static, strict=True))
    num = {k: sum(v[k]["timely"] - v[k]["fp"] for v in rows.values()) for k in ARMS}
    criteria = dict(
        no_fewer_timely=all(a["timely"] >= b["timely"] for a, b in pair),
        no_more_fp=all(a["fp"] <= b["fp"] for a, b in pair),
        at_least_two_seeds_strict=sum(
            a["timely"] > b["timely"] or a["fp"] < b["fp"] for a, b in pair
        )
        >= 2,
        beats_mechanistic_controls=all(
            num["guarded"] > num[k]
            for k in ("stale_guarded", "reversed_guarded", "score_guarded")
        ),
    )
    return dict(passed=all(criteria.values()), criteria=criteria, net_counts=num)


def witness(root, selected):
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    base, _ = r.load_parent(r.DEFAULT_PARENT)
    diag, heads = bank.frozen_fits(base), fit.load_heads(root / "heads.npz")
    checks = []
    with patch.object(
        r.io23, "read_outcome", side_effect=RuntimeError("CPU prefix stays blind")
    ):
        for q in selected:
            raw = r.simple_data.read_shot("J-TEXT", q["shot"])
            cpu = replay.predict(base, diag, heads, raw)
            with np.load(root / "val" / q["file"], allow_pickle=False) as a:
                err = float(np.abs(cpu["logits"] - a["logits"]).max())
                identical = np.array_equal(
                    cpu["logits"].astype(np.float64) >= fit.TAU,
                    a["logits"].astype(np.float64) >= fit.TAU,
                )
                if (
                    err > 1e-4
                    or not identical
                    or not np.array_equal(cpu["action"], a["action"])
                ):
                    raise ValueError(
                        "independent CPU replay differs from sealed CUDA path"
                    )
            checks.append(
                dict(shot=q["shot"], error=err, threshold_bits_equal=identical)
            )
    return checks


def verify(root):
    h, start = Hashes(), time.monotonic()
    cfg, seal, old = preflight(root, h)
    out = root / "val" / "audit.json"
    if out.exists():
        raise FileExistsError(out)
    retained = []
    diagnostic = {
        k: 0.0 for k in ("max_guarded_future_change", "max_guarded_reached_change")
    }
    picks = np.zeros((len(SEEDS), len(ARMS) - 1), dtype=np.int64)
    for p, q in zip(seal["rows"], old, strict=True):
        with (
            np.load(root / "val" / p["file"], allow_pickle=False) as a,
            np.load(OLD / q["file"], allow_pickle=False) as b,
        ):
            t, z = check_trace(a, b)
            retained.append((p, t, z))
            picks += (a["action"] >= 0).sum(0)
            diagnostic["max_guarded_future_change"] = max(
                diagnostic["max_guarded_future_change"],
                float(np.abs(z[:, :, ARMS.index("guarded")] - z[:, :, 0]).max()),
            )
            if len(a["past"]):
                diagnostic["max_guarded_reached_change"] = max(
                    diagnostic["max_guarded_reached_change"],
                    float(a["past"][:, :, ARMS.index("guarded") - 1].max()),
                )
    checked = witness(root, [q for q in seal["rows"] if q["rank"] in (3, 56, 119)])
    h.check()
    adv, scored = r.simple_data.advanced_times(), []
    for p, t, z in retained:
        sid = p["shot"]
        flag, down = r.io23.read_outcome("J-TEXT", sid)
        rec = dict(
            shot=sid,
            disruptive=flag,
            down=down,
            horizon_ms=r.io23.effective_horizon_ms(sid, adv)[0],
        )
        scored.append(fit.score_shot(rec, t, z))
    agg = fit.summarize(scored)
    g = gate(agg)
    h.check()
    r.write_new(
        out,
        dict(
            status="RECOUNTED_OUTPUT_TRUST_VAL",
            seal_sha256=h(root / "val" / "completed.json"),
            auditor_sha256=h(Path(__file__)),
            aggregate=agg,
            east_gate=g,
            selected=picks.tolist(),
            cpu_witness=checked,
            prefix=seal["prefix"],
            **diagnostic,
            rows=scored,
            seconds=time.monotonic() - start,
        ),
    )
    print("EAST gate", g, flush=True)
    print(
        "VAL scores",
        {
            s: {k: (q["timely"], q["fp"]) for k, q in v.items()}
            for s, v in agg["by_seed"].items()
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    verify(p.parse_args().root)
