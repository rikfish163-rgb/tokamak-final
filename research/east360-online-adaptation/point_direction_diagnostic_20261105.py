"""Post-run direction/lead diagnostic for the verified point feedback replay."""

from pathlib import Path

import numpy as np
import torch

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import (
    prepare_actual,
)
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    load_parent,
)
from joint_lora_hindsight_capacity_20261104 import read, sha, write_new
from local_affine_reach_20261105 import COLD
import trajectory_policy_20261105 as policy
from trajectory_replay_20261105 import features_at_arrival, make_views

ROOT = io23.ROOT / "probe_outputs"
POINT = ROOT / "trajectory_point_attribution_20261105_v1"
PARENT = ROOT / "trajectory_teaching360_20261105_v1"
AUDIT = ROOT / "trajectory_point_attribution_verification_20261105_v1.json"
OUT = ROOT / "point_direction_diagnostic_20261105_v1.json"


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.set_num_threads(1)
    audited, config, done = (
        read(AUDIT),
        read(POINT / "protocol.json"),
        read(POINT / "completed.json"),
    )
    if (
        audited["status"] != "PASS_RECOUNT_LOCKED_POINT_ATTRIBUTION"
        or audited["producer_completion_sha256"] != sha(POINT / "completed.json")
        or any(sha(POINT / k) != h for k, h in done["artifact_sha256"].items())
    ):
        raise ValueError("verified complete point replay required")
    base, parent = load_parent(DEFAULT_PARENT)
    threshold = config["threshold"]
    if parent["threshold"] != threshold:
        raise ValueError("unchanged parent threshold required")
    tau = np.log(threshold / (1 - threshold))
    mapping = read(COLD / "input_map.json")
    if (
        sha(COLD / "input_map.json")
        != config["input_sha256"][str(COLD / "input_map.json")]
    ):
        raise ValueError("unchanged input mapping required")
    fit = PARENT / "fit.npz"
    if sha(fit) != config["input_sha256"][str(fit)]:
        raise ValueError("fixed point direction changed")
    with np.load(fit, allow_pickle=False) as f:
        scale, matrix = [
            torch.from_numpy(f["point_real_" + k]).float() for k in ("scale", "matrix")
        ]
    cases = read(POINT / "per_shot.json")
    rescued = set(audited["paired"]["static"]["tp_gained"])
    rows, all_angles = [], []
    for rank, (sid, receipt) in enumerate(
        zip(config["order"][100:], done["sealed"], strict=True), 101
    ):
        stem = POINT / "shots" / f"{rank:04d}_{sid}"
        path, seal = stem.with_suffix(".npz"), stem.with_name(stem.name + "_seal.json")
        if sha(path) != receipt["curve_sha256"] or sha(seal) != receipt["seal_sha256"]:
            raise ValueError("sealed point curves changed")
        if sha(io23.path_for("EAST", sid)) != read(seal)["raw_sha256"]:
            raise ValueError("source raw record changed")
        raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
        signal = prepare_actual(raw, rank, mapping)
        with np.load(path, allow_pickle=False) as f:
            t, ref, real, lag = [
                f[k].copy() for k in ("times", "static", "point_real", "point_lagged")
            ]
        if not np.array_equal(t, raw["times"]):
            raise ValueError("signal clock changed")
        x, a = make_views(signal, "cpu")
        features = features_at_arrival(base, x, a, ref, tau)
        dr, dl = [policy.action(features[k], scale, matrix) for k in ("real", "lagged")]
        pair_norm = dr.norm(dim=1) * dl.norm(dim=1)
        if not (pair_norm > 0).all() or not torch.isfinite(pair_norm).all():
            raise ValueError("audited nonzero states required")
        angle = ((dr * dl).sum(1) / pair_norm).numpy()
        all_angles.append(angle)
        r, delayed, baseline = [
            cases[k][rank - 1] for k in ("point_real", "point_lagged", "static")
        ]
        common_tp = r["status"] == delayed["status"] == "tp"
        available = (np.arange(len(ref)) >= policy.FIRST) & (
            abs(ref - threshold) <= 0.1
        )
        a, b = real[available] - ref[available], lag[available] - ref[available]
        corr = (
            float(np.corrcoef(a, b)[0, 1])
            if len(a) > 1 and a.std() > 0 and b.std() > 0
            else None
        )
        info = dict(
            shot=sid,
            rank=rank,
            arrivals=len(angle),
            median_cosine=float(np.median(angle)),
            near_margin_points=len(a),
            near_margin_correlation=corr,
            common_tp=common_tp,
            lead_delta_ms=round(r["lead_ms"] - delayed["lead_ms"], 9)
            if common_tp
            else None,
            capped_both=bool(
                common_tp and max(r["lead_ms"], delayed["lead_ms"]) <= 3000
            ),
        )
        if sid in rescued:
            ok = np.round((r["down"] - t) * 1000, 9) > 30
            info["rescue"] = dict(
                static_lead=baseline["lead_ms"],
                real_lead=r["lead_ms"],
                lagged_lead=delayed["lead_ms"],
                static_margin=float(ref[ok].max() - threshold),
                real_margin=float(real[ok].max() - threshold),
                lagged_margin=float(lag[ok].max() - threshold),
                direction_cosine_at_real_alarm=float(
                    angle[(r["first_index"] - policy.FIRST) // policy.STRIDE]
                ),
            )
        rows.append(info)
    angles = np.concatenate(all_angles)
    both = np.array([r["lead_delta_ms"] for r in rows if r["common_tp"]], float)
    capped = np.array([r["lead_delta_ms"] for r in rows if r["capped_both"]], float)

    def summarize(d):
        return dict(
            n=len(d),
            better=int((d > 0).sum()),
            worse=int((d < 0).sum()),
            same=int((d == 0).sum()),
            quantiles=np.quantile(d, [0, 0.25, 0.5, 0.75, 1]).tolist(),
        )

    result = dict(
        status="PASS_POSTRUN_POINT_DIRECTION_DIAGNOSTIC",
        audit_sha256=sha(AUDIT),
        script_sha256=sha(Path(__file__)),
        n_shots=len(rows),
        compared_arrival_pairs=len(angles),
        cosine_quantiles=np.quantile(angles, [0, 0.25, 0.5, 0.75, 1]).tolist(),
        median_shot_near_margin_function_correlation=float(
            np.median(
                [
                    r["near_margin_correlation"]
                    for r in rows
                    if r["near_margin_correlation"] is not None
                ]
            )
        ),
        paired_lead_ms=summarize(both),
        paired_capped_lead_ms=summarize(capped),
        rescues=[r for r in rows if "rescue" in r],
        per_shot=rows,
    )
    write_new(OUT, result)
    print({k: v for k, v in result.items() if k != "per_shot"}, flush=True)


if __name__ == "__main__":
    run()
