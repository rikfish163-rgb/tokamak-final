"""Frozen J-TEXT TEST220 diagnostic-availability audit, with no online updates."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from . import io23, simple_data
from .causal_input import apply_input_contract
from .context_vector_20261030.source_run import source_gate
from .east_context_operator_20261031.run import configure
from .history_gradient_metric_20261102.run import read, sha, write
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from .source_event_tasks_20261102.context_pair import imported_code_hashes

ARMS = ("qualified_native", "contract_native", "no_MNM", "no_CIII", "no_MNM_CIII")


def diagnostic_view(signal, arm):
    if set(signal) != {"values", "available"} or arm not in ARMS:
        raise ValueError("signal-only flat23 and declared static mask view required")
    view = (
        {k: v.copy() for k, v in signal.items()}
        if arm == "qualified_native"
        else apply_input_contract(signal)
    )
    if arm in ("no_MNM", "no_MNM_CIII"):
        view["available"][:, 2] = False
    if arm in ("no_CIII", "no_MNM_CIII"):
        view["available"][:, 1] = False
    return view


def score_shot(sid, times, scores, flag, down, threshold):
    hits = np.flatnonzero(np.asarray(scores) >= threshold)
    lead = (
        round(float((down - times[hits[0]]) * 1000), 9) if flag and len(hits) else None
    )
    return dict(
        shot=int(sid),
        is_disrupt=bool(flag),
        alarm=bool(len(hits)),
        lead_ms=lead,
        timely=bool(lead is not None and lead > 10),
    )


def metrics(rows):
    nd = sum(r["is_disrupt"] for r in rows)
    ns = len(rows) - nd
    tp = sum(r["timely"] for r in rows)
    fp = sum(r["alarm"] and not r["is_disrupt"] for r in rows)
    leads = [r["lead_ms"] for r in rows if r["timely"]]
    return dict(
        tp=tp,
        fp=fp,
        n_disrupt=nd,
        n_safe=ns,
        tpr=tp / nd,
        fpr=fp / ns,
        J=tp / nd - fp / ns,
        median_lead_ms=float(np.median(leads)) if leads else None,
        early_over3000=sum(v > 3000 for v in leads),
    )


def main(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    configure()
    started = time.monotonic()
    model, parent = load_parent(DEFAULT_PARENT)
    model = model.cuda().eval().requires_grad_(False)
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ids = io23.source_roles()["test"]
    historical = read(DEFAULT_PARENT.parent / "test220.json")["jtext"]
    old = {r["shot"]: r for r in historical["predictions"]}
    if set(old) != set(ids):
        raise ValueError("qualified historical TEST220 identity changed")
    threshold = float(parent["threshold"])
    if threshold != historical["threshold"]:
        raise ValueError("qualified historical threshold changed")
    output.mkdir()
    (output / "shots").mkdir()
    code = imported_code_hashes()
    write(
        output / "protocol.json",
        dict(
            scope="FROZEN_SOURCE_TEST220_DIAGNOSTIC_MASK_ABLATION_NOT_ONLINE",
            ids=ids,
            arms=list(ARMS),
            source_sha256=sha(DEFAULT_PARENT),
            threshold=threshold,
            historical_result_sha256=sha(DEFAULT_PARENT.parent / "test220.json"),
            code_sha256=code,
            source_parameter_updates=0,
            jtext_online_updates=0,
            east_reads=0,
            batch_size=256,
            unit_contract="J-TEXT ip multiplied1000 once; no EAST map",
            native="original finite availability; no CIII sentinel intervention",
            other_views="same model/scaler/threshold; current pointwise CIII sentinel mask then named dropped channels",
            strict_timely_lead_ms=10,
            hard_outcome_authority="HDF5 after curve seal",
            no_test_threshold_selection=True,
            independent_confirmation=False,
        ),
    )
    rows = {a: [] for a in ARMS}
    hashes = {}
    parity = []
    availability = []
    with source_gate(ids):
        for index, sid in enumerate(ids, 1):
            raw = simple_data.read_shot("J-TEXT", sid)
            hashes[str(sid)] = sha(io23.path_for("J-TEXT", sid))
            signal = {k: raw[k] for k in ("values", "available")}
            curves = {}
            with (
                torch.no_grad(),
                patch.object(
                    io23,
                    "read_outcome",
                    side_effect=RuntimeError("no truth in prediction"),
                ),
            ):
                for arm in ARMS:
                    view = diagnostic_view(signal, arm)
                    probs = []
                    for lo in range(0, len(raw["times"]), 256):
                        windows = [
                            simple_data.window(view, j, 128)
                            for j in range(lo, min(lo + 256, len(raw["times"])))
                        ]
                        x, a = zip(*windows, strict=True)
                        xx = torch.as_tensor(np.stack(x), device="cuda")
                        aa = torch.as_tensor(np.stack(a), device="cuda")
                        probs.append(model(xx, aa)["logit"].sigmoid().cpu().numpy())
                    curves[arm] = np.concatenate(probs)
                    if not np.isfinite(curves[arm]).all():
                        raise FloatingPointError("nonfinite frozen-source prediction")
                    availability.append(
                        dict(
                            shot=sid,
                            arm=arm,
                            observed=view["available"].sum(0).tolist(),
                            samples=len(raw["times"]),
                        )
                    )
            path = output / "shots" / f"{index:04d}_{sid}.npz"
            np.savez_compressed(path, times=raw["times"], **curves)
            write(
                path.with_name(path.stem + "_seal.json"),
                dict(
                    shot=sid,
                    curve_sha256=sha(path),
                    current_outcome_used_in_prediction=False,
                ),
            )
            flag, down = io23.read_outcome("J-TEXT", sid)
            for arm in ARMS:
                rows[arm].append(
                    score_shot(sid, raw["times"], curves[arm], flag, down, threshold)
                )
            new = rows["qualified_native"][-1]
            ref = old[sid]
            same = all(new[k] == ref[k] for k in ("is_disrupt", "alarm", "timely"))
            same &= (new["lead_ms"] is None and ref["lead_ms"] is None) or (
                new["lead_ms"] is not None
                and ref["lead_ms"] is not None
                and abs(new["lead_ms"] - ref["lead_ms"]) <= 1e-6
            )
            parity.append(
                dict(shot=sid, matched=bool(same), current=new, historical=ref)
            )
            if index % 40 == 0:
                print(
                    dict(
                        stage="frozen_source_static_views",
                        shots=index,
                        total=len(ids),
                        seconds=time.monotonic() - started,
                    ),
                    flush=True,
                )
    m = {a: metrics(rs) for a, rs in rows.items()}
    if any(not torch.equal(v, model.state_dict()[k].cpu()) for k, v in frozen.items()):
        raise AssertionError("qualified source tensor changed")
    if any(sha(Path(p)) != h for p, h in code.items()):
        raise AssertionError("bound code changed")
    write(output / "per_shot.json", rows)
    write(output / "summary.json", m)
    write(output / "historical_parity.json", parity)
    write(output / "input_hashes.json", hashes)
    write(output / "availability.json", availability)
    completed = dict(
        status="COMPLETED_STATIC_SOURCE_AUDIT"
        if all(r["matched"] for r in parity)
        else "STOP_PARENT_REPRODUCTION_MISMATCH",
        shots=len(ids),
        arms=len(ARMS),
        source_unchanged=True,
        jtext_online_run=False,
        optimizer_steps=0,
        historical_mismatches=[r["shot"] for r in parity if not r["matched"]],
        seconds=time.monotonic() - started,
        metrics=m,
        artifact_sha256={f.name: sha(f) for f in output.iterdir() if f.is_file()},
    )
    write(output / "completed.json", completed)
    print(completed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
