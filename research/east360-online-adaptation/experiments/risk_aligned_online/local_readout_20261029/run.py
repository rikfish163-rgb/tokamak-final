"""Five-arm full360 run with cold20 included and sealed future-signal probes."""

import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..attention_basis_20261029.learning import MatrixAttention
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import read
from ..delayed_recurrent_memory_20261025.learning import original_predictions
from ..fixed_test_protocol import score_fixed_test, validate_test_ids
from ..latent_completion_20261028.stream import compact
from ..lifecycle_stream_20261025.protocol import AccessLedger
from ..qk_matched_head_20261029.run import snapshot
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform, fit_prefix_map
from .learning import ARMS, MODES, predict, load_head

PREVIOUS = io23.ROOT / "probe_outputs/qk_forecast_east360_v1"
READOUT = io23.ROOT / "probe_outputs/readout_identifiability_v2"


def pack_aux(rows):
    if not rows:
        return {}
    return {"aux_" + k: np.stack([r[k] for r in rows]) for k in rows[0]}


def score_aux(rows, signal, valid, center, scale):
    result = []
    for row in rows:
        c = row["clock"]
        raw = signal["values"][c : c + 25]
        mask = (
            signal["available"][c : c + 25]
            & np.isfinite(raw)
            & row["available"][None]
            & valid[c : c + 25, None]
        )
        truth = np.arcsinh((np.where(mask, raw, center).astype(float) - center) / scale)
        pred = {
            k: v
            for k, v in row.items()
            if k.endswith("_before") or k.endswith("_after")
        }
        pred["hold"] = np.repeat(row["last"][None], 25, axis=0)
        losses = {}
        for name, values in pred.items():
            error = np.where(mask, values.astype(float) - truth, 0.0)
            if not np.isfinite(error).all():
                raise FloatingPointError("invalid sealed auxiliary score")
            a = np.abs(error)
            loss = np.where(a < 1, error**2 / 2, a - 0.5)
            losses[name] = dict(
                sum=float(loss.sum()), channel_sums=loss.sum(0).tolist()
            )
        result.append(
            dict(
                clock=c,
                elements=int(mask.sum()),
                channel_counts=mask.sum(0).tolist(),
                losses=losses,
            )
        )
    return result


def summarize_aux(rows):
    result = {}
    methods = ["hold"] + [
        name + part for name in MODES for part in ("_before", "_after")
    ]
    for name in methods:
        shots = []
        for r in rows:
            eligible = [x for x in r["samples"] if x["elements"]]
            if eligible:
                shots.append(
                    dict(
                        shot=r["shot"],
                        windows=len(eligible),
                        loss=float(
                            np.mean(
                                [
                                    x["losses"][name]["sum"] / x["elements"]
                                    for x in eligible
                                ]
                            )
                        ),
                    )
                )
        if not shots:
            raise ValueError("no complete eligible auxiliary samples")
        values = np.array([s["loss"] for s in shots])
        result[name] = dict(
            shots=len(shots),
            windows=sum(s["windows"] for s in shots),
            mean=float(values.mean()),
            shot_sd=float(values.std(ddof=1)),
            per_shot=shots,
        )
    return result


def run(output):
    if output.exists() or output.absolute().parent != io23.ROOT / "probe_outputs":
        raise ValueError("fresh worktree output required")
    previous = read(PREVIOUS, "protocol.json")
    pa = read(PREVIOUS, "audit.json")
    ra = read(READOUT, "audit_with_fold_gap.json")
    rp = read(READOUT, "protocol.json")
    assert pa["complete"] and pa["complete_sha256"] == sha(PREVIOUS / "complete.json")
    assert ra["complete"] and ra["complete_sha256"] == sha(READOUT / "complete.json")
    assert (
        sha(READOUT / "cold_readouts.npz")
        == read(READOUT, "complete.json")["artifact_sha256"]["cold_readouts.npz"]
    )
    ref = Path(previous["reference"])
    ids = previous["order"]
    validate_test_ids("EAST", ids)
    assert rp["east_ids"] == ids[:20] and not rp["test_or_suffix_access"]
    for name, h in previous["reference_hashes"].items():
        assert sha(ref / name) == h, name
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("rl_env CUDA required")
    base, parent = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda())
    frozen = snapshot(net)
    threshold = float(parent["threshold"])
    assert threshold == previous["threshold"]
    mapping = read(ref, "input_map.json")
    hashes = read(ref, "input_hashes.json")
    assert mapping["calibration_ids"] == ids[:20]
    ledger = AccessLedger(ids, 20)
    slow = None
    head = None
    prefix = []
    files = {Path(__file__)} | {
        Path(m.__file__)
        for n, m in list(sys.modules.items())
        if n.startswith("experiments.risk_aligned_online.")
        and getattr(m, "__file__", None)
    }
    plan = io23.ROOT / "LOCAL_READOUT_STREAM_PLAN.md"
    protocol = dict(
        scope="DEVELOPMENT_FULL360_LOCAL_AUXILIARY_READOUT_NEURAL_INTRA",
        order=ids,
        arms=ARMS,
        source_sha256=PARENT_SHA256,
        threshold=threshold,
        warmup=20,
        step=0.05,
        history_rows=16,
        prior_precision=1.0,
        reference=str(ref),
        reference_hashes=previous["reference_hashes"],
        previous=str(PREVIOUS),
        previous_complete_sha256=sha(PREVIOUS / "complete.json"),
        readout=str(READOUT),
        readout_sha256=sha(READOUT / "cold_readouts.npz"),
        readout_audit_sha256=sha(READOUT / "audit_with_fold_gap.json"),
        plan_sha256=sha(plan),
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
        inter_updates_after_cold20=0,
        alarm_updates=0,
        reset_each_feedback=True,
        reset_each_shot=True,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        independent_confirmation=False,
    )
    output.mkdir()
    (output / "shots").mkdir()
    _durable_json(output / "protocol.json", protocol)
    with zipfile.ZipFile(output / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f, str(f.relative_to(io23.ROOT)))
        z.write(plan, plan.name)
    start = time.monotonic()
    rows = {k: [] for k in ARMS}
    receipts = []
    aux_rows = []

    def deny(*args, **kwargs):
        raise AssertionError("current outcome accessed before sealing")

    for i, sid in enumerate(ids):
        assert sha(io23.path_for("EAST", sid)) == hashes[str(sid)]
        active = ledger.ready
        with patch.object(io23, "read_outcome", side_effect=deny):
            raw = apply_input_contract(
                simple_data.read_shot("EAST", sid, rate_contract="nominal")
            )
            signal = {k: raw[k] for k in ("values", "available")}
            if active:
                if head is None or slow is None:
                    raise ValueError("unprepared cold state")
                signal = transform(signal, mapping)
                curves, info, aux = predict(net, signal, slow, head)
            else:
                curve = original_predictions(net.base, signal)
                curves = {k: curve.copy() for k in ARMS}
                info = {}
                aux = []
        name = f"{i + 1:04d}_{sid}"
        old_path = PREVIOUS / "shots" / f"{name}.npz"
        assert sha(old_path) == read(PREVIOUS / "shots", f"{name}_seal.json")["sha256"]
        with np.load(old_path, allow_pickle=False) as old:
            error = float(np.max(np.abs(old["cold"] - curves["cold"])))
            assert error <= 2e-6 and np.array_equal(old["times"], raw["times"])
            assert np.array_equal(old["cold"] >= threshold, curves["cold"] >= threshold)
        path = output / "shots" / f"{name}.npz"
        with path.open("xb") as f:
            np.savez_compressed(f, times=raw["times"], **curves, **pack_aux(aux))
        digest = sha(path)
        ledger.seal(sid, digest)
        receipt = dict(
            shot=sid,
            index=i + 1,
            sha256=digest,
            outcome_used=False,
            adaptation_active=active,
            updates=info,
            aux_samples=len(aux),
            cold_max_abs_error=error,
        )
        _durable_json(output / "shots" / f"{name}_seal.json", receipt)
        receipts.append(receipt)
        ledger.reveal(sid)
        labeled = simple_data.attach_outcome(raw, "EAST", {})
        valid = labeled["valid"]
        for k in ARMS:
            rows[k].append(
                dict(
                    shot=sid,
                    times=raw["times"][valid],
                    scores=curves[k][valid],
                    is_disrupt=labeled["is_disrupt"],
                    down=labeled["down"],
                )
            )
        if aux:
            aux_rows.append(
                dict(
                    shot=sid,
                    samples=score_aux(
                        aux,
                        signal,
                        valid,
                        net.base.center.cpu().numpy(),
                        net.base.scale.cpu().numpy(),
                    ),
                )
            )
        if i < 20:
            prefix.append(
                {k: raw[k] for k in ("shot", "device", "values", "available")}
            )
        if i == 19:
            assert (
                fit_prefix_map(
                    mapping["source_statistics"],
                    prefix,
                    order=ids,
                    count=20,
                    policy="density_gain",
                )
                == mapping
            )
            slow = torch.load(
                ref / "models/cold.pt", map_location="cuda", weights_only=True
            )["slow"]
            slow_before = slow.clone()
            head = load_head(READOUT / "cold_readouts.npz", "cuda")
            head_before = head.coef.clone()
            prefix = []
        if (i + 1) % 20 == 0:
            progress = dict(shots=i + 1, seconds=time.monotonic() - start)
            _durable_json(output / "progress.json", progress)
            print(progress, flush=True)
    decisions = {
        k: score_fixed_test("EAST", v, threshold)["predictions"]
        for k, v in rows.items()
    }
    results = {
        k: {
            s: compact(v, threshold)
            for s, v in (("full360", r), ("cold20", r[:20]), ("suffix340", r[20:]))
        }
        for k, r in rows.items()
    }
    changes = {}
    for name in MODES:
        rescued = []
        harmed = []
        for a, b in zip(decisions["cold"], decisions[name]):
            before = a["timely"] if a["is_disrupt"] else not a["alarm"]
            after = b["timely"] if b["is_disrupt"] else not b["alarm"]
            if after and not before:
                rescued.append(a["shot"])
            if before and not after:
                harmed.append(a["shot"])
        changes[name] = dict(rescued=rescued, harmed=harmed)
    assert len(ledger.history) == 360 and sha(DEFAULT_PARENT) == PARENT_SHA256
    assert torch.equal(slow, slow_before) and torch.equal(head.coef, head_before)
    assert all(torch.equal(v, frozen[k]) for k, v in snapshot(net).items())
    for f, h in protocol["code_sha256"].items():
        assert sha(io23.ROOT / f) == h, f
    _durable_json(output / "per_shot.json", decisions)
    _durable_json(output / "receipts.json", receipts)
    _durable_json(output / "auxiliary_per_shot.json", aux_rows)
    auxiliary = summarize_aux(aux_rows)
    _durable_json(output / "auxiliary_summary.json", auxiliary)
    b = results["cold"]["full360"]
    c = results["local_readout"]["full360"]
    screen = (
        c["tp"] >= b["tp"]
        and c["fpr"] <= 0.2
        and c["J"]
        > max(
            results[k]["full360"]["J"]
            for k in ("cold", "frozen_readout", "local_reverse", "local_hold")
        )
    )
    done = dict(
        complete=True,
        seconds=time.monotonic() - start,
        results=results,
        comparisons=changes,
        necessary_intra_screen=bool(screen),
        source_unchanged=True,
        scientific_goal_achieved=False,
        independent_confirmation=False,
        protocol_sha256=sha(output / "protocol.json"),
        per_shot_sha256=sha(output / "per_shot.json"),
        receipts_sha256=sha(output / "receipts.json"),
        auxiliary_summary_sha256=sha(output / "auxiliary_summary.json"),
        auxiliary_per_shot_sha256=sha(output / "auxiliary_per_shot.json"),
        update_counts={
            k: sum(r["updates"].get(k, {}).get("updates", 0) for r in receipts)
            for k in MODES
        },
    )
    _durable_json(output / "complete.json", done)
    print(
        dict(
            results={k: v["full360"] for k, v in results.items()},
            changes=changes,
            screen=screen,
            auxiliary={
                k: {n: v[n] for n in ("mean", "shots", "windows")}
                for k, v in auxiliary.items()
            },
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
