"""Fixed 3x3 fit/feedback factorial, all360 predictions sealed before outcomes."""

import argparse
from pathlib import Path
import shutil
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
from ..local_readout_20261029.learning import load_head
from ..local_readout_20261029.run import pack_aux, score_aux
from ..qk_matched_head_20261029.run import snapshot
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform, fit_prefix_map
from .learning import ARMS, MODES, CANDIDATE, COMPARATORS, predict

PREVIOUS = io23.ROOT / "probe_outputs/local_readout_east360_v1"


def summarize_aux(rows):
    result = {}
    for method in ["hold"] + [
        arm + part for arm in MODES for part in ("_before", "_after")
    ]:
        values = []
        for row in rows:
            selected = [s for s in row["samples"] if s["elements"]]
            if selected:
                values.append(
                    dict(
                        shot=row["shot"],
                        windows=len(selected),
                        loss=float(
                            np.mean(
                                [
                                    s["losses"][method]["sum"] / s["elements"]
                                    for s in selected
                                ]
                            )
                        ),
                    )
                )
        if not values:
            raise ValueError("no complete eligible auxiliary samples")
        loss = np.array([r["loss"] for r in values])
        result[method] = dict(
            shots=len(values),
            windows=sum(r["windows"] for r in values),
            mean=float(loss.mean()),
            shot_sd=float(loss.std(ddof=1)),
            per_shot=values,
        )
    return result


def run(output):
    if output.exists() or output.absolute().parent != io23.ROOT / "probe_outputs":
        raise ValueError("fresh worktree output required")
    if shutil.disk_usage(output.parent).free < 400 * 1024**2:
        raise RuntimeError(
            "factorial output needs at least400MiB free; no automatic deletion"
        )
    previous = read(PREVIOUS, "protocol.json")
    oldaudit = read(PREVIOUS, "audit.json")
    if not oldaudit["complete"] or oldaudit["complete_sha256"] != sha(
        PREVIOUS / "complete.json"
    ):
        raise ValueError("audited fixed-step predecessor required")
    ids = previous["order"]
    validate_test_ids("EAST", ids)
    ref = Path(previous["reference"])
    readout = Path(previous["readout"])
    for name, h in previous["reference_hashes"].items():
        assert sha(ref / name) == h, name
    assert sha(readout / "cold_readouts.npz") == previous["readout_sha256"]
    assert sha(readout / "audit_with_fold_gap.json") == previous["readout_audit_sha256"]
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
    slow = head = None
    prefix = []
    files = {Path(__file__)} | {
        Path(m.__file__)
        for n, m in list(sys.modules.items())
        if n.startswith("experiments.risk_aligned_online.")
        and getattr(m, "__file__", None)
    }
    plan = io23.ROOT / "LOCAL_READOUT_FACTORIAL_NEXT_PLAN.md"
    protocol = dict(
        scope="DEVELOPMENT_FULL360_LOCAL_READOUT_FACTORIAL_ARMIJO",
        order=ids,
        arms=ARMS,
        modes=MODES,
        candidate=CANDIDATE,
        comparators=COMPARATORS,
        source_sha256=PARENT_SHA256,
        threshold=threshold,
        warmup=20,
        step=0.05,
        history_rows=16,
        prior_precision=1.0,
        armijo_constant=0.0001,
        trial_factors=[2.0 ** (-i) for i in range(8)],
        reference=str(ref),
        reference_hashes=previous["reference_hashes"],
        previous=str(PREVIOUS),
        previous_complete_sha256=sha(PREVIOUS / "complete.json"),
        previous_audit_sha256=sha(PREVIOUS / "audit.json"),
        readout=str(readout),
        readout_sha256=sha(readout / "cold_readouts.npz"),
        readout_audit_sha256=sha(readout / "audit_with_fold_gap.json"),
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
        raise AssertionError("current outcome accessed before prediction seal")

    for i, sid in enumerate(ids):
        assert sha(io23.path_for("EAST", sid)) == hashes[str(sid)]
        active = ledger.ready
        with patch.object(io23, "read_outcome", side_effect=deny):
            raw = apply_input_contract(
                simple_data.read_shot("EAST", sid, rate_contract="nominal")
            )
            signal = {k: raw[k] for k in ("values", "available")}
            if active:
                if slow is None or head is None:
                    raise ValueError("cold state unavailable")
                signal = transform(signal, mapping)
                curves, info, aux = predict(net, signal, slow, head)
            else:
                curve = original_predictions(net.base, signal)
                curves = {k: curve.copy() for k in ARMS}
                info = {}
                aux = []
        name = f"{i + 1:04d}_{sid}"
        old_path = PREVIOUS / "shots" / f"{name}.npz"
        assert sha(old_path) == read(PREVIOUS / "shots", name + "_seal.json")["sha256"]
        with np.load(old_path, allow_pickle=False) as old:
            error = float(np.abs(old["cold"] - curves["cold"]).max())
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
        for arm in ARMS:
            rows[arm].append(
                dict(
                    shot=sid,
                    times=raw["times"][valid],
                    scores=curves[arm][valid],
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
            head = load_head(readout / "cold_readouts.npz", "cuda")
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
            name: compact(sub, threshold)
            for name, sub in (("full360", r), ("cold20", r[:20]), ("suffix340", r[20:]))
        }
        for k, r in rows.items()
    }
    comparisons = {}
    for arm in MODES:
        rescued = []
        harmed = []
        for old, new in zip(decisions["cold"], decisions[arm]):
            before = old["timely"] if old["is_disrupt"] else not old["alarm"]
            after = new["timely"] if new["is_disrupt"] else not new["alarm"]
            if after and not before:
                rescued.append(old["shot"])
            if before and not after:
                harmed.append(old["shot"])
        comparisons[arm] = dict(rescued=rescued, harmed=harmed)
    assert len(ledger.history) == 360 and sha(DEFAULT_PARENT) == PARENT_SHA256
    assert torch.equal(slow, slow_before) and torch.equal(head.coef, head_before)
    assert all(torch.equal(v, frozen[k]) for k, v in snapshot(net).items())
    for path, h in protocol["code_sha256"].items():
        assert sha(io23.ROOT / path) == h, path
    _durable_json(output / "per_shot.json", decisions)
    _durable_json(output / "receipts.json", receipts)
    _durable_json(output / "auxiliary_per_shot.json", aux_rows)
    auxiliary = summarize_aux(aux_rows)
    _durable_json(output / "auxiliary_summary.json", auxiliary)
    b = results["cold"]["full360"]
    c = results[CANDIDATE]["full360"]
    screen = (
        c["tp"] >= b["tp"]
        and c["fpr"] <= 0.2
        and c["J"] > max(results[k]["full360"]["J"] for k in COMPARATORS)
    )
    done = dict(
        complete=True,
        seconds=time.monotonic() - start,
        results=results,
        comparisons=comparisons,
        necessary_intra_screen=bool(screen),
        source_unchanged=True,
        scientific_goal_achieved=False,
        independent_confirmation=False,
        update_counts={
            k: sum(r["updates"].get(k, {}).get("updates", 0) for r in receipts)
            for k in MODES
        },
        **{
            name + "_sha256": sha(output / (name + ".json"))
            for name in (
                "protocol",
                "per_shot",
                "receipts",
                "auxiliary_summary",
                "auxiliary_per_shot",
            )
        },
    )
    _durable_json(output / "complete.json", done)
    print(
        dict(
            results={k: v["full360"] for k, v in results.items()},
            comparisons=comparisons,
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
