"""Complete sealed EAST360 comparison; auxiliary fitting only after cold20."""

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
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform, fit_prefix_map
from .learning import ARMS, UPDATES, STEP, predict, fit_head

SOURCE = io23.ROOT / "probe_outputs/qk_matched_source_v1"
PREVIOUS = io23.ROOT / "probe_outputs/attention_protected_east360_v1"


def run(output):
    if output.exists() or output.absolute().parent != io23.ROOT / "probe_outputs":
        raise ValueError("new worktree output required")
    prior = read(PREVIOUS, "protocol.json")
    old_audit = read(PREVIOUS, "audit.json")
    if (
        old_audit["complete_sha256"] != sha(PREVIOUS / "complete.json")
        or not old_audit["complete"]
    ):
        raise ValueError("audited comparator required")
    ref = Path(prior["reference"])
    ids = prior["order"]
    validate_test_ids("EAST", ids)
    for name, h in prior["reference_hashes"].items():
        if sha(ref / name) != h:
            raise ValueError("cold reference changed")
    trained = read(SOURCE, "complete.json")
    source_audit = read(SOURCE, "audit.json")
    if (
        not trained["complete"]
        or not trained["source_unchanged"]
        or trained["target_or_test_signals_read"] != 0
        or source_audit["complete_sha256"] != sha(SOURCE / "complete.json")
        or sha(SOURCE / "forecast_head.pt") != trained["heads"]["forecast"]
    ):
        raise ValueError("audited source-only ordinary forecast head required")
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("rl_env CUDA required")
    nets = {}
    for name in ("original", "source", "cold"):
        base, parent = load_parent(DEFAULT_PARENT)
        base.cuda()
        nets[name] = MatrixAttention(base)
        if name != "original":
            base.forecast_head.load_state_dict(
                torch.load(
                    SOURCE / "forecast_head.pt", map_location="cuda", weights_only=True
                )
            )
    frozen = snapshot(nets["original"])
    if any(
        not torch.equal(v, frozen[k])
        for net in nets.values()
        for k, v in snapshot(net).items()
    ):
        raise ValueError("classifiers differ between heads")
    threshold = float(parent["threshold"])
    mapping = read(ref, "input_map.json")
    hashes = read(ref, "input_hashes.json")
    if threshold != prior["threshold"] or mapping["calibration_ids"] != ids[:20]:
        raise ValueError("fixed threshold/prefix required")
    slow = None
    slow_before = None
    ledger = AccessLedger(ids, 20)
    prefix = []
    rows = {k: [] for k in ARMS}
    receipts = []
    files = {Path(__file__)} | {
        Path(m.__file__)
        for n, m in list(sys.modules.items())
        if n.startswith("experiments.risk_aligned_online.")
        and getattr(m, "__file__", None)
    }
    plan = io23.ROOT / "QK_FORECAST_STREAM_PLAN.md"
    protocol = dict(
        scope="DEVELOPMENT_FULL360_QK_ORDINARY_FORECAST_HEADS",
        order=ids,
        arms=ARMS,
        source_sha256=PARENT_SHA256,
        source_head_sha256=trained["heads"]["forecast"],
        source_root=str(SOURCE),
        source_complete_sha256=sha(SOURCE / "complete.json"),
        reference=str(ref),
        reference_hashes=prior["reference_hashes"],
        previous=str(PREVIOUS),
        previous_complete_sha256=sha(PREVIOUS / "complete.json"),
        threshold=threshold,
        warmup=20,
        step=STEP,
        reset_each_feedback=True,
        reset_each_shot=True,
        inter_updates_after_cold20=0,
        alarm_updates=0,
        head_fit_steps=256,
        head_fit_seed=131,
        head_fit_lr=1e-4,
        head_fit_weight_decay=0,
        head_fit_clip=1,
        head_fit_shots_per_step=4,
        plan_sha256=sha(plan),
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
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
    head_frozen = {}

    def deny(*args, **kwargs):
        raise AssertionError("outcome read before durable current-shot seal")

    for i, sid in enumerate(ids):
        if sha(io23.path_for("EAST", sid)) != hashes[str(sid)]:
            raise ValueError("raw input changed")
        active = ledger.ready
        with patch.object(io23, "read_outcome", side_effect=deny):
            raw = apply_input_contract(
                simple_data.read_shot("EAST", sid, rate_contract="nominal")
            )
            signal = {k: raw[k] for k in ("values", "available")}
            if active:
                if slow is None or not head_frozen:
                    raise ValueError("cold state/head not prepared")
                scores, info = predict(nets, transform(signal, mapping), slow)
            else:
                curve = original_predictions(nets["original"].base, signal)
                scores = {k: curve.copy() for k in ARMS}
                info = {}
        name = f"{i + 1:04d}_{sid}"
        old_path = PREVIOUS / "shots" / f"{name}.npz"
        old_seal = read(PREVIOUS / "shots", f"{name}_seal.json")
        if sha(old_path) != old_seal["sha256"] or old_seal["outcome_used"]:
            raise ValueError("prior seal invalid")
        with np.load(old_path) as old:
            error = float(np.max(np.abs(old["cold"] - scores["cold"])))
            if (
                error > 2e-6
                or not np.array_equal(old["times"], raw["times"])
                or not np.array_equal(
                    old["cold"] >= threshold, scores["cold"] >= threshold
                )
            ):
                raise ValueError(f"cold comparator changed on {sid}")
        path = output / "shots" / f"{name}.npz"
        with path.open("xb") as f:
            np.savez_compressed(f, times=raw["times"], **scores)
        digest = sha(path)
        ledger.seal(sid, digest)
        receipt = dict(
            shot=sid,
            index=i + 1,
            sha256=digest,
            outcome_used=False,
            updates=info,
            cold_max_abs_error=error,
            adaptation_active=active,
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
                    scores=scores[k][valid],
                    is_disrupt=labeled["is_disrupt"],
                    down=labeled["down"],
                )
            )
        if i < 20:
            prefix.append(
                {k: raw[k] for k in ("shot", "device", "values", "available")}
            )
        if i == 19:
            if (
                fit_prefix_map(
                    mapping["source_statistics"],
                    prefix,
                    order=ids,
                    count=20,
                    policy="density_gain",
                )
                != mapping
            ):
                raise ValueError("ended-prefix map not reproduced")
            slow = torch.load(
                ref / "models/cold.pt", map_location="cuda", weights_only=True
            )["slow"]
            slow_before = slow.clone()
            calibrated = [
                dict(
                    shot=s["shot"],
                    device="EAST",
                    **transform({k: s[k] for k in ("values", "available")}, mapping),
                )
                for s in prefix
            ]
            with (
                patch.object(io23, "read_outcome", side_effect=deny),
                patch.object(io23, "read_signals", side_effect=deny),
            ):
                fit = fit_head(nets["cold"], calibrated, slow)
            if fit["fit_ids"] != ids[:20]:
                raise ValueError("head fit accessed outside ended cold20")
            _durable_json(output / "cold_head_training.json", fit)
            torch.save(
                {
                    k: v.cpu()
                    for k, v in nets["cold"].base.forecast_head.state_dict().items()
                },
                output / "cold_head.pt",
            )
            head_frozen = {
                n: {
                    k: v.cpu().clone()
                    for k, v in net.base.forecast_head.state_dict().items()
                }
                for n, net in nets.items()
            }
            prefix = []
        if (i + 1) % 20 == 0:
            progress = dict(shots=i + 1, seconds=time.monotonic() - start)
            _durable_json(output / "progress.json", progress)
            print(progress, flush=True)
    predictions = {
        k: score_fixed_test("EAST", r, threshold)["predictions"]
        for k, r in rows.items()
    }
    results = {
        k: {
            s: compact(sub, threshold)
            for s, sub in (("full360", r), ("cold20", r[:20]), ("suffix340", r[20:]))
        }
        for k, r in rows.items()
    }
    comparisons = {}
    for k in UPDATES:
        rescued = []
        harmed = []
        for old, new in zip(predictions["cold"], predictions[k]):
            before = old["timely"] if old["is_disrupt"] else not old["alarm"]
            after = new["timely"] if new["is_disrupt"] else not new["alarm"]
            if after and not before:
                rescued.append(old["shot"])
            if before and not after:
                harmed.append(old["shot"])
        comparisons[k] = dict(rescued=rescued, harmed=harmed)
    for n, net in nets.items():
        if any(not torch.equal(v, frozen[k]) for k, v in snapshot(net).items()):
            raise ValueError("persistent classifier changed")
        if any(
            not torch.equal(v.cpu(), head_frozen[n][k])
            for k, v in net.base.forecast_head.state_dict().items()
        ):
            raise ValueError("auxiliary head changed after cold20")
    if (
        len(ledger.history) != 360
        or sha(DEFAULT_PARENT) != PARENT_SHA256
        or not torch.equal(slow, slow_before)
    ):
        raise ValueError("incomplete stream or changed parent")
    for f, h in protocol["code_sha256"].items():
        if sha(io23.ROOT / f) != h:
            raise ValueError("implementation changed during stream")
    screen = {}
    b = results["cold"]["full360"]
    for candidate in ("source", "cold"):
        c = results[f"{candidate}_forecast"]["full360"]
        screen[candidate] = bool(
            c["tp"] >= b["tp"]
            and c["fpr"] <= 0.20
            and c["J"]
            > max(
                b["J"],
                results[f"{candidate}_reverse"]["full360"]["J"],
                results[f"{candidate}_hold"]["full360"]["J"],
            )
        )
    _durable_json(output / "per_shot.json", predictions)
    _durable_json(output / "receipts.json", receipts)
    done = dict(
        complete=True,
        results=results,
        comparisons=comparisons,
        necessary_intra_screen=screen,
        seconds=time.monotonic() - start,
        protocol_sha256=sha(output / "protocol.json"),
        per_shot_sha256=sha(output / "per_shot.json"),
        receipts_sha256=sha(output / "receipts.json"),
        cold_head_sha256=sha(output / "cold_head.pt"),
        cold_head_training_sha256=sha(output / "cold_head_training.json"),
        update_counts={
            k: sum(r["updates"].get(k, {}).get("updates", 0) for r in receipts)
            for k in UPDATES
        },
        scientific_goal_achieved=False,
        independent_confirmation=False,
        source_unchanged=True,
    )
    _durable_json(output / "complete.json", done)
    print(
        dict(
            results={k: r["full360"] for k, r in results.items()},
            comparisons=comparisons,
            screen=screen,
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
