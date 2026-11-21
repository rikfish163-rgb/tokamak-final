"""Complete360 readout-by-routing factorial with matched causal context controls."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import load_context_memory
from ..delayed_recurrent_memory_20261025.learning import batch
from ..delayed_recurrent_memory_20261025.model import patch_signal, endpoint_corrections
from ..east_context_operator_20261031.run import configure, read
from ..lifecycle_stream_20261025.audit import classify, counts, flips
from ..lifecycle_stream_20261025.run import build_adapter
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..slow_fast_attribution_20261031.run import REFERENCE, REFERENCE_SHA
from ..warmstart_alignment_20261022.alignment import transform
from .learning import prepare
from .model import action, context_at, context_trace, token_basis, risk_with_shifts

PLAN = io23.ROOT / "MULTI_TOKEN_CONTEXT_PLAN_20261101.md"
ARMS = (
    "cold",
    "legacy",
    "per_token_real",
    "same_O_aligned_last",
    "same_O_tiled_norm",
    "per_token_reverse",
    "per_token_local",
    "per_token_constant",
    "last_trained_last",
    "last_trained_all",
)


@torch.no_grad()
def predict(model, signal, weights, constant):
    means, complete = patch_signal(signal, model.base.center, model.base.scale)
    real = context_trace(model.memory, means, complete)
    norm = real.norm(dim=1)
    contexts = dict(
        real=real,
        reverse=context_trace(
            model.memory, means, complete, mode="reverse", reference_norm=norm
        ),
        local=context_trace(
            model.memory, means, complete, mode="local", reference_norm=norm
        ),
        constant=norm[:, None] * constant,
    )
    legacy = model.memory.trace(means, complete, mode="memory")
    scores = {a: [] for a in ARMS}
    shifts = {a: [] for a in ARMS[2:]}
    zeros = []
    support = 0
    specs = {
        "per_token_real": ("per_token", "real", "per_token"),
        "same_O_aligned_last": ("per_token", "real", "aligned_last"),
        "same_O_tiled_norm": ("per_token", "real", "tiled_norm"),
        "per_token_reverse": ("per_token", "reverse", "per_token"),
        "per_token_local": ("per_token", "local", "per_token"),
        "per_token_constant": ("per_token", "constant", "per_token"),
        "last_trained_last": ("aligned_last", "real", "aligned_last"),
        "last_trained_all": ("aligned_last", "real", "per_token"),
    }
    for lo in range(0, len(signal["values"]), 32):
        ends = np.arange(lo, min(lo + 32, len(signal["values"])))
        x, m = batch(signal, ends, model.base.center.device)
        h, valid = model.tokens(x, m)
        b, supported = token_basis(
            model.memory, x, m, model.base.center, model.base.scale
        )
        support += int(supported.sum())
        off = model.risk(h, valid, h.new_zeros(len(h), 92))
        check = risk_with_shifts(model, h, valid, h.new_zeros(len(h), 16, 64))
        assert torch.equal(off, check), "zero action must exactly preserve cold"
        logs = dict(
            cold=off,
            legacy=model.risk(
                h,
                valid,
                endpoint_corrections(
                    legacy["delta"], torch.as_tensor(ends, device=h.device)
                ),
            ),
        )
        current = {k: context_at(c, ends) for k, c in contexts.items()}
        for arm, (weight, c, route) in specs.items():
            shift, info = action(weights[weight], b, current[c], valid, route)
            logs[arm] = risk_with_shifts(model, h, valid, shift)
            shifts[arm].append(info["norm"])
            if arm == "same_O_tiled_norm":
                zeros.append(info["tiled_zero_direction"])
        for arm, logit in logs.items():
            if not torch.isfinite(logit).all():
                raise FloatingPointError("nonfinite routed risk")
            scores[arm].append(logit.sigmoid().cpu().numpy())
    receipts = dict(
        supported_token_reads=support,
        context_max_norm=float(norm.max()) if len(norm) else None,
        nonzero_context_patches=int((norm > 0).sum()),
        first_possible_effect_index=16,
        canonical_basis_batch=32,
        tiled_zero_direction_points=int(torch.cat(zeros).sum()),
        action={
            a: dict(
                mean_norm=float(torch.cat(v).mean()),
                max_norm=float(torch.cat(v).max()),
                nonzero_points=int((torch.cat(v) > 0).sum()),
            )
            for a, v in shifts.items()
        },
    )
    return {a: np.concatenate(v) for a, v in scores.items()}, receipts


def main(root, *, preflight=False):
    if root.exists():
        raise FileExistsError(root)
    configure()
    start = time.monotonic()
    assert sha(REFERENCE / "completed.json") == REFERENCE_SHA
    p = read(REFERENCE / "protocol.json")
    assert p["source_memory"]["profile"] == "context8_v1"
    order = read(REFERENCE / "order.json")
    ids = [r["shot"] for r in order]
    hashes = read(REFERENCE / "input_hashes.json")
    mapping = read(REFERENCE / "input_map.json")
    assert mapping["calibration_ids"] == ids[:20]
    base, parent = load_parent(DEFAULT_PARENT)
    base.cuda().eval()
    model = build_adapter(base, readout="full", feedback="progressive", context=True)
    assert (
        load_context_memory(model, Path(p["source_memory"]["path"]))
        == p["source_memory"]
    )
    cold_path = REFERENCE / "models/cold.pt"
    model.restore(torch.load(cold_path, map_location="cpu", weights_only=True))
    model.eval().requires_grad_(False)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    threshold = float(parent["threshold"])
    assert threshold == p["threshold"]
    root.mkdir(parents=True)
    (root / "shots").mkdir()
    files = set(Path(__file__).parent.glob("*.py")) | {PLAN}
    for name, module in list(sys.modules.items()):
        if name.startswith("experiments.risk_aligned_online.") and getattr(
            module, "__file__", None
        ):
            files.add(Path(module.__file__))
    protocol = dict(
        scope="PREFIX20_PREFLIGHT"
        if preflight
        else "DEVELOPMENT_COMPLETE360_MULTI_TOKEN_CONTEXT",
        order=order,
        arms=ARMS,
        reference=str(REFERENCE),
        reference_completed_sha256=REFERENCE_SHA,
        cold_state=str(cold_path),
        cold_state_sha256=sha(cold_path),
        input_map_sha256=sha(REFERENCE / "input_map.json"),
        source_checkpoint=str(DEFAULT_PARENT),
        source_sha256=sha(DEFAULT_PARENT),
        context_source=p["source_memory"],
        threshold=threshold,
        strict_lead_ms=30,
        first20_cost_included=True,
        target_preparation=20,
        steps=2 if preflight else 256,
        trained_parameters_per_copy=5888,
        source_parameter_updates=0,
        context_rule_parameter_updates=0,
        suffix_supervised_updates=0,
        alarm_updates=0,
        context_update="analytical gradient update of c8 on arrived diagnostic changes; reset per shot",
        canonical_basis_batch=32,
        token_support="at least one fully observed8point channel",
        independent_confirmation=False,
        seed=131,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        plan_sha256=sha(PLAN),
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
    )
    _durable_json(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))
    pending, weights, constant = [], None, None
    rows = {a: [] for a in ARMS}
    receipts = []
    for index, sid in enumerate(ids[:20] if preflight else ids):
        prefix = f"{index + 1:04d}_{sid}"
        assert sha(io23.path_for("EAST", sid)) == hashes[str(sid)]
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = {k: raw[k] for k in ("values", "available")}
        with np.load(REFERENCE / "shots" / f"{prefix}.npz", allow_pickle=False) as ref:
            assert np.array_equal(raw["times"], ref["times"])
            if index < 20:
                curves, info = (
                    {a: ref["source"].copy() for a in ARMS},
                    dict(prepared=False),
                )
            else:
                curves, info = predict(
                    model, transform(signal, mapping), weights, constant
                )
                assert np.array_equal(curves["cold"], ref["cold"]), (
                    sid,
                    "cold mismatch",
                )
                assert np.array_equal(curves["legacy"], ref["intra"]), (
                    sid,
                    "legacy mismatch",
                )
        path = root / "shots" / f"{prefix}.npz"
        with path.open("xb") as handle:
            np.savez_compressed(handle, times=raw["times"], **curves)
            handle.flush()
            os.fsync(handle.fileno())
        receipt = dict(
            index=index + 1, shot=sid, sha256=sha(path), outcome_available=False
        )
        _durable_json(root / "shots" / f"{prefix}_seal.json", receipt)
        _durable_json(root / "shots" / f"{prefix}_action.json", info)
        receipts.append(receipt)
        outcome = simple_data.attach_outcome(raw, "EAST", {})
        for a, probability in curves.items():
            rows[a].append(
                dict(
                    shot=sid,
                    disruptive=outcome["is_disrupt"],
                    **classify(
                        raw["times"],
                        probability,
                        outcome["is_disrupt"],
                        outcome["down"],
                        threshold,
                    ),
                )
            )
        if index < 20:
            pending.append(outcome)
        if index == 19:
            _durable_json(
                root / "preparation_access.json",
                dict(
                    ended_ids=ids[:20],
                    first_affected_rank=21,
                    seals={str(r["shot"]): r["sha256"] for r in receipts},
                ),
            )
            pending = [
                dict(
                    s, **transform({k: s[k] for k in ("values", "available")}, mapping)
                )
                for s in pending
            ]
            weights, constant = prepare(
                model,
                pending,
                threshold,
                root / "preparation",
                steps=2 if preflight else 256,
            )
            if preflight:
                s = {k: pending[0][k] for k in ("values", "available")}
                means, complete = patch_signal(s, model.base.center, model.base.scale)
                full_trace = context_trace(model.memory, means, complete)
                cases = [0, 15, 16, 23, 24, 126, 127, 128, *range(144, 152)]
                for end in cases:
                    short = {k: v[: end + 1] for k, v in s.items()}
                    mm, cc = patch_signal(short, model.base.center, model.base.scale)
                    short_trace = context_trace(model.memory, mm, cc)
                    assert torch.equal(
                        context_at(full_trace, [end]), context_at(short_trace, [end])
                    ), (end, "CUDA prefix context mismatch")
                _durable_json(
                    root / "cuda_prefix_invariance.json",
                    dict(status="PASS", endpoints=cases),
                )
                # Ended prefix only: exercise all inference paths and the exact
                # zero-action comparison, without observing any suffix shot.
                _, preview = predict(
                    model,
                    {k: pending[0][k] for k in ("values", "available")},
                    weights,
                    constant,
                )
                _durable_json(root / "inference_preflight.json", preview)
            pending = []
        if (index + 1) % 20 == 0:
            progress = dict(
                stage="stream",
                shots=index + 1,
                total=20 if preflight else 360,
                seconds=time.monotonic() - start,
            )
            _durable_json(root / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in initial.items())
    metrics = {
        a: dict(
            full_stream=counts(r),
            warmup20=counts(r[:20]),
            suffix340=counts(r[20:]) if not preflight else None,
        )
        for a, r in rows.items()
    }
    _durable_json(root / "per_shot.json", rows)
    _durable_json(root / "receipts.json", receipts)
    main_score, off = (
        metrics["per_token_real"]["full_stream"],
        metrics["cold"]["full_stream"],
    )
    operational = (
        not preflight
        and main_score["fpr"] <= 0.20
        and main_score["tp"] >= off["tp"]
        and main_score["fp"] <= off["fp"]
        and (main_score["tp"] > off["tp"] or main_score["fp"] < off["fp"])
    )
    attribution = operational and all(
        main_score["J"] > metrics[a]["full_stream"]["J"]
        for a in (
            "same_O_aligned_last",
            "same_O_tiled_norm",
            "per_token_reverse",
            "per_token_constant",
        )
    )
    result = dict(
        status="COMPLETED_PREFLIGHT" if preflight else "COMPLETED",
        metrics=metrics,
        flips={a: flips(rows["cold"], r) for a, r in rows.items() if a != "cold"},
        operational_gate=operational,
        attribution_gate=attribution,
        source_and_rule_unchanged=True,
        first20_cost_included=True,
        seconds=time.monotonic() - start,
        scientific_goal_achieved=False,
        method_promoted=False,
        independent_confirmation=False,
    )
    _durable_json(root / "completed.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    main(args.output, preflight=args.preflight)
