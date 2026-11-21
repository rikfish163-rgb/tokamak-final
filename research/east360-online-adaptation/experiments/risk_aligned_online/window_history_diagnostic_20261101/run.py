"""Complete360 frozen cold-model history ablation and feedback timing census."""

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
from ..delayed_recurrent_memory_20261025.learning import batch
from ..east_context_operator_20261031.run import configure, read
from ..lifecycle_stream_20261025.audit import classify, counts, flips
from ..lifecycle_stream_20261025.run import build_adapter
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..slow_fast_attribution_20261031.run import REFERENCE, REFERENCE_SHA
from ..warmstart_alignment_20261022.alignment import transform
from .core import VIEWS, first_alarm_index, history_view, timing_bin

PLAN = io23.ROOT / "WINDOW_HISTORY_DIAGNOSTIC_PLAN_20261101.md"


@torch.no_grad()
def predict(model, signal):
    curves = {v: [] for v in VIEWS}
    for lo in range(0, len(signal["values"]), 32):
        ends = list(range(lo, min(lo + 32, len(signal["values"]))))
        values, mask = batch(signal, ends, model.base.center.device)
        for view in VIEWS:
            x, m = history_view(values, mask, ends, view)
            h, valid = model.tokens(x, m)
            z = model.risk(h, valid, h.new_zeros(len(h), 92))
            if not torch.isfinite(z).all():
                raise FloatingPointError("nonfinite history diagnostic")
            curves[view].append(z.sigmoid().cpu().numpy())
    return {k: np.concatenate(v) for k, v in curves.items()}


def main(root):
    if root.exists():
        raise FileExistsError(root)
    configure()
    start = time.monotonic()
    assert sha(REFERENCE / "completed.json") == REFERENCE_SHA
    p = read(REFERENCE / "protocol.json")
    order = read(REFERENCE / "order.json")
    ids = [r["shot"] for r in order]
    assert len(ids) == len(set(ids)) == 360
    base, parent = load_parent(DEFAULT_PARENT)
    base.cuda().eval()
    threshold = float(parent["threshold"])
    assert threshold == p["threshold"]
    model = build_adapter(base, readout="full", feedback="progressive", context=True)
    state_path = REFERENCE / "models/cold.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    model.restore(state)
    model.eval().requires_grad_(False)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    mapping = read(REFERENCE / "input_map.json")
    assert mapping["calibration_ids"] == ids[:20]
    input_hashes = read(REFERENCE / "input_hashes.json")
    saved = read(REFERENCE / "per_shot.json")
    assert set(saved) == set(p["arms"])
    root.mkdir(parents=True)
    (root / "shots").mkdir()
    files = set(Path(__file__).parent.glob("*.py")) | {PLAN}
    for name, module in list(sys.modules.items()):
        if name.startswith("experiments.risk_aligned_online.") and getattr(
            module, "__file__", None
        ):
            files.add(Path(module.__file__))
    protocol = dict(
        scope="COMPLETE360_POSTHOC_HISTORY_DEPENDENCE_DIAGNOSTIC",
        order=order,
        views=VIEWS,
        reference=str(REFERENCE),
        reference_completed_sha256=REFERENCE_SHA,
        cold_state=str(state_path),
        cold_state_sha256=sha(state_path),
        input_map_sha256=sha(REFERENCE / "input_map.json"),
        source_checkpoint=str(DEFAULT_PARENT),
        source_sha256=sha(DEFAULT_PARENT),
        threshold=threshold,
        strict_lead_ms=30,
        first_affected_rank=21,
        first20_original_source_cost_included=True,
        optimizer_steps=0,
        intra_parameter_updates=0,
        alarm_parameter_updates=0,
        independent_confirmation=False,
        input_perturbation_is_physical_trajectory=False,
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
    rows = {v: [] for v in VIEWS}
    timing_rows = {a: [] for a in saved}
    changes = {v: [] for v in VIEWS if v != "original"}
    receipts = []
    for index, sid in enumerate(ids):
        prefix = f"{index + 1:04d}_{sid}"
        assert sha(io23.path_for("EAST", sid)) == input_hashes[str(sid)]
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = {k: raw[k] for k in ("values", "available")}
        with np.load(
            REFERENCE / "shots" / f"{prefix}.npz", allow_pickle=False
        ) as previous:
            assert np.array_equal(raw["times"], previous["times"])
            if index < 20:
                curves = {v: previous["source"].copy() for v in VIEWS}
            else:
                curves = predict(model, transform(signal, mapping))
            assert np.array_equal(curves["original"], previous["cold"]), (
                sid,
                "cold replay drift",
            )
            old = {a: previous[a].copy() for a in saved}
        path = root / "shots" / f"{prefix}.npz"
        with path.open("xb") as handle:
            np.savez_compressed(handle, times=raw["times"], **curves)
            handle.flush()
            os.fsync(handle.fileno())
        receipt = dict(
            index=index + 1, shot=sid, sha256=sha(path), outcome_available=False
        )
        _durable_json(root / "shots" / f"{prefix}_seal.json", receipt)
        receipts.append(receipt)
        flag, down = io23.read_outcome("EAST", sid)
        valid = (
            np.ones(len(raw["times"]), bool)
            if not flag
            else (down - raw["times"]) * 1000 >= -1e-6
        )
        for v in VIEWS:
            decision = classify(raw["times"], curves[v], flag, down, threshold)
            rows[v].append(dict(shot=sid, disruptive=flag, **decision))
            if v != "original" and index >= 20:
                delta = np.abs(curves[v] - curves["original"])
                changes[v].append(
                    dict(
                        shot=sid,
                        points=len(delta),
                        mean_abs_probability_change=float(delta.mean()),
                        max_abs_probability_change=float(delta.max()),
                        points_changed_over1e4=int((delta > 1e-4).sum()),
                    )
                )
        for a, probability in old.items():
            decision = classify(raw["times"], probability, flag, down, threshold)
            assert dict(shot=sid, is_disrupt=flag, **decision) == saved[a][index]
            eligible_curve = np.where(valid, probability, -np.inf)
            crossing = np.flatnonzero(eligible_curve >= threshold)
            first = int(crossing[0]) if len(crossing) else None
            if not flag:
                assert first == first_alarm_index(probability, threshold)
            timing_rows[a].append(
                dict(
                    index=index + 1,
                    shot=sid,
                    disruptive=flag,
                    first_alarm_index=first,
                    first_alarm_ms_from_record_start=(
                        float((raw["times"][first] - raw["times"][0]) * 1000)
                        if first is not None
                        else None
                    ),
                    timing_bin=timing_bin(first),
                    **decision,
                )
            )
        if (index + 1) % 20 == 0:
            progress = dict(
                shots=index + 1, total=360, seconds=time.monotonic() - start
            )
            _durable_json(root / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in initial.items())
    all_metrics = {
        v: dict(full360=counts(r), warmup20=counts(r[:20]), suffix340=counts(r[20:]))
        for v, r in rows.items()
    }
    timing = {}
    for a, r in timing_rows.items():
        timing[a] = {}
        for scope, subset in (
            ("full360", r),
            ("warmup20", r[:20]),
            ("suffix340", r[20:]),
        ):
            safe = [t for t in subset if not t["disruptive"]]
            timing[a][scope] = dict(
                n_safe=len(safe),
                fp=sum(t["alarm"] for t in safe),
                safe_by_first_alarm={
                    b: [t["shot"] for t in safe if t["timing_bin"] == b]
                    for b in ("before16", "16to127", "from128", "no_alarm")
                },
            )
    _durable_json(root / "per_shot.json", rows)
    _durable_json(root / "timing_per_shot.json", timing_rows)
    _durable_json(root / "timing_summary.json", timing)
    _durable_json(root / "probability_changes.json", changes)
    _durable_json(root / "receipts.json", receipts)
    result = dict(
        status="COMPLETED",
        metrics=all_metrics,
        flips={v: flips(rows["original"], rows[v]) for v in VIEWS if v != "original"},
        reference_cold_curves_exact=360,
        reference_decisions_recounted=360 * len(saved),
        frozen_state_unchanged=True,
        seconds=time.monotonic() - start,
        method_promoted=False,
        scientific_goal_achieved=False,
    )
    _durable_json(root / "completed.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
