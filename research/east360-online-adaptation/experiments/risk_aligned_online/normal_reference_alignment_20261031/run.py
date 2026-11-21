"""Full360 frozen-source comparison of pooled and ended-SAFE input alignment."""

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
from ..delayed_recurrent_memory_20261025.learning import original_predictions
from ..east_context_operator_20261031.run import configure, read
from ..fixed_test_protocol import score_fixed_test
from ..lifecycle_stream_20261025.protocol import AccessLedger
from ..lifecycle_stream_20261025.run import compact, STATISTICS
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..slow_fast_attribution_20261031.run import REFERENCE, REFERENCE_SHA
from ..warmstart_alignment_20261022.alignment import fit_prefix_map, transform
from .core import fit_coordinates, make_transport, transport, ended_groups

ARMS = (
    "source",
    "density",
    "pooled_diagonal",
    "safe_diagonal",
    "pooled_correlation",
    "safe_correlation",
)
PLAN = io23.ROOT / "NORMAL_REFERENCE_ALIGNMENT_PLAN_20261031.md"


def main(root):
    if root.exists():
        raise FileExistsError(root)
    configure()
    started = time.monotonic()
    assert sha(REFERENCE / "completed.json") == REFERENCE_SHA
    order = read(REFERENCE / "order.json")
    ids = [r["shot"] for r in order]
    p = read(REFERENCE / "protocol.json")
    hashes = read(REFERENCE / "input_hashes.json")
    assert len(ids) == len(set(ids)) == 360
    base, parent = load_parent(DEFAULT_PARENT)
    base.cuda().eval()
    before = {k: v.cpu().clone() for k, v in base.state_dict().items()}
    center, scale = base.center.cpu().numpy(), base.scale.cpu().numpy()
    threshold = float(parent["threshold"])
    assert threshold == p["threshold"]
    ledger = AccessLedger(ids, 20)
    root.mkdir(parents=True)
    (root / "shots").mkdir()
    (root / "maps").mkdir()
    files = set(Path(__file__).parent.glob("*.py")) | {PLAN}
    for name, module in list(sys.modules.items()):
        if name.startswith("experiments.risk_aligned_online.") and getattr(
            module, "__file__", None
        ):
            files.add(Path(module.__file__))
    protocol = dict(
        scope="DEVELOPMENT_COMPLETE360_INPUT_ALIGNMENT_ONLY",
        arms=ARMS,
        order=order,
        source_sha256=PARENT_SHA256,
        source_checkpoint=str(DEFAULT_PARENT),
        reference=str(REFERENCE),
        reference_completed_sha256=REFERENCE_SHA,
        threshold=threshold,
        strict_lead_ms=30,
        warmup=20,
        warmup_in_primary=True,
        source_training=False,
        intra_neural_updates=0,
        inter_neural_updates=0,
        target_map_access="only ended first20; freeze from21; no live outcome selects map",
        source_reference="TRAIN only; pooled versus completed-safe labels",
        statistical_map_not_physical_calibration=True,
        independent_confirmation=False,
        center=center.tolist(),
        scale=scale.tolist(),
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
    source_rows = []
    source_safe = []
    source_hashes = {}
    for sid in io23.source_roles()["train"]:
        raw = simple_data.read_shot("J-TEXT", sid)
        row = {k: raw[k] for k in ("shot", "values", "available")}
        source_rows.append(row)
        flag, _ = io23.read_outcome("J-TEXT", sid)
        if not flag:
            source_safe.append(row)
        source_hashes[str(sid)] = sha(io23.path_for("J-TEXT", sid))
    source_fit = dict(
        pooled=fit_coordinates(source_rows, center, scale),
        safe=fit_coordinates(source_safe, center, scale),
    )
    _durable_json(root / "source_statistics.json", source_fit)
    _durable_json(root / "source_hashes.json", source_hashes)
    del source_rows, source_safe
    density_source = read(STATISTICS)
    assert sha(STATISTICS) == p["source_statistics_sha256"]
    pending = []
    prefix_flags = {}
    density_rows = []
    maps = {}
    density = None
    failed = {}
    rows = {name: [] for name in ARMS}
    receipts = []
    for index, sid in enumerate(ids):
        assert sha(io23.path_for("EAST", sid)) == hashes[str(sid)]
        prefix = f"{index + 1:04d}_{sid}"
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = {k: raw[k] for k in ("values", "available")}
        predictions = {"source": original_predictions(base, signal)}
        issues = {}
        if index < 20:
            predictions.update(
                {name: predictions["source"].copy() for name in ARMS[1:]}
            )
        else:
            predictions["density"] = original_predictions(
                base, transform(signal, density)
            )
            for name, mapping in maps.items():
                if name in failed:
                    continue  # Failed candidate is excluded from full-stream scoring, never filled by baseline.
                try:
                    mapped = transport(signal, mapping, center, scale)
                except FloatingPointError as error:
                    failed[name] = dict(
                        index=index + 1,
                        shot=sid,
                        reason=str(error),
                        partial_predictions_not_a_complete_result=True,
                    )
                    _durable_json(root / "candidate_failures.json", failed)
                    continue
                predictions[name] = original_predictions(base, mapped)
                issues[name] = {
                    io23.RAW23[c]: int(
                        ((mapped["values"][:, c] < 0) & mapped["available"][:, c]).sum()
                    )
                    for c in (17, 22)
                }
        with np.load(REFERENCE / "shots" / f"{prefix}.npz", allow_pickle=False) as ref:
            assert np.array_equal(ref["times"], raw["times"])
            assert np.array_equal(ref["source"], predictions["source"]), (
                sid,
                "source drift",
            )
            assert np.array_equal(ref["aligned"], predictions["density"]), (
                sid,
                "density drift",
            )
        path = root / "shots" / f"{prefix}.npz"
        with path.open("xb") as f:
            np.savez_compressed(f, times=raw["times"], **predictions)
            f.flush()
            os.fsync(f.fileno())
        digest = sha(path)
        ledger.seal(sid, digest)
        seal = dict(
            shot=sid,
            index=index + 1,
            sha256=digest,
            outcome_available=False,
            active_arms=sorted(predictions),
            negative_variance_outputs=issues,
        )
        _durable_json(root / "shots" / f"{prefix}_seal.json", seal)
        receipts.append(seal)
        ledger.reveal(sid)
        outcome = simple_data.attach_outcome(raw, "EAST", {})
        for name, scores in predictions.items():
            valid = outcome["valid"]
            rows[name].append(
                dict(
                    shot=sid,
                    times=raw["times"][valid],
                    scores=scores[valid],
                    is_disrupt=outcome["is_disrupt"],
                    down=outcome["down"],
                )
            )
        if index < 20:
            row = {k: raw[k] for k in ("shot", "values", "available")}
            pending.append(row)
            prefix_flags[sid] = bool(outcome["is_disrupt"])
            density_rows.append(
                {k: raw[k] for k in ("shot", "device", "values", "available")}
            )
        if index == 19:
            groups = ended_groups(
                pending,
                prefix_flags,
                order=ids,
                count=20,
                seals={r["shot"]: r["sha256"] for r in receipts},
            )
            target_fit = {
                group: fit_coordinates(records, center, scale)
                for group, records in groups.items()
            }
            _durable_json(root / "target_statistics.json", target_fit)
            for name in ARMS[2:]:
                group, mode = name.split("_")
                mapping = make_transport(source_fit[group], target_fit[group], mode)
                mapping.update(
                    first_affected_rank=21,
                    preceding_seals={str(r["shot"]): r["sha256"] for r in receipts},
                )
                maps[name] = mapping
                _durable_json(root / "maps" / f"{name}.json", mapping)
            density = fit_prefix_map(
                density_source, density_rows, order=ids, count=20, policy="density_gain"
            )
            assert density == read(REFERENCE / "input_map.json")
            _durable_json(root / "maps/density.json", density)
            pending = []
            prefix_flags = {}
            density_rows = []
        if (index + 1) % 20 == 0:
            progress = dict(
                shots=index + 1,
                total=360,
                seconds=time.monotonic() - started,
                failed_candidates=failed,
            )
            _durable_json(root / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    assert len(ledger.history) == 360 and sha(DEFAULT_PARENT) == PARENT_SHA256
    assert all(
        torch.equal(value, base.state_dict()[k].cpu()) for k, value in before.items()
    )
    results = {}
    per_shot = {}
    for name, values in rows.items():
        if name in failed:
            continue  # Preserve partial NPZ only; no partial metric masquerades as full360.
        assert len(values) == 360
        per_shot[name] = score_fixed_test("EAST", values, threshold)["predictions"]
        results[name] = dict(
            full360=compact(values, threshold),
            warmup20=compact(values[:20], threshold),
            suffix340=compact(values[20:], threshold),
        )
    _durable_json(root / "per_shot.json", per_shot)
    _durable_json(root / "receipts.json", receipts)
    done = dict(
        status="COMPLETED" if not failed else "COMPLETED_WITH_INVALID_CANDIDATES",
        metrics=results,
        failed_candidates=failed,
        seconds=time.monotonic() - started,
        source_bitwise_unchanged=True,
        reference_curves_exact=720,
        source_safe_shots=len(source_fit["safe"]["shot_ids"]),
        target_safe_shots=len(target_fit["safe"]["shot_ids"]),
        neural_updates=0,
        scientific_goal_achieved=False,
    )
    _durable_json(root / "completed.json", done)
    print(json.dumps(done), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
