"""Whole360 factorial: input parameter scope by mean/peak training objective."""

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
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..slow_fast_attribution_20261031.run import REFERENCE, REFERENCE_SHA
from ..warmstart_alignment_20261022.alignment import fit_prefix_map, transform
from .prepare import ARMS, prepare

NAMES = ("source", "density", *ARMS)
PLAN = io23.ROOT / "OPERATIONAL_INPUT_GAIN_PLAN_20261101.md"


def main(root):
    if root.exists():
        raise FileExistsError(root)
    configure()
    started = time.monotonic()
    assert sha(REFERENCE / "completed.json") == REFERENCE_SHA
    order = read(REFERENCE / "order.json")
    ids = [r["shot"] for r in order]
    parent_protocol = read(REFERENCE / "protocol.json")
    hashes = read(REFERENCE / "input_hashes.json")
    assert len(ids) == len(set(ids)) == 360
    base, parent = load_parent(DEFAULT_PARENT)
    base.cuda().eval()
    initial = {k: v.cpu().clone() for k, v in base.state_dict().items()}
    threshold = float(parent["threshold"])
    assert threshold == parent_protocol["threshold"]
    source_stats = read(STATISTICS)
    assert sha(STATISTICS) == parent_protocol["source_statistics_sha256"]
    root.mkdir(parents=True)
    (root / "shots").mkdir()
    files = set(Path(__file__).parent.glob("*.py")) | {PLAN}
    for name, module in list(sys.modules.items()):
        if name.startswith("experiments.risk_aligned_online.") and getattr(
            module, "__file__", None
        ):
            files.add(Path(module.__file__))
    protocol = dict(
        scope="DEVELOPMENT_COMPLETE360_INPUT_GAINS_OBJECTIVE_FACTORIAL",
        order=order,
        arms=NAMES,
        reference=str(REFERENCE),
        reference_completed_sha256=REFERENCE_SHA,
        source_checkpoint=str(DEFAULT_PARENT),
        source_sha256=sha(DEFAULT_PARENT),
        threshold=threshold,
        strict_lead_ms=30,
        warmup=20,
        warmup_in_primary=True,
        prepared_after="all first20 predictions sealed and HDF5 outcomes revealed",
        steps=256,
        learning_rate=0.03,
        gradient_clip=1.0,
        regularizer=".01 sum(theta squared)",
        target_loss="class-equal mean/peak margin loss; S full record, D strictly30..125ms positive interval",
        source_network_updates=0,
        learned_input_parameters=dict(density=1, structured=9),
        suffix_persistent_updates=0,
        intra_updates=0,
        physical_units_certified=False,
        seed=131,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        independent_confirmation=False,
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
        plan_sha256=sha(PLAN),
    )
    _durable_json(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))
    ledger = AccessLedger(ids, 20)
    density = None
    maps = {}
    pending = []
    map_records = []
    rows = {name: [] for name in NAMES}
    receipts = []
    for index, sid in enumerate(ids):
        assert sha(io23.path_for("EAST", sid)) == hashes[str(sid)]
        prefix = f"{index + 1:04d}_{sid}"
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = {k: raw[k] for k in ("values", "available")}
        curves = {"source": original_predictions(base, signal)}
        if index < 20:
            curves.update({name: curves["source"].copy() for name in NAMES[1:]})
        else:
            curves["density"] = original_predictions(base, transform(signal, density))
            for name, mapping in maps.items():
                curves[name] = original_predictions(base, transform(signal, mapping))
        if any(not np.isfinite(value).all() for value in curves.values()):
            raise FloatingPointError("nonfinite prediction")
        with np.load(
            REFERENCE / "shots" / f"{prefix}.npz", allow_pickle=False
        ) as reference:
            assert np.array_equal(raw["times"], reference["times"])
            assert np.array_equal(curves["source"], reference["source"]), (
                sid,
                "source drift",
            )
            assert np.array_equal(curves["density"], reference["aligned"]), (
                sid,
                "density drift",
            )
        path = root / "shots" / f"{prefix}.npz"
        with path.open("xb") as f:
            np.savez_compressed(f, times=raw["times"], **curves)
            f.flush()
            os.fsync(f.fileno())
        digest = sha(path)
        ledger.seal(sid, digest)
        receipt = dict(
            index=index + 1, shot=sid, sha256=digest, outcome_available=False
        )
        _durable_json(root / "shots" / f"{prefix}_seal.json", receipt)
        receipts.append(receipt)
        ledger.reveal(sid)
        outcome = simple_data.attach_outcome(raw, "EAST", {})
        for name, probability in curves.items():
            valid = outcome["valid"]
            rows[name].append(
                dict(
                    shot=sid,
                    times=raw["times"][valid],
                    scores=probability[valid],
                    is_disrupt=outcome["is_disrupt"],
                    down=outcome["down"],
                )
            )
        if index < 20:
            pending.append(outcome)
            map_records.append(
                {k: raw[k] for k in ("shot", "device", "values", "available")}
            )
        if index == 19:
            density = fit_prefix_map(
                source_stats, map_records, order=ids, count=20, policy="density_gain"
            )
            assert density == read(REFERENCE / "input_map.json") and ledger.ready
            _durable_json(root / "density.json", density)
            _durable_json(
                root / "preparation_access.json",
                dict(
                    ended_ids=ledger.history.copy(),
                    prediction_seals={str(r["shot"]): r["sha256"] for r in receipts},
                    first_affected_rank=21,
                ),
            )
            maps = prepare(
                base, pending, density, threshold, root / "preparation", steps=256
            )
            pending = []
            map_records = []
        if (index + 1) % 20 == 0:
            progress = dict(
                stage="stream",
                shots=index + 1,
                total=360,
                seconds=time.monotonic() - started,
            )
            _durable_json(root / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    assert (
        len(ledger.history) == 360 and sha(DEFAULT_PARENT) == protocol["source_sha256"]
    )
    assert all(
        torch.equal(value, base.state_dict()[k].cpu()) for k, value in initial.items()
    )
    metrics = {}
    per_shot = {}
    for name, values in rows.items():
        per_shot[name] = score_fixed_test("EAST", values, threshold)["predictions"]
        metrics[name] = dict(
            full360=compact(values, threshold),
            warmup20=compact(values[:20], threshold),
            suffix340=compact(values[20:], threshold),
        )
    _durable_json(root / "per_shot.json", per_shot)
    _durable_json(root / "receipts.json", receipts)
    result = dict(
        status="COMPLETED",
        metrics=metrics,
        seconds=time.monotonic() - started,
        source_network_bitwise_unchanged=True,
        reference_curves_exact=720,
        scientific_goal_achieved=False,
        input_parameters_trained=True,
        intra_neural_updates=0,
        suffix_persistent_updates=0,
    )
    _durable_json(root / "completed.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
