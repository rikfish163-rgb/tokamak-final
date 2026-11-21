"""Evaluate unchanged statistical maps without an overflowing inverse-asinh roundtrip."""

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
from ..delayed_recurrent_memory_20261025.learning import batch, original_predictions
from ..east_context_operator_20261031.run import configure, read
from ..fixed_test_protocol import score_fixed_test
from ..lifecycle_stream_20261025.protocol import AccessLedger
from ..lifecycle_stream_20261025.run import compact
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .run import ARMS
from .normalized_forward import coordinate_signal, forward_coordinates

OLD = io23.ROOT / "probe_outputs/normal_reference_alignment_v1"
OLD_SHA = "b7ced90e01b7b3b05f474316c075d240b4e3df1709f4470aca07a95279938b69"
PLAN = io23.ROOT / "NORMAL_REFERENCE_COORDINATES_PLAN_20261101.md"


@torch.no_grad()
def predict(base, raw_signal, mapping):
    center, scale = base.center.cpu().numpy(), base.scale.cpu().numpy()
    coordinate = coordinate_signal(raw_signal, mapping, center, scale)
    active = torch.as_tensor(mapping["active"], device=base.center.device)
    scores = []
    for lo in range(0, len(coordinate["values"]), 32):
        ends = range(lo, min(lo + 32, len(coordinate["values"])))
        x, m = batch(raw_signal, ends, base.center.device)
        levels, lmask = batch(coordinate, ends, base.center.device)
        assert torch.equal(m, lmask)
        scores.append(
            forward_coordinates(base, x, m, levels, active)["logit"]
            .sigmoid()
            .cpu()
            .numpy()
        )
    observed = coordinate["available"] & np.asarray(mapping["active"])
    observed_count = int(observed.sum())
    issues = dict(
        observed_active_points=observed_count,
        # No observed coordinate has no maximum; this is not a measured zero.
        max_abs_mapped_coordinate=(
            float(abs(coordinate["values"][observed]).max()) if observed_count else None
        ),
        negative_variance={},
    )
    for c in (17, 22):
        zero_coordinate = np.arcsinh(-float(center[c]) / float(scale[c]))
        issues["negative_variance"][io23.RAW23[c]] = int(
            (
                (coordinate["values"][:, c] < zero_coordinate)
                & coordinate["available"][:, c]
            ).sum()
        )
    return np.concatenate(scores), issues


def main(root):
    if root.exists():
        raise FileExistsError(root)
    assert sha(OLD / "completed.json") == OLD_SHA
    old_p = read(OLD / "protocol.json")
    assert read(OLD / "independent_audit.json")["status"] == "PASS"
    configure()
    started = time.monotonic()
    base, parent = load_parent(DEFAULT_PARENT)
    base.cuda().eval()
    initial = {k: v.cpu().clone() for k, v in base.state_dict().items()}
    assert sha(DEFAULT_PARENT) == old_p["source_sha256"]
    threshold = float(parent["threshold"])
    assert threshold == old_p["threshold"]
    ids = [r["shot"] for r in old_p["order"]]
    ledger = AccessLedger(ids, 20)
    reference = Path(old_p["reference"])
    hashes = read(reference / "input_hashes.json")
    root.mkdir(parents=True)
    (root / "shots").mkdir()
    files = set(Path(__file__).parent.glob("*.py")) | {PLAN}
    for name, module in list(sys.modules.items()):
        if name.startswith("experiments.risk_aligned_online.") and getattr(
            module, "__file__", None
        ):
            files.add(Path(module.__file__))
    protocol = dict(
        scope="DEVELOPMENT_COMPLETE360_SAME_MAP_STABLE_COORDINATE_INTERFACE",
        arms=ARMS,
        original_map_run=str(OLD),
        original_completed_sha256=OLD_SHA,
        original_audit_sha256=sha(OLD / "independent_audit.json"),
        source_checkpoint=str(DEFAULT_PARENT),
        source_sha256=old_p["source_sha256"],
        order=old_p["order"],
        threshold=threshold,
        strict_lead_ms=30,
        warmup=20,
        warmup_in_primary=True,
        first_affected_rank=21,
        same_map_no_refit=True,
        no_coordinate_clipping=True,
        neural_updates=0,
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
    maps = {}
    density = None
    rows = {name: [] for name in ARMS}
    comparisons = []
    receipts = []
    for index, sid in enumerate(ids):
        assert sha(io23.path_for("EAST", sid)) == hashes[str(sid)]
        prefix = f"{index + 1:04d}_{sid}"
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = {k: raw[k] for k in ("values", "available")}
        curves = {"source": original_predictions(base, signal)}
        issues = {}
        if index < 20:
            curves.update({name: curves["source"].copy() for name in ARMS[1:]})
        else:
            curves["density"] = original_predictions(base, transform(signal, density))
            for name, fit in maps.items():
                curves[name], issues[name] = predict(base, signal, fit)
        with np.load(OLD / "shots" / f"{prefix}.npz", allow_pickle=False) as old:
            assert np.array_equal(old["times"], raw["times"])
            for name in ("source", "density"):
                assert np.array_equal(old[name], curves[name]), (
                    sid,
                    name,
                    "reference drift",
                )
            for name in ARMS[2:]:
                if name in old.files:
                    comparisons.append(
                        dict(
                            shot=sid,
                            index=index + 1,
                            arm=name,
                            max_probability_error=float(
                                abs(old[name] - curves[name]).max()
                            ),
                            threshold_point_differences=int(
                                (
                                    (old[name] >= threshold)
                                    != (curves[name] >= threshold)
                                ).sum()
                            ),
                        )
                    )
        if any(not np.isfinite(v).all() for v in curves.values()):
            raise FloatingPointError("nonfinite complete risk curve")
        path = root / "shots" / f"{prefix}.npz"
        with path.open("xb") as handle:
            np.savez_compressed(handle, times=raw["times"], **curves)
            handle.flush()
            os.fsync(handle.fileno())
        digest = sha(path)
        ledger.seal(sid, digest)
        seal = dict(
            shot=sid,
            index=index + 1,
            sha256=digest,
            outcome_available=False,
            input_diagnostics=issues,
        )
        _durable_json(root / "shots" / f"{prefix}_seal.json", seal)
        receipts.append(seal)
        ledger.reveal(sid)
        outcome = simple_data.attach_outcome(raw, "EAST", {})
        for name, prediction in curves.items():
            valid = outcome["valid"]
            rows[name].append(
                dict(
                    shot=sid,
                    times=raw["times"][valid],
                    scores=prediction[valid],
                    is_disrupt=outcome["is_disrupt"],
                    down=outcome["down"],
                )
            )
        if index == 19:
            density = read(OLD / "maps/density.json")
            assert density["calibration_ids"] == ids[:20]
            for name in ARMS[2:]:
                fit = read(OLD / "maps" / f"{name}.json")
                assert fit["first_affected_rank"] == 21
                assert set(fit["target"]["shot_ids"]) <= set(ids[:20])
                maps[name] = fit
            _durable_json(
                root / "map_access.json",
                dict(
                    loaded_after_rank=20,
                    ids=ids[:20],
                    seals={str(r["shot"]): r["sha256"] for r in receipts},
                    map_sha256={
                        name: sha(OLD / "maps" / f"{name}.json")
                        for name in ("density", *maps)
                    },
                ),
            )
        if (index + 1) % 30 == 0:
            progress = dict(
                shots=index + 1, total=360, seconds=time.monotonic() - started
            )
            _durable_json(root / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    assert len(ledger.history) == 360 and all(
        torch.equal(v, base.state_dict()[k].cpu()) for k, v in initial.items()
    )
    assert (
        not base.embedding._forward_pre_hooks
        and sha(DEFAULT_PARENT) == old_p["source_sha256"]
    )
    result = {}
    per_shot = {}
    for name, values in rows.items():
        per_shot[name] = score_fixed_test("EAST", values, threshold)["predictions"]
        result[name] = dict(
            full360=compact(values, threshold),
            warmup20=compact(values[:20], threshold),
            suffix340=compact(values[20:], threshold),
        )
    _durable_json(root / "per_shot.json", per_shot)
    _durable_json(root / "receipts.json", receipts)
    _durable_json(root / "roundtrip_comparison.json", comparisons)
    completed = dict(
        status="COMPLETED",
        metrics=result,
        seconds=time.monotonic() - started,
        source_bitwise_unchanged=True,
        reference_curves_exact=720,
        neural_updates=0,
        no_hook_remaining=True,
        scientific_goal_achieved=False,
        existing_prefix_comparisons=len(comparisons),
        existing_prefix_max_error=max(r["max_probability_error"] for r in comparisons),
        existing_prefix_threshold_differences=sum(
            r["threshold_point_differences"] for r in comparisons
        ),
    )
    _durable_json(root / "completed.json", completed)
    print(json.dumps(completed), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
