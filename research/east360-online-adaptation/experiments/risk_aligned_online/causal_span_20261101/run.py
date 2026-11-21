"""One fixed d2 sensitivity experiment, scoring every original endpoint."""

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
from ..east_context_operator_20261031.run import configure, read
from ..lifecycle_stream_20261025.audit import classify, counts, flips
from ..lifecycle_stream_20261025.run import build_adapter
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..slow_fast_attribution_20261031.run import REFERENCE, REFERENCE_SHA
from ..warmstart_alignment_20261022.alignment import transform
from .core import span_windows

PLAN = io23.ROOT / "CAUSAL_SPAN_PLAN_20261101.md"
ARMS = ("d1", "d2")


@torch.no_grad()
def predict(model, signal):
    parts = {a: [] for a in ARMS}
    for lo in range(0, len(signal["values"]), 32):
        ends = np.arange(lo, min(lo + 32, len(signal["values"])))
        for arm, stride in (("d1", 1), ("d2", 2)):
            x, m, _ = span_windows(signal, ends, stride)
            x, m = torch.as_tensor(x, device="cuda"), torch.as_tensor(m, device="cuda")
            h, valid = model.tokens(x, m)
            z = model.risk(h, valid, h.new_zeros(len(h), 92))
            if not torch.isfinite(z).all():
                raise FloatingPointError("nonfinite history-span prediction")
            parts[arm].append(z.sigmoid().cpu().numpy())
    result = {a: np.concatenate(v) for a, v in parts.items()}
    assert np.array_equal(result["d1"][:254], result["d2"][:254])
    return result


def main(root):
    if root.exists():
        raise FileExistsError(root)
    configure()
    start = time.monotonic()
    assert sha(REFERENCE / "completed.json") == REFERENCE_SHA
    order = read(REFERENCE / "order.json")
    hashes = read(REFERENCE / "input_hashes.json")
    mapping = read(REFERENCE / "input_map.json")
    base, parent = load_parent(DEFAULT_PARENT)
    base.cuda().eval()
    model = build_adapter(base, readout="full", feedback="progressive", context=True)
    cold = REFERENCE / "models/cold.pt"
    model.restore(torch.load(cold, map_location="cpu", weights_only=True))
    model.eval().requires_grad_(False)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    threshold = float(parent["threshold"])
    root.mkdir(parents=True)
    (root / "shots").mkdir()
    files = set(Path(__file__).parent.glob("*.py")) | {PLAN}
    for name, module in list(sys.modules.items()):
        if name.startswith("experiments.risk_aligned_online.") and getattr(
            module, "__file__", None
        ):
            files.add(Path(module.__file__))
    protocol = dict(
        scope="DEVELOPMENT_COMPLETE360_FIXED_CAUSAL_SPAN_DIAGNOSTIC",
        order=order,
        arms=ARMS,
        reference=str(REFERENCE),
        reference_completed_sha256=REFERENCE_SHA,
        source_checkpoint=str(DEFAULT_PARENT),
        source_sha256=sha(DEFAULT_PARENT),
        cold_state=str(cold),
        cold_state_sha256=sha(cold),
        input_map_sha256=sha(REFERENCE / "input_map.json"),
        threshold=threshold,
        first20_source_cost_included=True,
        first_d2_endpoint=254,
        first_d2_shot_rank=21,
        insufficient_history="original d1 window, every endpoint retained",
        physical_time_axis_changed=False,
        neuron_parameter_updates=0,
        independent_confirmation=False,
        plan_sha256=sha(PLAN),
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
    )
    _durable_json(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))
    rows = {a: [] for a in ARMS}
    receipts = []
    for index, record in enumerate(order):
        sid = record["shot"]
        prefix = f"{index + 1:04d}_{sid}"
        assert sha(io23.path_for("EAST", sid)) == hashes[str(sid)]
        s = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        with np.load(REFERENCE / "shots" / f"{prefix}.npz", allow_pickle=False) as ref:
            assert np.array_equal(s["times"], ref["times"])
            if index < 20:
                curves = {a: ref["source"].copy() for a in ARMS}
            else:
                curves = predict(
                    model,
                    transform({k: s[k] for k in ("values", "available")}, mapping),
                )
            assert np.array_equal(curves["d1"], ref["cold"]), (sid, "cold drift")
        path = root / "shots" / f"{prefix}.npz"
        with path.open("xb") as handle:
            np.savez_compressed(handle, times=s["times"], **curves)
            handle.flush()
            os.fsync(handle.fileno())
        receipt = dict(
            index=index + 1,
            shot=sid,
            sha256=sha(path),
            outcome_available=False,
            full_d2_endpoints=max(0, len(s["values"]) - 254) if index >= 20 else 0,
        )
        _durable_json(root / "shots" / f"{prefix}_seal.json", receipt)
        receipts.append(receipt)
        flag, down = io23.read_outcome("EAST", sid)
        for a, probability in curves.items():
            rows[a].append(
                dict(
                    shot=sid,
                    disruptive=flag,
                    **classify(s["times"], probability, flag, down, threshold),
                )
            )
        if (index + 1) % 20 == 0:
            progress = dict(
                shots=index + 1, total=360, seconds=time.monotonic() - start
            )
            _durable_json(root / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in state.items())
    metrics = {
        a: dict(full360=counts(r), warmup20=counts(r[:20]), suffix340=counts(r[20:]))
        for a, r in rows.items()
    }
    _durable_json(root / "per_shot.json", rows)
    _durable_json(root / "receipts.json", receipts)
    result = dict(
        status="COMPLETED",
        metrics=metrics,
        flips=flips(rows["d1"], rows["d2"]),
        source_state_unchanged=True,
        reference_curves_exact=360,
        seconds=time.monotonic() - start,
        no_neural_adaptation=True,
        method_promoted=False,
        scientific_goal_achieved=False,
    )
    _durable_json(root / "completed.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
