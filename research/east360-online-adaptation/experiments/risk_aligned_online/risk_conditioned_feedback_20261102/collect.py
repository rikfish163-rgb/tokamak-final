"""Collect fixed-parent conditions using only each feedback's matured prefix."""

import argparse
import json
from numbers import Integral
from pathlib import Path
import sys
import time
import zipfile

import torch

from .. import io23, simple_data
from ..adaptivity_capacity_20261102.model import source_basis
from ..causal_input import apply_input_contract
from ..dual_history_ttt_20261101.run import load_model
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lora_stream_20261102.core import encode, forward
from ..warmstart_alignment_20261022.alignment import transform

SOURCE = io23.ROOT / "probe_outputs/channel_feedback_gradients20_v1"
SOURCE_SHA = "91b5793fae75961dd07bc62cac67866426c1d3e04a096eca1b0f8282a80d73d6"
PLAN = io23.ROOT / "RISK_CONDITIONED_FEEDBACK_PLAN_20261102.md"


def maturity_index(query):
    if isinstance(query, bool) or not isinstance(query, Integral) or query < 160:
        raise ValueError("supported integer query >=160 required")
    return int(query) - 1


def main(root):
    if root.exists():
        raise FileExistsError(root)
    started = time.monotonic()
    configure()
    assert sha(SOURCE / "completed.json") == SOURCE_SHA
    parent, done = read(SOURCE / "protocol.json"), read(SOURCE / "completed.json")
    assert sha(SOURCE / "records.json") == done["records_sha256"]
    rows, ids = read(SOURCE / "records.json"), parent["ids"]
    assert len(rows) == 100 and len(ids) == 20
    model, _, cold = load_model()
    assert sha(cold) == parent["cold_state_sha256"]
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    basis = source_basis(model)
    source_probe = Path(parent["source_probe"])
    source_protocol = read(source_probe / "protocol.json")
    mapping_path = Path(source_protocol["calibration_reference"]) / "input_map.json"
    assert sha(mapping_path) == parent["input_map_sha256"]
    mapping = read(mapping_path)
    files = set(Path(__file__).parent.glob("*.py")) | {PLAN}
    for name, module in list(sys.modules.items()):
        if name.startswith("experiments.risk_aligned_online.") and getattr(
            module, "__file__", None
        ):
            files.add(Path(module.__file__))
    root.mkdir(parents=True)
    protocol = dict(
        scope="ENDED20_CAUSAL_CONDITION_COLLECTION_NOT_ONLINE",
        source_collection=str(SOURCE),
        source_completed_sha256=SOURCE_SHA,
        records_sha256=done["records_sha256"],
        ids=ids,
        source_model_sha256=parent["source_model_sha256"],
        cold_state_sha256=sha(cold),
        input_map_sha256=sha(mapping_path),
        zero_temporary_state=True,
        max_signal_index="query-1",
        new_outcome_reads=0,
        plan_sha256=sha(PLAN),
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
    )
    write(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))
    collected = {}
    zero = torch.zeros(4, 64, device="cuda")
    for number, sid in enumerate(ids, 1):
        assert sha(io23.path_for("EAST", sid)) == parent["input_hashes"][str(sid)]
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
        for row in rows:
            if row["shot"] != sid:
                continue
            end = maturity_index(row["query"])
            # Physically truncate even the in-memory array. The encoder cannot
            # access a future row accidentally in this condition collection.
            prefix = {k: v[: end + 1] for k, v in signal.items()}
            pack = encode(model, prefix, [end], basis)
            with torch.no_grad():
                z = forward(model, pack, zero)["logit"]
            collected[(sid, row["query"])] = dict(
                shot=sid,
                query=row["query"],
                maturity=end,
                logit=float(z),
                condition=float(z.sigmoid()),
            )
        print(json.dumps(dict(completed_shots=number, total=20)), flush=True)
    conditions = [collected[(r["shot"], r["query"])] for r in rows]
    assert len(conditions) == 100 and all(0 <= r["condition"] <= 1 for r in conditions)
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
    write(root / "conditions.json", conditions)
    result = dict(
        status="COMPLETED_CONDITION_COLLECTION",
        n_samples=100,
        n_shots=20,
        conditions_sha256=sha(root / "conditions.json"),
        source_tensors_unchanged=True,
        raw_signal_ids_read=ids,
        raw_outcome_reads=0,
        online_gain_measured=False,
        seconds=time.monotonic() - started,
    )
    write(root / "completed.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
