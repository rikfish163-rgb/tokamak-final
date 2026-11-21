"""Reproduce fixed ended-shot cases and check actual first-FFN insertions."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .. import io23, simple_data
from ..adaptivity_capacity_20261102.model import source_basis
from ..causal_input import apply_input_contract
from ..dual_history_ttt_20261101.run import load_model
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lora_stream_20261102.core import direct
from ..warmstart_alignment_20261022.alignment import transform
from .pilot import build_example, evaluate

IDS = (71304, 69854, 70404, 75657)


def main(root):
    root = Path(root)
    target = root / "verification_neural.json"
    if target.exists():
        raise FileExistsError(target)
    p, done = read(root / "protocol.json"), read(root / "completed.json")
    assert done["status"] == "COMPLETED_LOCAL_NEURAL_SCREEN"
    assert read(root / "verification_records.json")["status"] == "PASS"
    assert sha(root / "records.json") == done["records_sha256"]
    for path, digest in p["bound_files_sha256"].items():
        assert sha(Path(path)) == digest
    for path, digest in p["code_sha256"].items():
        assert sha(io23.ROOT / path) == digest
    configure()
    started = time.monotonic()
    model, configuration, cold = load_model()
    assert (
        sha(cold) == p["cold_sha256"]
        and float(configuration["threshold"]) == p["threshold"]
    )
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    parent = read(
        io23.ROOT / "probe_outputs/difference_feedback_probe20_v1/protocol.json"
    )
    mapping = read(Path(parent["calibration_reference"]) / "input_map.json")
    context_heads = read(Path(p["context_source"]) / "context_heads.json")
    basis, records = source_basis(model), read(root / "records.json")
    checks, hooks = [], 0
    for sid in IDS:
        assert (
            sid in p["ids"]
            and sha(io23.path_for("EAST", sid)) == parent["input_hashes"][str(sid)]
        )
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
        selected = [r for r in records if r["shot"] == sid]
        assert len(selected) == 5
        for saved in selected:
            e = build_example(
                model,
                signal,
                basis,
                context_heads,
                sid=sid,
                query=saved["query"],
                flag=saved["disruptive"],
                lead=saved["actual_lead_ms"],
                point_label=saved["point_label125"],
            )
            path = root / f"fold{saved['fold']}_heads.json"
            assert sha(path) == done["fold_files_sha256"][path.name]
            heads = {
                a: torch.tensor(v, dtype=torch.float32, device="cuda")
                for a, v in read(path).items()
            }
            actual = dict(
                fold=saved["fold"], **evaluate(model, e, heads, p["threshold"])
            )
            assert actual == saved, (
                sid,
                e["query"],
                "state/gradient/risk/context replay",
            )
            if saved is selected[0]:
                for arm, value in saved["results"].items():
                    state = torch.tensor(
                        value["receipt"]["state256_after"], device="cuda"
                    ).reshape(4, 64)
                    with torch.no_grad():
                        z = direct(
                            model,
                            {k: v[: e["query"] + 1] for k, v in signal.items()},
                            np.asarray([e["query"]]),
                            basis,
                            state,
                        )["logit"]
                    assert float(z) == value["after_logit"], (
                        sid,
                        arm,
                        "uncached actual layer",
                    )
                    hooks += 1
        checks.append(dict(shot=sid, cases=5, methods=9))
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
    report = dict(
        status="PASS",
        fixed_ids=list(IDS),
        checks=checks,
        local_cases_replayed=20,
        neural_updates_replayed=180,
        actual_first_query_layer_hooks=hooks,
        risk_context_state_gradient_receipts_exact=True,
        raw_signal_shots_read=4,
        new_outcome_reads=0,
        suffix_shots_read=0,
        source_tensors_unchanged=True,
        completed_sha256=sha(root / "completed.json"),
        verifier_sha256=sha(Path(__file__)),
        seconds=time.monotonic() - started,
        scope="Fixed cases recomputed with deployed functions and 36 direct layer checks; not independent algorithm reimplementation or training rerun.",
    )
    write(target, report)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
