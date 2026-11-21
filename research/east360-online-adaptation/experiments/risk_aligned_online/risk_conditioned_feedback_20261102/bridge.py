"""Ended20 diagnostic: is the old-origin risk direction useful at the later query?

No parameter updates, fitting, or threshold selection. This is not online gain.
"""

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
from ..lora_stream_20261102.core import encode, forward
from ..warmstart_alignment_20261022.alignment import transform


def main(output):
    if output.exists():
        raise FileExistsError(output)
    configure()
    started = time.monotonic()
    source = io23.ROOT / "probe_outputs/difference_feedback_probe20_v1"
    assert (
        sha(source / "completed.json")
        == "2af94de43602e66cf7936b2fd9ccf7943759c42d31fb09a109941ad29d79fe5d"
    )
    done, parent = read(source / "completed.json"), read(source / "protocol.json")
    assert sha(source / "samples.json") == done["samples_sha256"]
    rows = read(source / "samples.json")
    model, _, cold = load_model()
    assert sha(cold) == parent["cold_state_sha256"]
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    basis = source_basis(model)
    mapping_path = Path(parent["calibration_reference"]) / "input_map.json"
    assert sha(mapping_path) == parent["input_map_sha256"]
    mapping = read(mapping_path)
    records = []
    for number, sid in enumerate(parent["ids"], 1):
        assert sha(io23.path_for("EAST", sid)) == parent["input_hashes"][str(sid)]
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
        for row in rows:
            if row["shot"] != sid:
                continue
            gradients = {}
            for mode, end in [("origin", row["origin"]), ("query", row["query"])]:
                pack = encode(
                    model, {k: v[: end + 1] for k, v in signal.items()}, [end], basis
                )
                w = torch.zeros(4, 64, device="cuda", requires_grad=True)
                logit = forward(model, pack, w)["logit"]
                gradients[mode] = (
                    torch.autograd.grad(logit.sum(), w)[0]
                    .detach()
                    .flatten()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
            p = row["results"]["difference_real"]["before_probability"]
            predicted_bce = gradients["query"] * (p - row["label"])
            actual = np.asarray(row["task_gradient256"])
            np.testing.assert_allclose(predicted_bce, actual, atol=2e-7, rtol=2e-4)
            first, later = gradients["origin"], gradients["query"]
            cosine = float(
                np.clip(
                    first @ later / np.linalg.norm(first) / np.linalg.norm(later), -1, 1
                )
            )
            records.append(
                dict(
                    shot=sid,
                    query=row["query"],
                    origin=row["origin"],
                    label=row["label"],
                    disruptive=row["disruptive"],
                    origin_query_logit_cosine=cosine,
                    max_bce_gradient_difference=float(
                        np.max(np.abs(predicted_bce - actual))
                    ),
                    origin_logit_gradient256=first.tolist(),
                    query_logit_gradient256=later.tolist(),
                )
            )
        print(json.dumps(dict(completed_shots=number, total=20)), flush=True)
    assert len(records) == 100
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in before.items())
    summary = {}
    for name in ("class0", "class1", "safe"):
        group = [
            r
            for r in records
            if (not r["disruptive"] if name == "safe" else r["label"] == int(name[-1]))
        ]
        ids = {r["shot"] for r in group}
        summary[name] = dict(
            n=len(group),
            n_shots=len(ids),
            shot_balanced_mean=float(
                np.mean(
                    [
                        np.mean(
                            [
                                r["origin_query_logit_cosine"]
                                for r in group
                                if r["shot"] == sid
                            ]
                        )
                        for sid in ids
                    ]
                )
            ),
            point_quantiles=np.quantile(
                [r["origin_query_logit_cosine"] for r in group], [0, 0.25, 0.5, 0.75, 1]
            ).tolist(),
        )
    result = dict(
        status="COMPLETED_ORIGIN_QUERY_RISK_BRIDGE_DIAGNOSTIC",
        summary=summary,
        records=records,
        source_tensors_unchanged=True,
        neural_updates=0,
        new_outcome_reads=0,
        raw_signal_ids=parent["ids"],
        source_samples_sha256=sha(source / "samples.json"),
        cold_state_sha256=sha(cold),
        static_source_sha256=parent["source_sha256"],
        code_sha256=sha(Path(__file__)),
        diagnostic_only=True,
        online_gain_measured=False,
        independent_future_confirmation=False,
        seconds=time.monotonic() - started,
    )
    write(output, result)
    print(
        json.dumps(
            dict(status=result["status"], summary=summary, seconds=result["seconds"])
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
