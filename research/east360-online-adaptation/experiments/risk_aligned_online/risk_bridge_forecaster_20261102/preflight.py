"""Four ended-shot numerical prototype; no head fitting or online result."""

import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..adaptivity_capacity_20261102.model import source_basis
from ..causal_input import apply_input_contract
from ..dual_history_ttt_20261101.run import load_model
from ..dual_history_ttt_20261101.stream import targets
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lora_stream_20261102.core import encode, forward
from ..warmstart_alignment_20261022.alignment import transform
from .core import auxiliary, outer_objective

PLAN = io23.ROOT / "RISK_BRIDGE_FORECASTER_PLAN_20261102.md"


def main(root):
    if root.exists():
        raise FileExistsError(root)
    configure()
    started = time.monotonic()
    source = io23.ROOT / "probe_outputs/difference_feedback_probe20_v1"
    assert (
        sha(source / "completed.json")
        == "2af94de43602e66cf7936b2fd9ccf7943759c42d31fb09a109941ad29d79fe5d"
    )
    parent, done = read(source / "protocol.json"), read(source / "completed.json")
    assert sha(source / "samples.json") == done["samples_sha256"]
    ids = [71304, 69854, 54157, 75657]
    rows = [r for r in read(source / "samples.json") if r["shot"] in ids]
    assert len(rows) == 20
    model, _, cold = load_model()
    assert sha(cold) == parent["cold_state_sha256"]
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    basis = source_basis(model)
    mapping_path = Path(parent["calibration_reference"]) / "input_map.json"
    assert sha(mapping_path) == parent["input_map_sha256"]
    mapping = read(mapping_path)
    files = set(Path(__file__).parent.glob("*.py")) | {PLAN}
    files.update(
        Path(m.__file__)
        for n, m in list(sys.modules.items())
        if n.startswith("experiments.risk_aligned_online.")
        and getattr(m, "__file__", None)
    )
    root.mkdir(parents=True)
    protocol = dict(
        scope="LOCAL_ZERO_HEAD_NUMERICAL_META_PREFLIGHT_NOT_PERFORMANCE",
        ids=ids,
        source_samples_sha256=sha(source / "samples.json"),
        cold_sha256=sha(cold),
        input_map_sha256=sha(mapping_path),
        source_sha256=parent["source_sha256"],
        head_parameters=575,
        head_optimizer_steps=0,
        raw_outcome_reads=0,
        plan_sha256=sha(PLAN),
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
    )
    write(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))
    records = []
    zero = torch.zeros(4, 64, device="cuda")
    for sid in ids:
        assert sha(io23.path_for("EAST", sid)) == parent["input_hashes"][str(sid)]
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
        for row in rows:
            if row["shot"] != sid:
                continue
            origin = encode(
                model,
                {k: v[: row["origin"] + 1] for k, v in signal.items()},
                [row["origin"]],
                basis,
            )
            q = row["query"]
            later = encode(
                model, {k: v[: q + 1] for k, v in signal.items()}, [q], basis
            )
            truth, mask = targets(signal, row["origin"], q - 1)
            table = torch.zeros(25, 23, device="cuda", requires_grad=True)
            with torch.no_grad():
                reference = forward(model, origin, zero, forecast=True)
                augmented = auxiliary(model, origin, zero, table)
                assert torch.equal(augmented, reference["forecast"])
            t, h, m = [
                torch.as_tensor(v, device="cuda")
                for v in (truth["real"], truth["hold"], mask)
            ]
            result = outer_objective(model, origin, later, t, h, m, row["label"], table)
            (gradient,) = torch.autograd.grad(result["meta"], table)
            assert torch.isfinite(gradient).all() and gradient.norm() > 0
            old = row["results"]["difference_real"]["receipt"]
            new = result["receipt"]
            np.testing.assert_allclose(
                new["gradient256"], old["gradient256"], atol=2e-7, rtol=2e-4
            )
            np.testing.assert_allclose(
                new["state256_after"], old["state256_after"], atol=2e-7, rtol=2e-4
            )
            assert new["accepted"] == old["accepted"]
            with torch.no_grad():
                # B is never an argument to risk forward; calling the
                # auxiliary branch with a nonzero B cannot mutate risk state.
                baseline = forward(model, later, zero)["logit"].clone()
                auxiliary(model, origin, zero, torch.ones_like(table) * 0.1)
                assert torch.equal(baseline, forward(model, later, zero)["logit"])
            records.append(
                dict(
                    shot=sid,
                    query=q,
                    label=row["label"],
                    accepted=new["accepted"],
                    head_meta_gradient_norm=float(gradient.double().norm()),
                    maximum_initial_gradient_difference=float(
                        np.max(
                            np.abs(np.array(new["gradient256"]) - old["gradient256"])
                        )
                    ),
                    head_meta_gradient575=gradient.detach().flatten().cpu().tolist(),
                    risk_before=float(result["risk_before"]),
                    risk_after=float(result["risk_after"].detach()),
                )
            )
        print(json.dumps(dict(completed_shot=sid, records=len(records))), flush=True)
    assert len(records) == 20
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
    write(root / "records.json", records)
    result = dict(
        status="COMPLETED_NUMERICAL_META_PREFLIGHT",
        samples=20,
        raw_signal_ids_read=ids,
        head_optimizer_steps=0,
        source_tensors_unchanged=True,
        meta_gradients_finite_nonzero=True,
        initial_candidate_matches_previous=True,
        numerical_prototype_only=True,
        neural_online_gain_measured=False,
        independent_future_confirmation=False,
        records_sha256=sha(root / "records.json"),
        seconds=time.monotonic() - started,
    )
    write(root / "completed.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
