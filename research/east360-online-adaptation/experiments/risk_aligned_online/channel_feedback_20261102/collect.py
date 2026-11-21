"""Ended20-only GPU collection of channelwise signal-gradient contributions."""

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
from ..dual_history_ttt_20261101.run import digest_array, load_model
from ..dual_history_ttt_20261101.stream import targets
from ..east_context_operator_20261031.run import configure, read
from ..lora_stream_20261102.core import encode, forward
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .loss import channel_losses

SOURCE = io23.ROOT / "probe_outputs/difference_feedback_probe20_v1"
SOURCE_SHA = "2af94de43602e66cf7936b2fd9ccf7943759c42d31fb09a109941ad29d79fe5d"
PLAN = io23.ROOT / "CHANNEL_FEEDBACK_PLAN_20261102.md"


def main(root):
    if root.exists():
        raise FileExistsError(root)
    configure()
    started = time.monotonic()
    assert sha(SOURCE / "completed.json") == SOURCE_SHA
    parent, done = read(SOURCE / "protocol.json"), read(SOURCE / "completed.json")
    assert read(SOURCE / "verification_records.json")["status"] == "PASS"
    assert read(SOURCE / "verification_neural_v2.json")["status"] == "PASS"
    assert sha(SOURCE / "samples.json") == done["samples_sha256"]
    samples, ids = read(SOURCE / "samples.json"), parent["ids"]
    assert (
        len(ids) == 20
        and len(samples) == 100
        and {r["shot"] for r in samples} == set(ids)
    )
    model, _, cold = load_model()
    assert sha(cold) == parent["cold_state_sha256"]
    original = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    basis = source_basis(model)
    calibration = Path(parent["calibration_reference"])
    assert sha(calibration / "input_map.json") == parent["input_map_sha256"]
    mapping = read(calibration / "input_map.json")
    files = set(Path(__file__).parent.glob("*.py")) | {PLAN}
    for name, module in list(sys.modules.items()):
        if name.startswith("experiments.risk_aligned_online.") and getattr(
            module, "__file__", None
        ):
            files.add(Path(module.__file__))
    root.mkdir(parents=True)
    protocol = dict(
        scope="ENDED20_CHANNEL_SIGNAL_GRADIENT_COLLECTION_NOT_ONLINE",
        source_probe=str(SOURCE),
        source_completed_sha256=SOURCE_SHA,
        samples_sha256=done["samples_sha256"],
        ids=ids,
        feature_names=list(io23.RAW23),
        source_model_sha256=parent["source_sha256"],
        cold_state_sha256=sha(cold),
        input_map_sha256=parent["input_map_sha256"],
        input_hashes=parent["input_hashes"],
        modes=["real", "reverse"],
        zero_state=True,
        new_outcome_reads=0,
        source_task_gradients_reused=True,
        plan_sha256=sha(PLAN),
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
    )
    _durable_json(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for f in sorted(files):
            archive.write(f, str(f.relative_to(io23.ROOT)))
    gradients = {mode: [] for mode in ("real", "reverse")}
    counts, records = [], []
    for i, sid in enumerate(ids, 1):
        assert sha(io23.path_for("EAST", sid)) == parent["input_hashes"][str(sid)]
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({key: raw[key] for key in ("values", "available")}, mapping)
        for row in samples:
            if row["shot"] != sid:
                continue
            q, origin = row["query"], row["origin"]
            point = encode(model, signal, [origin], basis)
            truth, available = targets(signal, origin, q - 1)
            assert digest_array(available) == row["mask_sha256"]
            variable = torch.zeros(4, 64, device="cuda", requires_grad=True)
            prediction = forward(model, point, variable, forecast=True)["forecast"]
            assert (
                digest_array(prediction.detach().cpu().numpy())
                == row["forecast_sha256"]
            )
            mask = torch.as_tensor(available, device="cuda")
            hold = torch.as_tensor(truth["hold"], device="cuda")
            errors = {}
            for mode in ("real", "reverse"):
                assert digest_array(truth[mode]) == row["target_sha256"][mode]
                values = channel_losses(
                    model,
                    prediction,
                    torch.as_tensor(truth[mode], device="cuda"),
                    hold,
                    mask,
                )
                parts = []
                for channel in range(23):
                    retain = not (mode == "reverse" and channel == 22)
                    value = torch.autograd.grad(
                        values[channel], variable, retain_graph=retain
                    )[0]
                    parts.append(value.detach().flatten().cpu().numpy())
                array = np.stack(parts)
                reference = np.asarray(
                    row["results"]["difference_" + mode]["receipt"]["gradient256"]
                )
                summed = array.astype(np.float64).sum(0)
                assert np.allclose(summed, reference, atol=2e-7, rtol=2e-4)
                errors[mode] = float(np.max(np.abs(summed - reference)))
                gradients[mode].append(array)
            observed_counts = available.sum((0, 1))
            assert np.all(
                np.stack([gradients[m][-1] for m in gradients])[:, observed_counts == 0]
                == 0
            )
            counts.append(observed_counts)
            records.append(
                dict(
                    shot=sid,
                    query=q,
                    label=row["label"],
                    disruptive=row["disruptive"],
                    channel_sum_max_absolute=errors,
                )
            )
        print(
            json.dumps(dict(completed_shots=i, total=20, samples=len(records))),
            flush=True,
        )
    assert len(records) == 100
    assert all(
        torch.equal(value, model.state_dict()[key].cpu())
        for key, value in original.items()
    )
    source_rows = {(r["shot"], r["query"]): r for r in samples}
    task = np.asarray(
        [source_rows[(r["shot"], r["query"])]["task_gradient256"] for r in records],
        np.float32,
    )
    with (root / "gradients.npz").open("xb") as handle:
        np.savez_compressed(
            handle,
            real=np.stack(gradients["real"]),
            reverse=np.stack(gradients["reverse"]),
            task=task,
            observations=np.stack(counts),
        )
    _durable_json(root / "records.json", records)
    result = dict(
        status="COMPLETED_CHANNEL_GRADIENT_COLLECTION",
        n_shots=20,
        n_samples=100,
        source_tensors_unchanged=True,
        raw_signal_ids_read=ids,
        raw_outcome_reads=0,
        gradients_sha256=sha(root / "gradients.npz"),
        records_sha256=sha(root / "records.json"),
        max_sum_difference={
            mode: max(r["channel_sum_max_absolute"][mode] for r in records)
            for mode in gradients
        },
        online_gain_measured=False,
        independent_future_confirmation=False,
        seconds=time.monotonic() - started,
    )
    _durable_json(root / "completed.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
