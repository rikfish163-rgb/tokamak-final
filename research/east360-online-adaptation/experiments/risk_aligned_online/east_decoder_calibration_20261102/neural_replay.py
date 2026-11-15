"""Replay four fixed complete trajectories and actual first-FFN hook batches."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..direct_decoder_meta_20261102.core import clone_head
from ..dual_history_ttt_20261101.run import load_model
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..lora_stream_20261102.core import direct, encode, forward
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT
from ..source_event_tasks_20261102.data import SOURCE_SHA
from ..source_event_tasks_20261102.trial import source_basis
from ..warmstart_alignment_20261022.alignment import transform
from .core import LowRankDecoder
from .run import SOURCE
from .stream360 import ARMS, CV, REFERENCE, predict

IDS = (74087, 73403, 63656, 62573)


def state_at(records, query):
    if query < 0:
        raise ValueError("nonnegative query required")
    state = np.zeros((4, 64), dtype=np.float32)
    previous = -1
    for r in records:
        if r["first_affected"] != r["maturity"] + 1 or r["maturity"] <= previous:
            raise ValueError("strictly ordered seal-then-update clock required")
        previous = r["maturity"]
        if r["first_affected"] <= query:
            value = np.asarray(r["state256_after"], dtype=np.float32)
            if value.shape != (256,) or not np.isfinite(value).all():
                raise ValueError("finite first-FFN state required")
            state = value.reshape(4, 64)
    return state


def main(root):
    root = Path(root)
    output = root / "verification_neural.json"
    if output.exists():
        raise FileExistsError(output)
    done, protocol = read(root / "completed.json"), read(root / "protocol.json")
    if done["status"] != "COMPLETED":
        raise ValueError("completed full360 experiment required")
    for path, digest in protocol["bound_code_sha256"].items():
        if sha(Path(path)) != digest:
            raise ValueError("bound code changed")
    if sha(root / "heads.pt") != read(root / "calibration.json")["heads_sha256"]:
        raise ValueError("calibrated decoder changed")
    configure()
    start = time.monotonic()
    model, _, cold = load_model()
    if sha(cold) != protocol["cold_sha256"] or sha(DEFAULT_PARENT) != SOURCE_SHA:
        raise ValueError("cold/source changed")
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    reference = read(REFERENCE / "protocol.json")
    mapping_path = Path(reference["calibration_reference"]) / "input_map.json"
    if sha(mapping_path) != protocol["input_map_sha256"]:
        raise ValueError("input mapping changed")
    mapping = read(mapping_path)
    hashes = read(Path(reference["calibration_reference"]) / "input_hashes.json")
    prior = clone_head(model)
    prior.load_state_dict(
        torch.load(SOURCE / "decoders.pt", map_location="cpu", weights_only=True)[
            "heads"
        ]["forecast_only"],
        strict=True,
    )
    prior.requires_grad_(False)
    basis = torch.load(
        CV / "diagnostic_basis.pt", map_location="cuda", weights_only=True
    )
    saved = torch.load(root / "heads.pt", map_location="cuda", weights_only=True)
    heads = {name: LowRankDecoder(prior, basis) for name in saved}
    for name, head in heads.items():
        with torch.no_grad():
            head.delta.copy_(saved[name])
        head.requires_grad_(False)
    heads["prior"] = LowRankDecoder(prior, basis).requires_grad_(False)
    ffn_basis = source_basis(model)
    ids = [r["shot"] for r in protocol["order"]]
    checks, actual_batches = [], 0
    max_step_rms, max_total_rms = 0.0, 0.0
    for sid in IDS:
        index = ids.index(sid) + 1
        if index <= 20 or sha(io23.path_for("EAST", sid)) != hashes[str(sid)]:
            raise ValueError("fixed suffix shot/raw identity changed")
        path = root / "shots" / f"{index:04d}_{sid}.npz"
        records_path = path.with_name(path.stem + "_updates.json")
        seal = read(path.with_name(path.stem + "_seal.json"))
        if (
            sha(path) != seal["curve_sha256"]
            or sha(records_path) != seal["updates_sha256"]
        ):
            raise ValueError("sealed replay artifact changed")
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        signal = transform({k: raw[k] for k in ("values", "available")}, mapping)
        with patch.object(
            io23,
            "read_outcome",
            side_effect=RuntimeError("outcomes forbidden in neural replay"),
        ):
            curves, receipts = predict(model, signal, ffn_basis, heads)
        if receipts != read(records_path):
            raise AssertionError("neural gradient/state/acceptance receipts changed")
        for arm in ARMS[1:]:
            before = model.base.center.new_zeros(4, 64)
            for r in receipts[arm]:
                end, origin = r["maturity"], r["origin"]
                lo = end // 32 * 32
                prefix = {k: v[: end + 1] for k, v in signal.items()}
                pack = encode(model, prefix, np.arange(lo, end + 1), ffn_basis)
                point = {k: v[origin - lo : origin - lo + 1] for k, v in pack.items()}
                after = torch.as_tensor(r["state256_after"], device="cuda").reshape(
                    4, 64
                )
                clean = point["phi"].double() @ (after - before).double()
                valid = point["valid"][:, :16]
                squared = clean.masked_fill(~valid[:, :, None], 0).square().sum()
                step_rms = float((squared / (valid.sum().clamp_min(1) * 64)).sqrt())
                with torch.no_grad():
                    total_rms = float(forward(model, point, after)["rms"].max())
                if step_rms > 0.05 + 1e-7 or total_rms > 0.25 + 1e-7:
                    raise AssertionError((sid, arm, end, "observed FFN RMS bound"))
                max_step_rms = max(max_step_rms, step_rms)
                max_total_rms = max(max_total_rms, total_rms)
                before = after
        with np.load(path, allow_pickle=False) as saved_curves:
            for arm in ARMS:
                if not np.array_equal(curves[arm], saved_curves[arm]):
                    raise AssertionError((sid, arm, "full trajectory mismatch"))
            for query in (0, 160, len(raw["times"]) - 1):
                lo = query // 32 * 32
                hi = min(lo + 32, len(raw["times"]))
                prefix = {k: v[:hi] for k, v in signal.items()}
                for arm in ARMS:
                    state = state_at([] if arm == "static" else receipts[arm], query)
                    state = torch.as_tensor(state, device="cuda")
                    with torch.no_grad():
                        actual = (
                            direct(model, prefix, np.arange(lo, hi), ffn_basis, state)[
                                "logit"
                            ]
                            .sigmoid()
                            .cpu()
                            .numpy()
                        )
                    if not np.array_equal(actual, curves[arm][lo:hi]):
                        raise AssertionError((sid, arm, query, "actual FFN mismatch"))
                    actual_batches += 1
        checks.append(
            dict(
                shot=sid,
                index=index,
                points=len(raw["times"]),
                events=len(receipts["worst_real"]),
            )
        )
        print(dict(replayed_shot=sid, complete=len(checks)), flush=True)
    if any(
        not torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items()
    ):
        raise AssertionError("source tensors changed during replay")
    result = dict(
        status="PASS",
        checks=checks,
        fixed_preselected_shots=list(IDS),
        curves_replayed=len(IDS) * len(ARMS),
        actual_ffn_hook_batches=actual_batches,
        max_replayed_step_hidden_rms=max_step_rms,
        max_replayed_origin_total_hidden_rms=max_total_rms,
        curve_gradient_state_acceptance_exact=True,
        source_unchanged=True,
        completed_sha256=sha(root / "completed.json"),
        verifier_sha256=sha(Path(__file__)),
        seconds=time.monotonic() - start,
        outcome_reads=0,
        scope="Same deployed algorithm replay plus actual FFN hooks; not independent algorithm reimplementation or training reproduction.",
    )
    write(output, result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
