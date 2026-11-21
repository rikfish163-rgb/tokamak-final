"""Prepare the diagnostic branch without changing the risk parent."""

from pathlib import Path
import time
from unittest.mock import patch

import torch

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.cold20_compare_20261104.data import make_schedule
from experiments.risk_aligned_online.direct_fast_memory_20261104.timeline import (
    window_batch,
)
from experiments.risk_aligned_online.history_gradient_metric_20261102.run import write
from experiments.risk_aligned_online.joint_lora_20261104.model import JointLoRA
from experiments.risk_aligned_online.joint_lora_20261104.run import PRIOR, load_initial
from experiments.risk_aligned_online.joint_lora_20261104.training import (
    ended_records,
    training_batch,
)
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import (
    load_inputs,
    prepare_actual,
)
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    load_parent,
)
from joint_lora_gradient_probe_20261104 import sha, state_digest
from joint_lora_gradient_span_20261104 import RUN, read_json

ROOT = io23.ROOT
PLAN = ROOT / "JOINT_LORA_DECODER_ONLY_PLAN_20261104.md"
OUTPUT = ROOT / "probe_outputs/joint_lora_decoder_only_warm_20261104_v1"


def fit_decoder(model, latent, prior, target, mask, schedule):
    if (
        latent.shape != (len(prior), 192)
        or prior.shape != (len(prior), 168)
        or target.shape != prior.shape
        or mask.shape != prior.shape
        or mask.dtype != torch.bool
    ):
        raise ValueError("matched decoder-only targets and features required")
    if not all(torch.isfinite(x).all() for x in (latent, prior, target[mask])):
        raise ValueError("nonfinite observed decoder data")
    latent, prior, target, mask = (x.detach() for x in (latent, prior, target, mask))
    model.eval().requires_grad_(False)
    decoder = model.residual_decoder.requires_grad_(True)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)
    history = []
    for i, draw in enumerate(schedule, 1):
        ids = draw["indices"]
        if draw["step"] != i or not ids or min(ids) < 0 or max(ids) >= len(prior):
            raise ValueError("invalid fixed decoder schedule")
        support = mask[ids]
        if not support.any():
            raise ValueError("fixed decoder batch has no observed diagnostics")
        pred = prior[ids] + decoder(latent[ids])
        # Mask before squaring, so an unavailable NaN cannot contaminate grads.
        error = torch.where(support, pred - target[ids], 0.0)
        mse = error.square().sum() / support.sum()
        optimizer.zero_grad(set_to_none=True)
        (0.1 * mse).backward()
        norm = torch.nn.utils.clip_grad_norm_(
            decoder.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        history.append(dict(step=i, mse=float(mse.detach()), gradient_norm=float(norm)))
    optimizer.zero_grad(set_to_none=True)
    model.requires_grad_(False)
    return history


def non_decoder_digest(model):
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("residual_decoder.")
    }


@torch.no_grad()
def cache_records(base, model, shots, records):
    latent, prior, targets, masks = [], [], [], []
    for lo in range(0, len(records), 128):
        x, a, y, m, _ = training_batch(
            model, shots, records[lo : lo + 128], device="cpu"
        )
        result = model(x, a)
        if not torch.equal(result["logit"], base(x, a)["logit"]):
            raise AssertionError(
                "zero LoRA must preserve native risk on each real train window"
            )
        for buffer, value in zip(
            (latent, prior, targets, masks),
            (result["latent"], result["diagnostic"], y, m),
            strict=True,
        ):
            buffer.append(value)
    return tuple(torch.cat(parts) for parts in (latent, prior, targets, masks))


@torch.no_grad()
def early_probe(base, model, shots):
    count = 0
    for signal in shots.values():
        endpoints = range(min(135, len(signal["values"])))
        x, a = window_batch(signal, endpoints, device="cpu")
        if not torch.equal(model(x, a)["logit"], base(x, a)["logit"]):
            raise AssertionError("decoder must not influence early risk")
        count += len(endpoints)
    return count


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    inputs = load_inputs()
    ids = inputs["ids"][:20]
    paths = [
        DEFAULT_PARENT,
        RUN / "warm.json",
        RUN / "warm.pt",
        RUN / "protocol.json",
        PRIOR,
        PLAN,
        Path(__file__),
    ]
    bound = {**inputs["bound"], **{str(p.resolve()): sha(p) for p in paths}}
    warm = read_json(RUN / "warm.json")
    protocol = read_json(RUN / "protocol.json")
    if warm["label_shots"] != ids or protocol["order"] != inputs["order"]:
        raise ValueError("exact same original warm20 and order required")
    reader, outcome_reader = io23.read_signals, io23.read_outcome
    open_signals, open_outcomes = [], []

    def signals(device, sid, *args, **kwargs):
        if device != "EAST" or sid not in ids:
            raise ValueError("only original warm20 may be read")
        open_signals.append(sid)
        return reader(device, sid, *args, **kwargs)

    def outcomes(device, sid):
        if device != "EAST" or sid not in ids or sid not in open_signals:
            raise ValueError("only original sealed warm20 outcomes allowed")
        open_outcomes.append(sid)
        return outcome_reader(device, sid)

    shots, truth = {}, {}
    with (
        patch.object(io23, "read_signals", side_effect=signals),
        patch.object(io23, "read_outcome", side_effect=outcomes),
    ):
        for i, sid in enumerate(ids, 1):
            stem = RUN / "shots" / f"{i:04d}_{sid}"
            seal = read_json(stem.with_name(stem.name + "_seal.json"))
            if (
                seal["shot"] != sid
                or seal["current_outcome_used_in_prediction"] is not False
                or sha(stem.with_suffix(".npz")) != seal["curve_sha256"]
            ):
                raise ValueError(
                    "sealed startup prediction required before outcome read"
                )
            if sha(io23.path_for("EAST", sid)) != inputs["hashes"][str(sid)]:
                raise ValueError("ended raw signals changed")
            raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
            shots[sid] = dict(
                prepare_actual(raw, 21, inputs["mapping"]),
                device="EAST",
                times=raw["times"],
            )
            truth[sid] = io23.read_outcome("EAST", sid)
    records, pools = ended_records(shots, truth, ids)
    schedule = make_schedule(pools, steps=512, batch_size=128, seed=131)
    if records != warm["records"] or schedule != warm["schedule"]:
        raise ValueError("matched record/schedule changed")
    base, parent = load_parent(DEFAULT_PARENT)
    base.eval().requires_grad_(False)
    source_before = state_digest(base)
    if (
        source_before != protocol["source_state_sha256"]
        or parent["threshold"] != protocol["threshold"]
    ):
        raise ValueError("source state or threshold changed")
    w0 = load_initial(ids, "cpu")
    model = JointLoRA(base, w0).eval().requires_grad_(False)
    original = non_decoder_digest(model)
    decoder_before = state_digest(model.residual_decoder)
    cache = cache_records(base, model, shots, records)
    history = fit_decoder(model, *cache, schedule)
    decoder_after = state_digest(model.residual_decoder)
    if (
        decoder_before == decoder_after
        or state_digest(base) != source_before
        or any(
            not torch.equal(model.state_dict()[k], value)
            for k, value in original.items()
        )
    ):
        raise AssertionError("only the decoder may change, and it must actually change")
    early = early_probe(base, model, shots)
    if (
        open_signals != ids
        or open_outcomes != ids
        or any(sha(Path(p)) != v for p, v in bound.items())
    ):
        raise AssertionError("source/input/access changed")
    OUTPUT.mkdir()
    write(
        OUTPUT / "warm.json",
        dict(
            label_shots=ids,
            records=records,
            schedule=schedule,
            history=history,
            fitted_state_sha256=state_digest(model),
            threshold=parent["threshold"],
            source_state_sha256=source_before,
            decoder_before=decoder_before,
            decoder_after=decoder_after,
            lora_updates=0,
            training_risk_parity_points=len(records),
            early_risk_parity_points=early,
            independent_validation=False,
            input_sha256=bound,
        ),
    )
    torch.save(model.state_dict(), OUTPUT / "warm.pt")
    write(
        OUTPUT / "completed.json",
        dict(
            status="COMPLETED_RISK_NEUTRAL_DECODER_ONLY_PREPARATION",
            elapsed_seconds=time.monotonic() - started,
            artifact_sha256={
                name: sha(OUTPUT / name) for name in ("warm.json", "warm.pt")
            },
        ),
    )
    print(
        dict(
            records=len(records),
            steps=len(history),
            early_risk_parity_points=early,
            final_mse=history[-1]["mse"],
            elapsed_seconds=time.monotonic() - started,
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
