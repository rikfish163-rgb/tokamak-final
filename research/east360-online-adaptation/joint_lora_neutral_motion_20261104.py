"""Posthoc replay of the common 65944 rescue: separate loss energy and motion."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.hdf5_score_recheck_20261103 import score_curve
from experiments.risk_aligned_online.history_gradient_metric_20261102.run import (
    read,
    sha,
)
from experiments.risk_aligned_online.joint_lora_20261104.timeline import (
    _masked_mse,
    _project,
    _risk_segment,
    _targets,
)
from experiments.risk_aligned_online.mask_repair_20261104.evaluation import _write
from joint_lora_neutral_stream_20261104 import (
    DEFAULT_PARENT,
    WARM,
    load_inputs,
    load_parent,
    load_risk_neutral,
    prepare_actual,
    state_digest,
)

MODES = ("real", "hold", "reverse", "motion", "anti")
ROOT = io23.ROOT / "probe_outputs"
RUN = ROOT / "joint_lora_neutral_stream360_20261104_v1"
AUDIT = ROOT / "joint_lora_neutral_prefix130_analysis_20261104_v1.json"
OUTPUT = ROOT / "joint_lora_neutral_motion_65944_20261104_v1"
SHOT, INDEX = 65944, 74


def replay_loss(parent, signal, mode):
    if mode not in MODES:
        raise ValueError("fixed matched diagnostic losses only")
    model = deepcopy(parent).eval().requires_grad_(False)
    parameters = model.neural_parameters()
    for param in parameters:
        param.requires_grad_(True)
    initial = tuple(p.detach().clone() for p in parameters)
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    length = len(signal["values"])
    origins = list(range(127, length, 32))
    mature = {o + 8: o for o in origins if o + 8 < length}
    bounds = sorted({0, length, *origins, *mature})
    curve = np.empty(length, dtype=np.float32)
    updates = 0
    for index, stop in zip(bounds, bounds[1:]):
        if index in mature:
            # This origin's prediction is unchanged between issue and arrival:
            # horizon8 < stride32, so there is no intervening update.
            x, a, y, mask = _targets(model, signal, mature[index], device="cpu")
            support = mask.sum()
            if support:
                before = tuple(p.detach().clone() for p in parameters)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x, a)["diagnostic"]
                if mode == "motion" or mode == "anti":
                    sign = -2 if mode == "motion" else 2
                    loss = sign * (prediction * y).masked_fill(~mask, 0).sum() / support
                else:
                    target = (
                        y
                        if mode == "real"
                        else (-y if mode == "reverse" else torch.zeros_like(y))
                    )
                    loss = _masked_mse(prediction, target, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
                optimizer.step()
                _project(parameters, initial, 0.5)
                updates += any(
                    not torch.equal(p, q)
                    for p, q in zip(parameters, before, strict=True)
                )
        _risk_segment(model, signal, index, stop, curve, device="cpu")
    return curve, updates


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    audit = read(AUDIT)
    if (
        audit["status"] != "VERIFIED_PREFIX_ONLY_NOT_FINAL"
        or audit["scored_shots"] != 130
        or any(
            audit["paired"][a + "_vs_static"]["capped_gained"] != [SHOT]
            for a in ("intra", "hold", "reverse")
        )
    ):
        raise ValueError("only the confirmed common prefix rescue is in scope")
    inputs = load_inputs()
    if inputs["ids"][INDEX - 1] != SHOT:
        raise ValueError("fixed selected shot and order changed")
    prefix = f"{INDEX:04d}_{SHOT}"
    shot = RUN / "shots" / prefix
    seal = shot.with_name(prefix + "_seal.json")
    archive = shot.with_suffix(".npz")
    hdf = io23.path_for("EAST", SHOT)
    bound = {p: sha(p) for p in (AUDIT, seal, archive, hdf, Path(__file__))}
    if (
        bound[seal] != audit["seal_sha256"][seal.name]
        or read(seal)["curve_sha256"] != bound[archive]
        or bound[hdf] != inputs["hashes"][str(SHOT)]
    ):
        raise ValueError("selected sealed shot or raw signal changed")
    base, receipt = load_parent(DEFAULT_PARENT)
    base.eval().requires_grad_(False)
    threshold = receipt["threshold"]
    ids = inputs["ids"][:20]
    parent = load_risk_neutral(base, WARM, ids, ids, threshold)
    initial = state_digest(parent)
    if initial != read(seal)["starting_fixed_state_sha256"]:
        raise ValueError("wrong risk-neutral parent")
    raw = simple_data.read_shot("EAST", SHOT, rate_contract="nominal")
    signal = prepare_actual(raw, INDEX, inputs["mapping"])
    curves, updates = {}, {}
    for name in MODES:
        curves[name], updates[name] = replay_loss(parent, signal, name)
    if state_digest(parent) != initial or any(
        not np.array_equal(v[:135], curves["real"][:135]) for v in curves.values()
    ):
        raise ValueError("temporary update escaped or changed the fixed prefix")
    # Outcome used only to score the replay, never to compute a gradient.
    flag, down = io23.read_outcome("EAST", SHOT)
    scores = {
        k: score_curve(raw["times"], v, flag, down, threshold)
        for k, v in curves.items()
    }
    gaps = {}
    with np.load(archive, allow_pickle=False) as saved:
        if not np.array_equal(saved["times"], raw["times"]):
            raise ValueError("raw clock differs from sealed curve")
        for name, control in (
            ("real", "intra"),
            ("hold", "hold"),
            ("reverse", "reverse"),
        ):
            other = saved[control]
            gaps[name] = float(np.max(np.abs(other - curves[name])))
            if score_curve(raw["times"], other, flag, down, threshold) != scores[name]:
                raise ValueError(
                    "CPU control must reproduce the sealed CUDA alarm and lead"
                )
        baseline = saved["static"].copy()
    if any(sha(path) != digest for path, digest in bound.items()):
        raise ValueError("bound input changed")
    OUTPUT.mkdir()
    with (OUTPUT / "curves.npz").open("xb") as f:
        np.savez_compressed(f, times=raw["times"], static=baseline, **curves)
    result = dict(
        status="COMPLETE_POSTHOC_SINGLE_CASE_NOT_ONLINE_GAIN",
        shot=SHOT,
        index=INDEX,
        threshold=threshold,
        hdf_down_s=down,
        retrospective_case_selection=True,
        feedback_labels_used=False,
        static=score_curve(raw["times"], baseline, flag, down, threshold),
        scores=scores,
        updates=updates,
        max_cpu_cuda_control_gap=gaps,
        max_risk={k: float(v.max()) for k, v in curves.items()},
        curve_sha256=sha(OUTPUT / "curves.npz"),
        input_sha256={str(k): v for k, v in bound.items()},
    )
    _write(OUTPUT / "result.json", result)
    print(
        {
            k: result[k]
            for k in (
                "status",
                "scores",
                "updates",
                "max_cpu_cuda_control_gap",
                "max_risk",
            )
        }
    )


if __name__ == "__main__":
    main()
