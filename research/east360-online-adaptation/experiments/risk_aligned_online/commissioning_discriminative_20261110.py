"""Fit a delayed observation update on completed shots and evaluate exact replay."""

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import io23, simple_data
from .commissioning_calibration_20261108 import prefix_mapping
from .commissioning_innovation_fast_20261110 import arrival_inputs, lagged_forward
from .commissioning_intra_20261110 import (
    CALIBRATION, CAPS, HORIZON, PARENT_ARM, PARENT_RUN, START, STARTUP, STRIDE,
    feedback_blocks, load_frozen,
)
from .commissioning_transfer_20261108 import (
    TARGET_FACTORIAL, combine, contrast, features, fit_readouts, make_rows, predict,
)
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_repair_20261104.east_evaluation import configure, prepare_actual
from .mask_separated_intra_20261103.run import state_digest
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .source_anticipation_20261104.static_stream import timing_decomposition


ARMS = ("learned_real", "learned_reversed", "learned_unordered")
NAMES = ("static", *ARMS)
DECAY = math.exp(-STRIDE / (START + 1))
RANK, STEPS, RIDGE = 4, 128, .03


@dataclass
class Episode:
    shot: int
    disruptive: bool
    history: torch.Tensor
    slot: torch.Tensor
    logit: torch.Tensor
    jacobian: torch.Tensor
    target: torch.Tensor


@dataclass
class Rule:
    scale: torch.Tensor
    basis: torch.Tensor
    weight: torch.Tensor

    def coordinates(self, history):
        value = (history / self.scale) @ self.weight.T
        return value / value.norm(dim=-1, keepdim=True).clamp_min(1.)

    def states(self, history):
        return self.coordinates(history) @ self.basis.T


def causal_history(evidence, n):
    """Weighted least-squares sufficient statistic, initialized per discharge."""
    first = np.array([start for _, start, _ in feedback_blocks(n)], dtype=np.int64)
    if evidence.shape != (len(first), 46):
        raise ValueError("one matured innovation vector per release")
    out = np.zeros((len(first) + 1, 46), dtype=np.float32)
    for i, observation in enumerate(evidence, 1):
        out[i] = DECAY * out[i - 1] + (1 - DECAY) * observation
    return out, np.searchsorted(first, np.arange(n), side="right")


def make_folds(ids, outcomes):
    positive = [s for s in ids if outcomes[s][0]]
    safe = [s for s in ids if not outcomes[s][0]]
    if (len(positive), len(safe)) != (8, 12):
        raise ValueError("the original ended 8 disruptive/12 safe shots are required")
    held = [set(positive[k::4] + safe[k::4]) for k in range(4)]
    return [([s for s in ids if s not in group], [s for s in ids if s in group]) for group in held]


def project_logits(episode, rule):
    theta = rule.coordinates(episode.history)[episode.slot]
    return episode.logit + ((episode.jacobian @ rule.basis) * theta).sum(-1)


def fit_rule(episodes, cutoff):
    """Optimize shot-level peaks in a zero-state linearization; replay is nonlinear."""
    device = episodes[0].logit.device
    gram = torch.stack([a.jacobian.T @ a.jacobian / len(a.logit) for a in episodes]).mean(0)
    basis = torch.linalg.eigh(gram).eigenvectors[:, -RANK:].contiguous()
    scale = torch.stack([a.history.square().mean(0) for a in episodes]).mean(0).sqrt().clamp_min(.05)
    weights = torch.nn.Parameter(torch.zeros(basis.shape[1], len(scale), device=device))
    rule = Rule(scale, basis, weights)
    prepared = [(a, a.jacobian @ basis) for a in episodes]
    boundary = math.log(cutoff / (1 - cutoff))
    if not any(a.disruptive and bool(a.target.any()) for a in episodes) or not any(
            not a.disruptive for a in episodes):
        raise ValueError("timely positive and safe training support required")
    optim = torch.optim.Adam([weights], lr=.025)
    for _ in range(STEPS):
        losses = ([], [])
        for a, j in prepared:
            theta = rule.coordinates(a.history)[a.slot]
            z = a.logit + (j * theta).sum(-1)
            peak = z[a.target].amax() if a.disruptive else z.amax()
            margin = peak - boundary
            losses[int(a.disruptive)].append(F.softplus(-margin if a.disruptive else margin))
        loss = .5 * sum(torch.stack(v).mean() for v in losses) + RIDGE * weights.square().mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    if not torch.isfinite(weights).all():
        raise FloatingPointError("nonfinite observation update fit")
    return Rule(scale.detach(), basis.detach(), weights.detach())


def history_for(signal, base):
    values, valid = signal["values"], signal["available"]
    q = np.asarray([q for q, _, _ in feedback_blocks(len(values))], np.int64)
    x = arrival_inputs(values, valid, base.center.detach().cpu().numpy(),
                       base.scale.detach().cpu().numpy(), q)
    return x, len(values)


def window_arrays(signal, device):
    return tuple(F.pad(torch.as_tensor(signal[key], device=device), (0, 0, START, 0))
                 .unfold(0, START + 1, 1).transpose(1, 2) for key in ("values", "available"))


def risk_input(base, x, m, c, coef, state):
    captured = []
    handle = base.risk_head.register_forward_pre_hook(lambda _m, args: captured.append(args[0]))
    try:
        lagged_forward(base, x, m, state)
    finally:
        handle.remove()
    return torch.cat((captured[0], c), dim=-1) @ coef[:-1] + coef[-1]


def episode(base, coef, signal, times, disruptive, down):
    device = coef.device
    data, n = history_for(signal, base)
    history, slot = causal_history(data, n)
    first = START + HORIZON + 1
    w, a = window_arrays(signal, device)
    c = torch.as_tensor(contrast(signal["values"], signal["available"]), device=device)
    logits, jac = [], []
    for i in range(0, n, 256):
        stop = min(n, i + 256)
        state = coef.new_zeros((stop - i, base.cfg.dim), requires_grad=True)
        z = risk_input(base, w[i:stop], a[i:stop], c[i:stop], coef, state)
        (j,) = torch.autograd.grad(z.sum(), state)
        logits.append(z.detach())
        jac.append(j.detach())
    j = torch.cat(jac)
    j[:first] = 0.
    lead = np.round((down - times) * 1000., 9) if disruptive else np.zeros(n)
    mask = (lead > 30.) & (lead <= 125.)
    if disruptive and not mask.any():
        raise ValueError(f"no observable 30-to-125ms positive training interval: {signal['shot']}")
    return Episode(signal["shot"], disruptive,
                   torch.as_tensor(history, device=device), torch.as_tensor(slot, device=device),
                   torch.cat(logits), j, torch.as_tensor(mask, device=device))


def exact_replay(base, coef, signal, rule, static):
    """Fitted state acts at the first encoder block; no risk-logit correction."""
    device = coef.device
    real, n = history_for(signal, base)
    reversed_order = real.copy()
    reversed_order[:, 23:] *= -1
    unordered = real.copy()
    unordered[:, 23:] = 0.
    evidence = [causal_history(x, n) for x in (real, reversed_order, unordered)]
    histories = torch.as_tensor(np.stack([v[0] for v in evidence]), device=device)
    slot = torch.as_tensor(evidence[0][1], device=device)
    states = rule.states(histories)
    change = (states[:, 1:] != states[:, :-1]).any(-1).sum(-1)
    curves = torch.as_tensor(static, device=device).expand(len(ARMS), -1).clone()
    if not len(real):
        return dict(zip(ARMS, curves.cpu().numpy(), strict=True)), dict.fromkeys(ARMS, 0)
    w, a = window_arrays(signal, device)
    c = torch.as_tensor(contrast(signal["values"], signal["available"]), device=device)
    with torch.no_grad():
        for i in range(START + HORIZON + 1, n, 256):
            s = slice(i, min(n, i + 256))
            z = risk_input(base, w[s].repeat(len(ARMS), 1, 1), a[s].repeat(len(ARMS), 1, 1),
                           c[s].repeat(len(ARMS), 1), coef, states[:, slot[s]].flatten(0, 1))
            curves[:, s] = z.sigmoid().reshape(len(ARMS), -1)
    if not torch.isfinite(curves).all():
        raise FloatingPointError("nonfinite actual risk curve")
    return (dict(zip(ARMS, curves.cpu().numpy(), strict=True)),
            dict(zip(ARMS, change.cpu().tolist(), strict=True)))


def fit_fold_parent(base, signals, ids, metadata, cutoff):
    rows = []
    for index, sid in enumerate(ids):
        sig, times = signals[sid]
        x, _ = features(base, sig, "cuda")
        rows.append(make_rows(x, times, *metadata[sid], 125., 30., index))
    coef, _ = fit_readouts(base, None, combine(rows), cutoff, "cuda",
                           arms={PARENT_ARM: TARGET_FACTORIAL[PARENT_ARM]})
    return coef[PARENT_ARM]


def loaded_shot(raw, index, mapping):
    return prepare_actual(raw, index, mapping), raw["times"]


def train_from_ended(base, coef, signals, ids, metadata, cutoff):
    return fit_rule([episode(base, coef, *signals[s], *metadata[s]) for s in ids], cutoff)


def leave_shots_out(base, inputs, raw, metadata, original, out):
    rows, folds = {arm: [] for arm in NAMES}, []
    for number, (train, held) in enumerate(make_folds(list(raw), metadata), 1):
        mapping = prefix_mapping(inputs["mapping"]["source_statistics"], [raw[s] for s in train])
        # Do not compute any held-out features until all fits have completed.
        signals = {s: loaded_shot(raw[s], STARTUP + 1, mapping) for s in train}
        coef = fit_fold_parent(base, signals, train, metadata, original)
        rule = train_from_ended(base, coef, signals, train, metadata, original)
        folds.append(dict(fit=train, held=held, cutoff=original))
        for sid in held:
            sig, times = loaded_shot(raw[sid], STARTUP + 1, mapping)
            static = predict(base, sig, {PARENT_ARM: coef}, "cuda")[PARENT_ARM]
            curves, _ = exact_replay(base, coef, sig, rule, static)
            flag, down = metadata[sid]
            lead = np.round((down - times) * 1000., 9) if flag else None
            measure = ((lead > 30.) & (lead <= 125.) if flag
                       else np.ones(len(times), bool))
            for arm, score in {"static": static, **curves}.items():
                row = score_curve(times, score, flag, down, original)
                rows[arm].append(dict(shot=sid, disruptive=flag, **row,
                                      target_peak=(float(score[measure].max())
                                                   if measure.any() else None)))
        print(dict(stage="held_fold", fold=number, held=held), flush=True)
    write(out / "held20.json", dict(
        scope="ENDED20_DISJOINT_FOLDS_ORIGINAL_CUTOFF_NOT_FUTURE_CONFIRMATION",
        folds=folds, metrics={k: timing_decomposition(v) for k, v in rows.items()},
        paired={k: flips(rows["static"], rows[k]) for k in ARMS}, rows=rows,
    ))


def run(output):
    started = time.monotonic()
    configure()
    inputs, coef, original, cutoffs, bound = load_frozen()
    base, ckpt = load_parent(DEFAULT_PARENT)
    base.cuda()
    before = state_digest(base)
    if ckpt["threshold"] != original or before != inputs["baseline_protocol"]["source_state_sha256"]:
        raise ValueError("frozen source contract mismatch")
    ids = inputs["ids"]
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    write(output / "protocol.json", dict(
        scope="DEVELOPMENT_COLD20_FIT_THEN_CAUSAL_INTRA_ONLY",
        **bound, source_sha256=PARENT_SHA256, order=ids, fit_shots=ids[:STARTUP],
        original=original, suffix_thresholds=cutoffs, fit_threshold=original, modes=NAMES,
        decay=DECAY, rank=RANK, ridge=RIDGE, steps=STEPS,
        first_origin=START, horizon=HORIZON, stride=STRIDE,
        core_sha256=sha(Path(__file__).with_name("commissioning_innovation_fast_20261110.py")),
        producer_sha256=sha(Path(__file__)),
    ))
    raw, metadata = {}, {}
    for i, sid in enumerate(ids[:STARTUP], 1):
        if not (PARENT_RUN / "shots" / f"{i:04d}_{sid}.npz").is_file():
            raise ValueError("startup outcome is allowed only after original prediction seal")
        path = io23.path_for("EAST", sid)
        if sha(path) != inputs["hashes"][str(sid)]:
            raise ValueError("startup waveform changed")
        raw[sid] = simple_data.read_shot("EAST", sid, rate_contract="nominal")
        metadata[sid] = io23.read_outcome("EAST", sid)
    leave_shots_out(base, inputs, raw, metadata, original, output)
    signals = {s: loaded_shot(raw[s], STARTUP + 1, inputs["mapping"]) for s in raw}
    rule = train_from_ended(base, coef, signals, ids[:STARTUP], metadata, original)
    with (output / "rule.npz").open("xb") as f:
        np.savez(f, scale=rule.scale.cpu().numpy(), basis=rule.basis.cpu().numpy(),
                 weight=rule.weight.cpu().numpy())
    print(dict(stage="fit20_complete", seconds=time.monotonic()-started), flush=True)

    seals, updates = {}, dict.fromkeys(ARMS, 0)
    with patch.object(io23, "read_outcome", side_effect=RuntimeError("outcome before sealing")):
        for i, sid in enumerate(ids, 1):
            name = f"{i:04d}_{sid}.npz"
            with np.load(PARENT_RUN / "shots" / name, allow_pickle=False) as reference:
                times, static = reference["times"], reference[PARENT_ARM]
            if i <= STARTUP:
                np.testing.assert_array_equal(times, raw[sid]["times"])
                curves = {k: static for k in ARMS}
            else:
                if sha(io23.path_for("EAST", sid)) != inputs["hashes"][str(sid)]:
                    raise ValueError(f"waveform changed: {sid}")
                sig, new_times = loaded_shot(simple_data.read_shot("EAST", sid, rate_contract="nominal"),
                                             i, inputs["mapping"])
                np.testing.assert_array_equal(times, new_times)
                cold = predict(base, sig, {PARENT_ARM: coef}, "cuda")[PARENT_ARM]
                np.testing.assert_array_equal(cold, static)
                curves, applied = exact_replay(base, coef, sig, rule, cold)
                for arm, n in applied.items():
                    updates[arm] += n
            path = output / "shots" / name
            with path.open("xb") as f:
                np.savez_compressed(f, times=times, static=static, **curves)
                f.flush()
                os.fsync(f.fileno())
            seals[name] = sha(path)
            if i % 20 == 0:
                print(dict(stage="sealed", shots=i, seconds=time.monotonic()-started), flush=True)
    if state_digest(base) != before or any(v.grad is not None for v in base.parameters()):
        raise AssertionError("source parameters must stay frozen")
    write(output / "seal.json", dict(curve_sha256=seals, updates=updates,
                                      rule_sha256=sha(output / "rule.npz")))
    per = {cap: {arm: [] for arm in NAMES} for cap in CAPS}
    for i, sid in enumerate(ids, 1):
        name = f"{i:04d}_{sid}.npz"
        path = output / "shots" / name
        if sha(path) != seals[name]:
            raise ValueError("sealed curve changed")
        flag, down = metadata[sid] if i <= STARTUP else io23.read_outcome("EAST", sid)
        with np.load(path, allow_pickle=False) as record:
            for arm in NAMES:
                for cap, cutoff in cutoffs.items():
                    row = score_curve(record["times"], record[arm], flag, down,
                                      original if i <= STARTUP else cutoff)
                    per[cap][arm].append(dict(shot=sid, disruptive=flag, down_s=down, **row))
    ref = read(CALIBRATION / "per_shot.json")
    if any(per[c]["static"] != ref[c][PARENT_ARM] for c in CAPS):
        raise AssertionError("matched 360 control differs from sealed parent")
    metrics = {c: {k: timing_decomposition(v) for k, v in a.items()} for c, a in per.items()}
    paired = {c: {k: flips(a["static"], a[k]) for k in ARMS} for c, a in per.items()}
    write(output / "per_shot.json", per)
    write(output / "result.json", dict(status="COMPLETED_DISCRIMINATIVE_DEVELOPMENT",
                                       metrics=metrics, paired=paired, updates=updates,
                                       seconds=time.monotonic() - started))
    print({c: {k: (v["tp"], v["fp"]) for k, v in a.items()} for c, a in metrics.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output.resolve())
