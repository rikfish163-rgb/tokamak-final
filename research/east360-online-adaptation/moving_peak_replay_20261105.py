"""Matched extra training on native vs current-policy safety peaks, fixed EAST360."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from confirmation_raw_replay_20261105 import signature
from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.east_context_operator_20261031.run import configure
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import (
    prepare_actual,
)
from experiments.risk_aligned_online.mask_separated_intra_20261103.run import (
    state_digest,
)
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    load_parent,
)
from joint_lora_hindsight_capacity_20261104 import (
    SOURCE,
    THRESHOLD,
    read,
    sha,
    timeline,
    write_new,
)
import latent_trajectory_adapt_20261105 as latent
from local_affine_reach_20261105 import COLD
from run_latent_trajectory_adapt_20261105 import save_arrays, sections
import trajectory_policy_20261105 as policy
from trajectory_replay_20261105 import BATCH, reference

ROOT = io23.ROOT
PLAN = ROOT / "MOVING_PEAK_REPLAY_PLAN_20261105.md"
PARENT = ROOT / "probe_outputs/latent_trajectory_adapt360_20261105_v1"
PARENT_DONE = "ed2cf8202b12d232d437001900bfa4b6f823e6362ed3131d761829315d733511"
INITIAL_FIT = "ad52728e43623c813a15525a23ab084b4345da64ed67168610acea6c9c883d4e"
BUDGET, ROUNDS, EPOCHS_PER_ROUND, SEED = 20, 3, 4, 131
TRAIN_ARMS = ("fixed_peak", "moving_peak")
ARMS = ("static", "history20_reference", *TRAIN_ARMS, "fixed_stale", "moving_stale")


def mining_queries(times, native, adapted, flag, down, arm):
    if (
        arm not in TRAIN_ARMS
        or times.ndim != 1
        or times.shape != native.shape
        or times.shape != adapted.shape
    ):
        raise ValueError("aligned curves and a specified training arm required")
    if not np.isfinite(times).all() or any(
        not np.isfinite(v).all() or ((v < 0) | (v > 1)).any() for v in (native, adapted)
    ):
        raise ValueError("finite clock and valid probabilities required")
    curve = native if flag or arm == "fixed_peak" else adapted
    q, _ = latent.labelled_queries(times, curve, flag, down)
    return q


def shared_orders(n):
    generator = torch.Generator().manual_seed(SEED)
    return [
        [torch.randperm(n, generator=generator) for _ in range(EPOCHS_PER_ROUND)]
        for _ in range(ROUNDS)
    ]


def load_rule(path, device, *, namespace=""):
    with np.load(path, allow_pickle=False) as data:
        state = {
            k[len(namespace) :]: torch.as_tensor(data[k], device=device)
            for k in data.files
            if k.startswith(namespace)
        }
    rule = latent.Generator(state["mean"], state["scale"]).to(device)
    rule.load_state_dict(state, strict=True)
    return rule.eval().requires_grad_(False)


def forecast(base, ctx, targets):
    """Execute shot-local writes and original internal attention, without labels."""
    x, a, p, _, _, _, _ = ctx
    traces, stats = {}, {}
    for name, target in targets.items():
        trace = p.copy()
        with torch.enable_grad():
            fast = policy.write_fast(target) if len(target) else target
        norm = fast.norm(dim=1)
        if not torch.isfinite(norm).all() or (norm > latent.RADIUS + 1e-6).any():
            raise FloatingPointError("unbounded or nonfinite temporary update")
        stats[name] = dict(
            nonzero=int((norm > 0).sum()),
            max_norm=float(norm.max()) if len(norm) else 0.0,
        )
        with torch.no_grad():
            for start in range(policy.FIRST, len(x), BATCH):
                stop = min(start + BATCH, len(x))
                v, mask = x[start:stop], a[start:stop]
                q = torch.arange(start, stop, device=x.device)
                trace[start:stop] = (
                    policy.routed_pair(
                        base,
                        v,
                        mask,
                        latent.routes_all(mask),
                        fast[(q - policy.FIRST) // policy.STRIDE],
                    )
                    .sigmoid()
                    .cpu()
                    .numpy()
                )
        traces[name] = trace
    return traces, stats


@torch.no_grad()
def actions(rule, ctx, *, stale=False):
    state, history, previous = ctx[4:]
    real = rule(torch.cat((state, history), dim=1))
    if not stale:
        return real
    delayed = rule(torch.cat((state, previous), dim=1))
    matched = policy.match_norm(delayed, real)
    if (abs(matched.norm(dim=1) - real.norm(dim=1)) > 1e-6).any():
        raise ValueError("stale control must match each real action norm")
    return real, matched


def train_round(base, rule, optim, history, curves, permutations, arm):
    parts = []
    selection = {}
    for sid, ctx, times, flag, down in history:
        x, a, p, _, s, h, _ = ctx
        q = mining_queries(times, p, curves[sid], flag, down, arm)
        if len(q) != 8:
            raise ValueError(f"exact eight matched teaching windows required: {sid}")
        idx = torch.as_tensor(q, device=x.device)
        k = (idx - policy.FIRST) // policy.STRIDE
        selection[str(sid)] = q.tolist()
        parts.append(
            (
                x[idx],
                a[idx],
                torch.cat((s[k], h[k]), dim=1),
                np.full(len(q), sid),
                np.full(len(q), flag),
            )
        )
    xx, aa, features = (torch.cat([v[j] for v in parts]) for j in range(3))
    ids, y = (np.concatenate([v[j] for v in parts]) for j in (3, 4))
    w = torch.as_tensor(
        latent.weights_by_class_shot(ids, y), device=xx.device, dtype=xx.dtype
    )
    signs = torch.as_tensor(np.where(y, -1.0, 1.0), device=xx.device, dtype=xx.dtype)
    threshold_logit = float(np.log(THRESHOLD / (1 - THRESHOLD)))

    def objective(idx):
        delta = rule(features[idx])
        v = policy.routed_pair(
            base, xx[idx], aa[idx], latent.routes_all(aa[idx]), delta
        )
        regularizer = 0.005 * delta.square().sum(dim=1) / latent.RADIUS**2
        return (
            w[idx] * (F.softplus(signs[idx] * (v - threshold_logit)) + regularizer)
        ).sum()

    def full_objective():
        with torch.no_grad():
            return sum(
                float(objective(i))
                for i in torch.arange(len(xx), device=xx.device).split(latent.MINIBATCH)
            )

    initial = full_objective()
    steps = 0
    rule.requires_grad_(True)
    for order in permutations:
        for idx in order.to(xx.device).split(latent.MINIBATCH):
            optim.zero_grad(set_to_none=True)
            loss = objective(idx) * len(xx) / len(idx)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite matched extra optimization")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                rule.parameters(), 5.0, error_if_nonfinite=True
            )
            optim.step()
            steps += 1
    rule.requires_grad_(False)
    return dict(
        steps=steps,
        selection=selection,
        initial_loss=initial,
        loss_on_same_selected_points=full_objective(),
    )


def fit_paired(base, history, output, initial_fit):
    if len(history) != BUDGET or sum(x[3] for x in history) != 8:
        raise ValueError("exact ended 8D/12S fitting prefix required")
    device = base.center.device
    rules = {
        k: load_rule(initial_fit, device, namespace="history.") for k in TRAIN_ARMS
    }
    for k, v in rules["fixed_peak"].state_dict().items():
        if not torch.equal(v, rules["moving_peak"].state_dict()[k]):
            raise AssertionError("unequal initial parameters")
    optim = {
        k: torch.optim.Adam(v.parameters(), lr=latent.LR) for k, v in rules.items()
    }
    observations = {f"{sid}_native": ctx[2] for sid, ctx, *_ in history}
    report = {k: dict(rounds=[]) for k in TRAIN_ARMS}
    batches = shared_orders(BUDGET * 8)
    for stage in range(1, ROUNDS + 2):
        for arm, rule in rules.items():
            by_shot = {}
            for sid, ctx, *_ in history:
                pred, _ = forecast(base, ctx, {arm: actions(rule, ctx)})
                by_shot[sid] = pred[arm]
                observations[f"{sid}_{arm}_{stage}"] = pred[arm]
            if stage <= ROUNDS:
                rd = train_round(
                    base, rule, optim[arm], history, by_shot, batches[stage - 1], arm
                )
                if rd["steps"] != 12:
                    raise AssertionError("unexpected per-round optimizer budget")
                report[arm]["rounds"].append(dict(round=stage, **rd))
                print(
                    "trained round",
                    stage,
                    arm,
                    rd["initial_loss"],
                    rd["loss_on_same_selected_points"],
                    flush=True,
                )
    save_arrays(output / "mining_curves.npz", **observations)
    for arm, rule in rules.items():
        path = output / f"fit_{arm}.npz"
        save_arrays(
            path, **{k: v.detach().cpu().numpy() for k, v in rule.state_dict().items()}
        )
        report[arm].update(
            steps=sum(r["steps"] for r in report[arm]["rounds"]), fit_sha256=sha(path)
        )
    write_new(
        output / "fit_receipt.json",
        dict(
            ids=[r[0] for r in history],
            initial_fit_sha256=sha(initial_fit),
            arms=report,
        ),
    )
    return rules, {k: v["fit_sha256"] for k, v in report.items()}


def run(output):
    start = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    configure()
    order, cfg, _ = reference()
    initial_fit = PARENT / "fit20.npz"
    if sha(PARENT / "completed.json") != PARENT_DONE or sha(initial_fit) != INITIAL_FIT:
        raise ValueError("the verified prior 20-shot parent is required")
    prior = read(PARENT / "completed.json")
    old_protocol = read(PARENT / "protocol.json")
    old_fit = read(PARENT / "fit_receipts.json")["20"]
    if (
        old_protocol["order"] != order
        or old_protocol["threshold"] != THRESHOLD
        or old_protocol["raw_sha256"] != cfg["raw_sha256"]
        or old_fit["params_sha256"] != INITIAL_FIT
        or set(old_fit["ids"]) != set(order[:BUDGET])
        or not old_fit["after_all_historical_reveals"]
        or any(sha(PARENT / k) != v for k, v in prior["artifact_sha256"].items())
    ):
        raise ValueError("frozen parent replay or label budget changed")
    mapping_path = COLD / "input_map.json"
    mapping = read(mapping_path)
    if (
        mapping["calibration_ids"] != order[:BUDGET]
        or sha(mapping_path) != cfg["input_sha256"][str(mapping_path)]
    ):
        raise ValueError("frozen input mapping changed")
    base, parent = load_parent(DEFAULT_PARENT)
    source_digest = state_digest(base)
    if source_digest != cfg["source_state_sha256"] or parent["threshold"] != THRESHOLD:
        raise ValueError("frozen source state or threshold changed")
    base.cuda()
    initial_rule = load_rule(initial_fit, base.center.device, namespace="history.")
    deps = [
        Path(__file__),
        PLAN,
        initial_fit,
        DEFAULT_PARENT,
        mapping_path,
        io23.EAST_COHORT,
        SOURCE / "protocol.json",
        SOURCE / "completed.json",
        Path(latent.__file__),
        Path(policy.__file__),
        Path(latent.make_views.__code__.co_filename),
        Path(prepare_actual.__code__.co_filename),
        Path(io23.__file__),
        Path(simple_data.__file__),
        PARENT / "completed.json",
        PARENT / "protocol.json",
        PARENT / "fit_receipts.json",
    ]
    hashes = {str(p.resolve()): sha(p) for p in deps}
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    write_new(
        output / "protocol.json",
        dict(
            order=order,
            threshold=THRESHOLD,
            first_anchor=policy.FIRST,
            stride=policy.STRIDE,
            arms=ARMS,
            input_sha256=hashes,
            raw_sha256=cfg["raw_sha256"],
            label_budget=BUDGET,
            rounds=ROUNDS,
            steps_per_arm=36,
            independent_future_validation=False,
            rounds_use_same_historical_labels_only=True,
            equal_full_trajectory_mining_forward_passes=True,
            source_state_sha256=source_digest,
            initial_fit_sha256=INITIAL_FIT,
            radius=latent.RADIUS,
            train_seed=SEED,
            epochs_per_round=EPOCHS_PER_ROUND,
            minibatch=latent.MINIBATCH,
            lr=latent.LR,
            device=torch.cuda.get_device_name(),
        ),
    )
    rows = {k: [] for k in ARMS}
    seals, pending = [], []
    rules, fit_hash = {}, {}
    maxima = {"static": 0.0, "history20_reference": 0.0}
    print("started fixed360 moving peaks", flush=True)
    for rank, sid in enumerate(order, 1):
        raw_path = io23.path_for("EAST", sid)
        before = signature(raw_path)
        raw_hash = cfg["raw_sha256"][str(sid)]
        with patch.object(
            io23,
            "read_outcome",
            side_effect=RuntimeError("current outcome unavailable before seal"),
        ):
            raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
            times = raw["times"]
            ctx = latent.native_context(base, prepare_actual(raw, rank, mapping))
            native = ctx[2]
            targets = {}
            if rank > BUDGET:
                targets["history20_reference"] = actions(initial_rule, ctx)
                for arm in TRAIN_ARMS:
                    real, stale = actions(rules[arm], ctx, stale=True)
                    targets[arm] = real
                    targets[arm.replace("_peak", "_stale")] = stale
            traces, updates = forecast(base, ctx, targets)
            curves = {k: traces[k] if k in traces else native for k in ARMS}
        stem = f"{rank:04d}_{sid}"
        original = PARENT / "shots" / f"{stem}.npz"
        if sha(original) != prior["sealed"][rank - 1]["curve_sha256"]:
            raise ValueError(f"archived comparison changed: {sid}")
        with np.load(original, allow_pickle=False) as p:
            if not np.array_equal(p["times"], times):
                raise ValueError(f"changed HDF clock: {sid}")
            for new, old in (
                ("static", "static"),
                ("history20_reference", "history20"),
            ):
                diff = float(np.max(abs(curves[new] - p[old])))
                if diff > 4e-6 or not np.array_equal(
                    curves[new] >= THRESHOLD, p[old] >= THRESHOLD
                ):
                    raise ValueError(
                        f"unchanged replay disagrees: {sid}/{new}, {diff:g}"
                    )
                maxima[new] = max(maxima[new], diff)
        path = output / "shots" / f"{stem}.npz"
        save_arrays(path, times=times, **curves)
        curve_hash = sha(path)
        seal = path.with_name(stem + "_seal.json")
        write_new(
            seal,
            dict(
                shot=sid,
                rank=rank,
                raw_sha256=raw_hash,
                curve_sha256=curve_hash,
                active_fit_sha256=fit_hash.copy(),
                current_outcome_used_in_prediction=False,
                updates=updates,
            ),
        )
        seal_hash = sha(seal)
        if signature(raw_path) != before or sha(raw_path) != raw_hash:
            raise ValueError(f"raw changed: {sid}")
        flag, down = io23.read_outcome("EAST", sid)
        if signature(raw_path) != before:
            raise ValueError(f"raw changed during reveal: {sid}")
        scored = {
            k: dict(
                shot=sid,
                disruptive=flag,
                down=down,
                **timeline(times, v, flag, down, THRESHOLD),
            )
            for k, v in curves.items()
        }
        for k, v in scored.items():
            rows[k].append(v)
        reveal = path.with_name(stem + "_revealed.json")
        write_new(reveal, dict(shot=sid, after_seal_sha256=seal_hash, rows=scored))
        seals.append(
            dict(
                shot=sid,
                curve_sha256=curve_hash,
                seal_sha256=seal_hash,
                revealed_sha256=sha(reveal),
            )
        )
        if rank <= BUDGET:
            pending.append((sid, raw, flag, down))
            if rank == BUDGET:
                history = [
                    (
                        s,
                        latent.native_context(base, prepare_actual(a, 21, mapping)),
                        a["times"],
                        f,
                        d,
                    )
                    for s, a, f, d in pending
                ]
                rules, fit_hash = fit_paired(base, history, output, initial_fit)
                pending.clear()
                history.clear()
        if rank % 25 == 0 or rank == len(order):
            print("sealed and revealed", rank, len(order), flush=True)
    if (
        state_digest(base) != source_digest
        or any(p.grad is not None for p in base.parameters())
        or any(sha(p) != h for p, h in hashes.items())
    ):
        raise AssertionError("frozen source or an input changed")
    summary = {k: v for k, v in sections(rows).items() if k in ("full360", "after20")}
    write_new(output / "per_shot.json", rows)
    write_new(output / "summary.json", summary)
    files = (
        "protocol.json",
        "fit_receipt.json",
        "mining_curves.npz",
        "fit_fixed_peak.npz",
        "fit_moving_peak.npz",
        "per_shot.json",
        "summary.json",
    )
    write_new(
        output / "completed.json",
        dict(
            status="COMPLETE_MOVING_PEAK_REPLAY",
            n_shots=len(order),
            n_curves=len(order) * len(ARMS),
            sealed=seals,
            artifact_sha256={k: sha(output / k) for k in files},
            parent_parity_max_error=maxima,
            seconds=time.monotonic() - start,
        ),
    )
    print("complete", summary["full360"]["metrics"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
