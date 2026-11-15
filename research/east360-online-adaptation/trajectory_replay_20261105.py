"""Delayed-activation development replay with a fixed historical teaching prefix."""

from collections import Counter
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import (
    prepare_actual,
)
from experiments.risk_aligned_online.mask_separated_intra_20261103.run import (
    state_digest,
)
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    PARENT_SHA256,
    load_parent,
)
from joint_lora_hindsight_capacity_20261104 import (
    SOURCE,
    SOURCE_DONE_SHA,
    read,
    sha,
    timeline,
    write_new,
)
from local_affine_reach_20261105 import COLD, local_weights
import trajectory_policy_20261105 as policy
from trajectory_teaching_loss_20261105 import teaching_loss

ROOT = io23.ROOT
PLAN = ROOT / "TRAJECTORY_TEACHING_PLAN_20261105.md"
FIT_AFTER = 100
BATCH = 256
RIDGE = 0.005
ARMS = (
    "static",
    "point_real",
    "trajectory_real",
    "trajectory_lagged",
    "trajectory_reversed",
    "trajectory_state",
)
SPEC = (
    ("point_real", "real", "point"),
    ("trajectory_real", "real", "trajectory"),
    ("trajectory_state", "state", "trajectory"),
)


def reference():
    if sha(SOURCE / "completed.json") != SOURCE_DONE_SHA:
        raise ValueError("original sealed reference changed")
    done = read(SOURCE / "completed.json")
    if sha(SOURCE / "protocol.json") != done["artifact_sha256"]["protocol.json"]:
        raise ValueError("original reference protocol changed")
    cfg = read(SOURCE / "protocol.json")
    order = [r["shot"] for r in cfg["order"]]
    if (
        len(order) != 360
        or len(set(order)) != 360
        or set(order) != set(io23.cohort_ids("EAST"))
        or cfg["source_sha256"] != PARENT_SHA256
        or len(done["seals"]) != 360
    ):
        raise ValueError("fixed EAST cohort and qualified source required")
    return order, cfg, done


def baseline_for(rank, sid, done):
    root = SOURCE / "shots" / f"{rank:04d}_{sid}"
    seal = read(root.with_name(root.name + "_seal.json"))
    path = root.with_suffix(".npz")
    if (
        seal != done["seals"][rank - 1]
        or seal["shot"] != sid
        or seal["current_outcome_used_in_prediction"] is not False
        or sha(path) != seal["curve_sha256"]
    ):
        raise ValueError(f"sealed reference mismatch: {sid}")
    with np.load(path, allow_pickle=False) as archive:
        return archive["times"].copy(), archive["map_after20"].copy()


def make_views(signal, device):
    return policy.window_views(
        torch.as_tensor(signal["values"], device=device),
        torch.as_tensor(signal["available"], device=device),
    )


@torch.no_grad()
def features_at_arrival(base, views, available, original_scores, tau):
    m = np.arange(policy.FIRST, len(views), policy.STRIDE, dtype=np.int64)
    if not len(m):
        return {name: views.new_zeros((0, 84)) for name in ("real", "lagged", "state")}
    indices = torch.as_tensor(m, device=views.device)
    v, a = views[indices], available[indices]
    asinh = torch.asinh((torch.where(a, v, base.center) - base.center) / base.scale)
    p = torch.as_tensor(original_scores[m], dtype=torch.float64, device=views.device)
    gate = (torch.logit(p.clamp(1e-8, 1 - 1e-8)) - tau).tanh().to(v)
    return policy.contrast_features(asinh, a, gate)


def routes_for(available, queries):
    maturities = (
        policy.FIRST + ((queries - policy.FIRST) // policy.STRIDE) * policy.STRIDE
    )
    return torch.stack(
        [
            local_weights(available, queries, maturities, "recent"),
            local_weights(available, queries, maturities, "cls"),
        ],
        dim=1,
    )


def collect_teaching(base, signal, times, scores, truth, alarm, tau, device):
    flag, down = truth
    leads = np.round((down - times) * 1000, 9) if flag else np.zeros_like(times)
    query = policy.chronological_queries(scores, leads, disrupted=flag)
    x, a = make_views(signal, device)
    feats = features_at_arrival(base, x, a, scores, tau)
    before = query < policy.FIRST
    z = np.empty(len(query), dtype=np.float32)
    g = np.zeros((len(query), 256), dtype=np.float32)
    if before.any():
        ix = torch.as_tensor(query[before], device=device)
        with torch.no_grad():
            z[before] = base(x[ix], a[ix])["logit"].cpu().numpy()
    for i in range(int(before.sum()), len(query), BATCH):
        q = query[i : i + BATCH]
        ix = torch.as_tensor(q, device=device)
        v, m = x[ix], a[ix]
        zt, gt = policy.effect_gradient(base, v, m, routes_for(m, q))
        z[i : i + len(q)], g[i : i + len(q)] = zt.cpu().numpy(), gt.cpu().numpy()
    error = np.max(np.abs(1 / (1 + np.exp(-z.astype(np.float64))) - scores[query]))
    if error > 4e-6:
        raise ValueError(f"teaching zero-state mismatch: {error}")
    per_endpoint = {}
    for k in ("real", "state"):
        arr = np.zeros((len(query), 84), dtype=np.float32)
        if (~before).any():
            ix = torch.as_tensor(
                (query[~before] - policy.FIRST) // policy.STRIDE, device=device
            )
            arr[~before] = feats[k][ix].cpu().numpy()
        per_endpoint[k] = arr
    return dict(
        shot=signal["shot"],
        label=flag,
        query=query,
        lead=leads[query],
        z=z,
        g=g,
        inputs=per_endpoint,
        alarm=alarm,
        parity=float(error),
    )


def fit(episodes, threshold_logit, output):
    n, width = len(episodes), max(len(r["query"]) for r in episodes)
    valid = torch.zeros(n, width, dtype=torch.bool)
    leads = torch.zeros(n, width, dtype=torch.float64)
    disruptive = torch.tensor([r["label"] for r in episodes], dtype=torch.bool)
    groups = [
        "d_good"
        if r["alarm"] == "tp"
        else "d_bad"
        if r["label"]
        else "s_bad"
        if r["alarm"] == "fp"
        else "s_good"
        for r in episodes
    ]
    census = Counter(groups)
    if set(census) != {"d_good", "d_bad", "s_bad", "s_good"}:
        raise ValueError(f"both historical correct/error strata required: {census}")
    shot_weight = torch.tensor([0.25 / census[x] for x in groups], dtype=torch.float64)
    for i, ep in enumerate(episodes):
        n_endpoint = len(ep["query"])
        valid[i, :n_endpoint] = True
        leads[i, :n_endpoint] = torch.from_numpy(ep["lead"])
    row_weight = np.concatenate(
        [
            np.full(len(r["query"]), float(w) / len(r["query"]))
            for r, w in zip(episodes, shot_weight, strict=True)
        ]
    )
    packed = {
        "z": np.concatenate([r["z"] for r in episodes]),
        "g": np.concatenate([r["g"] for r in episodes]),
        "lead": leads.numpy(),
        "valid": valid.numpy(),
        "disruptive": disruptive.numpy(),
        "shot_weight": shot_weight.numpy(),
        "ids": np.array([r["shot"] for r in episodes]),
        "query": np.concatenate([r["query"] for r in episodes]),
        **{
            key: np.concatenate([r["inputs"][key] for r in episodes])
            for key in ("real", "state")
        },
    }
    z, g = [torch.as_tensor(packed[k], dtype=torch.float64) for k in ("z", "g")]
    results, models = {}, {}
    for name, key, mode in SPEC:
        x = torch.as_tensor(packed[key], dtype=torch.float64)
        wt = torch.as_tensor(row_weight, dtype=torch.float64)
        rms = (wt @ x.square()).sqrt()
        scale = torch.where(rms > 1e-10, rms, 1.0)
        matrix = torch.zeros((84, 256), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            (matrix,),
            max_iter=80,
            tolerance_grad=1e-7,
            tolerance_change=1e-10,
            line_search_fn="strong_wolfe",
        )

        def objective():
            action = policy.action(x, scale, matrix)
            risk = z + (g * action).sum(dim=1)
            matrix_scores = torch.zeros_like(leads).masked_scatter(valid, risk)
            losses = teaching_loss(
                matrix_scores, leads, disruptive, valid, threshold_logit, mode
            )
            return shot_weight @ losses + (RIDGE / 2) * matrix.square().sum()

        calls = 0

        def closure():
            nonlocal calls
            calls += 1
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            loss.backward()
            return loss

        before = float(objective().detach())
        optimizer.step(closure)
        after = float(objective().detach())
        if not np.isfinite(after) or after > before + 1e-10:
            raise FloatingPointError("finite descent required for historical teaching")
        with torch.no_grad():
            shift = policy.action(x, scale, matrix)
            logit_shift = (g * shift).sum(dim=1)
        results[name] = dict(
            mode=mode,
            feature=key,
            before=before,
            after=after,
            closure_calls=calls,
            mean_absolute_linearized_logit_change=float(logit_shift.abs().mean()),
            nonzero_actions=int((shift.norm(dim=1) > 0).sum()),
        )
        models[name] = (scale.detach().clone(), matrix.detach().clone())
        packed[name + "_scale"], packed[name + "_matrix"] = [
            x.numpy() for x in models[name]
        ]
        print("fit", name, results[name], flush=True)
    with (output / "fit.npz").open("xb") as f:
        np.savez_compressed(f, **packed)
    result = dict(
        scopes=[21, FIT_AFTER],
        n_shots=n,
        n_endpoints=len(z),
        statuses=dict(Counter(r["alarm"] for r in episodes)),
        n_zero_jacobian=int((g.norm(dim=1) == 0).sum()),
        training_only=True,
        maximum_baseline_parity=max(r["parity"] for r in episodes),
        fit_results=results,
        fit_sha256=sha(output / "fit.npz"),
    )
    write_new(output / "fit_receipt.json", result)
    return models, result


def predict(base, signal, original, tau, models, device):
    x, a = make_views(signal, device)
    feats = features_at_arrival(base, x, a, original, tau)
    matrices = {
        k: tuple(w.to(device=device, dtype=x.dtype) for w in m)
        for k, m in models.items()
    }
    with torch.no_grad():
        real = policy.action(feats["real"], *matrices["trajectory_real"])
        lagged = policy.action(feats["lagged"], *matrices["trajectory_real"])
        state = policy.action(feats["state"], *matrices["trajectory_state"])
        target = dict(
            point_real=policy.action(feats["real"], *matrices["point_real"]),
            trajectory_real=real,
            trajectory_lagged=policy.match_norm(lagged, real),
            trajectory_reversed=-real,
            trajectory_state=policy.match_norm(state, real),
        )
        # A deterministic small shadow confirms the original fixed comparison.
        sample = torch.tensor(
            sorted(set([0, min(len(x) - 1, 135), len(x) // 2, len(x) - 1])),
            device=device,
        )
        shadow = base(x[sample], a[sample])["logit"].sigmoid().cpu().numpy()
        mismatch = float(np.max(np.abs(shadow - original[sample.cpu().numpy()])))
        if mismatch > 4e-6:
            raise ValueError(
                f"frozen-source replay disagrees with its sealed curve: {mismatch}"
            )
    fast = {name: policy.write_fast(v) for name, v in target.items()}
    arrays = {k: original.copy() for k in ARMS}
    for start in range(policy.FIRST, len(x), BATCH):
        end = min(start + BATCH, len(x))
        q = np.arange(start, end, dtype=np.int64)
        v, mask = x[start:end], a[start:end]
        weights = routes_for(mask, q)
        k = torch.as_tensor((q - policy.FIRST) // policy.STRIDE, device=device)
        with torch.no_grad():
            for name, trace in fast.items():
                values = policy.routed_pair(base, v, mask, weights, trace[k]).sigmoid()
                arrays[name][start:end] = values.cpu().numpy()
    if any(
        not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) for p in arrays.values()
    ):
        raise FloatingPointError("finite probabilities required")
    norm = {name: v.norm(dim=1).cpu().numpy() for name, v in fast.items()}
    if any(np.any(v > policy.RADIUS + 1e-6) for v in norm.values()):
        raise ValueError("temporary-state radius exceeded")
    stats = {
        k: dict(
            nonzero=int((v > 0).sum()),
            maximum=float(v.max()) if len(v) else 0.0,
            median=float(np.median(v)) if len(v) else 0.0,
        )
        for k, v in norm.items()
    }
    return arrays, dict(
        updates=stats,
        n_maturities=len(real),
        native_shadow_error=mismatch,
        norm_mismatches={
            k: int((abs(v - norm["trajectory_real"]) > 1e-6).sum())
            for k, v in norm.items()
            if k not in ("point_real", "trajectory_real")
        },
    )


def counts(rows):
    n_d = sum(r["disruptive"] for r in rows)
    n_s = len(rows) - n_d
    tp = sum(r["status"] == "tp" for r in rows)
    fp = sum(r["status"] == "fp" for r in rows)
    early = sum(r["status"] == "tp" and r["lead_ms"] > 3000 for r in rows)
    return dict(
        n_disruptive=n_d,
        n_safe=n_s,
        tp=tp,
        fp=fp,
        tpr=tp / n_d,
        fpr=fp / n_s,
        j=tp / n_d - fp / n_s,
        early_over3000=early,
        tp_30_to_3000=tp - early,
    )


def flips(real, control):
    pairs = list(zip(real, control, strict=True))
    if any(
        (a["shot"], a["disruptive"]) != (b["shot"], b["disruptive"]) for a, b in pairs
    ):
        raise ValueError("paired shot and outcome required")
    return {
        name: [a["shot"] for a, b in pairs if a["status"] == pos and b["status"] != pos]
        if name.endswith(("gained", "added"))
        else [a["shot"] for a, b in pairs if a["status"] != pos and b["status"] == pos]
        for name, pos in (
            ("tp_gained", "tp"),
            ("tp_lost", "tp"),
            ("fp_added", "fp"),
            ("fp_removed", "fp"),
        )
    }


def run(output):
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError("this fixed replay requires local CUDA")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device("cuda")
    order, cfg, source_done = reference()
    threshold = cfg["threshold"]
    tau = float(np.log(threshold / (1 - threshold)))
    mapping = read(COLD / "input_map.json")
    if mapping["calibration_ids"] != order[:20]:
        raise ValueError("original map and first20 changed")
    map_path = COLD / "input_map.json"
    declared_mapping = cfg["input_sha256"]
    if declared_mapping.get(str(map_path)) != sha(map_path):
        raise ValueError("numeric mapping is not the exact sealed reference mapping")
    base, parent = load_parent(DEFAULT_PARENT)
    if (
        parent["threshold"] != threshold
        or state_digest(base) != cfg["source_state_sha256"]
    ):
        raise ValueError("original source and operating threshold required")
    base = base.to(device)
    code = [
        Path(__file__),
        Path(policy.__file__),
        Path(teaching_loss.__code__.co_filename),
        Path(local_weights.__code__.co_filename),
        Path(io23.__file__),
        Path(simple_data.__file__),
        Path(prepare_actual.__code__.co_filename),
        DEFAULT_PARENT,
        PLAN,
        map_path,
        SOURCE / "protocol.json",
        SOURCE / "completed.json",
    ]
    hashes = {str(f.resolve()): sha(f) for f in code}
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    write_new(
        output / "protocol.json",
        dict(
            scope="CAUSAL_SUFFIX_DEV_REPLAY_AFTER_FIXED_ENDED_TEACHING_PREFIX",
            input_sha256=hashes,
            order=order,
            threshold=threshold,
            arms=ARMS,
            fit_indices=[21, FIT_AFTER],
            evaluation_indices=[FIT_AFTER + 1, 360],
            whole_stream_includes_prefix=True,
            fixed_source_sha256=PARENT_SHA256,
            fixed_source_state_sha256=cfg["source_state_sha256"],
            label_authority="HDF5",
            temporary_radius=policy.RADIUS,
            first=policy.FIRST,
            stride=policy.STRIDE,
            risk_head_frozen=True,
            target_labels_after_prediction_seal=True,
            independent_validation=False,
            ridge=RIDGE,
            torch=torch.__version__,
            gpu=torch.cuda.get_device_name(),
        ),
    )
    print("started", output, flush=True)
    history, models, fit_info, sealed = [], None, None, []
    all_rows = {name: [] for name in ARMS}
    for rank, sid in enumerate(order, 1):
        path = io23.path_for("EAST", sid)
        signature = path.stat()
        raw_sha = sha(path)
        if raw_sha != cfg["raw_sha256"][str(sid)]:
            raise ValueError("raw HDF changed since original reference")
        t, p = baseline_for(rank, sid, source_done)
        raw = simple_data.read_shot("EAST", sid, rate_contract="nominal")
        if not np.array_equal(raw["times"], t):
            raise ValueError("raw signal and original sealed clocks differ")
        signal = prepare_actual(raw, rank, mapping)
        if rank <= FIT_AFTER:
            traces, stats = {arm: p for arm in ARMS}, dict(frozen_prefix=True)
        else:
            if models is None:
                raise RuntimeError("all history must be sealed/revealed before suffix")
            traces, stats = predict(base, signal, p, tau, models, device)
        stem = output / "shots" / f"{rank:04d}_{sid}"
        curve = stem.with_suffix(".npz")
        with curve.open("xb") as handle:
            np.savez_compressed(handle, times=t, **traces)
        seal_path = stem.with_name(stem.name + "_seal.json")
        seal = dict(
            shot=sid,
            rank=rank,
            curve_sha256=sha(curve),
            raw_sha256=raw_sha,
            current_outcome_used_in_prediction=False,
            action_stats=stats,
            fit_sha256=fit_info["fit_sha256"] if fit_info else None,
        )
        write_new(seal_path, seal)
        flag, down = io23.read_outcome("EAST", sid)
        actual = path.stat()
        if (signature.st_size, signature.st_mtime_ns) != (
            actual.st_size,
            actual.st_mtime_ns,
        ):
            raise ValueError("raw HDF changed during replay")
        per = {}
        for arm, probabilities in traces.items():
            row = dict(
                shot=sid,
                disruptive=flag,
                down=down,
                **timeline(t, probabilities, flag, down, threshold),
            )
            all_rows[arm].append(row)
            per[arm] = row
        reveal_path = stem.with_name(stem.name + "_revealed.json")
        write_new(
            reveal_path, dict(shot=sid, after_seal_sha256=sha(seal_path), rows=per)
        )
        sealed.append(
            dict(
                shot=sid,
                curve_sha256=seal["curve_sha256"],
                seal_sha256=sha(seal_path),
                revealed_sha256=sha(reveal_path),
            )
        )
        if 21 <= rank <= FIT_AFTER:
            history.append(
                collect_teaching(
                    base,
                    signal,
                    t,
                    p,
                    (flag, down),
                    per["static"]["status"],
                    tau,
                    device,
                )
            )
        if rank == FIT_AFTER:
            if (
                len(history) != 80
                or [r["shot"] for r in history] != order[20:FIT_AFTER]
            ):
                raise ValueError("exact ended teaching cohort required")
            models, fit_info = fit(history, tau, output)
            del history
        if rank % 10 == 0:
            print("sealed and revealed", rank, "of 360", flush=True)
    summary = {
        name: {arm: counts(rows[selection]) for arm, rows in all_rows.items()}
        for name, selection in (
            ("full360", slice(None)),
            ("prefix100", slice(0, FIT_AFTER)),
            ("suffix260", slice(FIT_AFTER, None)),
        )
    }
    if (summary["full360"]["static"]["tp"], summary["full360"]["static"]["fp"]) != (
        116,
        29,
    ):
        raise ValueError("static fixed baseline counts changed")
    paired = {
        arm: flips(all_rows["trajectory_real"], rows)
        for arm, rows in all_rows.items()
        if arm != "trajectory_real"
    }
    if state_digest(base) != cfg["source_state_sha256"] or any(
        sha(Path(f)) != h for f, h in hashes.items()
    ):
        raise ValueError("source state or input artifact changed during replay")
    write_new(output / "summary.json", summary)
    write_new(output / "paired.json", paired)
    write_new(output / "per_shot.json", all_rows)
    write_new(
        output / "completed.json",
        dict(
            status="COMPLETE_DELAYED_TRAJECTORY_TEACHING360",
            seconds=time.monotonic() - started,
            n_shots=len(sealed),
            n_curves=len(sealed) * len(ARMS),
            sealed=sealed,
            artifact_sha256={f.name: sha(f) for f in output.iterdir() if f.is_file()},
        ),
    )
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
