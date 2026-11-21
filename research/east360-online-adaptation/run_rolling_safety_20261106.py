"""Train on released cold20, freeze decisions, and score separate EAST replays."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import moving_peak_replay_20261105 as r
import rolling_safety_policy_20261106 as model
from run_layer0_action_credit_history_20261106 import HISTORY, release_rows
from run_layer0_query_memory_20261105 import parents, raw_for
from verify_confirmation_raw_replay_20261105 import counts, pair, raw_truth, score


ROOT = r.ROOT / "probe_outputs"
HISTORY_CURVES = ROOT / "layer0_action_credit_history_20261106_v2"
GATES = ("peak1", "peak8", "rolling", "risk_clock")
MODES = ("static", "original", *GATES, "rolling_lagged", "rolling_scalar")
COHORTS = {
    "fixed360": ("layer_timing_control360_20261105_v2", "layer0", "layer0_stale", 360, 20),
    "conditional835": ("layer0_conditional835_20261106_v1", "intra", "lagged", 835, 0),
}
WITNESSES = {"fixed360": (21, 181, 359), "conditional835": (1, 418, 835)}


def open_history(order):
    cold = order[:20]
    block = HISTORY / "blocks"
    done = r.read(HISTORY / "completed.json")
    committed, revealed = (block / f"01_{k}.json" for k in ("commit", "reveals"))
    first = done["blocks"][0]
    if r.sha(committed) != first["commit_sha256"] or r.sha(revealed) != first["reveals_sha256"]:
        raise ValueError("original cold20 release changed")
    train = release_rows(1, done["sealed"][:20], r.read(committed), r.read(revealed), r.sha(committed))
    if [t["shot"] for t in train] != cold or sum(t["disruptive"] for t in train) != 8:
        raise ValueError("only the released cold20 can supply fitting labels")
    history = r.read(HISTORY_CURVES / "completed.json")
    shots, support = [], []
    for row, note in zip(train, history["sealed"][:20], strict=True):
        sid = row["shot"]
        path = HISTORY_CURVES / note["file"]
        fold = r.read(HISTORY_CURVES / "folds" / f"{note['rank']:04d}_{sid}_fit.json")
        if (note["shot"] != sid or note["group"] != "cold_loo" or r.sha(path) != note["curve_sha256"]
                or fold["held"] != sid or set(fold["training_ids"]) != set(cold) - {sid}):
            raise ValueError("historical curves must leave their own shot out")
        with np.load(path, allow_pickle=False) as f:
            t, static, adapted = (f[k] for k in ("times", "static", "adapted"))
            x = model.features(t, static, f["features"][:, 4:])
            teacher = model.make_teacher(t, static, adapted, row["down"], r.THRESHOLD)
            shots.append((x, teacher))
            support.append(dict(shot=sid, blocks=len(x), disruptive=row["disruptive"]))
    return shots, support


def cohort(name):
    label, real, stale, n, warm = COHORTS[name]
    path = ROOT / label
    cfg, done = (r.read(path / f"{k}.json") for k in ("protocol", "completed"))
    if (len(cfg["order"]) != n or len(set(cfg["order"])) != n
            or [s["shot"] for s in done["sealed"]] != cfg["order"]
            or (cfg["threshold"], cfg["first"], cfg["stride"]) != (r.THRESHOLD, model.FIRST, model.STRIDE)):
        raise ValueError("fixed original cohort and action clock required")
    return path, cfg, done, (real, stale, warm)


def recorded(path, note, rank, real, stale):
    file = path / "shots" / f"{rank:04d}_{note['shot']}.npz"
    if r.sha(file) != note["curve_sha256"]:
        raise ValueError("archived menu curve changed")
    with np.load(file, allow_pickle=False) as f:
        return tuple(f[k] for k in ("times", "static", real, stale))


@torch.no_grad()
def observe(base, raw, rank, mapping, static):
    """Compute the frozen native CLS only at the prespecified action anchors."""
    t = raw["times"]
    ix = np.arange(model.FIRST, len(t), model.STRIDE)
    if not len(ix):
        return model.features(t, static, np.empty((0, 128)))
    x, a = r.latent.make_views(r.prepare_actual(raw, rank, mapping), base.center.device)
    seen, hidden, probabilities = [], [], []
    hook = base.risk_head.register_forward_pre_hook(lambda _, args: seen.append(args[0]))
    try:
        for i in range(0, len(ix), r.BATCH):
            k = ix[i:i+r.BATCH]
            seen.clear()
            p = base(x[k].contiguous(), a[k].contiguous(),
                     base.initial_fast(len(k)).detach())["logit"].sigmoid()
            if len(seen) != 1 or seen[0].shape != (len(k), 64):
                raise ValueError("wrong frozen representation")
            probabilities.append(p.cpu().numpy())
            hidden.append(seen[0])
    finally:
        hook.remove()
    require_parity(np.concatenate(probabilities), static[ix])
    h, _ = r.latent.causal_history(torch.cat(hidden))
    return model.features(t, static, h.cpu().numpy())


def require_parity(actual, expected):
    difference = float(np.max(np.abs(actual - expected), initial=0.))
    if (difference > 5e-6 or not np.array_equal(actual.astype(float) >= r.THRESHOLD,
                                                 expected.astype(float) >= r.THRESHOLD)):
        raise ValueError(f"actual frozen neural write disagrees with recorded menu: {difference:g}")
    return difference


def witness(base, rule, raw, rank, mapping, static, real, stale, decisions, gates):
    n = model.FIRST + max(0, (len(decisions)-1)//2) * model.STRIDE + 1
    truncated = {**raw, **{k: raw[k][:n] for k in ("times", "values", "available")}}
    x = observe(base, truncated, rank, mapping, static[:n])
    partial = np.column_stack([gates[k].select(x) for k in GATES])
    np.testing.assert_array_equal(partial, decisions[:len(x)])

    ctx = r.latent.native_context(base, r.prepare_actual(raw, rank, mapping))
    true, wrong = r.actions(rule, ctx, stale=True)
    actual = true[:, :128].contiguous()
    other = r.policy.match_norm(wrong[:, :128].contiguous(), actual)
    torch.testing.assert_close(actual.norm(dim=1), other.norm(dim=1), rtol=0., atol=1e-6)
    select = actual.new_tensor(decisions[:, GATES.index("rolling")])[:, None]
    live, _ = r.forecast(base, ctx, {k: torch.cat((select*u, torch.zeros_like(u)), dim=1)
                                  for k,u in (("true",actual),("wrong",other))})
    use = decisions[:, GATES.index("rolling")]
    # One genuine test uses the trained selector; a second deliberately mixes
    # on/off when the witness is a constant gate, to test action composition.
    mask = use if use.any() and not use.all() else np.arange(len(use)) % 2 == 0
    if not np.array_equal(mask, use):
        swap = actual.new_tensor(mask)[:, None]
        mixed, _ = r.forecast(base, ctx, {"mix": torch.cat((swap*actual, torch.zeros_like(actual)), dim=1)})
        mixed_error = require_parity(mixed["mix"], model.splice(static,real,mask))
    else:
        mixed_error = 0.
    error = max(require_parity(live[k], model.splice(static,z,use))
                for k,z in (("true",real),("wrong",stale)))
    return dict(shot=raw["shot"], prefix_anchors=len(x), max_error=max(error,mixed_error),
                rolling_writes=int(use.sum()), available=len(use))


def predict(out):
    start = time.monotonic()
    r.configure()
    order, _, _, base, source, mapping, _, fit_path = parents()
    rule = r.load_rule(fit_path, base.center.device)
    shots, train = open_history(order)
    gates, traces = {}, {}
    for name in GATES:
        gates[name], traces[name] = model.fit(shots, name if name != "risk_clock" else "rolling",
                                             feedback=name != "risk_clock")
    out.mkdir(parents=True, exist_ok=False)
    r.write_new(out/"protocol.json", dict(
        status="COLD20_ONLY_ROLLING_SAFETY", training=train, train_labels=20, new_query_labels=0,
        fit=traces, threshold=r.THRESHOLD, source_state=source, first=model.FIRST,
        stride=model.STRIDE, modes=MODES, gates=GATES,
        temperature=model.TEMPERATURE, ridge=model.RIDGE, evaluation_outcomes_during_prediction=0,
    ))
    r.save_arrays(out/"gates.npz", **{f"{k}_{p}":getattr(v,p) for k,v in gates.items()
                                     for p in ("mean","scale","weight")})
    archive = {}
    with patch.object(r.io23, "read_outcome", side_effect=RuntimeError("evaluation outcomes sealed off")):
        for name in COHORTS:
            folder, cfg, done, (real, stale, warm) = cohort(name)
            if (cfg["source_state_sha256"] != source or cfg["fit_sha256"] != r.sha(fit_path)
                    or (warm and cfg["order"] != order) or (not warm and set(order) & set(cfg["order"]))):
                raise ValueError("wrong frozen parent, original direction, or cohort")
            decisions, witnesses, sizes = [], [], []
            for rank, note in enumerate(done["sealed"], 1):
                t, static, active, lagged = recorded(folder, note, rank, real, stale)
                m = len(np.arange(model.FIRST, len(t), model.STRIDE))
                if rank <= warm:
                    select = np.zeros((m, len(GATES)), dtype=bool)
                else:
                    raw = raw_for(note["shot"], cfg["raw_sha256"])
                    np.testing.assert_array_equal(raw["times"], t)
                    x = observe(base, raw, rank if warm else 21, mapping, static)
                    select = np.column_stack([gates[k].select(x) for k in GATES])
                    if rank in WITNESSES[name]:
                        witnesses.append(witness(base, rule, raw, rank if warm else 21, mapping,
                                                 static, active, lagged, select, gates))
                decisions.append(select)
                sizes.append(m)
                if rank % 120 == 0:
                    print(name, rank, flush=True)
            p = out/f"{name}.npz"
            z = np.concatenate(decisions)
            r.save_arrays(p, order=np.array(cfg["order"]), sizes=np.array(sizes), decisions=z)
            archive[name] = dict(decisions_sha256=r.sha(p), parent_sha256=r.sha(folder/"completed.json"),
                                 writes=dict(zip(GATES,z.sum(axis=0).tolist(),strict=True)),
                                 eligible_blocks=sum(sizes[warm:]), witnesses=witnesses)
    if any(p.grad is not None for p in base.parameters()) or r.state_digest(base) != source:
        raise ValueError("frozen source model changed")
    r.write_new(out/"completed.json", dict(status="EAST_ROLLING_SAFETY_PREDICTIONS_SEALED",
        seconds=time.monotonic()-start, protocol_sha256=r.sha(out/"protocol.json"), cohorts=archive))
    print("predictions sealed", archive, flush=True)


def recount(out):
    start = time.monotonic()
    done, plan = (r.read(out / f"{k}.json") for k in ("completed", "protocol"))
    if (done["status"] != "EAST_ROLLING_SAFETY_PREDICTIONS_SEALED"
            or done["protocol_sha256"] != r.sha(out/"protocol.json") or plan["threshold"] != r.THRESHOLD):
        raise ValueError("seal predictions before any evaluation outcome access")
    result = {}
    for name in COHORTS:
        folder, cfg, parent, (real, stale, warm) = cohort(name)
        p = out/f"{name}.npz"
        sealed = done["cohorts"][name]
        if r.sha(p) != sealed["decisions_sha256"] or r.sha(folder/"completed.json") != sealed["parent_sha256"]:
            raise ValueError("decisions or their reference curves changed")
        with np.load(p, allow_pickle=False) as f:
            np.testing.assert_array_equal(f["order"], cfg["order"])
            sizes = f["sizes"]
            write = np.split(f["decisions"], np.cumsum(sizes)[:-1])
        rows = {key: [] for key in MODES}
        for rank, (note, choose) in enumerate(zip(parent["sealed"], write, strict=True), 1):
            t, s, a, lag = recorded(folder, note, rank, real, stale)
            if rank <= warm and choose.any():
                raise ValueError("cold predictions must remain static")
            traces = {"static":s,"original":a}
            traces.update({g:model.splice(s,a,choose[:,j]) for j,g in enumerate(GATES)})
            use = choose[:, GATES.index("rolling")]
            traces["rolling_lagged"] = model.splice(s,lag,use)
            traces["rolling_scalar"] = model.splice(s,model.scalar_control(s,a),use)
            sid = note["shot"]
            truth, times = raw_truth(r.io23.path_for("EAST", sid), sid)
            np.testing.assert_array_equal(times, t)
            for mode in MODES:
                rows[mode].append(score(t, traces[mode].astype(np.float64), truth, r.THRESHOLD))
        metrics = {k: counts(v) for k,v in rows.items()}
        expected = ((116,29,120,33) if warm else (257,68,299,85))
        q, w = metrics["static"], metrics["original"]
        if (q["tp"],q["fp"],w["tp"],w["fp"]) != expected:
            raise ValueError("static or original internal update no longer matches independent HDF5 scores")
        comparisons = {k: dict(strict=pair(rows["rolling"],rows[k]),
                               capped=pair(rows["rolling"],rows[k],capped=True))
                       for k in MODES if k != "rolling"}
        result[name] = dict(metrics=metrics, comparisons=comparisons, rows=rows, **sealed)
        print(name,{k:(v["tp"],v["tp_30_to_3000"],v["fp"]) for k,v in metrics.items()},flush=True)
    main, transfer = (result[x]["metrics"] for x in COHORTS)
    primary = main["rolling"]
    gate = (primary["tp"] > main["static"]["tp"] and primary["fpr"] <= .20
            and all(primary["j"] > main[k]["j"] for k in ("peak1","peak8","risk_clock","rolling_lagged","rolling_scalar"))
            and transfer["rolling"]["fpr"] <= .20 and transfer["rolling"]["tp"] >= transfer["static"]["tp"])
    r.write_new(out/"result.json", dict(status="EAST_HDF5_RECOUNT_COMPLETE", promote=gate,
        seconds=time.monotonic()-start, cohorts=result))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("predict", "score"))
    p.add_argument("output", type=Path)
    a = p.parse_args()
    (predict if a.phase == "predict" else recount)(a.output)
