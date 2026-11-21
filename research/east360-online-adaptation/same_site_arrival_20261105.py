"""Learn an innovation-driven, shot-local action at its actual insertion site."""

from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.direct_fast_memory_20261104.model import levels
from experiments.risk_aligned_online.direct_fast_memory_20261104.timeline import (
    window_batch,
)
from experiments.risk_aligned_online.mask_separated_intra_20261103.run import (
    state_digest,
)
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    load_parent,
)

import local_affine_reach_20261105 as reach
import source_supported_internal_update_20261104 as previous
import timed_token_norm_20261105 as timed

ROOT = io23.ROOT
OUT = ROOT / "probe_outputs/same_site_arrival_20261105_v2"
PLAN = ROOT / "SAME_SITE_ARRIVAL_PLAN_20261105.md"
RADIUS, RIDGE = 0.5, 0.005
CHANNELS = [i for i in range(23) if i not in (2, 10)]
SITES = (("early_recent", 0, "recent"), ("last_cls", 1, "cls"))
DATASETS = ("east20", "mixed")
INPUTS = ("observed", "innovation")
FEEDBACK = ("real", "hold", "reversed", "negative")
GROUPS = ("safe", "earlier", "positive")


def arrival_features(base, shot, origin, maturity, tau):
    if origin < 127 or maturity != origin + 8 or maturity >= len(shot["values"]):
        raise ValueError("exactly the eight newly arrived targets are required")
    xo, ao = window_batch(shot, [origin], device="cpu")
    with torch.no_grad():
        outputs = base(xo, ao)
        obs = torch.as_tensor(shot["values"][origin + 1 : maturity + 1])
        avail = torch.as_tensor(shot["available"][origin + 1 : maturity + 1])
        observed = levels(base, obs, avail)[:, CHANNELS]
        forecast = outputs["forecast"][0, :8, CHANNELS]
        available = avail[:, CHANNELS]
        means = []
        for values in (observed, forecast):
            halves = values.reshape(2, 4, -1)
            mask = available.reshape(2, 4, -1)
            count = mask.sum(1)
            supported = (count >= 2).all(0)
            mean = halves.masked_fill(~mask, 0).sum(1) / count.clamp_min(1)
            means.append(torch.where(supported, mean[1] - mean[0], 0.0))
        state = torch.tanh(outputs["logit"][0] - tau)
        obs, pred = means
        return {
            name: torch.cat([v, v * state]).double().numpy()
            for name, v in (
                ("observed", obs),
                ("innovation", obs - pred),
                ("hold", -pred),
                ("reversed", -obs - pred),
            )
        }


def _bounded(raw):
    return RADIUS * raw / (1 + raw.square().sum(dim=1, keepdim=True)).sqrt()


def fit_action(features, offset, jacobian, y, weights):
    x, z, g, y, w = (
        torch.as_tensor(a, dtype=torch.float64)
        for a in (features, offset, jacobian, y, weights)
    )
    if (
        x.ndim != 2
        or g.ndim != 2
        or len(g) != len(x)
        or z.shape != y.shape
        or w.shape != z.shape
        or z.shape != (len(x),)
        or not all(torch.isfinite(a).all() for a in (x, z, g, y, w))
        or ((y != 0) & (y != 1)).any()
        or (w < 0).any()
        or abs(float(w.sum()) - 1) > 1e-12
    ):
        raise ValueError(
            "finite paired features, risks, derivatives and fit weights required"
        )
    scale = (w @ x.square()).sqrt()
    scale = torch.where(scale > 1e-12, scale, 1.0)
    x = x / scale
    matrix = torch.zeros(
        (x.shape[1], g.shape[1]), dtype=torch.float64, requires_grad=True
    )
    opt = torch.optim.LBFGS(
        (matrix,),
        max_iter=160,
        tolerance_grad=1e-8,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def loss():
        offset = z + (g * _bounded(x @ matrix)).sum(1)
        return (
            w @ F.binary_cross_entropy_with_logits(offset, y, reduction="none")
            + RIDGE * matrix.square().sum() / 2
        )

    def closure():
        opt.zero_grad(set_to_none=True)
        objective = loss()
        objective.backward()
        return objective

    start = float(loss().detach())
    opt.step(closure)
    end = float(loss().detach())
    if not np.isfinite(end) or end > start + 1e-10:
        raise FloatingPointError(
            "finite descent on the predeclared fit objective required"
        )
    return dict(
        scale=scale.numpy(), matrix=matrix.detach().numpy(), before=start, after=end
    )


def action(fit, x):
    raw = (x / fit["scale"]) @ fit["matrix"]
    return RADIUS * raw / np.sqrt(1 + np.square(raw).sum(axis=1, keepdims=True))


def match_norm(control, real):
    norm = np.linalg.norm(control, axis=1, keepdims=True)
    target = np.linalg.norm(real, axis=1, keepdims=True)
    return control * (target / np.where(norm > 0, norm, 1))


def _weight(rows):
    count = Counter((r["group"], r["shot"]) for r in rows)
    per_group = Counter(g for g, _ in count)
    if set(per_group) != set(GROUPS):
        raise ValueError("all training groups are required")
    return np.array(
        [1 / (3 * per_group[r["group"]] * count[r["group"], r["shot"]]) for r in rows]
    )


def _context(base, signal, query, tau):
    x, a = window_batch(signal, [query], device="cpu")
    m = query - 24
    obs = arrival_features(base, signal, m - 8, m, tau)
    weights = {
        key: timed.support_weights(a, [query], [m], route) for key, _, route in SITES
    }
    with torch.no_grad():
        bare = float(base(x, a)["logit"])
    return x, a, obs, weights, bare


def _teaching(base, signal, query, tau):
    x, a, obs, weights, bare = _context(base, signal, query, tau)
    gradients = {}
    for site, layer, _ in SITES:
        z, g = reach.value_and_gradient(base, x, a, weights[site], layer)
        if float(z) != bare:
            raise ValueError("teacher zero state must preserve the frozen classifier")
        gradients[site] = g[0].double().numpy()
    return dict(z=bare - tau, features={k: obs[k] for k in INPUTS}, gradients=gradients)


def east_fit_record(base, signal, sid, case, tau):
    q = case["future"]
    if case["arrival"] != q - 24 or case["origin"] != q - 32:
        raise ValueError("EAST teaching record does not match the arrival clock")
    return dict(
        shot=sid,
        query=q,
        group=case["group"],
        label=case["label"],
        **_teaching(base, signal, q, tau),
    )


def _source(base, tau, records, receipts, allowed):
    groups = defaultdict(list)
    for row in records:
        if row["shot"] not in allowed or row["role"] != "train":
            raise ValueError("no source validation/test fit signal access")
        groups[row["shot"]].append(row)
    advanced = io23.advanced_times()
    fit = []
    sha = reach.sealed.sha
    for j, (sid, rows) in enumerate(sorted(groups.items()), 1):
        path = io23.path_for("J-TEXT", sid)
        if sha(path) != receipts[sid]:
            raise ValueError("archived source raw binding changed")
        raw = simple_data.read_shot("J-TEXT", sid)
        flag, down = io23.read_outcome("J-TEXT", sid)
        y, valid = io23.labels_at(
            raw["times"], flag, down, io23.effective_horizon_ms(sid, advanced)[0]
        )
        raw["available"][:, 2] = False
        for r in rows:
            q = r["query"]
            if (
                q < 159
                or not valid[q]
                or r["label"] != y[q]
                or r["actual_hdf_disruptive"] != flag
            ):
                raise ValueError("archived TRAIN query or label differs from HDF")
            group = ("positive" if y[q] else "earlier") if flag else "safe"
            fit.append(
                dict(
                    shot=sid,
                    group=group,
                    label=int(y[q]),
                    query=q,
                    **_teaching(base, raw, q, tau),
                )
            )
        if sha(path) != receipts[sid]:
            raise ValueError("source HDF changed during fit extraction")
        if j % 300 == 0:
            print("source TRAIN read", j, flush=True)
    if len(groups) != 1299 or len(fit) != 1410:
        raise ValueError("archived fixed TRAIN corpus size changed")
    return fit


def _fit(source, target):
    models = {}
    arrays = {}
    for scope in DATASETS:
        rows = target if scope == "east20" else source + target
        weights = (
            _weight(target)
            if scope == "east20"
            else np.r_[0.5 * _weight(source), 0.5 * _weight(target)]
        )
        z, y = (
            np.asarray([r["z"] for r in rows]),
            np.asarray([r["label"] for r in rows]),
        )
        arrays.update(
            {
                scope + "_z": z,
                scope + "_y": y,
                scope + "_weight": weights,
                scope + "_shot": np.asarray([r["shot"] for r in rows]),
                scope + "_query": np.asarray([r["query"] for r in rows]),
            }
        )
        for site, _, _ in SITES:
            g = np.stack([r["gradients"][site] for r in rows])
            arrays[scope + "_" + site + "_jacobian"] = g
            for kind in INPUTS:
                features = np.stack([r["features"][kind] for r in rows])
                arrays[scope + "_" + kind + "_features"] = features
                key = scope + "__" + site + "__" + kind
                model = fit_action(features, z, g, y, weights)
                models[key] = model
                arrays[key + "_scale"], arrays[key + "_matrix"] = (
                    model["scale"],
                    model["matrix"],
                )
    return models, arrays


def _record(z, energy=0.0, norm=0.0):
    return dict(z=z, energy=energy, norm=norm)


def _predict(base, signal, query, tau, models):
    x, a, obs, weights, bare = _context(base, signal, query, tau)
    arms = {"none": _record(bare)}
    for site, layer, _ in SITES:
        w = weights[site]
        for scope in DATASETS:
            for kind in INPUTS:
                key = scope + "__" + site + "__" + kind
                model = models[key]
                real = action(model, obs[kind][None])
                placebo = (
                    np.zeros_like(real)
                    if kind == "observed"
                    else action(model, obs["hold"][None])
                )
                reverse_signal = (
                    -obs["observed"] if kind == "observed" else obs["reversed"]
                )
                reverse = action(model, reverse_signal[None])
                directions = (
                    real,
                    match_norm(placebo, real),
                    match_norm(reverse, real),
                    -real,
                )
                for mode, vector in zip(FEEDBACK, directions, strict=True):
                    value = x.new_tensor(vector if w.any() else np.zeros_like(vector))
                    param = torch.zeros_like(value, requires_grad=True)
                    # Linear surrogate for a single SGD step with lr=1.
                    loss = -(param * value).sum()
                    loss.backward()
                    with torch.no_grad():
                        param.sub_(param.grad)
                        logit, energy = reach.routed_logit(base, x, a, w, param, layer)
                        if (
                            not torch.isfinite(logit).all()
                            or not torch.isfinite(energy).all()
                        ):
                            raise FloatingPointError("nonfinite real model forward")
                    arms[key + "__" + mode] = _record(
                        float(logit[0]), float(energy[0]), float(param.detach().norm())
                    )
    return arms


def _summaries(rows, models, tau):
    results = {}
    for group in GROUPS:
        subset = [r for r in rows if r["group"] == group]
        by = defaultdict(list)
        for r in subset:
            by[r["shot"]].append(r)
        comparisons = {}
        for key in models:
            real = key + "__real"
            comp = {}
            for ref in ("none", *FEEDBACK[1:]):
                arm = ref if ref == "none" else key + "__" + ref
                d = np.asarray(
                    [
                        np.mean(
                            [
                                np.logaddexp(
                                    0, (1 - 2 * r["label"]) * r["arms"][real]["z"]
                                )
                                - np.logaddexp(
                                    0, (1 - 2 * r["label"]) * r["arms"][arm]["z"]
                                )
                                for r in cases
                            ]
                        )
                        for cases in by.values()
                    ]
                )
                comp[ref] = dict(
                    delta_bce=float(d.mean()) if len(d) else None,
                    shot_se=float(d.std(ddof=1) / np.sqrt(len(d)))
                    if len(d) > 1
                    else None,
                    improved=int((d < 0).sum()),
                    worse=int((d > 0).sum()),
                    tied=int((d == 0).sum()),
                    norm_mismatches=sum(
                        abs(r["arms"][real]["norm"] - r["arms"][arm]["norm"]) > 1e-6
                        for r in subset
                    )
                    if ref != "none"
                    else None,
                )
            comp["crossings"] = dict(
                up=sum(
                    r["arms"]["none"]["z"] < tau <= r["arms"][real]["z"] for r in subset
                ),
                down=sum(
                    r["arms"][real]["z"] < tau <= r["arms"]["none"]["z"] for r in subset
                ),
            )
            comp["median_step_norm"] = (
                float(np.median([r["arms"][real]["norm"] for r in subset]))
                if subset
                else None
            )
            comparisons[key] = comp
        results[group] = dict(shots=len(by), cases=len(subset), models=comparisons)
    return results


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    data, sha = previous.data, reach.sealed.sha
    for folder in (previous.OUT, timed.OUT, reach.OUT):
        previous.verify_saved(folder)
    reads = (
        Path(__file__),
        PLAN,
        Path(reach.__file__),
        Path(timed.__file__),
        Path(previous.__file__),
        Path(data.__file__),
        Path(io23.__file__),
        Path(simple_data.__file__),
        DEFAULT_PARENT,
        io23.JTEXT_INDEX,
        io23.JTEXT_ADVANCED_TIME_PATH,
        data.MAP,
        data.RAW_HASHES,
        data.RUN / "protocol.json",
        previous.OUT / "source_records.json",
        previous.OUT / "source_inputs.json",
        timed.OUT / "rows.json",
        reach.OUT / "completed.json",
        reach.FRESH / "hdf5_metadata.json",
    )
    bound = {str(f.resolve()): sha(f) for f in reads}
    if sha(reach.FRESH / "hdf5_metadata.json") != reach.sealed.V5_METADATA_SHA:
        raise ValueError("fresh HDF reconciliation differs from the sealed inventory")
    saved = data.read_json(data.RUN / "protocol.json")
    ids = [r["shot"] for r in saved["order"]]
    mapping, hashes = data.read_json(data.MAP), data.read_json(data.RAW_HASHES)
    if (
        mapping["calibration_ids"] != ids[:20]
        or len(ids) != 360
        or len(set(ids)) != 360
    ):
        raise ValueError("locked order and previous preparation required")
    base, parent = load_parent(DEFAULT_PARENT)
    initial = state_digest(base)
    threshold = parent["threshold"]
    if initial != saved["source_state_sha256"] or threshold != saved["threshold"]:
        raise ValueError("qualified source and its threshold must not change")
    tau = float(np.log(threshold / (1 - threshold)))
    OUT.mkdir()
    reach.sealed.write_new(
        OUT / "protocol.json",
        dict(
            scope="OFFLINE_SAME_SITE_ARRIVAL_UTILITY_NOT_FULL_ONLINE",
            sites=SITES,
            radius=RADIUS,
            ridge=RIDGE,
            threshold=threshold,
            first20_fit=ids[:20],
            held80=ids[20:100],
            later20=ids[100:120],
            no_held_labels_fit=True,
            source_train_only=True,
            origin_to_maturity=8,
            maturity_to_query=24,
            independent_validation=False,
            input_sha256=bound,
        ),
    )
    src = _source(
        base,
        tau,
        data.read_json(previous.OUT / "source_records.json"),
        {
            r["shot"]: r["raw_sha256"]
            for r in data.read_json(previous.OUT / "source_inputs.json")
        },
        set(io23.source_roles()["train"]),
    )
    fit, receipts = [], []
    for idx, sid in enumerate(ids[:20], 1):
        signal, cases, receipt = data._read_sealed(idx, sid, mapping, hashes)
        receipts.append(receipt)
        for case in cases:
            fit.append(east_fit_record(base, signal, sid, case, tau))
    if len(fit) != 60:
        raise ValueError("fixed first20 fit cases changed")
    models, fit_arrays = _fit(src, fit)
    fit_end = time.monotonic()
    timed_baseline = data.read_json(timed.OUT / "rows.json")
    all_rows = []
    for idx, sid in enumerate(ids[20:120], 21):
        signal, cases, receipt = data._read_sealed(idx, sid, mapping, hashes)
        receipts.append(receipt)
        for case in cases:
            q = case["future"]
            if case["arrival"] != q - 24 or case["origin"] != q - 32:
                raise ValueError("identical causal arrival clock required")
            arms = _predict(base, signal, q, tau, models)
            if idx <= 100:
                old = timed_baseline[len(all_rows)]
                if (sid, q, case["label"], arms["none"]["z"]) != (
                    old["shot"],
                    old["future"],
                    old["label"],
                    old["arms"]["none"]["logit"],
                ):
                    raise ValueError(
                        "held comparison did not preserve the case or zero state"
                    )
            all_rows.append(dict(shot=sid, stream_index=idx, **case, arms=arms))
        if idx % 20 == 0:
            print("scored through stream rank", idx, flush=True)
    if sum(r["stream_index"] <= 100 for r in all_rows) != 240:
        raise ValueError("held80 case count changed")
    if len({r["shot"] for r in all_rows}) != 100:
        raise ValueError("full held and later-development rosters required")
    if initial != state_digest(base) or any(
        sha(Path(f)) != h for f, h in bound.items()
    ):
        raise ValueError("source or bound inputs changed")
    summary = {
        name: _summaries(
            [r for r in all_rows if ((r["stream_index"] <= 100) == held)], models, tau
        )
        for name, held in (("held80", True), ("later20", False))
    }
    np.savez_compressed(OUT / "fit.npz", **fit_arrays)
    for f, value in (
        ("rows.json", all_rows),
        ("summary.json", summary),
        ("east_inputs.json", receipts),
        (
            "fit_objective.json",
            {k: {s: v[s] for s in ("before", "after")} for k, v in models.items()},
        ),
    ):
        reach.sealed.write_new(OUT / f, value)
    reach.sealed.write_new(
        OUT / "completed.json",
        dict(
            status="COMPLETE_SAME_SITE_ARRIVAL_DEVELOPMENT",
            source_cases=len(src),
            east_fit_cases=len(fit),
            east_scored_cases=len(all_rows),
            seconds=time.monotonic() - started,
            fit_elapsed=fit_end - started,
            artifact_sha256={f.name: sha(f) for f in OUT.iterdir()},
        ),
    )
    print(
        {
            sl: {g: v["cases"] for g, v in groups.items()}
            for sl, groups in summary.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    run()
