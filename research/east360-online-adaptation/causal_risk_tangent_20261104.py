"""Counterfactual internal step preserving causal past risk tangents."""

from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import odd_internal_feedback_20261104 as odd

p = odd.p
ROOT = p.ROOT
OUT = ROOT / "probe_outputs/causal_risk_tangent_20261104_v1"
PLAN = ROOT / "CAUSAL_RISK_TANGENT_PLAN_20261104.md"
RULES = ("east20", "mixed_uniform")
LAGS = (128, 96, 64, 32)
ARMS = ("raw", "shrink", "real", "reverse", "permuted")


def remove_anchor_directions(jacobian, vector):
    j, v = np.asarray(jacobian, dtype=float), np.asarray(vector, dtype=float)
    if (
        j.ndim != 2
        or v.shape != (j.shape[1],)
        or not all(np.isfinite(x).all() for x in (j, v))
    ):
        raise ValueError("finite matching past-risk gradients and direction required")
    _, s, vh = np.linalg.svd(j, full_matrices=False)
    if not len(s) or s[0] == 0:
        return v.copy()
    basis = vh[s > s[0] * 1e-10]
    return v - basis.T @ (basis @ v)


def fixed_unit_step(v):
    n = np.linalg.norm(v)
    return 0.05 * v / n if n else np.zeros_like(v)


def apply_step(params, gradient):
    chunks = np.split(np.asarray(gradient), np.cumsum([q.numel() for q in params])[:-1])
    loss = sum(
        (q * torch.as_tensor(v, dtype=q.dtype).reshape_as(q)).sum()
        for q, v in zip(params, chunks, strict=True)
    )
    loss.backward()
    optimizer = torch.optim.SGD(params, lr=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def anchor_jacobian(model, params, x, a):
    logits = model(x, a)["logit"]
    rows = []
    for i, z in enumerate(logits):
        grad = torch.autograd.grad(z, params, retain_graph=i < len(logits) - 1)
        rows.append(torch.cat([v.reshape(-1) for v in grad]).detach().double().numpy())
    return np.stack(rows), logits.detach()


def aggregate(rows, threshold):
    report = {}
    for rule in RULES:
        groups = {}
        for group in ("safe", "earlier", "positive"):
            subset = [r for r in rows if r["group"] == group]
            ids = sorted({r["shot"] for r in subset})
            item = dict(shots=len(ids), cases=len(subset))
            for a, b in (
                ("real", "none"),
                ("real", rule + "__raw"),
                ("real", rule + "__shrink"),
                ("real", rule + "__reverse"),
                ("real", rule + "__permuted"),
            ):
                key = rule + "__" + a
                v = np.array(
                    [
                        np.mean(
                            [
                                r["scores"][key]["bce"] - r["scores"][b]["bce"]
                                for r in subset
                                if r["shot"] == sid
                            ]
                        )
                        for sid in ids
                    ]
                )
                item["real_vs_" + b.split("__")[-1]] = dict(
                    delta_bce=float(v.mean()),
                    better=int((v < 0).sum()),
                    worse=int((v > 0).sum()),
                    shot_se=float(v.std(ddof=1) / np.sqrt(len(v))),
                )
            key = rule + "__real"
            item["up"] = sum(
                r["scores"]["none"]["p"] < threshold <= r["scores"][key]["p"]
                for r in subset
            )
            item["down"] = sum(
                r["scores"][key]["p"] < threshold <= r["scores"]["none"]["p"]
                for r in subset
            )
            item["max_past_anchor_delta"] = max(
                r["scores"][key]["past_max_delta"] for r in subset
            )
            item["median_step_norm"] = float(
                np.median([r["scores"][key]["norm"] for r in subset])
            )
            groups[group] = item
        report[rule] = groups
    return report


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    for folder in (odd.OUT, odd.FIT, p.original.FEATURES):
        p.verify_saved(folder)
    previous = p.data.read_json(odd.OUT / "rows.json")
    protocol = p.data.read_json(p.data.RUN / "protocol.json")
    threshold = protocol["threshold"]
    ids = [r["shot"] for r in protocol["order"]]
    bound = {
        str(path): p.data.sha(path)
        for path in (
            PLAN,
            Path(__file__),
            Path(odd.__file__),
            Path(p.__file__),
            odd.OUT / "completed.json",
            odd.OUT / "rows.json",
            odd.FIT / "fit.npz",
            p.original.FEATURES / "features_and_logits.npz",
            p.data.RUN / "protocol.json",
            p.data.DEFAULT_PARENT,
            p.data.MAP,
            p.data.RAW_HASHES,
        )
    }
    OUT.mkdir()
    p.data.write_json(
        OUT / "protocol.json",
        dict(
            scope="CAUSAL_PAST_RISK_TANGENT_FIXED_HELD80_NOT_ONLINE",
            anchor_lags=LAGS,
            rules=RULES,
            held_ids=ids[20:100],
            input_sha256=bound,
            parameter_scope=256,
            max_norm=0.05,
            adaptive_threshold=False,
        ),
    )
    with np.load(odd.FIT / "fit.npz", allow_pickle=False) as f:
        fits = {
            rule: {k: f[rule + "_" + k] for k in ("mean", "scale", "coef")}
            for rule in RULES
        }
    with np.load(
        p.original.FEATURES / "features_and_logits.npz", allow_pickle=False
    ) as f:
        m, mperm = f["held_long"], f["held_permuted"]
    motions = {
        rule: {
            mode: odd.odd_direction(fits[rule], v)
            for mode, v in (
                ("real", m),
                ("reverse", -m),
                ("permuted", mperm),
            )
        }
        for rule in RULES
    }
    base, parent = p.data.load_parent(p.data.DEFAULT_PARENT)
    base.eval().requires_grad_(False)
    state = p.data.state_digest(base)
    if state != protocol["source_state_sha256"] or parent["threshold"] != threshold:
        raise ValueError("source or threshold changed")
    model = p.original.make_adapter(base, "norm")
    params = tuple(q for q in model.parameters() if q.requires_grad)
    if sum(q.numel() for q in params) != 256:
        raise ValueError("fixed 256 trainable internal parameters required")
    initial = tuple(q.detach().clone() for q in params)
    mapping, hashes = p.data.read_json(p.data.MAP), p.data.read_json(p.data.RAW_HASHES)
    rows, receipts, k = [], [], 0
    for index, sid in enumerate(ids[20:100], 21):
        signal, cases, receipt = p.data._read_sealed(index, sid, mapping, hashes)
        receipts.append(receipt)
        for case in cases:
            old = previous[k]
            if (sid, case["origin"], case["label"]) != (
                old["shot"],
                old["origin"],
                old["label"],
            ):
                raise ValueError("same held cases required")
            anchors = [case["arrival"] - lag for lag in LAGS]
            if min(anchors) < 0 or max(anchors) >= case["arrival"] - 31:
                raise ValueError("anchors must precede the observed trend")
            xa, aa = p.original.window_batch(signal, anchors, device="cpu")
            p.reset(params, initial)
            jac, past = anchor_jacobian(model, params, xa, aa)
            actions = {}
            for rule, directions in motions.items():
                raw = {
                    mode: fixed_unit_step(arr[k]) for mode, arr in directions.items()
                }
                steps = {
                    mode: remove_anchor_directions(jac, v) for mode, v in raw.items()
                }
                if max(np.linalg.norm(v) for v in steps.values()) > 0.05 + 1e-12:
                    raise ValueError("projection increased the step budget")
                steps["raw"] = raw["real"]
                steps["shrink"] = raw["real"] * (np.linalg.norm(steps["real"]) / 0.05)
                actions.update({rule + "__" + a: v for a, v in steps.items()})
            # The action map is already fixed before constructing the future query.
            xq, aq = p.original.window_batch(signal, [case["future"]], device="cpu")
            scores = {}
            for arm, step in (("none", np.zeros(256)), *actions.items()):
                p.reset(params, initial)
                if arm != "none":
                    apply_step(params, step)
                with torch.no_grad():
                    z = model(xq, aq)["logit"]
                    pa = model(xa, aa)["logit"]
                    loss = F.binary_cross_entropy_with_logits(
                        z, torch.full_like(z, float(case["label"]))
                    )
                scores[arm] = dict(
                    bce=float(loss),
                    p=float(z.sigmoid()),
                    logit=float(z),
                    norm=float(np.linalg.norm(step)),
                    past_max_delta=float((pa - past).abs().max()),
                    linear_past_max=float(np.max(np.abs(jac @ step))),
                )
            if (scores["none"]["bce"], scores["none"]["logit"]) != (
                old["arms"]["none"]["bce"],
                old["arms"]["none"]["logit"],
            ):
                raise ValueError("fixed no-update baseline changed")
            for rule in RULES:
                if (
                    abs(
                        scores[rule + "__raw"]["logit"]
                        - old["arms"][rule + "__real"]["logit"]
                    )
                    > 2e-6
                ):
                    raise ValueError("original odd-direction control changed")
            rows.append(dict(shot=sid, **case, anchors=anchors, scores=scores))
            k += 1
    p.reset(params, initial)
    if (
        k != 240
        or p.data.state_digest(base) != state
        or any(p.data.sha(Path(f)) != h for f, h in bound.items())
    ):
        raise ValueError("incomplete or changed fixed inputs")
    summary = aggregate(rows, threshold)
    for name, value in (
        ("rows.json", rows),
        ("summary.json", summary),
        ("held_inputs.json", receipts),
    ):
        p.data.write_json(OUT / name, value)
    p.data.write_json(
        OUT / "completed.json",
        dict(
            status="COMPLETE_CAUSAL_TANGENT_HELD_DIAGNOSTIC",
            seconds=time.monotonic() - started,
            artifact_sha256={
                f.name: p.data.sha(f) for f in OUT.iterdir() if f.is_file()
            },
        ),
    )
    print(summary, flush=True)


if __name__ == "__main__":
    run()
