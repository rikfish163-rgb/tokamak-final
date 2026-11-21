"""Matched absolute-clock routing for a fixed learned internal update."""

from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import odd_internal_feedback_20261104 as odd

p = odd.p
ROOT = p.ROOT
PLAN = ROOT / "TIMED_TOKEN_NORM_PLAN_20261105.md"
OUT = ROOT / "probe_outputs/timed_token_norm_20261105_v1"
RULES = ("east20", "mixed_uniform")
MODES = {
    "recent": ("recent", "real"),
    "older": ("older", "real"),
    "global": ("global", "real"),
    "cls": ("cls", "real"),
    "reverse": ("recent", "reverse"),
    "permuted": ("recent", "permuted"),
}


def support_weights(available, queries, maturities, route):
    if (
        available.ndim != 3
        or available.shape[1:] != (128, 23)
        or available.dtype != torch.bool
    ):
        raise ValueError("matched source windows and boolean availability required")
    q = torch.as_tensor(queries, device=available.device)
    t = torch.as_tensor(maturities, device=available.device)
    if q.shape != t.shape or q.shape != (len(available),) or (q < t).any():
        raise ValueError("route clocks must match and not anticipate an update")
    if route not in ("recent", "older", "global", "cls"):
        raise ValueError("fixed route required")
    ends = q[:, None] - 8 * torch.arange(15, -1, -1, device=q.device)[None]
    starts = ends - 7
    valid = available.reshape(len(available), 16, 8, 23).any(dim=(2, 3)) & (starts >= 0)
    recent = valid & (starts >= t[:, None] - 31) & (ends <= t[:, None])
    older = valid & (starts >= t[:, None] - 95) & (ends <= t[:, None] - 64)
    complete = (recent.sum(1) == 4) & (older.sum(1) == 4)
    if route == "cls":
        mask = torch.cat((torch.zeros_like(valid), complete[:, None]), dim=1)
    else:
        selected = {"recent": recent, "older": older, "global": valid}[route]
        mask = torch.cat(
            (selected & complete[:, None], torch.zeros_like(complete[:, None])), dim=1
        )
    weights = (
        mask.to(torch.float32) * (4 / mask.sum(1, keepdim=True).clamp_min(1)).sqrt()
    )
    return weights


def timed_logit(base, x, a, weights, delta):
    if weights.shape != (len(x), 17) or delta.shape != (128,):
        raise ValueError("matched token weights and norm1 affine state required")
    norm1 = base.encoder.layers[-1].norm1
    dg, db = delta.chunk(2)
    shifts = []

    def hook(module, args, original):
        z = F.layer_norm(args[0], module.normalized_shape, eps=module.eps)
        d = weights[..., None] * (z * dg + db)
        shifts.append(d)
        return original + d

    handle = norm1.register_forward_hook(hook)
    try:
        z = base(x, a)["logit"]
    finally:
        handle.remove()
    if len(shifts) != 1:
        raise RuntimeError(
            "one explicit internal attention input intervention required"
        )
    return z, shifts[0].flatten(1).norm(dim=1)


def set_direction(delta, vector):
    g = torch.as_tensor(vector, device=delta.device, dtype=delta.dtype)
    if g.shape != delta.shape or not torch.isfinite(g).all():
        raise ValueError("finite same-scope gradient required")
    with torch.no_grad():
        delta.zero_()
    delta.grad = None
    norm = g.norm()
    if norm:
        (delta * (g / norm).detach()).sum().backward()
        opt = torch.optim.SGD((delta,), lr=0.05)
        opt.step()
        opt.zero_grad(set_to_none=True)


def report(rows, threshold):
    out = {}
    for rule in RULES:
        groups = {}
        for group in ("safe", "earlier", "positive"):
            cases = [r for r in rows if r["group"] == group]
            ids = sorted({r["shot"] for r in cases})
            values = {}
            for ref in ("none", "older", "global", "cls", "reverse", "permuted"):
                a, b = rule + "__recent", "none" if ref == "none" else rule + "__" + ref
                delta = np.asarray(
                    [
                        np.mean(
                            [
                                r["arms"][a]["bce"] - r["arms"][b]["bce"]
                                for r in cases
                                if r["shot"] == sid
                            ]
                        )
                        for sid in ids
                    ]
                )
                values[ref] = dict(
                    mean=float(delta.mean()),
                    se=float(delta.std(ddof=1) / np.sqrt(len(delta))),
                    improved=int((delta < 0).sum()),
                    worse=int((delta > 0).sum()),
                    tied=int((delta == 0).sum()),
                )
            groups[group] = dict(
                shots=len(ids),
                windows=len(cases),
                supported=sum(r["supported"] for r in cases),
                recent_vs=values,
                route_vs_none={
                    route: float(
                        np.mean(
                            [
                                np.mean(
                                    [
                                        r["arms"][rule + "__" + route]["bce"]
                                        - r["arms"]["none"]["bce"]
                                        for r in cases
                                        if r["shot"] == sid
                                    ]
                                )
                                for sid in ids
                            ]
                        )
                    )
                    for route in MODES
                },
                energy_medians={
                    route: float(
                        np.median(
                            [
                                r["arms"][rule + "__" + route]["activation_l2"]
                                for r in cases
                            ]
                        )
                    )
                    for route in MODES
                },
                crossings={
                    route: {
                        "up": sum(
                            r["arms"]["none"]["p"]
                            < threshold
                            <= r["arms"][rule + "__" + route]["p"]
                            for r in cases
                        ),
                        "down": sum(
                            r["arms"][rule + "__" + route]["p"]
                            < threshold
                            <= r["arms"]["none"]["p"]
                            for r in cases
                        ),
                    }
                    for route in MODES
                },
            )
        out[rule] = groups
    return out


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    for directory in (odd.OUT, odd.FIT, p.original.FEATURES):
        p.verify_saved(directory)
    previous = p.data.read_json(odd.OUT / "rows.json")
    source_protocol = p.data.read_json(p.data.RUN / "protocol.json")
    ids = [x["shot"] for x in source_protocol["order"]]
    bound = {
        str(path): p.data.sha(path)
        for path in (
            Path(__file__),
            PLAN,
            Path(odd.__file__),
            Path(p.__file__),
            Path(p.data.__file__),
            odd.OUT / "completed.json",
            odd.OUT / "rows.json",
            odd.FIT / "fit.npz",
            p.original.FEATURES / "features_and_logits.npz",
            p.data.DEFAULT_PARENT,
            p.data.MAP,
            p.data.RAW_HASHES,
            p.data.RUN / "protocol.json",
        )
    }
    OUT.mkdir()
    p.data.write_json(
        OUT / "protocol.json",
        dict(
            scope="FIXED_DIRECTION_ABSOLUTE_PATCH_CLOCK_INTERVENTION_HELD80",
            held_shots=ids[20:100],
            source_params_unchanged=True,
            norm_scope="last block pre-attention norm1 only",
            dimensions=128,
            modes=MODES,
            rules=RULES,
            input_sha256=bound,
            step_norm=0.05,
            mask_l2=2,
            threshold=source_protocol["threshold"],
        ),
    )
    with np.load(odd.FIT / "fit.npz", allow_pickle=False) as f:
        fits = {
            k: {name: f[k + "_" + name] for name in ("coef", "scale")} for k in RULES
        }
    with np.load(
        p.original.FEATURES / "features_and_logits.npz", allow_pickle=False
    ) as f:
        m, perm = f["held_long"], f["held_permuted"]
    directions = {
        k: {
            kind: odd.odd_direction(fit, v)[:, :128]
            for kind, v in (
                ("real", m),
                ("reverse", -m),
                ("permuted", perm),
            )
        }
        for k, fit in fits.items()
    }
    base, parent = p.data.load_parent(p.data.DEFAULT_PARENT)
    base.eval().requires_grad_(False)
    state = p.data.state_digest(base)
    threshold = parent["threshold"]
    if (
        state != source_protocol["source_state_sha256"]
        or threshold != source_protocol["threshold"]
    ):
        raise ValueError("unchanged source required")
    # Match the first two 64D slots of the archived direction to norm1 γ/β.
    adapter = p.original.make_adapter(base, "norm")
    block = adapter.encoder.layers[-1]
    params = tuple(q for q in adapter.parameters() if q.requires_grad)
    if (
        len(params) != 4
        or params[0] is not block.norm1.weight
        or params[1] is not block.norm1.bias
    ):
        raise ValueError("archived direction's affine parameter order differs")
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
                raise ValueError("fixed held case identity changed")
            q, t = case["future"], case["arrival"]
            if q != t + 24:
                raise ValueError("fixed update and score clocks differ")
            x, a = p.original.window_batch(signal, [q], device="cpu")
            weights = {
                route: support_weights(a, [q], [t], route)
                for route in ("recent", "older", "global", "cls")
            }
            delta = torch.nn.Parameter(torch.zeros(128))
            with torch.no_grad():
                z0 = base(x, a)["logit"]
                z_zero, e_zero = timed_logit(base, x, a, weights["recent"], delta)
            if (
                not torch.equal(z0, z_zero)
                or float(e_zero) != 0
                or float(z0) != old["arms"]["none"]["logit"]
            ):
                raise ValueError("frozen baseline must be bit-identical")

            def entry(z, energy):
                target = torch.full_like(z, float(case["label"]))
                return dict(
                    logit=float(z),
                    p=float(z.sigmoid()),
                    bce=float(F.binary_cross_entropy_with_logits(z, target)),
                    activation_l2=float(energy),
                )

            arms = {"none": entry(z0, e_zero)}
            supported = bool(weights["recent"].any())
            for rule in RULES:
                for mode, (route, kind) in MODES.items():
                    set_direction(
                        delta, directions[rule][kind][k] if supported else np.zeros(128)
                    )
                    with torch.no_grad():
                        z, energy = timed_logit(base, x, a, weights[route], delta)
                    arms[rule + "__" + mode] = dict(
                        **entry(z, energy), step_norm=float(delta.detach().norm())
                    )
            rows.append(dict(shot=sid, **case, supported=supported, arms=arms))
            k += 1
    if (
        k != 240
        or state != p.data.state_digest(base)
        or any(p.data.sha(Path(path)) != h for path, h in bound.items())
    ):
        raise ValueError("source, fixed case set or sealed input changed")
    summary = report(rows, threshold)
    for name, value in (
        ("rows.json", rows),
        ("summary.json", summary),
        ("held_inputs.json", receipts),
    ):
        p.data.write_json(OUT / name, value)
    p.data.write_json(
        OUT / "completed.json",
        dict(
            status="COMPLETE_TIMED_TOKEN_NORM_ROUTE_SCREEN",
            seconds=time.monotonic() - started,
            cases=k,
            supported=sum(r["supported"] for r in rows),
            artifact_sha256={
                f.name: p.data.sha(f) for f in OUT.iterdir() if f.is_file()
            },
        ),
    )
    print(
        {
            rule: {g: v["route_vs_none"] for g, v in x.items()}
            for rule, x in summary.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    run()
