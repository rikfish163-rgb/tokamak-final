"""Shot-held-out weight learning, then actual finite LoRA steps on ended20."""

from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.direct_fast_memory_20261104.timeline import (
    window_batch,
)
from experiments.risk_aligned_online.history_gradient_metric_20261102.run import write
from experiments.risk_aligned_online.joint_lora_20261104.run import frozen_digest
from experiments.risk_aligned_online.joint_lora_20261104.timeline import (
    _project,
    _targets,
)
from experiments.risk_aligned_online.joint_lora_resume_20261104.state import (
    restore_warm,
)
from experiments.risk_aligned_online.mask_repair_20261104.east_evaluation import (
    prepare_actual,
)
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    DEFAULT_PARENT,
    load_parent,
)
from joint_lora_gradient_probe_20261104 import centered_risk_loss, sha, state_digest
from joint_lora_gradient_span_20261104 import MAP, PRIOR, RUN, OUTPUT as SPAN, read_json

ROOT = io23.ROOT
FEATURES = ROOT / "probe_outputs/joint_lora_gate_features_20261104_v1"
OUTPUT = ROOT / "probe_outputs/joint_lora_gate_screen_20261104_v1"
PLAN = ROOT / "JOINT_LORA_GATE_SCREEN_PLAN_20261104.md"
GROUPS = ("safe", "earlier", "positive")
STEPS = 64


class Gate(nn.Module):
    def __init__(self, conditional):
        super().__init__()
        self.conditional = conditional
        self.b = nn.Parameter(torch.zeros(21, dtype=torch.float64))
        if conditional:
            generator = torch.Generator().manual_seed(131)
            self.V = nn.Parameter(
                0.02 * torch.randn(2, 192, generator=generator, dtype=torch.float64)
            )
            self.U = nn.Parameter(torch.zeros(21, 2, dtype=torch.float64))

    def forward(self, h):
        if h.ndim != 2 or h.shape[-1] != 192:
            raise ValueError("gate requires causal latent rows of width 192")
        z = self.b.expand(len(h), -1)
        if self.conditional:
            z = z + F.linear(F.linear(F.layer_norm(h.detach(), (192,)), self.V), self.U)
        bounded = z.tanh()
        return 1 + 0.25 * (bounded - bounded.mean(dim=-1, keepdim=True))


def adam_delta(g):
    """Ideal differentiable first step, before float32 parameter rounding."""
    clipped = g * (1.0 / (g.norm(dim=-1, keepdim=True) + 1e-6)).clamp(max=1.0)
    return -0.0001 * clipped / (clipped.abs() + 1e-8)


def folds(ids, disruptive):
    d, s = ([sid for sid in ids if disruptive[sid] == value] for value in (True, False))
    if len(d) != 8 or len(s) != 12 or len(set(ids)) != 20:
        raise ValueError("predeclared warm20 8/12 outcome composition required")
    return [d[i::5] + s[(i + 2) % 5 :: 5] for i in range(5)]


def group_weights(rows):
    count = Counter((r["group"], r["shot"]) for r in rows)
    nshots = Counter(g for g, _ in count)
    if set(nshots) != set(GROUPS):
        raise ValueError("every training fold must include all three risk strata")
    return torch.tensor(
        [1 / (3 * nshots[r["group"]] * count[r["group"], r["shot"]]) for r in rows],
        dtype=torch.float64,
    )


def fit(h, g, r, rows, conditional):
    model = Gate(conditional)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    balance = group_weights(rows)
    history = []
    for _ in range(STEPS):
        w = model(h)
        diag = torch.bmm(g, w[..., None]).squeeze(-1)
        objective = (
            2
            - F.cosine_similarity(diag, r, dim=-1, eps=1e-12)
            + F.cosine_similarity(adam_delta(diag), r, dim=-1, eps=1e-12)
            + 0.01 * (w - 1).square().mean(-1)
        )
        loss = (balance * objective).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite gate teacher objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return model.eval().requires_grad_(False), history


def verify_artifacts(directory, names):
    receipt = read_json(directory / "completed.json")
    for name in names:
        if sha(directory / name) != receipt["artifact_sha256"][name]:
            raise ValueError(f"bound artifact changed: {directory / name}")
    return [directory / name for name in (*names, "completed.json")]


def finite_steps(model, signal, case, old_case, weights, h, uniform_grad, threshold):
    parameters = model.neural_parameters()
    start = tuple(p.detach().clone() for p in parameters)
    x, a, target, mask = _targets(model, signal, case["origin"], device="cpu")
    if int(mask.sum()) != case["supported"]:
        raise ValueError("diagnostic support changed")
    with torch.no_grad():
        latent = model(x, a)["latent"].reshape(-1)
        torch.testing.assert_close(latent, h, rtol=1e-5, atol=2e-6)
    # All updates receive origin data and matured diagnostics only. The
    # risk query/label is evaluated after each candidate parameter is set.
    results = {}
    for name, w in [("none", None), *weights.items()]:
        with torch.no_grad():
            for parameter, initial in zip(parameters, start, strict=True):
                parameter.copy_(initial)
        displacement = 0.0
        norm = None
        if w is not None:
            optimizer = torch.optim.Adam(parameters, lr=0.0001)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, a)["diagnostic"]
            desired = (
                torch.zeros_like(target)
                if name in ("hold", "context_hold")
                else -target
                if name in ("reverse", "context_reverse")
                else target
            )
            terms = (pred - desired).square().masked_fill(~mask, 0).reshape(8, 21)
            loss = (terms * w.to(terms)).sum() / mask.sum()
            loss.backward()
            if name == "uniform":
                g = torch.cat([p.grad.reshape(-1) for p in parameters])
                torch.testing.assert_close(g, uniform_grad, rtol=2e-4, atol=5e-8)
            norm = float(
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            )
            optimizer.step()
            displacement = _project(parameters, start, 0.5)
        with torch.no_grad():
            xr, ar = window_batch(signal, [case["future_risk_point"]], device="cpu")
            logit = model(xr, ar)["logit"]
            risk = float(centered_risk_loss(logit, case["label"], threshold))
        if not np.isfinite(risk):
            raise FloatingPointError("nonfinite finite-step risk")
        results[name] = dict(
            bce=risk, logit=float(logit), displacement=displacement, grad_norm=norm
        )
    with torch.no_grad():
        for parameter, initial in zip(parameters, start, strict=True):
            parameter.copy_(initial)
            parameter.grad = None
    if not np.isclose(
        results["none"]["bce"], old_case["risk_loss"], rtol=2e-4, atol=2e-7
    ):
        raise ValueError("held-out no-update risk differs from frozen parent probe")
    return results


def summarize(rows):
    groups = {}
    for group in GROUPS:
        subset = [r for r in rows if r["group"] == group]
        ids = sorted({r["shot"] for r in subset})
        if not ids:
            raise ValueError("all preregistered strata required")

        def pair(a, b):
            values = [
                float(
                    np.mean(
                        [
                            r["arms"][a]["bce"] - r["arms"][b]["bce"]
                            for r in subset
                            if r["shot"] == sid
                        ]
                    )
                )
                for sid in ids
            ]
            return dict(
                mean_delta=float(np.mean(values)),
                improved=sum(v < 0 for v in values),
                worse=sum(v > 0 for v in values),
                ties=sum(v == 0 for v in values),
            )

        groups[group] = dict(
            shots=len(ids),
            cases=len(subset),
            vs_none={arm: pair(arm, "none") for arm in subset[0]["arms"]},
            context_vs={
                arm: pair("context", arm)
                for arm in (
                    "uniform",
                    "global",
                    "context_wrong",
                    "context_hold",
                    "context_reverse",
                )
            },
        )
    return groups


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    paths = (
        verify_artifacts(SPAN, ("cases.json", "summary.json", "gradients.npz"))
        + verify_artifacts(PRIOR, ("cases.json", "summary.json"))
        + verify_artifacts(FEATURES, ("features.npz", "protocol.json"))
        + [
            RUN / "warm.json",
            RUN / "warm.pt",
            MAP,
            DEFAULT_PARENT,
            PLAN,
            Path(__file__),
        ]
    )
    bound = {str(p.resolve()): sha(p) for p in paths}
    cases = read_json(SPAN / "cases.json")
    old = read_json(PRIOR / "cases.json")
    prior = read_json(PRIOR / "summary.json")
    warm = read_json(RUN / "warm.json")
    ids = warm["label_shots"]
    for field in ("warm_pt", "warm_json", "density_map", "source_checkpoint"):
        path = dict(
            warm_pt=RUN / "warm.pt",
            warm_json=RUN / "warm.json",
            density_map=MAP,
            source_checkpoint=DEFAULT_PARENT,
        )[field]
        if sha(path) != prior["input_sha256"][field]:
            raise ValueError("probe parent/map changed")
    keys = (
        "shot",
        "origin",
        "arrival",
        "future_risk_point",
        "label",
        "group",
        "supported",
    )
    if (
        len(cases) != 60
        or len(old) != 60
        or Counter(c["shot"] for c in cases) != {s: 3 for s in ids}
        or any(any(a[k] != b[k] for k in keys) for a, b in zip(cases, old, strict=True))
    ):
        raise ValueError("exact previous cases required")
    with np.load(FEATURES / "features.npz", allow_pickle=False) as f:
        if (
            f["latent"].shape != (60, 192)
            or f["latent"].dtype != np.float32
            or any(
                not np.array_equal(f[key], [c[key] for c in cases])
                for key in ("shot", "origin")
            )
        ):
            raise ValueError("causal feature identity changed")
        latent = torch.from_numpy(f["latent"])
    with np.load(SPAN / "gradients.npz", allow_pickle=False) as f:
        g, r, uniform = (torch.from_numpy(f[key]) for key in ("G", "risk", "uniform"))
    if (
        g.shape != (60, 512, 21)
        or r.shape != (60, 512)
        or uniform.shape != (60, 512)
        or not all(torch.isfinite(x).all() for x in (g, r, uniform, latent))
    ):
        raise ValueError("finite fixed-shape gradient/feature assets required")
    g, r, h = g.double(), r.double(), latent.double()
    split = folds(ids, {c["shot"]: c["disruptive"] for c in cases})
    policies = {}
    receipts = []
    for number, held in enumerate(split):
        training = [i for i, c in enumerate(cases) if c["shot"] not in held]
        testing = [i for i, c in enumerate(cases) if c["shot"] in held]
        matched = [cases[i] for i in training]
        global_gate, global_loss = fit(
            h[training], g[training], r[training], matched, False
        )
        context_gate, context_loss = fit(
            h[training], g[training], r[training], matched, True
        )
        for position, i in enumerate(testing):
            j = testing[(position + 3) % len(testing)]
            if cases[i]["shot"] == cases[j]["shot"] or i in policies:
                raise ValueError("invalid disjoint fold/context decoy")
            with torch.no_grad():
                w = context_gate(h[i : i + 1])[0]
                ones = torch.ones(21, dtype=torch.float64)
                policies[i] = (
                    number,
                    j,
                    dict(
                        uniform=ones,
                        global_=global_gate(h[i : i + 1])[0],
                        context=w,
                        context_wrong=context_gate(h[j : j + 1])[0],
                        hold=ones,
                        reverse=ones,
                        context_hold=w,
                        context_reverse=w,
                        oracle=torch.tensor(cases[i]["oracle_weights"]),
                    ),
                )
        receipts.append(
            dict(
                fold=number,
                held_shots=held,
                train_shots=[sid for sid in ids if sid not in held],
                global_loss=global_loss,
                context_loss=context_loss,
            )
        )
    if set(policies) != set(range(60)):
        raise ValueError("every case needs exactly one held-shot policy")

    base, parent = load_parent(DEFAULT_PARENT)
    base.eval().requires_grad_(False)
    model = restore_warm(base, RUN / "warm.pt", warm, device="cpu")
    fixed = state_digest(model)
    base_hash, frozen = state_digest(base), frozen_digest(model)
    for p in model.neural_parameters():
        p.requires_grad_(True)
    mapping = read_json(MAP)
    saved_stats = {r["shot"]: r for r in prior["shots"]}
    if ids != prior["label_shots"] or ids != mapping["calibration_ids"]:
        raise ValueError("ended shot membership changed")
    rows = []
    reader = io23.read_signals
    calls = []

    def checked(device, sid, *args, **kwargs):
        if device != "EAST" or sid not in ids:
            raise ValueError("only the original ended20 EAST signals may be opened")
        calls.append(sid)
        return reader(device, sid, *args, **kwargs)

    with (
        patch.object(io23, "read_signals", side_effect=checked),
        patch.object(
            io23, "read_outcome", side_effect=RuntimeError("no new HDF outcomes")
        ),
    ):
        for sid in ids:
            path = io23.path_for("EAST", sid)
            s = saved_stats[sid]
            before = path.stat()
            if (before.st_size, before.st_mtime_ns) != (s["size"], s["mtime_ns"]):
                raise ValueError("ended signals changed since gradient probe")
            signal = prepare_actual(
                simple_data.read_shot("EAST", sid, rate_contract="nominal"), 21, mapping
            )
            for i, case in enumerate(cases):
                if case["shot"] != sid:
                    continue
                fold, decoy, weights = policies[i]
                weights = {
                    "global" if name == "global_" else name: w
                    for name, w in weights.items()
                }
                arms = finite_steps(
                    model,
                    signal,
                    case,
                    old[i],
                    weights,
                    latent[i],
                    uniform[i],
                    float(parent["threshold"]),
                )
                rows.append(
                    dict(
                        case,
                        fold=fold,
                        decoy_shot=cases[decoy]["shot"],
                        weights={k: w.tolist() for k, w in weights.items()},
                        arms=arms,
                    )
                )
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != (
                before.st_size,
                before.st_mtime_ns,
            ):
                raise ValueError("ended signals changed during measurement")
            print(f"finite-risk cases {len(rows)}/60", flush=True)
    if (
        len(rows) != 60
        or calls != ids
        or state_digest(model) != fixed
        or state_digest(base) != base_hash
        or frozen_digest(model) != frozen
        or any(sha(Path(p)) != v for p, v in bound.items())
    ):
        raise AssertionError("parent, bound inputs or access scope changed")
    groups = summarize(rows)
    required = ("uniform", "global", "context_hold", "context_reverse")
    passed = all(
        groups[group]["context_vs"][arm]["mean_delta"] < 0
        for group in ("safe", "positive")
        for arm in required
    )
    OUTPUT.mkdir()
    write(OUTPUT / "cases.json", rows)
    write(
        OUTPUT / "summary.json",
        dict(
            scope="HELD_SHOT_GATE_ONLY_ON_COMMON_WARM20_NOT_INDEPENDENT",
            groups=groups,
            thresholds_unchanged=True,
            actual_adam_steps_per_nonstatic_case=1,
            folds=receipts,
            passes_required_loss_gate=passed,
            new_outcomes_read=0,
            online_performance_claim=False,
            input_sha256=bound,
        ),
    )
    write(
        OUTPUT / "completed.json",
        dict(
            status="COMPLETED_ENDED20_FINITE_STEP_GATE_SCREEN",
            artifact_sha256={
                k: sha(OUTPUT / k) for k in ("cases.json", "summary.json")
            },
        ),
    )
    print(dict(passes=passed, groups=groups), flush=True)


if __name__ == "__main__":
    run()
