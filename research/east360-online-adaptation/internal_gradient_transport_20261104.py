"""Past-ended shots teach internal gradient directions; fixed held80 screen."""

from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import joint_lora_subspace_probe_20261104 as data
import joint_lora_persistent_motion_probe_20261104 as motion
from experiments.risk_aligned_online.cold20_compare_20261104.adapters import (
    make_adapter,
)
from experiments.risk_aligned_online.direct_fast_memory_20261104.timeline import (
    window_batch,
)

ROOT = data.ROOT
FEATURES = ROOT / "probe_outputs/joint_lora_persistent_motion_probe_20261104_v2"
OUT = ROOT / "probe_outputs/internal_gradient_transport_20261104_v1"
PLAN = ROOT / "INTERNAL_GRADIENT_TRANSPORT_PLAN_20261104.md"
ARMS = ("none", "state", "real", "hold", "reverse")


def fit_transport(x, gradients, weights):
    x, y, w = (np.asarray(v, dtype=np.float64) for v in (x, gradients, weights))
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or w.shape != (len(x),):
        raise ValueError(
            "matching features, gradient targets and shot weights required"
        )
    if (
        not all(np.isfinite(a).all() for a in (x, y, w))
        or (w < 0).any()
        or not np.isclose(w.sum(), 1)
    ):
        raise ValueError("finite inputs and unit nonnegative weights required")
    mean = w @ x
    scale = np.sqrt(w @ (x - mean) ** 2)
    scale[scale < 1e-12] = 1.0
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    penalty = np.eye(design.shape[1]) * 0.1
    penalty[0, 0] = 0
    coef = np.linalg.solve(
        design.T @ (design * w[:, None]) + penalty, design.T @ (y * w[:, None])
    )
    return dict(mean=mean, scale=scale, coef=coef)


def predict_transport(fit, x):
    return (
        np.column_stack([np.ones(len(x)), (x - fit["mean"]) / fit["scale"]])
        @ fit["coef"]
    )


def apply_direction(parameters, gradient):
    vector = torch.as_tensor(gradient, dtype=torch.float32)
    size = sum(p.numel() for p in parameters)
    if vector.shape != (size,) or not torch.isfinite(vector).all():
        raise ValueError("one finite learned gradient per internal parameter required")
    norm = vector.norm()
    if not norm:
        return 0
    vector = vector / norm
    loss, offset = None, 0
    for param in parameters:
        param.grad = None
        piece = vector[offset : offset + param.numel()].reshape_as(param)
        term = (param * piece.detach()).sum()
        loss = term if loss is None else loss + term
        offset += param.numel()
    loss.backward()
    optimizer = torch.optim.SGD(parameters, lr=0.05)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return size


def summaries(rows):
    out = {}
    for group in ("safe", "earlier", "positive"):
        subset = [r for r in rows if r["group"] == group]
        ids = sorted({r["shot"] for r in subset})
        pairs = {}
        for a, b in (
            ("real", "none"),
            ("state", "none"),
            ("real", "state"),
            ("real", "hold"),
            ("real", "reverse"),
        ):
            delta = np.array(
                [
                    np.mean(
                        [
                            r["arms"][a]["bce"] - r["arms"][b]["bce"]
                            for r in subset
                            if r["shot"] == sid
                        ]
                    )
                    for sid in ids
                ]
            )
            pairs[a + "_vs_" + b] = dict(
                delta_bce=float(delta.mean()),
                improved=int((delta < 0).sum()),
                worse=int((delta > 0).sum()),
                ties=int((delta == 0).sum()),
                shot_se=float(delta.std(ddof=1) / np.sqrt(len(delta)))
                if len(delta) > 1
                else None,
            )
        out[group] = dict(shots=len(ids), cases=len(subset), pairs=pairs)
    return out


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    completion = data.read_json(FEATURES / "completed.json")
    if completion["status"] != "COMPLETED_VERSION2_PERSISTENT_MOTION_DEVELOPMENT_PROBE":
        raise ValueError("verified mapped motion features required")
    for name, value in completion["artifact_sha256"].items():
        if data.sha(FEATURES / name) != value:
            raise ValueError("feature artifact changed")
    base, source = data.load_parent(data.DEFAULT_PARENT)
    base.eval().requires_grad_(False)
    before = data.state_digest(base)
    protocol = data.read_json(data.RUN / "protocol.json")
    ids = [r["shot"] for r in protocol["order"]]
    if (
        before != protocol["source_state_sha256"]
        or source["threshold"] != protocol["threshold"]
    ):
        raise ValueError("source must remain fixed")
    mapping, hashes = data.read_json(data.MAP), data.read_json(data.RAW_HASHES)
    cases = data.read_json(FEATURES / "cases.json")
    sets = {
        stage: [r for r in cases if r["stage"] == stage] for stage in ("fit", "held")
    }
    with np.load(FEATURES / "features_and_logits.npz", allow_pickle=False) as values:
        features = {
            stage: np.concatenate(
                [
                    values[stage + "_endpoint"],
                    values[stage + "_endpoint_mask"],
                    values[stage + "_common"],
                    values[stage + "_native_logit"][:, None],
                    values[stage + "_long"],
                ],
                axis=1,
            ).astype(float)
            for stage in ("fit", "held")
        }
    bound = {
        str(p): data.sha(p)
        for p in (
            Path(__file__),
            PLAN,
            FEATURES / "completed.json",
            FEATURES / "features_and_logits.npz",
            data.RUN / "protocol.json",
            data.DEFAULT_PARENT,
            data.MAP,
            data.RAW_HASHES,
            Path(data.__file__),
            Path(motion.__file__),
        )
    }
    OUT.mkdir()
    data.write_json(
        OUT / "protocol.json",
        dict(
            scope="ENDED20_INTERNAL_DIRECTION_REGRESSION_SEALED80_DEVELOPMENT",
            input_sha256=bound,
            fit_shots=ids[:20],
            test_shots=ids[20:100],
            fixed_step_norm=0.05,
            ridge=0.1,
            actual_backprop=True,
            future_risk_target_offset=24,
            labels_not_deployment_input=True,
        ),
    )
    model = make_adapter(base, "norm")
    params = tuple(p for p in model.parameters() if p.requires_grad)
    if sum(p.numel() for p in params) != 256:
        raise ValueError("exact last-layer norm1/norm2 scope required")
    initial = tuple(p.detach().clone() for p in params)

    def reset():
        with torch.no_grad():
            for p, saved in zip(params, initial, strict=True):
                p.copy_(saved)
                p.grad = None

    teachers, receipts = [], []
    for index, sid in enumerate(ids[:20], 1):
        signal, selected, receipt = data._read_sealed(index, sid, mapping, hashes)
        receipts.append(receipt)
        expected = [r for r in sets["fit"] if r["shot"] == sid]
        if [(r["origin"], r["label"]) for r in expected] != [
            (r["origin"], r["label"]) for r in selected
        ]:
            raise ValueError("training coordinates or HDF targets changed")
        for case in selected:
            x, a = window_batch(signal, [case["future"]], device="cpu")
            z = model(x, a)["logit"]
            loss = F.binary_cross_entropy_with_logits(
                z, torch.full_like(z, float(case["label"]))
            )
            grad = torch.autograd.grad(loss, params)
            vector = torch.cat([g.flatten() for g in grad]).detach().double()
            norm = vector.norm()
            teachers.append((vector / norm if norm else vector).numpy())
    teachers = np.stack(teachers)
    weights = motion.balanced_weights(sets["fit"])
    transport = fit_transport(features["fit"], teachers, weights)
    state = fit_transport(features["fit"][:, :64], teachers, weights)
    x = features["held"]
    held = x.copy()
    held[:, 64:] = 0
    reversed_ = x.copy()
    reversed_[:, 64:] *= -1
    directions = dict(
        state=predict_transport(state, x[:, :64]),
        real=predict_transport(transport, x),
        hold=predict_transport(transport, held),
        reverse=predict_transport(transport, reversed_),
    )
    print("warm20 direction rule frozen; only now evaluate later80", flush=True)
    result, row_index = [], 0
    for index, sid in enumerate(ids[20:100], 21):
        signal, selected, receipt = data._read_sealed(index, sid, mapping, hashes)
        receipts.append(receipt)
        for case in selected:
            meta = sets["held"][row_index]
            if (sid, case["origin"], case["label"]) != (
                meta["shot"],
                meta["origin"],
                meta["label"],
            ):
                raise ValueError("held coordinates changed")
            xq, aq = window_batch(signal, [case["future"]], device="cpu")
            output = {}
            for arm in ARMS:
                reset()
                if arm != "none":
                    apply_direction(params, directions[arm][row_index])
                # HDF risk target is used for evaluation only after update.
                with torch.no_grad():
                    z = model(xq, aq)["logit"]
                    target = torch.full_like(z, float(case["label"]))
                    output[arm] = dict(
                        bce=float(F.binary_cross_entropy_with_logits(z, target)),
                        logit=float(z),
                        probability=float(z.sigmoid()),
                    )
            result.append(dict(shot=sid, **case, arms=output))
            row_index += 1
        if index % 10 == 0:
            print(dict(shots=index, cases=row_index), flush=True)
    reset()
    if (
        row_index != 240
        or data.state_digest(base) != before
        or any(data.sha(Path(p)) != v for p, v in bound.items())
    ):
        raise ValueError("case, parent or bound files changed")
    summary = summaries(result)
    data.write_json(OUT / "result.json", result)
    data.write_json(OUT / "summary.json", summary)
    data.write_json(OUT / "receipts.json", receipts)
    with (OUT / "fit.npz").open("xb") as f:
        np.savez_compressed(
            f,
            teachers=teachers,
            weights=weights,
            **{
                k + "_" + key: v
                for k, m in (("state", state), ("real", transport))
                for key, v in m.items()
            },
        )
    data.write_json(
        OUT / "completed.json",
        dict(
            status="COMPLETE_FIXED_INTERNAL_DIRECTION_SCREEN_NOT_ONLINE",
            seconds=time.monotonic() - started,
            artifact_sha256={p.name: data.sha(p) for p in OUT.iterdir() if p.is_file()},
        ),
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
