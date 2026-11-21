"""Seal equal-norm, layer-0 direction interventions without reading outcomes."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import layer_interaction_replay_20261105 as layer
import moving_peak_replay_20261105 as r
from run_layer0_query_memory_20261105 import parents, raw_for


PLAN = r.ROOT / "LAYER0_DIRECTION_SPAN_PLAN_20261106.md"
NEXT = r.ROOT / "LAYER0_DIRECTION_SPAN_NEXT_20261106.md"
FAMILIES = ("full", "in_span", "out_span", "random")
ARMS = (
    "static",
    "intra",
    *(f"{key}_{sign}" for key in FAMILIES for sign in ("pos", "neg")),
)
FIRST, STRIDE, WIDTH = 135, 32, 128
RANK, SEED, REL_EPS, GRAD_BATCH = 33, 131, 1e-10, 128
MODES = {"fixed": ("fixed360", 360, 21), "conditional": ("conditional835", 835, 1)}


def basis_from_columns(columns):
    a = np.asarray(columns, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != WIDTH or not a.shape[1] or not np.isfinite(a).all():
        raise ValueError("finite 128-D generator output columns required")
    q, s, _ = np.linalg.svd(a, full_matrices=False)
    keep = s > s[0] * max(a.shape) * np.finfo(a.dtype).eps
    if not keep.any():
        raise ValueError("generator output has no direction space")
    return q[:, keep], s[keep]


def load_basis(fit, device):
    with np.load(fit, allow_pickle=False) as a:
        w, b = (a[k].astype(np.float64) for k in ("layers.2.weight", "layers.2.bias"))
    if w.shape != (2 * WIDTH, 32) or b.shape != (2 * WIDTH,):
        raise ValueError("frozen generator output layer geometry required")
    q, s = basis_from_columns(np.c_[w[:WIDTH], b[:WIDTH]])
    if q.shape != (WIDTH, RANK):
        raise ValueError("expected the audited rank-33 output subspace")
    return torch.as_tensor(q, device=device), s


def random_anchors(n, shot):
    # Prefix draws are identical for any cut of the same shot; IDs only seed this control.
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence((SEED, int(shot))))
    ).standard_normal((n, WIDTH))


@torch.no_grad()
def direction_targets(original, gradient, q, *, shot):
    if (
        original.ndim != 2
        or original.shape[1] != WIDTH
        or not original.is_floating_point()
        or gradient.shape != original.shape
        or not gradient.is_floating_point()
        or gradient.device != original.device
        or q.device != original.device
        or q.ndim != 2
        or q.shape[0] != WIDTH
        or not 0 < q.shape[1] <= WIDTH
        or not all(torch.isfinite(v).all() for v in (original, gradient, q))
    ):
        raise ValueError("finite actions, gradients and matching basis required")
    gd, q = gradient.double(), q.double()
    if not torch.allclose(
        q.T @ q,
        torch.eye(q.shape[1], dtype=q.dtype, device=q.device),
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("projection basis must be orthonormal")
    part = (gd @ q) @ q.T
    rem = gd - part
    random = torch.as_tensor(
        random_anchors(len(original), shot), device=original.device
    )
    originals = original.double().norm(dim=1)
    full = gd.norm(dim=1)
    if (
        not torch.isfinite(originals).all()
        or (originals > r.latent.RADIUS + 1e-6).any()
    ):
        raise ValueError("original action exceeds fixed radius")
    grad_norms = torch.stack((full, part.norm(dim=1), rem.norm(dim=1)), dim=1)
    if not torch.isfinite(grad_norms).all() or not torch.allclose(
        grad_norms[:, 1:].square().sum(1), full.square(), rtol=1e-10, atol=0
    ):
        raise ValueError("in/out gradient energies must decompose the full gradient")

    def pad(v):
        return torch.cat((v, torch.zeros_like(v)), dim=1)

    targets = {"intra": pad(original)}
    actual_norms, flags = [original.norm(dim=1)], []
    for name, v in zip(FAMILIES, (gd, part, rem, random), strict=True):
        norms = v.norm(dim=1)
        if name == "random":
            if (norms == 0).any() or not torch.isfinite(norms).all():
                raise ValueError("invalid seeded random control")
            valid = originals > 0
        else:
            valid = (originals > 0) & (full > 0) & (norms > REL_EPS * full)
        a = torch.zeros_like(original)
        a[valid] = (v[valid] / norms[valid, None] * originals[valid, None]).to(
            original.dtype
        )
        actual = a.norm(dim=1)
        if (
            not torch.isfinite(a).all()
            or (abs(actual[valid] - originals[valid]) > 1e-6).any()
            or (actual[~valid] != 0).any()
        ):
            raise ValueError(f"same-norm direction check failed: {name}")
        targets[f"{name}_pos"], targets[f"{name}_neg"] = pad(a), pad(-a)
        actual_norms.append(actual)
        flags.append(valid)
    evidence = dict(
        anchor_norms=torch.stack(actual_norms, dim=1).cpu().numpy(),
        anchor_gradient_norms=grad_norms.cpu().numpy(),
        anchor_valid=torch.stack(flags, dim=1).cpu().numpy(),
    )
    return targets, evidence


def anchor_gradient(base, ctx):
    """Only differentiate with respect to the temporary first-layer vector."""
    if any(p.requires_grad for p in base.parameters()) or any(
        m.training for m in base.modules()
    ):
        raise ValueError("gradient probe requires frozen eval source")
    windows, mask, native = ctx[:3]
    if len(windows) <= FIRST:
        return windows.new_empty((0, WIDTH))
    idx = torch.arange(FIRST, len(windows), STRIDE, device=windows.device)
    gradients, anchor_p = [], []
    with torch.enable_grad():
        for q in idx.split(GRAD_BATCH):
            x, available = windows[q].contiguous(), mask[q].contiguous()
            u = x.new_zeros((len(q), WIDTH), requires_grad=True)
            z = r.policy.routed_pair(
                base,
                x,
                available,
                r.latent.routes_all(available),
                torch.cat((u, torch.zeros_like(u)), dim=1),
            )
            (g,) = torch.autograd.grad(z.sum(), u)
            gradients.append(g.detach())
            anchor_p.append(z.detach().sigmoid().cpu().numpy())
    layer.require_curve_parity(
        np.concatenate(anchor_p), native[idx.cpu().numpy()], r.THRESHOLD
    )
    result = torch.cat(gradients)
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite first-layer action gradient")
    return result


def predict(base, raw, map_rank, active, mapping, rule, basis):
    ctx = r.latent.native_context(base, r.prepare_actual(raw, map_rank, mapping))
    if not active:
        return dict.fromkeys(ARMS, ctx[2]), {}
    original = r.actions(rule, ctx)[:, :WIDTH].contiguous()
    targets, receipt = direction_targets(
        original, anchor_gradient(base, ctx), basis, shot=raw["shot"]
    )
    p, _ = r.forecast(base, ctx, targets)
    return {"static": ctx[2], **p}, receipt


def load_reference(mode, fixed_order, source, fixed_raw):
    tag, n, active_rank = MODES[mode]
    path = r.ROOT / "probe_outputs" / f"layer0_signed_{tag}_20261106_v1"
    cfg, done = [r.read(path / p) for p in ("protocol.json", "completed.json")]
    order = cfg["order"]
    if (
        cfg["mode"] != mode
        or done["mode"] != mode
        or done["status"] != "COMPLETE_LAYER0_SIGNED_PROJECTION"
        or done["protocol_sha256"] != r.sha(path / "protocol.json")
        or len(order) != n
        or len(set(order)) != n
        or len(done["sealed"]) != n
        or done["n_shots"] != n
        or cfg["threshold"] != r.THRESHOLD
        or cfg["source_state_sha256"] != source
        or cfg["bank_active_rank"] != active_rank
        or (cfg["first"], cfg["stride"]) != (FIRST, STRIDE)
        or (
            mode == "fixed" and (order != fixed_order or cfg["raw_sha256"] != fixed_raw)
        )
        or (mode == "conditional" and set(order) & set(fixed_order))
    ):
        raise ValueError("exact locked signed-replay signal cohort required")
    return path, cfg, done, active_rank


def run(mode, output):
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    r.configure()
    order, source_cfg, _, base, source, mapping, map_path, fit = parents()
    path, cfg, ref, active_rank = load_reference(
        mode, order, source, source_cfg["raw_sha256"]
    )
    rule = r.load_rule(fit, base.center.device)
    basis, singular = load_basis(fit, base.center.device)
    deps = [
        Path(__file__),
        PLAN,
        NEXT,
        Path(r.__file__),
        Path(r.latent.__file__),
        Path(r.policy.__file__),
        Path(r.io23.__file__),
        Path(r.simple_data.__file__),
        Path(r.prepare_actual.__code__.co_filename),
        Path(parents.__code__.co_filename),
        Path(layer.__file__),
        r.DEFAULT_PARENT,
        map_path,
        fit,
        path / "protocol.json",
        path / "completed.json",
    ]
    hashes = {str(p.resolve()): r.sha(p) for p in deps}
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    r.save_arrays(
        output / "basis.npz", basis=basis.cpu().numpy(), singular_values=singular
    )
    basis_hash = r.sha(output / "basis.npz")
    order = cfg["order"]
    raw_sha256 = cfg["raw_sha256"]
    r.write_new(
        output / "protocol.json",
        dict(
            mode=mode,
            order=order,
            arms=ARMS,
            raw_sha256=raw_sha256,
            threshold=r.THRESHOLD,
            initial_ns=time.monotonic_ns(),
            first=FIRST,
            stride=STRIDE,
            active_rank=active_rank,
            basis_rank=RANK,
            basis_sha256=basis_hash,
            source_state_sha256=source,
            source_fit_sha256=r.sha(fit),
            source_map_sha256=r.sha(map_path),
            input_sha256=hashes,
            development_only=True,
            new_fit_labels=0,
            inter_updates=0,
            outcome_reads_during_prediction=0,
            random_seed=SEED,
            relative_direction_epsilon=REL_EPS,
        ),
    )
    sealed, previous, parity = [], None, dict(static=0.0, intra=0.0)
    with patch.object(
        r.io23,
        "read_outcome",
        side_effect=RuntimeError("no outcome access before span seal"),
    ):
        for rank, sid in enumerate(order, 1):
            raw = raw_for(sid, raw_sha256)
            active = rank >= active_rank
            p, receipt = predict(
                base, raw, rank if mode == "fixed" else 21, active, mapping, rule, basis
            )
            if tuple(p) != ARMS or any(
                not np.isfinite(v).all() or (v < 0).any() or (v > 1).any()
                for v in p.values()
            ):
                raise ValueError(
                    "valid span probabilities in frozen arm order required"
                )
            stem = f"{rank:04d}_{sid}"
            old_path = path / "shots" / (stem + ".npz")
            if r.sha(old_path) != ref["sealed"][rank - 1]["curve_sha256"]:
                raise ValueError("archived control changed")
            with np.load(old_path, allow_pickle=False) as old:
                if not np.array_equal(raw["times"], old["times"]):
                    raise ValueError("matched raw clock required")
                for k in parity:
                    parity[k] = max(
                        parity[k], layer.require_curve_parity(p[k], old[k], r.THRESHOLD)
                    )
            out = output / "shots" / (stem + ".npz")
            note = output / "shots" / (stem + "_seal.json")
            r.save_arrays(out, times=raw["times"], **p, **receipt)
            row = dict(
                rank=rank,
                shot=sid,
                raw_sha256=raw_sha256[str(sid)],
                curve_sha256=r.sha(out),
                previous_seal_sha256=previous,
                active=active,
                seal_ns=time.monotonic_ns(),
            )
            r.write_new(note, row)
            previous = r.sha(note)
            sealed.append(dict(row, seal_sha256=previous))
            if rank % 50 == 0 or rank == len(order):
                print(f"direction span {mode} {rank}/{len(order)}", flush=True)
    if (
        r.state_digest(base) != source
        or any(p.grad is not None for p in base.parameters())
        or r.sha(output / "basis.npz") != basis_hash
        or any(r.sha(p) != h for p, h in hashes.items())
    ):
        raise ValueError("direction replay modified a frozen input")
    r.write_new(
        output / "completed.json",
        dict(
            status="COMPLETE_LAYER0_DIRECTION_SPAN",
            mode=mode,
            n_shots=len(order),
            n_curves=len(order) * len(ARMS),
            sealed=sealed,
            max_reference_error=parity,
            protocol_sha256=r.sha(output / "protocol.json"),
            completed_ns=time.monotonic_ns(),
            seconds=time.monotonic() - started,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=tuple(MODES), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.mode, a.output)
