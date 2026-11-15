"""A bounded, shot-local attention update conditioned on causal latent history."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import trajectory_policy_20261105 as policy
from trajectory_replay_20261105 import BATCH, make_views

RADIUS, EPOCHS, MINIBATCH, LR = 2.0, 12, 64, 0.002
CONTRAST_LAG, WIDTH = 4, 194


def causal_history(h):
    if h.ndim != 2 or h.shape[1] != 64:
        raise ValueError("chronological native CLS states required")
    if not len(h):
        empty = h.new_empty((0, 128))
        return empty, empty
    i = torch.arange(len(h), device=h.device)
    previous = (i - CONTRAST_LAG).clamp_min(0)
    n = (i + 1).clamp_max(CONTRAST_LAG)
    early = h[:CONTRAST_LAG].cumsum(0)[n - 1] / n[:, None]
    real = torch.cat((h - h[previous], h - early), dim=1)
    return real, real[previous]


def labelled_queries(times, p, disruptive, down):
    """Call only on ended shots: legitimate scoring/training selection, not routing."""
    if times.ndim != 1 or times.shape != p.shape:
        raise ValueError("matching one-dimensional clock and probability required")
    valid = np.arange(policy.FIRST, len(times), dtype=np.int64)
    if disruptive:
        lead = np.round((down - times) * 1000, 9)
        valid = valid[(lead[valid] > 30) & (lead[valid] <= 125)]
        q = valid[np.linspace(0, len(valid) - 1, min(8, len(valid)), dtype=int)]
        return q, lead[q]
    q = np.asarray(
        [b[np.argmax(p[b])] for b in np.array_split(valid, 8) if len(b)], dtype=np.int64
    )
    return q, np.full(len(q), np.nan)


def weights_by_class_shot(shots, positive):
    if shots.ndim != 1 or shots.shape != positive.shape or positive.dtype != bool:
        raise ValueError("matching shot IDs and binary classes required")
    result = np.zeros(len(shots), dtype=np.float64)
    groups = [set(shots[positive]), set(shots[~positive])]
    if not all(groups) or groups[0] & groups[1]:
        raise ValueError("separate and nonempty positive and safe shot groups required")
    for ids in groups:
        for sid in ids:
            same = shots == sid
            result[same] = 0.5 / (len(ids) * int(same.sum()))
    return result


def routes_all(available):
    data = available.reshape(len(available), 16, 8, 23).any(dim=(2, 3))
    tokens = torch.cat((data, torch.ones_like(data[:, :1])), dim=1)
    return tokens[:, None, :].expand(-1, 2, -1)


def bounded_delta(raw):
    return RADIUS * raw / (1 + raw.square().sum(dim=1, keepdim=True)).sqrt()


class Generator(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)
        self.layers = nn.Sequential(nn.Linear(WIDTH, 32), nn.SiLU(), nn.Linear(32, 256))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features):
        return bounded_delta(self.layers((features - self.mean) / self.scale))


@torch.no_grad()
def native_context(base, signal):
    x, a = make_views(signal, base.center.device)
    m = np.arange(policy.FIRST, len(x), policy.STRIDE, dtype=np.int64)
    native_p = np.empty(len(x), dtype=np.float32)
    native_z = np.empty_like(native_p)
    states = x.new_empty((len(m), 64))
    seen = []

    def capture(_module, args):
        seen.append(args[0])

    init = base.initial_fast(1).detach()
    hook = base.risk_head.register_forward_pre_hook(capture)
    try:
        for lo in range(0, len(x), BATCH):
            hi = min(lo + BATCH, len(x))
            seen.clear()
            z = base(
                x[lo:hi].contiguous(), a[lo:hi].contiguous(), init.expand(hi - lo, -1)
            )["logit"]
            if (
                len(seen) != 1
                or seen[0].shape != (hi - lo, 64)
                or not torch.isfinite(z).all()
            ):
                raise RuntimeError(
                    "exactly one finite native logit and CLS per endpoint required"
                )
            native_p[lo:hi] = z.sigmoid().cpu().numpy()
            native_z[lo:hi] = z.cpu().numpy()
            mlo, mhi = np.searchsorted(m, (lo, hi))
            idx = torch.as_tensor(m[mlo:mhi] - lo, device=x.device)
            states[mlo:mhi] = seen[0][idx]
    finally:
        hook.remove()
    aux = np.stack((native_z[m], np.log1p(m / 1000)), axis=1)
    state = torch.cat(
        (torch.as_tensor(aux, device=x.device, dtype=x.dtype), states), dim=1
    )
    history, stale = causal_history(states)
    return x, a, native_p, native_z, state, history, stale


def episode(ctx, times, flag, down, sid, rank, *, keep_windows):
    x, a, p, z, state, history, _ = ctx
    q, lead = labelled_queries(times, p, flag, down)
    k = torch.as_tensor((q - policy.FIRST) // policy.STRIDE, device=x.device)
    info = dict(
        shot=np.full(len(q), sid, np.int64),
        rank=np.full(len(q), rank, np.int64),
        query=q,
        positive=np.full(len(q), flag, bool),
        lead_ms=lead,
        native_logit=z[q],
        state=state[k].cpu().numpy(),
        history=history[k].cpu().numpy(),
    )
    if keep_windows:
        idx = torch.as_tensor(q, device=x.device)
        return info, (x[idx].cpu(), a[idx].cpu())
    return info, None


def pack(episodes):
    return {k: np.concatenate([v[k] for v in episodes], axis=0) for k in episodes[0]}


def fit_budget(base, episodes, windows, threshold, seed=131):
    bank = pack(episodes)
    device = base.center.device
    x, a = (torch.cat(items).to(device) for items in zip(*windows, strict=True))
    w = torch.as_tensor(
        weights_by_class_shot(bank["shot"], bank["positive"]),
        device=device,
        dtype=x.dtype,
    )
    signs = torch.as_tensor(
        np.where(bank["positive"], -1.0, 1.0), device=device, dtype=x.dtype
    )
    clock = float(np.log(threshold / (1 - threshold)))
    static = torch.as_tensor(bank["native_logit"], device=device)
    initial_loss = float((F.softplus(signs * (static - clock)) * w).sum())
    models, report = {}, {}
    for name in ("state", "history"):
        observed = (
            bank["history"] if name == "history" else np.zeros_like(bank["history"])
        )
        f = torch.as_tensor(
            np.concatenate((bank["state"], observed), axis=1), device=device
        )
        torch.manual_seed(seed)
        model = Generator(f.mean(0), f.std(0, correction=0).clamp_min(0.05)).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=LR)
        generator = torch.Generator().manual_seed(seed)

        def objective(idx):
            delta = model(f[idx])
            moved = policy.routed_pair(base, x[idx], a[idx], routes_all(a[idx]), delta)
            norm = delta.square().sum(dim=1) / RADIUS**2
            return (
                w[idx] * (F.softplus(signs[idx] * (moved - clock)) + 0.005 * norm)
            ).sum()

        steps = 0
        for _ in range(EPOCHS):
            order = torch.randperm(len(x), generator=generator).to(device)
            for idx in order.split(MINIBATCH):
                optim.zero_grad(set_to_none=True)
                loss = objective(idx) * len(x) / len(idx)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite exact-forward training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0, error_if_nonfinite=True
                )
                optim.step()
                steps += 1
        model.eval().requires_grad_(False)
        with torch.no_grad():
            loss = sum(
                float(objective(idx))
                for idx in torch.arange(len(x), device=device).split(MINIBATCH)
            )
        models[name] = model
        report[name] = dict(
            initial_loss=initial_loss, fitted_loss=loss, optimizer_steps=steps
        )
    return models, report


def predict(base, ctx, budgets):
    """Temporary weights are written from observable mature features, not shot labels."""
    x, a, p, _, state, history, stale = ctx
    traces, writes = {}, {}
    for budget, models in budgets.items():
        with torch.no_grad():

            def features(h):
                return torch.cat((state, h), dim=1)

            real = models["history"](features(history))
            targets = dict(
                history=real,
                state=models["state"](features(torch.zeros_like(history))),
                stale=policy.match_norm(models["history"](features(stale)), real),
            )
        for k, target in targets.items():
            name = f"{k}{budget}"
            traces[name] = p.copy()
            with torch.enable_grad():
                fast = policy.write_fast(target) if len(target) else target
            norm = fast.norm(dim=1)
            if not torch.isfinite(norm).all() or (norm > RADIUS + 1e-6).any():
                raise FloatingPointError("out of bounds temporary write")
            writes[name] = dict(
                nonzero=int((norm > 0).sum()),
                maximum=float(norm.max()) if len(norm) else 0.0,
                real_norm_mismatch=int((abs(norm - real.norm(dim=1)) > 1e-6).sum())
                if k == "stale"
                else 0,
            )
            with torch.no_grad():
                for lo in range(policy.FIRST, len(x), BATCH):
                    hi = min(lo + BATCH, len(x))
                    v, mask = x[lo:hi], a[lo:hi]
                    i = torch.arange(lo, hi, device=x.device)
                    d = fast[(i - policy.FIRST) // policy.STRIDE]
                    traces[name][lo:hi] = (
                        policy.routed_pair(base, v, mask, routes_all(mask), d)
                        .sigmoid()
                        .cpu()
                        .numpy()
                    )
    return traces, writes
