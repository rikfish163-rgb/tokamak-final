"""Historical EAST output tangents protect arrived-signal encoder updates.

OGD-style first-order protection at the frozen parent, not a physical law.
Controls share the history arm's *unlabelled* step length, not its direction.
"""
import math

import numpy as np
import torch

from .. import simple_data
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..entropy_stream_20261023.learning import validate
from ..paired_feedback_20261022.model import initial_encoder
from ..shot_objective_alignment_20261023.learning import candidate_ends

ARMS = ('history', 'norm_only', 'permuted')


def flat(values):
    return torch.cat([v.reshape(-1) for v in values])


def basis_from_rows(rows):
    if (rows.ndim != 2 or min(rows.shape) < 1 or not torch.isfinite(rows).all()
            or (rows.norm(dim=1) == 0).any()):
        raise ValueError('finite nonzero historical output gradients required')
    r = rows.detach().cpu().double()
    r = r / r.norm(dim=1, keepdim=True)
    _, s, vh = torch.linalg.svd(r, full_matrices=False)
    rank = int((s > s[0]*1e-7).sum())
    if not rank:
        raise ValueError('no historical tangent identified')
    return vh[:rank].to(rows).contiguous()


def permute_basis(basis, params):
    if basis.ndim != 2 or basis.shape[1] != sum(p.numel() for p in params.values()):
        raise ValueError('basis must match ordered encoder parameters')
    rng = torch.Generator(device='cpu').manual_seed(131)
    indexes = []; offset = 0
    for p in params.values():
        indexes.append(torch.randperm(p.numel(), generator=rng)+offset)
        offset += p.numel()
    return basis[:, torch.cat(indexes).to(basis.device)].contiguous()


def project(g, basis):
    if (g.ndim != 1 or basis.ndim != 2 or basis.shape[1] != len(g)
            or g.device != basis.device or g.dtype != basis.dtype
            or not torch.isfinite(g).all() or not torch.isfinite(basis).all()):
        raise ValueError('matching finite gradient and orthonormal basis required')
    # Second pass removes floating-point residue; the caller verifies QQ'=I.
    v = g-basis.T@(basis@g)
    v = v-basis.T@(basis@v)
    return v


def matched_step(direction, length):
    if not math.isfinite(float(length)) or length < 0 or not torch.isfinite(direction).all():
        raise ValueError('finite direction and nonnegative step length required')
    norm = direction.double().norm()
    if norm == 0 or length == 0:
        # A zero direction contains no usable update; never substitute a raw one.
        return torch.zeros_like(direction)
    return direction * (float(length)/norm).to(direction)


def logit(model, x, mask, fast):
    h = model.predict_fast(x, mask, fast, mode='risk_state')['risk_state']
    return model.base.risk_head(h).squeeze(-1)


def history_basis(model, shots, *, device='cuda'):
    if not shots or len({s['shot'] for s in shots}) != len(shots):
        raise ValueError('distinct authorized historical shots required')
    fast = initial_encoder(model); rows = []; anchors = []; receipts = []
    for s in shots:
        ends = candidate_ends(s); signal = {k:s[k] for k in ('values', 'available')}
        best = -math.inf; chosen = None
        for start in range(0, len(ends), 32):
            chunk = ends[start:start+32]
            windows = [simple_data.window(signal, int(e), 128) for e in chunk]
            x, m = [torch.as_tensor(np.stack(a), device=device) for a in zip(*windows)]
            z = logit(model, x, m, fast).detach()
            if not torch.isfinite(z).all():
                raise FloatingPointError('nonfinite historical risk')
            idx = int(z.argmax())
            if float(z[idx]) > best:
                best = float(z[idx]); chosen = int(chunk[idx])
        x, m = [torch.as_tensor(a[None], device=device) for a in simple_data.window(signal, chosen, 128)]
        z = logit(model, x, m, fast)
        if abs(float(z.detach()[0])-best) > 2e-5:
            raise AssertionError('historical peak single-window replay mismatch')
        gradient = flat(torch.autograd.grad(z.sum(), tuple(fast.values()))).detach()
        rows.append(gradient); anchors.append((x[0].cpu(), m[0].cpu()))
        receipts.append(dict(shot=int(s['shot']), is_disrupt=bool(s['is_disrupt']),
                             end=chosen, candidate_count=len(ends), parent_logit=float(z.detach()[0]),
                             gradient_norm=float(gradient.double().norm())))
    rows = torch.stack(rows); q = basis_from_rows(rows)
    torch.testing.assert_close(q @ q.T, torch.eye(len(q), device=device), atol=2e-6, rtol=2e-6)
    x, m = [torch.stack(a).to(device) for a in zip(*anchors)]
    # Store batched parent values: drift checks use identical batch dispatch.
    z = logit(model, x, m, fast).detach()
    return dict(history=q, permuted=permute_basis(q, fast)), dict(x=x, mask=m, logits=z), rows, receipts


def apply_step(fast, vector):
    out = {}; offset = 0
    for k, v in fast.items():
        n = v.numel()
        out[k] = (v-vector[offset:offset+n].reshape_as(v)).detach().requires_grad_(True)
        offset += n
    if offset != len(vector):
        raise ValueError('step shape does not match parameters')
    return out


def update_all(model, signal, count, states, bases, *, device):
    validate(signal)
    x, mask, target, tm, ends = forecast_examples(signal, count, model.cfg, device)
    observed = int((tm & torch.isfinite(target)).sum())
    receipts = {}; target_length = None; next_states = {}
    parent = dict(model.named_parameters())
    for arm in ARMS:
        fast = states[arm]
        r = dict(last_observation=count-1, first_affected_prediction=count,
                 forecast_origins=ends, observed_targets=observed, gradient_steps=0,
                 parameter_delta_norm=0., requested_step_norm=0.)
        if not observed:
            r['reason'] = 'no_observed_targets'
            next_states[arm] = fast; receipts[arm] = r
            continue
        prediction = model.predict_fast(x, mask, fast)['forecast']
        loss = model.base.forecast_loss(prediction, target, tm)
        g = flat(torch.autograd.grad(loss, tuple(fast.values()))).detach()
        if not torch.isfinite(loss) or not torch.isfinite(g).all():
            raise FloatingPointError('nonfinite arrived-signal gradient')
        raw = float(g.double().norm())
        d = g if arm == 'norm_only' else project(g, bases[arm])
        projected = float(d.double().norm())
        if arm == 'history':
            target_length = .001*projected/max(1., raw)
        step = matched_step(d, target_length)
        updated = apply_step(fast, step)
        actual = flat([updated[k].detach()-v.detach() for k, v in fast.items()])
        with torch.no_grad():
            after = model.base.forecast_loss(model.predict_fast(x, mask, updated)['forecast'], target, tm)
        r.update(gradient_steps=int(bool(step.any())), requested_step_norm=target_length,
                 parameter_delta_norm=float(actual.double().norm()), raw_gradient_norm=raw,
                 projected_gradient_norm=projected, loss_before=float(loss.detach()), loss_after=float(after),
                 parent_distance=float(flat([v.detach()-parent[k].detach() for k,v in updated.items()]).double().norm()),
                 historical_tangent_step_norm=float((bases['history']@step).double().norm()),
                 historical_tangent_actual_norm=float((bases['history']@actual).double().norm()),
                 zero_direction=projected == 0.)
        next_states[arm] = updated; receipts[arm] = r
    return next_states, receipts


def predict(model, signal, bases, *, device='cuda', anchors=None):
    validate(signal)
    if model.cfg.window != 128 or model.cfg.horizon != 25 or set(bases) != {'history','permuted'}:
        raise ValueError('W128 H25 model and two declared bases required')
    states = {a:initial_encoder(model) for a in ARMS}
    size = sum(v.numel() for v in states['history'].values())
    for q in bases.values():
        if q.ndim != 2 or q.shape[1] != size:
            raise ValueError('tangent does not match encoder')
        torch.testing.assert_close(q@q.T, torch.eye(len(q),device=q.device), atol=2e-6, rtol=2e-6)
    curves = {a:[] for a in ARMS}; receipts = {a:[] for a in ARMS}
    n = len(signal['values'])
    for lo in range(0, n, 128):
        hi = min(lo+128, n)
        for start in range(lo, hi, 32):
            ws = [simple_data.window(signal, t, 128) for t in range(start, min(start+32, hi))]
            x, m = [torch.as_tensor(np.stack(a), device=device) for a in zip(*ws)]
            for arm in ARMS:
                curves[arm].append(logit(model,x,m,states[arm]).sigmoid().detach().cpu().numpy())
        if hi >= 256 and hi < n:
            states, step_receipts = update_all(model,signal,hi,states,bases,device=device)
            for arm in ARMS:
                receipts[arm].append(step_receipts[arm])
    drift = {}
    if anchors is not None:
        for arm in ARMS:
            z = logit(model, anchors['x'], anchors['mask'], states[arm]).detach()
            drift[arm] = (z-anchors['logits']).cpu().tolist()
    curves = {a:np.concatenate(v) for a,v in curves.items()}
    if not all(np.isfinite(v).all() for v in curves.values()):
        raise FloatingPointError('nonfinite complete prediction')
    return curves, receipts, drift
