"""Learn an internal correction from causal, partially supported temporal views."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import commissioning_weak_shape_data_20261111 as shape
from . import commissioning_weak_shape_source_20261111 as weak
from . import simple_data
from .commissioning_transfer_20261108 import features, shot_weights


SHORT_VIEW, FULL_VIEW, RAMP_END = 64, 128, 160
CORRECTION = .60
STEPS, BATCH, SEED = 384, 192, 131


def history_weight(samples):
    """Smooth contribution with eight arrived samples as the zero anchor."""
    x = np.clip((np.asarray(samples, dtype=np.float32) - 8.) / (RAMP_END - 8.), 0., 1.)
    return x*x*(3.-2.*x)


def at_width(z, m, c, cm, ends, width):
    """Apply the existing feature formula to an explicitly censored history."""
    if width == FULL_VIEW:
        return shape.inputs(z, m, c, cm, ends)
    if not 1 <= width < FULL_VIEW:
        raise ValueError('width must be between 1 and the fixed full window')
    blocks = []
    offsets = np.arange(1-FULL_VIEW, 1)
    for start in range(0, len(ends), 256):
        at = ends[start:start+256]
        ids = at[:, None]+offsets
        visible = ((ids >= 0) & (offsets > -width))[..., None]
        ids = np.maximum(ids, 0)
        blocks.append(shape.inputs(
            z[ids].reshape(-1, 10), (m[ids] & visible).reshape(-1, 10),
            c[ids].reshape(-1, 2), (cm[ids] & visible).reshape(-1, 2),
            np.arange(FULL_VIEW-1, FULL_VIEW*len(at), FULL_VIEW)))
    if not blocks:
        return shape.inputs(z, m, c, cm, ends)
    return tuple(np.concatenate(x) for x in zip(*blocks, strict=True))


def coverage(x):
    mask = x[:, 100:180].reshape(-1, 8, 10)
    groups = np.asarray(shape.GROUPS)
    return np.column_stack([mask[:, :, groups == i].mean((1, 2)) for i in range(4)])


def inputs(full, short, context, cf, cs, samples):
    return torch.cat((full.tanh(), short.tanh(), context, cf, cs,
                      (samples / RAMP_END).clamp_max(1.)[..., None]), dim=-1)


def latent(pre, weight, correction):
    return weight[..., None] * (pre+correction).tanh()


@torch.no_grad()
def record(net, signal, center, scale, ends, match_scale):
    z, m, c, cm = weak.shot_coordinates(signal, center, scale)
    x, context = at_width(z, m, c, cm, ends, FULL_VIEW)
    short, _ = at_width(z, m, c, cm, ends, SHORT_VIEW)
    pre = net.stem(torch.from_numpy(x)).numpy()
    prior = net.stem(torch.from_numpy(short)).numpy()
    key = .5*np.column_stack((np.ones(len(ends)), np.tanh(context[:, 0]), np.tanh(context[:, 1]),
                             np.tanh(4.*context[:, 2]))).astype(np.float32)
    return dict(full=pre, short=prior, context=np.tanh(context/match_scale).astype(np.float32),
                cf=coverage(x), cs=coverage(short), samples=(ends+1).astype(np.float32), key=key)


def source_records(net, ids, advanced, center, scale, match_scale, role):
    rows = []
    for i, sid in enumerate(ids, 1):
        raw = simple_data.read_shot('J-TEXT', sid)
        s = simple_data.attach_outcome(raw, 'J-TEXT', advanced)
        y, t = s['y'], s['times']
        eligible = s['valid'].copy()
        if s['is_disrupt']:
            eligible &= (y == 0.) | (np.round((s['down']-t)*1000., 9) > 10.)
        legal = np.flatnonzero(eligible)
        q = np.union1d(legal[::16], np.flatnonzero((y > 0) & eligible)[::4])
        q = np.union1d(q, legal[legal < RAMP_END][::8])
        if len(q):
            r = record(net, raw, center, scale, q, match_scale)
            rows.append(dict(r, risk=y[q], shot=np.full(len(q), sid)))
        if i % 450 == 0 or i == len(ids):
            print(dict(stage=f'prefix_{role}', shots=i, total=len(ids)), flush=True)
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


@torch.no_grad()
def cold_record(base, net, raw, mapped, center, scale, match_scale, coef):
    """Construct ended-shot early-positive and safe-peak training rows."""
    s = simple_data.attach_outcome(raw, 'EAST', None)
    x, _ = features(base, mapped, 'cuda')
    raw_z = torch.from_numpy(x).cuda() @ coef[:-25] + coef[-1]
    t = s['times']
    if s['is_disrupt']:
        lead = np.round((s['down']-t)*1000., 9)
        q = np.flatnonzero(s['valid'] & (s['y'] > 0.) & (lead > 30.))
    else:
        r = record(net, mapped, center, scale, np.arange(len(t)), match_scale)
        old = torch.from_numpy(r['full']).cuda().tanh()
        old[:127] = 0.
        ramp = torch.from_numpy(history_weight(np.arange(len(t))+1)).cuda()
        unconstrained = torch.from_numpy(r['full']).cuda().tanh() * ramp[:, None]
        candidates = torch.stack((raw_z+old@coef[-25:-1], raw_z+unconstrained@coef[-25:-1]))
        peak = candidates.topk(min(128, len(t)), dim=1).indices.cpu().numpy().ravel()
        q = np.union1d(np.arange(0, len(t), 16), peak)
        q = np.union1d(q, np.arange(min(len(t), RAMP_END)))
        r = {k: v[q] for k, v in r.items()}
    if not len(q):
        raise ValueError(f'no legal cold decision support for {raw["shot"]}')
    if s['is_disrupt']:
        r = record(net, mapped, center, scale, q, match_scale)
    return dict(r, raw_z=raw_z[q].cpu().numpy(), disruptive=s['is_disrupt'], sid=raw['shot'])


def _tensor(r, device):
    return [torch.as_tensor(r[k], device=device)
            for k in ('full', 'short', 'context', 'cf', 'cs', 'samples')]


def correction(layer, r):
    return CORRECTION * layer(inputs(*r)).tanh()


def fit(source, cold, net, beta, cutoff):
    """Paired same-end censoring on TRAIN, then fixed-boundary cold teaching."""
    device = 'cuda'
    src = _tensor(source, device)
    sbeta = net.risk.weight.detach().flatten().cuda()
    sbias = net.risk.bias.detach().cuda()
    y = torch.from_numpy(source['risk']).cuda()
    r = [_tensor(row, device) for row in cold]
    raw = [torch.from_numpy(row['raw_z']).cuda() for row in cold]
    cuts = [torch.from_numpy(history_weight(row['samples'])).cuda() for row in cold]
    lengths = [len(row['full']) for row in cold]
    packed = [torch.cat(col) for col in zip(*r, strict=True)]
    ro, rw = torch.cat(raw), torch.cat(cuts)
    design = inputs(*packed)
    layer = nn.Linear(design.shape[-1], src[0].shape[-1], device=device)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    weights = torch.from_numpy(shot_weights(source['shot'], source['risk'].astype(bool))).cuda()
    rng = torch.Generator(device=device).manual_seed(SEED)
    draw = torch.multinomial(weights, STEPS * BATCH, True, generator=rng).reshape(STEPS, BATCH)
    pos = torch.tensor([row['disruptive'] for row in cold], dtype=torch.bool, device=device)
    if not pos.any() or pos.all():
        raise ValueError('cold warning fit needs both outcomes')
    target_boundary = float(np.log(cutoff/(1.-cutoff)))
    opt = torch.optim.Adam(layer.parameters(), lr=.006)
    # The same acquired endpoint and label are used after hiding older
    # history; the complete view is a TRAIN-only same-end representation target.
    for step, ix in enumerate(draw, 1):
        opt.zero_grad(set_to_none=True)
        full, short, context, cf, cs, samples = (x[ix] for x in src)
        censor = (torch.arange(len(ix), device=device) % 2 == 0) & (samples > SHORT_VIEW)
        use_pre = torch.where(censor[:, None], short, full)
        use_cf = torch.where(censor[:, None], cs, cf)
        age = torch.where(censor, samples.clamp_max(SHORT_VIEW), samples)
        a = ((age-8.)/(RAMP_END-8.)).clamp(0., 1.)
        a = a*a*(3.-2.*a)
        offset = correction(layer, [use_pre, short, context, use_cf, cs, age])
        h = latent(use_pre, a, offset)
        source_loss = F.binary_cross_entropy_with_logits(h@sbeta+sbias, y[ix])
        stability = (h - a[:, None]*full.tanh()).square().mean()
        hz = latent(packed[0], rw, CORRECTION*layer(design).tanh())
        zz = ro + hz@beta
        peaks = torch.stack([z.max() for z in zz.split(lengths)])
        cold_loss = .5*(F.softplus(target_boundary-peaks[pos]).mean()
                        + F.softplus(peaks[~pos]-target_boundary).mean())
        loss = .35*source_loss + .25*stability + cold_loss + .006*layer.weight.square().sum()
        loss.backward()
        opt.step()
        if step % 128 == 0:
            print(dict(stage='prefix_fit', steps=step, loss=float(loss.detach()),
                       cold=float(cold_loss.detach()), source=float(source_loss.detach())), flush=True)
    if not all(torch.isfinite(t).all() for t in layer.parameters()):
        raise FloatingPointError('nonfinite prefix adapter')
    return layer.cpu().eval().requires_grad_(False)


@torch.no_grad()
def held_source(val, layer, net):
    r = _tensor(val, 'cpu')
    a = torch.from_numpy(history_weight(val['samples']))
    f = r[0].clone()
    f[val['samples'] < FULL_VIEW] = 0.
    b = net.risk.weight.detach().flatten()
    off = net.risk.bias.detach()
    z = dict(fixed=f.tanh()@b+off,
             ramp=latent(r[0], a, torch.zeros_like(r[0]))@b+off,
             learned=latent(r[0], a, correction(layer, r))@b+off)
    y = torch.from_numpy(val['risk'])
    out = {}
    for name, take in (('all', np.ones(len(y), bool)), ('short_prefix', val['samples'] < FULL_VIEW)):
        out[name] = dict(rows=int(take.sum()), positives=int((val['risk'][take] > 0).sum()),
                         bce={k: float(F.binary_cross_entropy_with_logits(v[take], y[take])) for k,v in z.items()})
    return out
