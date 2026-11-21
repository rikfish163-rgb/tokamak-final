"""Fit a source-only gradient transform on finite first-layer writes and replay EAST."""

import argparse
from pathlib import Path
from shutil import copyfile

import numpy as np
from scipy.special import logit
import torch
from torch.nn import functional as F

from . import commissioning_group_token_20261112 as group
from . import io23, simple_data
from .history_gradient_metric_20261102.run import read, sha
from .mask_separated_intra_20261103.run import state_digest


lag, layer = group.lag, group.layer
TEACHER = io23.ROOT/'probe_outputs/commissioning_signed_handoff_east360_20261112_v1'
RANK, PEAKS, EPOCHS, BATCH, SEED = 4, 4, 18, 16, 143
LR, TEMPERATURE, PENALTY = .02, .15, .04


def source_peaks(logits, release, eligible, valid, patch, *, lead_ms=None):
    """Select peaks from chronological sections of mature, source-labeled support."""
    endpoints = np.arange(len(logits))
    slot = np.searchsorted(release, endpoints, side='right')
    ok = valid & np.r_[False, eligible][slot] & (endpoints-np.r_[0, release][slot]>=patch)
    if lead_ms is not None:
        ok &= np.round(lead_ms, 9)>10.
    sections = np.array_split(np.flatnonzero(ok), PEAKS)
    return np.array([r[np.argmax(logits[r])] for r in sections if len(r)], dtype=np.int64)


def shot_max(values, owner, size):
    return values.new_full((size,), -torch.inf).scatter_reduce(0, owner, values, 'amax')


def shot_loss(peak, positive, threshold, *, mass=None):
    terms = F.softplus(torch.where(positive, threshold-peak, peak-threshold)/TEMPERATURE)
    return terms.mean() if mass is None else (mass*terms).sum()


def source_episode(raw, truth, args, decoder, path, weights):
    base = args[0]
    q, h, z, beta, prior, phase, correction = group.parent_track(raw, args, decoder, path, source=True)
    eligible, fields, waveform = group.feedback_waveform(raw, q, args[2], beta.device)
    states, common = group._estimate_innovation(base, raw, q, eligible, phase, correction, prior, fields,
                                               waveform, decoder, weights)
    positive = truth['is_disrupt']
    lead = (truth['down']-raw['times'])*1000. if positive else None
    valid = truth['valid'] & truth['y'].astype(bool) if positive else truth['valid']
    ends = source_peaks(z.cpu().numpy(), q+lag.H+1, common, valid, base.cfg.patch, lead_ms=lead)
    if not len(ends):
        return None
    slot, _ = lag.delayed_slots(q, ends)
    x, mask = lag._windows(base, raw, ends)
    # Preserve the native risk head's offset and the unadapted logits.
    baseline = z[ends]-h[ends]@beta
    current, lagged = states[-2:, slot]
    record = dict(shot=raw['shot'], positive=positive, endpoints=ends.tolist(),
                  release=(q[slot-1]+lag.H+1).tolist(),
                  lead_ms=lead[ends].tolist() if positive else None)
    return (x, mask, phase[slot], correction[slot], baseline, current, lagged), record


def batch_views(base, episodes, factors, *, controls):
    data, records = zip(*episodes, strict=True)
    x, mask, phase, correction, baseline, current, lagged = map(torch.cat, zip(*data, strict=True))
    mapped = group.gradient_transform(current, factors)
    moves = (current, lagged, mapped, group.gradient_transform(lagged, factors)) if controls else (mapped,)
    hidden, dose, err = group.forward_hidden(base, x, mask, layer.patcher(phase, correction),
                                           writes=torch.stack(moves))
    logits = baseline + hidden @ base.risk_head.weight.detach().flatten()
    counts = torch.tensor([len(r['endpoints']) for r in records], device=x.device)
    owners = torch.repeat_interleave(torch.arange(len(records), device=x.device), counts)
    positive = torch.tensor([r['positive'] for r in records], device=x.device, dtype=torch.bool)
    return torch.stack([shot_max(v, owners, len(records)) for v in logits]), positive, dose, err


def fit(base, examples, threshold, *, mass=None):
    width = (base.cfg.window//base.cfg.patch-1)*len(group.TRUSTED)
    rng = torch.Generator().manual_seed(SEED)
    direction = torch.linalg.qr(torch.randn(width, RANK, generator=rng)).Q.to(base.center.device)
    factors = torch.nn.Parameter(torch.stack((torch.zeros_like(direction), direction)))
    opt = torch.optim.Adam([factors], lr=LR)
    trace = []
    for epoch in range(EPOCHS):
        order = torch.randperm(len(examples), generator=rng).tolist()
        losses = []
        for start in range(0, len(order), BATCH):
            opt.zero_grad(set_to_none=True)
            selected = order[start:start+BATCH]
            rows = [examples[i] for i in selected]
            weight = None if mass is None else mass[selected]*(len(examples)/len(selected))
            peaks, positive, _, _ = batch_views(base, rows, factors, controls=False)
            left, right = factors.unbind()
            loss = (shot_loss(peaks[1], positive, threshold, mass=weight)
                    +(left@right.T).square().sum()*PENALTY/RANK)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        trace.append(float(np.mean(losses)))
        if epoch in (0, EPOCHS-1):
            print(dict(stage='source_gradient_fit', epoch=epoch+1, loss=trace[-1]), flush=True)
    if not torch.isfinite(factors).all():
        raise FloatingPointError('nonfinite source gradient factors')
    return factors.detach(), trace


@torch.no_grad()
def validate(base, examples, factors, threshold, *, arms=('parent', *group.METRIC[1:])):
    if not examples:
        return dict(count=0)
    view, label, written, errors = [], [], [], []
    for start in range(0, len(examples), BATCH):
        peaks, positive, dose, err = batch_views(base, examples[start:start+BATCH], factors, controls=True)
        view.append(peaks)
        label.append(positive)
        written.append(dose>0.)
        errors.append(float(err))
    logits, positive = torch.cat(view, dim=1), torch.cat(label)
    delta = logits-logits[0]
    if max(errors)>1e-5 or not torch.isfinite(logits).all():
        raise ValueError('invalid finite source writes')
    return dict(count=len(examples), positive=int(positive.sum()), safe=int((~positive).sum()),
                modified_endpoints=int(torch.cat(written).sum()), max_dose_error=max(errors),
                scope='four source-only eligible sampled peaks per shot; not a full shot evaluation',
                metrics={k: dict(positive_crossing=int((positive & (v>=threshold)).sum()),
                                 safe_crossing=int((~positive & (v>=threshold)).sum()),
                                 mean_delta_positive=float(d[positive].mean()) if positive.any() else None,
                                 mean_delta_safe=float(d[~positive].mean()) if (~positive).any() else None,
                                 loss=float(shot_loss(v, positive, threshold)))
                         for k, v, d in zip(arms, logits, delta, strict=True)})


def teach():
    lag.configure()
    lock = lag.load_inputs()
    p, _, args, decoder, path = group.load_reference(lock)
    base = args[0]
    frozen = state_digest(base)
    teacher = read(TEACHER/'teaching.json')
    tproto = read(TEACHER/'protocol.json')
    teacher_hash = sha(TEACHER/'teaching.json')
    history = group.REFERENCE/'historical.json'
    if (tproto['scope'] != 'EAST360_SIGNED_HANDOFF_DEVELOPMENT'
            or tproto['teaching_sha256'] != teacher_hash or teacher['history_sha256'] != sha(history)
            or teacher['fit_code_sha256'] != sha(TEACHER/'executed_teacher.py')
            or teacher['source_sha256'] != p['source_sha256']):
        raise ValueError('gradient transform must reuse the locked source training artifacts')
    weights = torch.as_tensor(teacher['weights'], device=base.center.device, dtype=torch.float32)
    manifests, roles, advanced = read(history)['orientation'], io23.source_roles(), io23.advanced_times()
    records, episodes = {}, {}
    for role in ('train', 'val'):
        manifest, allowed = manifests[role]['manifest'], set(roles[role])
        rows, missing = [], []
        for entry in manifest:
            sid = entry['shot']
            if sid not in allowed:
                raise ValueError('gradient transform history outside the source split')
            raw = simple_data.read_shot('J-TEXT', sid)
            truth = simple_data.attach_outcome(raw, 'J-TEXT', advanced)
            if truth['is_disrupt'] != entry['positive']:
                raise ValueError('source historical label differs from raw HDF5')
            episode = source_episode(raw, truth, args, decoder, path, weights)
            if episode is None:
                missing.append(sid)
            else:
                rows.append(episode)
        records[role] = dict(registered=len(manifest), used=[r for _, r in rows], missing=missing)
        episodes[role] = rows
        print(dict(stage='source_gradient_cases', role=role, count=len(rows), skipped=len(missing)), flush=True)
    if len({r['positive'] for _, r in episodes['train']})!=2:
        raise ValueError('source TRAIN requires supported positive and safe histories')
    threshold = float(logit(p['original']))
    factors, trace = fit(base, episodes['train'], threshold)
    for role in ('train', 'val'):
        records[role]['evaluation'] = validate(base, episodes[role], factors, threshold)
    if state_digest(base)!=frozen:
        raise ValueError('source backbone changed during gradient teaching')
    record = dict(scope='JTEXT_TRAIN_ONLY_FINITE_WRITE_GRADIENT_TRANSFORM',
                  source_sha256=p['source_sha256'], teacher_sha256=teacher_hash,
                  history_sha256=sha(history), fit_code_sha256=sha(Path(__file__)), rank=RANK,
                  factors=factors.cpu().tolist(), threshold_logit=threshold, loss_trace=trace,
                  fit=dict(seed=SEED, epochs=EPOCHS, batch=BATCH, lr=LR, temperature=TEMPERATURE, penalty=PENALTY),
                  evidence=records)
    print(dict(stage='source_gradient_frozen', train=records['train']['evaluation'],
               val=records['val']['evaluation']), flush=True)
    return teacher, record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    dest = parser.parse_args().out.resolve()
    teacher, record = teach()
    group.run(dest, teaching=teacher, innovation=True, metric=record)
    copyfile(Path(__file__), dest/'executed_metric.py')
    copyfile(TEACHER/'executed_teacher.py', dest/'executed_teacher.py')
    group.score(dest)
