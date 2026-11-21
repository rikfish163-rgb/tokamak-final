"""Compare earlier/later states of the SAME historical disruptive shot.

Earlier means outside the fixed 125ms target, not physically safe. Only past
authorized labels are used. Current arrived signals route encoder gradients;
this is supervised-history reuse, not a label-free gradient objective.
"""
import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..event_pairing_20261024.learning import features
from ..event_routed_20261024.learning import embed
from ..history_tangent_20261023.learning import flat, logit
from ..paired_feedback_20261022.model import initial_encoder
from ..trajectory_memory_20261024.learning import observed_window, select_weights, combine_step, predict as stream_predict

ARMS = ('point', 'augmented', 'paired', 'unpaired', 'reverse', 'uniform')


def pair_loss(late, early, mode):
    if mode == 'rank':
        return F.softplus(early - late)
    if mode == 'augmented':
        return .5 * (F.softplus(early) + F.softplus(-late))
    raise ValueError('rank or augmented objective required')


def earlier_partner(signal, times, down, late_end, center, scale):
    """Nearest average observed state, same shot; no model-score selection."""
    candidates = np.arange(127, late_end, 16)
    candidates = candidates[(down - times[candidates]) * 1000 > 250. + 1e-6]
    if not len(candidates):
        raise ValueError('no full earlier historical window outside 250ms')
    late = features(*simple_data.window(signal, int(late_end), 128), center, scale)
    f = np.stack([features(*simple_data.window(signal, int(e), 128), center, scale) for e in candidates])
    common = (f[:, 207:].reshape(-1, 8, 23).sum(1) > 0) & (late[207:].reshape(8, 23).sum(0) > 0)
    n = common.sum(1)
    d = (((f[:, :23] - late[:23]) ** 2) * common).sum(1) / np.maximum(n, 1)
    d[n < 8] = np.inf
    if not np.isfinite(d).any():
        raise ValueError('no earlier partner with eight joint observed channels')
    best = int(d.argmin())
    return int(candidates[best]), float(d[best]), int(np.isfinite(d).sum())


def permuted_partners(shots, labels):
    disruptive = np.unique(shots[labels == 1])
    if len(disruptive) < 2:
        raise ValueError('at least two historical disruptive shots required')
    groups = [np.flatnonzero((shots == sid) & (labels == 1)) for sid in disruptive]
    if len({len(g) for g in groups}) != 1:
        raise ValueError('equal row counts required for exact marginal-preserving control')
    indexes = np.arange(len(shots))
    for group, other in zip(groups, groups[1:] + groups[:1]):
        indexes[group] = other
    return indexes


def prepare_pairs(model, bank, histories, device='cuda'):
    """Build exact parent gradients; safe-shot singleton BCE is unchanged."""
    early_x = bank['x'].copy(); early_m = bank['mask'].copy(); pairs = []
    for i, row in enumerate(bank['rows']):
        if not row['label']:
            continue
        h = histories[row['shot']]
        signal = {k:h[k] for k in ('values', 'available')}
        lead = (h['down'] - h['times'][row['end']]) * 1000
        if not 30 < lead <= 125 + 1e-6:
            raise ValueError('late bank endpoint outside authorized timely target')
        end, distance, count = earlier_partner(signal, h['times'], h['down'], row['end'], bank['center'], bank['scale'])
        early_x[i], early_m[i] = simple_data.window(signal, end, 128)
        pairs.append(dict(row=i, shot=row['shot'], early_end=end, late_end=row['end'],
            early_lead_ms=float((h['down'] - h['times'][end])*1000), late_lead_ms=float(lead),
            mean_state_distance=distance, candidates=count))
    perm = permuted_partners(bank['shots'], bank['labels'])
    gradients = {k:bank['gradients'].clone() for k in ('augmented', 'paired', 'unpaired')}
    fast = initial_encoder(model)
    for r in pairs:
        i = r['row']; probs = {}
        for mode in gradients:
            j = int(perm[i]) if mode == 'unpaired' else i
            xx = torch.as_tensor(np.stack([early_x[j], bank['x'][i]]), device=device)
            mm = torch.as_tensor(np.stack([early_m[j], bank['mask'][i]]), device=device)
            z = logit(model, xx, mm, fast)
            loss = pair_loss(z[1], z[0], 'augmented' if mode == 'augmented' else 'rank')
            g = flat(torch.autograd.grad(loss, tuple(fast.values()))).detach()
            if not torch.isfinite(g).all():
                raise FloatingPointError('nonfinite historical paired gradient')
            gradients[mode][i] = g
            probs[mode] = dict(early=float(z[0].sigmoid().detach()), late=float(z[1].sigmoid().detach()),
                               loss=float(loss.detach()), gradient_norm=float(g.double().norm()))
        r['parent'] = probs
        r['unpaired_early_shot'] = int(bank['shots'][perm[i]])
        r['unpaired_early_row'] = int(perm[i])
    bank['pair_gradients'] = gradients
    return dict(early_x=early_x, early_mask=early_m, permutation=perm), pairs


def update_all(model, signal, count, bank, device='cuda'):
    x, m = observed_window(signal, count)
    query = features(x, m, bank['center'], bank['scale'])[None]
    enough = int((query[0, 207:].reshape(8, 23).sum(0) > 0).sum()) >= 8
    distance = {}
    for mode in ('risk', 'reverse'):
        q = embed(query, bank['metric_models']['risk'], bank['normalizer']['mean'],
                  bank['normalizer']['scale'], mode, device)[0]
        distance[mode] = ((bank['metric_bank']['risk'] - q) ** 2).sum(1)
    distance['uniform'] = np.full(len(bank['shots']), np.inf)
    for sid in np.unique(bank['shots']):
        ids = np.flatnonzero(bank['shots'] == sid)
        distance['uniform'][ids[len(ids)//2]] = 0.
    states = {}; receipts = {}; length = None; parent = initial_encoder(model)
    for arm in ARMS:
        d = distance[arm if arm in ('reverse', 'uniform') else 'risk'].copy()
        if not enough:
            d[:] = np.inf
        ix, w = select_weights(d, bank['shots'], bank['labels'], bank['metric_temperature']['risk'])
        grad = bank['gradients'] if arm == 'point' else bank['pair_gradients'][arm if arm in ('augmented', 'unpaired') else 'paired']
        if not len(ix):
            states[arm] = initial_encoder(model)
            r = dict(gradient_norm=0., requested_step_norm=0., step_norm=0., reason='fewer_than8_observed_channels')
        else:
            states[arm], r = combine_step(parent, grad[ix], torch.as_tensor(w, device=device), learning_rate=.01, length=length)
        if arm == 'point':
            length = r['requested_step_norm']
        receipts[arm] = dict(r, selected=ix.tolist(), weights=w.tolist(), distances=d[ix].tolist(),
                             last_observation=count-1, first_affected_prediction=count)
    return states, receipts


def predict(model, signal, bank, device='cuda'):
    return stream_predict(model, signal, bank, device, arms=ARMS, updater=update_all)
