"""Source-only matched partial feedback, with causal whole-shot outer risk.

The classifier remains frozen. Only the auxiliary direct head is fitted offline.
No source point-label changes; the >30ms auxiliary risk objective is separate.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data
from ..simple_train import metrics
from ..redesign_20261004.intra import _batch_windows
from ..decoder_intra_20261005.whole_meta import select_peak
from ..decoder_intra_20261005.matched_meta import maintenance_loss
from .feedback import event_clock, segment_step
from .stream import snapshot, predict_shot


def causal_logit(model, shot, query, *, mode, create_graph=False):
    if shot['device'] != 'J-TEXT':
        raise ValueError('offline source auxiliary development only')
    if type(query) is not int or not 0 <= query < len(shot['times']):
        raise ValueError('query must be an existing endpoint index')
    if mode not in ('partial', 'delayed'):
        raise ValueError('declare partial/delayed clock')
    signals = {k: shot[k] for k in ('values', 'available')}
    device = model.center.device; events = []; missing = 0
    for origin in range(0, query, 26):
        for cutoff, update, effect, segment in event_clock(origin, mode):
            if effect > query:
                continue
            truth, observed = snapshot(signals, origin, segment, device, model.center.dtype)
            level = observed if segment == 'early' else observed[:, 1:]
            if not level.any():
                missing += 1
                continue
            events.append((origin, segment, truth, observed))
    # Keep four EXECUTED updates. All earlier real updates form detached burn-in.
    # This covers a different duration than the previous H25 four-event rule.
    burn = max(0, len(events) - 4); delta = model.initial_delta()
    for j, (origin, segment, truth, observed) in enumerate(events):
        x, a = _batch_windows(simple_data, signals, [origin], model, device)
        delta, _ = segment_step(model, delta, x, a, truth, observed, segment,
                                create_graph=create_graph and j >= burn)
        if j < burn or not create_graph:
            delta = delta.detach().requires_grad_()
    x, a = _batch_windows(simple_data, signals, [query], model, device)
    return model(x, a, delta)['logit'].reshape(()), dict(mature_events=len(events) + missing,
        executed_updates=len(events), missing_target_events=missing, burn_in_updates=burn,
        differentiated_updates=min(4, len(events)) if create_graph else 0,
        delta_norm=float(delta.detach().double().norm()))


def risk_from_probability(probability, label, boundary):
    if not math.isfinite(boundary):
        raise ValueError('finite frozen boundary required')
    p = probability.double().clamp(float(np.float32(1e-7)), float(np.float32(1-1e-7)))
    margin = torch.logit(p) - boundary
    return F.softplus(-margin if label else margin)


def whole_risk(model, shot, *, mode, boundary, create_graph=False):
    scores, _ = predict_shot(model, shot, mode=mode, offline_source=True)
    query = select_peak(shot, scores)
    z, row = causal_logit(model, shot, query, mode=mode, create_graph=create_graph)
    probability = z.sigmoid()
    if abs(float(probability.detach()) - float(scores[query])) > 2e-6:
        raise RuntimeError('selected risk differs from actual causal stream')
    risk = risk_from_probability(probability, shot['is_disrupt'], boundary)
    if not torch.isfinite(risk):
        raise FloatingPointError('nonfinite source risk')
    return risk, dict(**row, shot=shot['shot'], label=int(shot['is_disrupt']), peak_index=query,
        peak_probability=float(probability.detach()), risk_loss=float(risk.detach()),
        before_first_update=row['executed_updates'] == 0)


def selection_report(model, pairs, threshold, *, mode):
    """Whole and former later-segment criteria, BOTH on actually emitted traces.

    Partial updates can occur inside the old later segment. Rescoring all its
    points with one frozen terminal delta would no longer be a causal measure.
    """
    boundary = math.log(threshold/(1-threshold))
    rows = []; whole = []; later = []; maintenance = []; updates = 0
    for shot, episode in pairs:
        if shot['device'] != 'J-TEXT' or episode['shot'] != shot['shot']:
            raise ValueError('matched source episode required')
        scores, audit = predict_shot(model, shot, mode=mode, offline_source=True)
        q = select_peak(shot, scores)
        risk = float(risk_from_probability(torch.tensor(scores[q]), shot['is_disrupt'], boundary))
        whole.append(dict(shot=shot['shot'], label=int(shot['is_disrupt']), peak_index=q,
                          peak_probability=float(scores[q]), risk_loss=risk))
        queries = list(episode['queries'])
        if not shot['is_disrupt']:
            queries += [o + 26 for o in episode['origins']]
        indices = np.array(sorted(set(queries)), dtype=int)
        if not len(indices) or indices.min() < 0 or indices.max() >= len(scores):
            raise ValueError('nonempty in-shot later segment required')
        lp = float(scores[indices].max())
        later.append(dict(shot=shot['shot'], label=int(shot['is_disrupt']),
            outer_segment_loss=float(risk_from_probability(torch.tensor(lp), shot['is_disrupt'], boundary)),
            peak_probability=lp))
        valid = shot['valid'].copy()
        if shot['is_disrupt']:
            valid &= shot['times'] <= shot['down']
        rows.append(dict(shot=shot['shot'], is_disrupt=shot['is_disrupt'], down=shot['down'],
                         times=shot['times'][valid], scores=scores[valid]))
        maintenance.append(float(maintenance_loss(model, shot, episode).detach()))
        updates += audit['accepted_updates']
    if {r['label'] for r in whole} != {0, 1}:
        raise ValueError('both source classes required')
    return dict(whole=dict(metrics30=metrics(rows, threshold, lead=30.),
        metrics10=metrics(rows, threshold, lead=10.), records=whole, updates=updates,
        balanced_loss=float(np.mean([np.mean([r['risk_loss'] for r in whole if r['label'] == label]) for label in (0, 1)]))),
        later=dict(mean_loss=float(np.mean([r['outer_segment_loss'] for r in later])), records=later),
        maintenance=float(np.mean(maintenance)), scope='source auxiliary development, not JTEXT online deployment')


def train_pair(model, pairs, optimizer, *, objective, boundary):
    if objective not in ('signal_only', 'risk_aligned'):
        raise ValueError('signal_only or risk_aligned objective required')
    allowed = {id(p) for p in model.base.forecast_head.parameters()}
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    if set(actual) != allowed or len(actual) != len(allowed) or {id(p) for p in model.parameters() if p.requires_grad} != allowed:
        raise ValueError('only auxiliary forecast head may be optimized')
    if len(pairs) != 2 or sorted(int(s['is_disrupt']) for s, _ in pairs) != [0, 1]:
        raise ValueError('balanced source pair required')
    if any(s['device'] != 'J-TEXT' or s['shot'] != e['shot'] or int(s['is_disrupt']) != e['label'] for s, e in pairs):
        raise ValueError('matching source identity and label required')
    model.zero_grad(set_to_none=True); rows = []
    for shot, episode in pairs:
        if objective == 'risk_aligned':
            risk, row = whole_risk(model, shot, mode='partial', boundary=boundary, create_graph=True)
        else:
            risk = model.center.new_zeros(())
            row = dict(shot=shot['shot'], label=int(shot['is_disrupt']))
        signal = maintenance_loss(model, shot, episode)
        loss = risk + .20 * signal
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite source head objective')
        (loss/2).backward()
        rows.append(dict(**row, maintenance_loss=float(signal.detach()), objective=float(loss.detach())))
    norm = torch.nn.utils.clip_grad_norm_(model.base.forecast_head.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(head_gradient_norm=float(norm), records=rows, objective=float(np.mean([r['objective'] for r in rows])))
