"""Offline source-only auxiliary fitting; never update the qualified classifier."""
import math

import numpy as np
import torch

from .. import simple_data
from ..simple_train import metrics
from ..redesign_20261004.intra import _batch_windows
from ..decoder_intra_20261005.whole_meta import select_peak
from ..decoder_intra_20261005.matched_meta import maintenance_loss
from ..partial_intra_20261005.feedback import event_clock
from ..partial_intra_20261005.source import risk_from_probability
from ..partial_intra_20261005.stream import snapshot
from .feedback import segment_step
from .stream import predict_shot


def enable_decoder_only(model):
    model.requires_grad_(False)
    model.decoder.requires_grad_(True)
    model.eval()


def causal_logit(model, shot, query, *, use_task, create_graph=False):
    if shot['device'] != 'J-TEXT' or type(query) is not int or not 0 <= query < len(shot['times']):
        raise ValueError('offline source endpoint required')
    signals = {k: shot[k] for k in ('values', 'available')}
    device = model.center.device; events = []; missing = 0
    for origin in range(0, query, 26):
        early = None
        for _, _, effect, segment in event_clock(origin, 'partial'):
            if effect > query:
                continue
            if segment == 'early':
                early = snapshot(signals, origin, segment, device, model.center.dtype)
                truth, observed = early
            else:
                late = snapshot(signals, origin, segment, device, model.center.dtype)
                truth = torch.cat((early[0], late[0][:, 1:]), dim=1)
                observed = torch.cat((early[1], late[1][:, 1:]), dim=1)
            new_valid = observed if segment == 'early' else observed[:, 5:]
            if not new_valid.any():
                missing += 1
                continue
            events.append((origin, segment, truth, observed))
    burn = max(0, len(events) - 4); delta = model.initial_delta()
    for j, (origin, segment, truth, observed) in enumerate(events):
        x, a = _batch_windows(simple_data, signals, [origin], model, device)
        delta, _ = segment_step(model, delta, x, a, truth, observed, segment, use_task=use_task,
                                create_graph=create_graph and j >= burn)
        if j < burn or not create_graph:
            delta = delta.detach().requires_grad_()
    x, a = _batch_windows(simple_data, signals, [query], model, device)
    return model(x, a, delta)['logit'].reshape(()), dict(executed_updates=len(events),
        missing_target_events=missing, burn_in_updates=burn,
        differentiated_updates=min(4, len(events)) if create_graph else 0,
        delta_norm=float(delta.detach().double().norm()))


def whole_risk(model, shot, *, use_task, boundary):
    scores, _ = predict_shot(model, shot, use_task=use_task, offline_source=True)
    query = select_peak(shot, scores)
    z, row = causal_logit(model, shot, query, use_task=use_task, create_graph=True)
    probability = z.sigmoid()
    if abs(float(probability.detach()) - float(scores[query])) > 2e-6:
        raise RuntimeError('replayed source peak differs from sealed trace')
    risk = risk_from_probability(probability, shot['is_disrupt'], boundary)
    return risk, dict(**row, shot=shot['shot'], label=int(shot['is_disrupt']), peak_index=query,
        peak_probability=float(probability.detach()), risk_loss=float(risk.detach()),
        before_first_update=row['executed_updates'] == 0)


def selection_report(model, pairs, threshold, *, use_task, intra=True):
    rows = []; records = []; maintenance = []; updates = 0
    boundary = math.log(threshold / (1 - threshold))
    for shot, episode in pairs:
        if shot['device'] != 'J-TEXT' or episode['shot'] != shot['shot']:
            raise ValueError('matched source episode required')
        scores, audit = predict_shot(model, shot, use_task=use_task, intra=intra, offline_source=True)
        q = select_peak(shot, scores)
        risk = float(risk_from_probability(torch.tensor(scores[q]), shot['is_disrupt'], boundary))
        records.append(dict(shot=shot['shot'], label=int(shot['is_disrupt']), peak_index=q,
            peak_probability=float(scores[q]), risk_loss=risk))
        valid = shot['valid'].copy()
        if shot['is_disrupt']: valid &= shot['times'] <= shot['down']
        rows.append(dict(shot=shot['shot'], is_disrupt=shot['is_disrupt'], down=shot['down'],
                         times=shot['times'][valid], scores=scores[valid]))
        maintenance.append(float(maintenance_loss(model, shot, episode).detach()))
        updates += audit['accepted_updates']
    if {r['label'] for r in records} != {0, 1}:
        raise ValueError('both source classes required')
    by_class = {str(label): float(np.mean([r['risk_loss'] for r in records if r['label'] == label]))
                for label in (0, 1)}
    return dict(metrics30=metrics(rows, threshold, lead=30.), metrics10=metrics(rows, threshold, lead=10.),
        records=records, updates=updates, balanced_loss=float(np.mean(list(by_class.values()))),
        class_loss=by_class, maintenance=float(np.mean(maintenance)),
        scope='source auxiliary development, not JTEXT online deployment')


def train_pair(model, pairs, optimizer, *, use_task, boundary):
    allowed = {id(p) for p in model.decoder.parameters()}
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    if set(actual) != allowed or len(actual) != len(allowed) or {id(p) for p in model.parameters() if p.requires_grad} != allowed:
        raise ValueError('only the auxiliary CLS decoder may be optimized')
    if len(pairs) != 2 or sorted(int(s['is_disrupt']) for s, _ in pairs) != [0, 1]:
        raise ValueError('balanced source pair required')
    if any(s['device'] != 'J-TEXT' or s['shot'] != e['shot'] or int(s['is_disrupt']) != e['label'] for s, e in pairs):
        raise ValueError('matched source identity/outcome required')
    model.zero_grad(set_to_none=True); rows = []
    for shot, episode in pairs:
        risk, row = whole_risk(model, shot, use_task=use_task, boundary=boundary)
        signal = maintenance_loss(model, shot, episode)
        loss = risk + .20 * signal
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite source objective')
        (loss / 2).backward()
        rows.append(dict(**row, maintenance_loss=float(signal.detach()), objective=float(loss.detach())))
    norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(decoder_gradient_norm=float(norm), records=rows,
                objective=float(np.mean([r['objective'] for r in rows])))
