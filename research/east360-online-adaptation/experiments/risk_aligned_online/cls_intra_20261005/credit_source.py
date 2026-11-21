"""Selectable backward credit for source-only offline causal replay.

Forward execution is unchanged from source.causal_logit: every legal event
with new observed targets is replayed, including detached burn-in and finite
zero-gradient updates. This module does not alter the online stream or the
qualified classifier. Full credit is for offline gradient audits, not a new
training or deployment policy.
"""
import torch

from .source import simple_data, _batch_windows, event_clock, snapshot, segment_step


def credit_causal_logit(model, shot, query, *, use_task, credit_steps=4, create_graph=False):
    """Replay all arrivals; differentiate the last N executed updates or all.

    ``credit_steps`` is a positive Python integer, or ``None`` for full credit.
    Burn-in is executed before its state is detached, never omitted. With
    ``create_graph=False`` each update is detached immediately, irrespective
    of the requested window, so no update-history graph is retained. The
    endpoint readout remains differentiable just as in the original helper.
    """
    if credit_steps is not None and (type(credit_steps) is not int or credit_steps <= 0):
        raise ValueError('credit_steps must be a positive integer or None')
    if shot['device'] != 'J-TEXT' or type(query) is not int or not 0 <= query < len(shot['times']):
        raise ValueError('offline source endpoint required')
    signals = {k: shot[k] for k in ('values', 'available')}
    device = model.center.device
    events = []
    missing = 0
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
    burn = 0 if credit_steps is None else max(0, len(events) - credit_steps)
    delta = model.initial_delta()
    for j, (origin, segment, truth, observed) in enumerate(events):
        x, a = _batch_windows(simple_data, signals, [origin], model, device)
        delta, _ = segment_step(model, delta, x, a, truth, observed, segment, use_task=use_task,
                                create_graph=create_graph and j >= burn)
        if j < burn or not create_graph:
            delta = delta.detach().requires_grad_()
    x, a = _batch_windows(simple_data, signals, [query], model, device)
    return model(x, a, delta)['logit'].reshape(()), dict(
        executed_updates=len(events), missing_target_events=missing, burn_in_updates=burn,
        differentiated_updates=len(events) - burn if create_graph else 0,
        delta_norm=float(delta.detach().double().norm()))
