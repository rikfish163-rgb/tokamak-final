"""Source-only update-geometry training through up to FOUR causal signal steps.

Risk supervision targets worst safety risk / best genuinely timely positive
among source anchors, not mean easy-negative BCE. Original source network and
deployment threshold remain fixed. Anchors/labels NEVER enter the inner step.
"""
import math

import torch
from torch.nn import functional as F

from .ffn_subspace import one_step


def chain_origins(endpoint):
    if type(endpoint) is not int or endpoint < 26:
        raise ValueError('risk endpoint must be at least 26 samples after shot start')
    last = ((endpoint - 26) // 100) * 100
    return list(range(max(0, last - 300), last + 1, 100))


def operational_loss(logits, is_disrupt, labels, lead_ms, *, threshold_logit):
    if (logits.ndim != 1 or not len(logits) or logits.shape != labels.shape
            or logits.shape != lead_ms.shape or not math.isfinite(threshold_logit)):
        raise ValueError('finite threshold and aligned source risk anchors required')
    if is_disrupt:
        eligible = (labels == 1) & (lead_ms > 10.)
        if not eligible.any(): raise ValueError('at least one truly timely positive source anchor required')
        return F.softplus(threshold_logit - logits[eligible].max())
    if labels.any(): raise ValueError('safe source shot must have zero labels')
    return F.softplus(logits.max() - threshold_logit)


def meta_sequence_step(model, optimizer, chains, *, mode, threshold_logit):
    if mode not in ('risk', 'signal') or not chains:
        raise ValueError('nonempty source chains and risk/signal mode required')
    if [id(p) for group in optimizer.param_groups for p in group['params']] != [id(model.basis)]:
        raise ValueError('optimizer must contain only the update basis')
    model.zero_grad(set_to_none=True)
    rows = []
    for chain in chains:
        steps = chain['steps']
        if not 1 <= len(steps) <= 4: raise ValueError('one to four linked source signal steps required')
        current = model.initial_fast(1).requires_grad_()
        logits, labels, leads, signal_losses = [], [], [], []
        before_logits = []; accepted = 0
        for e in steps:
            current, row = one_step(model, e, fast=current, create_graph=True)
            accepted += row['accepted']
            size = len(e['segment_x'])
            after = model(e['segment_x'], e['segment_a'], current.expand(size, -1))['logit']
            zero = model.initial_fast(size).requires_grad_()
            before = model(e['segment_x'], e['segment_a'], zero)['logit'].detach()
            logits.append(after); before_logits.append(before)
            labels.append(e['segment_y']); leads.append(e['segment_lead'])
            signal_losses.append(model.forecast_loss(
                model(e['x'], e['a'], current)['forecast'], e['truth'], e['mask']))
        labels, leads = torch.cat(labels), torch.cat(leads)
        risk_after = operational_loss(torch.cat(logits), chain['is_disrupt'], labels, leads,
                                      threshold_logit=threshold_logit)
        risk_before = operational_loss(torch.cat(before_logits), chain['is_disrupt'], labels, leads,
                                       threshold_logit=threshold_logit)
        objective = risk_after if mode == 'risk' else torch.stack(signal_losses).mean()
        if not torch.isfinite(objective): raise FloatingPointError('nonfinite source chain objective')
        (objective / len(chains)).backward()
        rows.append(dict(accepted=accepted, signal_steps=len(steps),
            risk_before=float(risk_before), risk_after=float(risk_after.detach()),
            signal_after=float(torch.stack(signal_losses).detach().mean())))
    if model.basis.grad is None: raise RuntimeError('missing trajectory update-geometry gradient')
    norm = torch.nn.utils.clip_grad_norm_([model.basis], 1., error_if_nonfinite=True)
    optimizer.step(); model.project_basis()
    return dict(basis_grad_norm=float(norm), chains=len(rows),
        accepted=sum(r['accepted'] for r in rows), signal_steps=sum(r['signal_steps'] for r in rows),
        risk_before=sum(r['risk_before'] for r in rows) / len(rows),
        risk_after=sum(r['risk_after'] for r in rows) / len(rows),
        signal_after=sum(r['signal_after'] for r in rows) / len(rows))
