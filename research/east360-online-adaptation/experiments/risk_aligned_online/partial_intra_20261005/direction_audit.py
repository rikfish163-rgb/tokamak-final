"""Post-hoc source-only gradient diagnosis of three fixed clock-harm cases.

Uses source outcomes to EXPLAIN an update, never to choose an online update.
Channel gradient attribution is a local model derivative, NOT physical causality.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .. import simple_data, io23
from ..simple_online import _hash, _durable_json
from ..fast_update import normalized_step
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT
from ..redesign_20261004.intra import _batch_windows
from ..ffn_intra_20261005.model import FfnModel
from ..ffn_intra_20261005.loss import weighted_signal_loss
from .feedback import event_clock, segment_step, segment_components
from .stream import snapshot
from .source import causal_logit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--counterfactual', action='store_true', help='post-hoc single-update channel intervention; not a new stream method')
    args = p.parse_args(); output = args.run/('direction_counterfactual.json' if args.counterfactual else 'direction_audit.json')
    if output.exists(): raise FileExistsError(output)
    audit = json.loads((args.run/'class_clock_audit.json').read_text())
    checkpoint = torch.load(args.run/'candidate_060.pt', map_location='cpu', weights_only=True)
    if checkpoint['protocol']['parent_sha256'] != _hash(DEFAULT_PARENT) or checkpoint['iteration'] != 60:
        raise ValueError('fixed source/candidate identity required')
    ids = (1055789, 1055436, 1055492)  # Fixed AFTER source class audit: 2S + 1D, not a performance sample.
    known = {r['shot']: r for r in audit['results']['risk60_partial']['records']}
    if not set(ids) <= set(simple_data.source_roles()['train']) or not set(ids) <= known.keys():
        raise ValueError('three fixed source development cases required')
    torch.set_num_threads(4); base, state = load_parent(DEFAULT_PARENT); model = FfnModel(base.to('cuda'))
    base.forecast_head.load_state_dict(checkpoint['forecast_head'])
    boundary = math.log(state['threshold']/(1-state['threshold']))
    advanced = simple_data.advanced_times(); rows = []
    for sid in ids:
        shot = simple_data.attach_outcome(simple_data.read_shot('J-TEXT', sid), 'J-TEXT', advanced)
        query = known[sid]['peak']; origin = query//26*26
        if not 6 <= query-origin <= 25: raise ValueError('clock-harm peak must be between early and late effects')
        signals = {k: shot[k] for k in ('values', 'available')}; device = model.center.device
        delta = model.initial_delta()
        for o in range(0, origin, 26):
            x, a = _batch_windows(simple_data, signals, [o], model, device)
            for _, _, effect, segment in event_clock(o, 'delayed'):
                if effect <= query:
                    truth, observed = snapshot(signals, o, segment, device, model.center.dtype)
                    delta, _ = segment_step(model, delta, x, a, truth, observed, segment)
                    delta = delta.detach().requires_grad_()
        qx, qa = _batch_windows(simple_data, signals, [query], model, device)
        before = model(qx, qa, delta)['logit'].reshape(())
        sign = -1 if shot['is_disrupt'] else 1
        risk = F.softplus(sign*(before-boundary)); risk_gradient, = torch.autograd.grad(risk, delta)
        x, a = _batch_windows(simple_data, signals, [origin], model, device)
        truth, observed = snapshot(signals, origin, 'early', device, model.center.dtype)
        parts = segment_components(model(x, a, delta)['forecast'], truth, observed, model.center, model.scale, 'early')
        valid = parts['valid']; count = valid.sum(-1).clamp_min(1)
        per_channel = (parts['level'].masked_fill(~valid, 0.)
            + .25*parts['difference'].masked_fill(~(valid & parts['difference_valid']), 0.)) / count[:, None]
        torch.testing.assert_close(per_channel.sum(-1), weighted_signal_loss(parts))
        gradients = [torch.autograd.grad(per_channel[0, c], delta, retain_graph=True)[0] for c in range(23)]
        total, = torch.autograd.grad(weighted_signal_loss(parts).sum(), delta)
        torch.testing.assert_close(sum(gradients), total, rtol=2e-4, atol=2e-7)
        updated, _ = segment_step(model, delta, x, a, truth, observed, 'early')
        after = model(qx, qa, updated)['logit'].reshape(())
        expected_before, _ = causal_logit(model, shot, query, mode='delayed')
        expected_after, _ = causal_logit(model, shot, query, mode='partial')
        torch.testing.assert_close(before, expected_before, rtol=0, atol=2e-6)
        torch.testing.assert_close(after, expected_after, rtol=0, atol=2e-6)
        actual = float((F.softplus(sign*(after-boundary))-risk).detach())
        norm = (total.square().sum()+1e-12).sqrt()
        channels = [dict(feature=name, signal_loss=float(per_channel[0, c].detach()),
            risk_dot_signal_gradient=float((risk_gradient*gradient).sum().detach()),
            unprojected_first_order_risk_change=float((-.02*(risk_gradient*gradient).sum()/norm).detach()))
            for c, (name, gradient) in enumerate(zip(io23.RAW23, gradients))]
        row = dict(shot=sid, label=int(shot['is_disrupt']), origin=origin, query=query,
            truth_last_index=origin+5, actual_risk_change=actual,
            signal_risk_cosine=float(F.cosine_similarity(risk_gradient, total, dim=0).detach()),
            projected_first_order_risk_change=float((risk_gradient*(updated-delta)).sum().detach()),
            before_delta_norm=float(delta.detach().norm()), after_delta_norm=float(updated.detach().norm()),
            gradient_decomposition_verified=True, causal_state_parity_verified=True,
            channels=sorted(channels, key=lambda r: r['unprojected_first_order_risk_change'], reverse=True))
        if args.counterfactual:
            index = io23.RAW23.index('mode_number_n')
            controls = {}
            for name, gradient in [('without_mode_number_n', total-gradients[index]),
                                   ('only_mode_number_n', gradients[index])]:
                changed = normalized_step(delta[None], torch.zeros_like(delta)[None], gradient[None], step=.02, radius=1.)[0]
                z = model(qx, qa, changed)['logit'].reshape(())
                controls[name] = dict(risk_change=float((F.softplus(sign*(z-boundary))-risk).detach()),
                    delta_step_norm=float((changed-delta).detach().norm()))
            row['single_update_controls'] = controls
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'channels'}), flush=True)
    _durable_json(output, dict(source_only=True, posthoc=True, cases_selected_after_class_audit=list(ids),
        candidate_sha256=_hash(args.run/'candidate_060.pt'), rows=rows,
        interpretation='local derivative explanation, not a policy, feature exclusion rule or physical mechanism proof'))


if __name__ == '__main__':
    main()
