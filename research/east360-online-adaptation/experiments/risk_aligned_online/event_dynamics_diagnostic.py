"""Posthoc local counterfactuals on completed state-loss error trajectories.

Outcome labels are available ONLY in this forensic entry point. They neither
choose nor alter an online update. Every full replay must match its saved
curve exactly. Selected errors cannot estimate a population harm frequency.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import simple_data
from .alignment import fit_ended_shot_statistics
from .causal_input import apply_input_contract
from .ended_density_replay import EndedDensityReplay
from .event_dynamics import EventDynamicsFfnModel
from .event_dynamics_experiment import load_coordinates
from .fast_update import normalized_step
from .redesign_20261004 import intra
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .simple_online import _durable_json, _hash


def measure_step(model, x, a, truth, observed, qx, qa, fast, label):
    if label not in (0, 1):
        raise ValueError('explicit posthoc binary outcome required')
    current = fast.detach().clone().requires_grad_(True)
    output = model(x, a, current)
    loss, _ = model.prepare_matured_forecast_loss(output['forecast'].detach(), truth, observed,
                                                values=x, observed=a)
    before = loss(output['forecast'], truth, observed)
    gradient, = torch.autograd.grad(before, current)
    candidate = normalized_step(current.detach(), torch.zeros_like(current), gradient.detach(),
                                step=.05, radius=1.).detach()
    logit = model(qx, qa, current)['logit']
    target = logit.new_full(logit.shape, float(label))
    risk_before = F.binary_cross_entropy_with_logits(logit, target)
    risk_gradient, = torch.autograd.grad(risk_before, current)
    with torch.no_grad():
        after_output = model(x, a, candidate)
        after = loss(after_output['forecast'], truth, observed)
        next_logit = model(qx, qa, candidate)['logit']
        risk_after = F.binary_cross_entropy_with_logits(next_logit, target)
        raw_before = model.forecast_loss(output['forecast'].detach(), truth, observed)
        raw_after = model.forecast_loss(after_output['forecast'], truth, observed)
    delta = (candidate - current.detach()).double()
    rg = risk_gradient.detach().double()
    cosine = -(rg * delta).sum() / (rg.norm() * delta.norm()).clamp_min(1e-20)
    return dict(offline_outcome_used=True, state_loss_before=float(before.detach()),
        state_loss_after=float(after), raw_signal_loss_before=float(raw_before),
        raw_signal_loss_after=float(raw_after), probability_before=float(logit.detach().sigmoid()[0]),
        probability_after=float(next_logit.sigmoid()[0]), risk_bce_before=float(risk_before.detach()),
        risk_bce_after=float(risk_after), risk_bce_change=float(risk_after - risk_before.detach()),
        risk_descent_cosine=float(cosine), gradient_norm=float(gradient.detach().double().norm()),
        step_norm=float(delta.norm()), candidate_sha256=model.summarize_fast(candidate)['sha256']), candidate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', type=int, nargs='+', required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('preserve existing diagnostic; new output required')
    verification = json.loads((args.root / 'independent_verification.json').read_text())
    if verification['verification'] != 'PASS': raise ValueError('completed verified pilot required')
    order = json.loads((args.root / 'event/order.json').read_text())
    ids = [r['shot'] for r in order]
    if len(set(args.cases)) != len(args.cases) or not set(args.cases) <= set(ids):
        raise ValueError('distinct diagnostic cases in locked stream required')
    torch.set_num_threads(4)
    base, parent = load_parent(DEFAULT_PARENT)
    model = EventDynamicsFfnModel(base, *load_coordinates('event'), mode='event').to('cuda').eval()
    model.requires_grad_(False)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    prefix = [apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal')) for sid in ids[:30]]
    stats = fit_ended_shot_statistics(prefix, ended_ids=ids[:30])
    provider = EndedDensityReplay(parent['source_center'], parent['source_scale'],
        output_dir=args.root / 'event/stream', order=ids, preserve_observed_channels=True)
    args.output.mkdir(parents=True)
    protocol = dict(scope='POSTHOC_SELECTED_FLIP_MECHANISM_DIAGNOSTIC', cases=args.cases,
        parent_sha256=PARENT_SHA256, verification_sha256=_hash(args.root / 'independent_verification.json'),
        code_sha256=_hash(Path(__file__)), labels_used_only_after_experiment=True,
        outcome_target='shot outcome, at valid next endpoints with lead>30ms for D',
        current_point125_labels_not_used=True, updated_model_or_alarm=False,
        population_harm_rate=False, selection='all seven lost TP and three removed FP versus static')
    _durable_json(args.output / 'protocol.json', protocol)
    original_attempt = intra._attempt
    rows = []
    for sid in args.cases:
        index = ids.index(sid) + 1
        folder = args.root / 'event/stream'
        seal = json.loads((folder / f'{index:04d}_{sid}_seal.json').read_text())
        path = folder / seal['npz']
        if _hash(path) != seal['sha256']: raise ValueError('sealed curve changed')
        with np.load(path, allow_pickle=False) as f:
            expected_scores, expected_times = f['scores'].copy(), f['times'].copy()
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        provider.statistics = stats if index > 30 else None
        shot = provider.reencode(raw)
        outcome = simple_data.attach_outcome(shot, 'EAST', {})
        assert np.array_equal(expected_times, shot['times'])
        events = []

        def explained(m, pending, signals, times, fast, step, radius, objective, device, previous=None):
            updated, receipt = original_attempt(m, pending, signals, times, fast, step, radius,
                                                objective, device, previous=previous)
            q = receipt['maturity_index'] + 1
            lead = ((outcome['down'] - times[q]) * 1000
                    if q < len(times) and outcome['is_disrupt'] else None)
            if receipt['accepted'] and q < len(times) and outcome['valid'][q] and (lead is None or lead > 30):
                _, x, a, _ = pending
                start, end = receipt['origin_index'] + 1, receipt['maturity_index'] + 1
                truth = torch.as_tensor(signals['values'][start:end], device=device, dtype=m.center.dtype)[None]
                mask = torch.as_tensor(signals['available'][start:end], device=device)[None]
                qx, qa = intra._batch_windows(simple_data, signals, [q], m, device)
                result, candidate = measure_step(m, x, a, truth, mask, qx, qa, fast, int(outcome['is_disrupt']))
                if result['candidate_sha256'] != m.summarize_fast(updated)['sha256']:
                    raise AssertionError('diagnostic candidate differs from actual accepted update')
                result.update(origin=receipt['origin_index'], query=q, query_time=float(times[q]),
                              lead_ms=None if lead is None else float(lead), candidate_parity=True)
                events.append(result)
            return updated, receipt

        with patch.object(intra, '_attempt', explained):
            scores, audit = intra.predict_shot(model, shot, intra=True, device='cuda')
        delta = float(np.max(np.abs(scores - expected_scores)))
        if delta != 0: raise AssertionError('forensic replay changed full risk trajectory')
        row = dict(shot=sid, is_disrupt=bool(outcome['is_disrupt']), replay_curve_max_error=delta,
            accepted_updates=audit['accepted_updates'], events=events)
        rows.append(row)
        _durable_json(args.output / f'{sid}.json', row)
        print(json.dumps(dict(shot=sid, measured=len(events),
            locally_harmful=sum(e['risk_bce_change'] > 0 for e in events), curve_error=delta)), flush=True)
    assert _hash(DEFAULT_PARENT) == PARENT_SHA256
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v.detach().cpu(), initial[k], rtol=0, atol=0)
    _durable_json(args.output / 'completed.json', dict(protocol=protocol, rows=rows,
        source_and_teacher_unchanged=True, all_curves_exact=True,
        interpretation='selected local interventions; not population risk or a new policy'))


if __name__ == '__main__':
    main()
