"""Posthoc geometry of state feedback at already identified EAST failures.

This is not an online policy or an achievable rescue bound. At a fixed fast
state, any differentiable scalar loss of the same25 predicted state values
has a parameter gradient in their Jacobian row span. A learned loss metric
can change the direction inside that span, not the span itself. Nonlinear
finite updates, radial projection, or a changed teacher are outside this
local statement. Future query windows and outcomes are forensic input only.
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
from .event_dynamics import EventDynamicsFfnModel, arrived_windows, forecast_windows
from .event_dynamics_experiment import load_coordinates
from .redesign_20261004 import intra
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .simple_online import _durable_json, _hash


def gradient_geometry(jacobian, query_gradient, *, rtol):
    if (jacobian.ndim != 2 or query_gradient.ndim != 1 or not jacobian.shape[0]
            or jacobian.shape[1] != query_gradient.numel() or not 0 < rtol < 1
            or not torch.isfinite(jacobian).all() or not torch.isfinite(query_gradient).all()):
        raise ValueError('finite nonempty coordinate Jacobian and aligned query gradient required')
    j, g = jacobian.detach().cpu().double(), query_gradient.detach().cpu().double()
    _, singular, vh = torch.linalg.svd(j, full_matrices=False)
    keep = singular > singular[0] * rtol
    component = vh[keep] @ g
    norm = float(g.norm())
    fraction = float(component.norm()) / norm if norm > 0 else None
    return dict(rtol=rtol, rank=int(keep.sum()), query_gradient_norm=norm,
        projected_gradient_fraction=fraction,
        projected_gradient_energy_fraction=None if fraction is None else fraction**2,
        singular_values=singular.tolist())


def measure_geometry(model, x, a, truth, mask, qx, qa, fast, *, save_path=None):
    if x.shape[0] != 1 or qx.shape[0] != 1:
        raise ValueError('one forensic episode at a time')
    current = fast.detach().clone().requires_grad_(True)
    actual, actual_mask, common = arrived_windows(x, a, truth, mask, model.center)
    if not common.any():
        raise ValueError('no observed state supervision; geometry undefined for this episode')
    target = model.coordinates(actual, actual_mask).detach()
    forecast = model(x, a, current)['forecast']
    virtual, vm = forecast_windows(x, a, forecast, common, model.center, model.scale)
    states = model.coordinates(virtual, vm)
    loss = F.smooth_l1_loss(states, target)
    coefficient, = torch.autograd.grad(loss, states, retain_graph=True)
    state_gradient, = torch.autograd.grad(loss, current, retain_graph=True)
    jacobian = torch.stack([torch.autograd.grad(s, current, retain_graph=True)[0].flatten()
                            for s in states.flatten()]).detach()
    logit = model(qx, qa, current)['logit'].sum()
    query_gradient, = torch.autograd.grad(logit, current)
    j, g, qg = (t.detach().cpu().double() for t in (jacobian, state_gradient.flatten(), query_gradient.flatten()))
    reconstructed = j.T @ coefficient.detach().cpu().double().flatten()
    norm = float(g.norm())
    relative = float((g - reconstructed).norm()) / norm if norm > 0 else None
    if relative is not None and relative > 1e-4:
        raise AssertionError('coordinate-chain gradient does not reconstruct actual loss gradient')
    denominator = float(g.norm() * qg.norm())
    # Descent direction is -gradient. Positive cosine locally raises logit.
    cosine = -float(g @ qg) / denominator if denominator > 0 else None
    if save_path is not None:
        with Path(save_path).open('xb') as handle:
            np.savez_compressed(handle, jacobian=jacobian.cpu().numpy(),
                query_gradient=query_gradient.detach().cpu().numpy().flatten(),
                state_gradient=state_gradient.detach().cpu().numpy().flatten(),
                coefficients=coefficient.detach().cpu().numpy().flatten(),
                fast=fast.detach().cpu().numpy())
    return dict(state_coordinates=states.numel(), temporary_parameters=current.numel(),
        state_loss_before=float(loss.detach()), query_logit=float(logit.detach()),
        state_gradient_norm=norm, chain_rule_relative_error=relative,
        state_descent_to_logit_increase_cosine=cosine,
        rank_sensitivity=[gradient_geometry(j, qg, rtol=t) for t in (1e-6, 1e-4, 1e-3)],
        query_after_maturity=True, deployment_policy=False,
        matrices_sha256=None if save_path is None else _hash(save_path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('new diagnostic directory required')
    verified = json.loads((args.root / 'independent_verification.json').read_text())
    prior = json.loads((args.root / 'step_diagnostic/completed.json').read_text())
    assert verified['verification'] == 'PASS' and prior['all_curves_exact']
    # Every previously identified lost timely TP, no selection by this result.
    cases = [r['shot'] for r in prior['rows'] if r['is_disrupt']]
    assert set(cases) == {71310, 96506, 75238, 60557, 66437, 78891, 64413}
    order = json.loads((args.root / 'event/order.json').read_text())
    ids = [r['shot'] for r in order]
    torch.set_num_threads(4)
    base, parent = load_parent(DEFAULT_PARENT)
    model = EventDynamicsFfnModel(base, *load_coordinates('event'), mode='event').cuda().eval().requires_grad_(False)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    prefix = [apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal')) for sid in ids[:30]]
    stats = fit_ended_shot_statistics(prefix, ended_ids=ids[:30])
    folder = args.root / 'event/stream'
    provider = EndedDensityReplay(parent['source_center'], parent['source_scale'],
        output_dir=folder, order=ids, preserve_observed_channels=True)
    args.output.mkdir(parents=True)
    protocol = dict(scope='POSTHOC_LOCAL_GRADIENT_SPAN_NOT_NEW_POLICY', cases=cases,
        selection='all lost timely TP of event-state loss versus static; accepted steps with 30<lead<=125 ms',
        parent_sha256=PARENT_SHA256, prior_diagnostic_sha256=_hash(args.root / 'step_diagnostic/completed.json'),
        code_sha256=_hash(Path(__file__)), unchanged_online_update=True, source_model_changed=False,
        rank_rtol_primary=1e-4, rank_rtol_sensitivity=[1e-6, 1e-3],
        future_query_used_only_for_diagnostic=True, outcomes_used_only_for_posthoc_selection=True,
        inference_scope='local first-order sensitivity; not an achievable rescue bound or a population estimate')
    _durable_json(args.output / 'protocol.json', protocol)
    original_attempt, rows = intra._attempt, []
    for sid in cases:
        index = ids.index(sid) + 1
        seal = json.loads((folder / f'{index:04d}_{sid}_seal.json').read_text())
        path = folder / seal['npz']
        assert _hash(path) == seal['sha256']
        with np.load(path, allow_pickle=False) as f:
            expected = f['scores'].copy()
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        provider.statistics = stats if index > 30 else None
        shot = provider.reencode(raw)
        outcome = simple_data.attach_outcome(shot, 'EAST', {})
        events = []

        def explained(m, pending, signals, times, fast, step, radius, objective, device, previous=None):
            updated, receipt = original_attempt(m, pending, signals, times, fast, step, radius,
                                               objective, device, previous=previous)
            q = receipt['maturity_index'] + 1
            lead = (outcome['down'] - times[q]) * 1000 if q < len(times) else None
            if receipt['accepted'] and lead is not None and 30 < lead <= 125 and outcome['valid'][q]:
                _, x, a, _ = pending
                start, end = receipt['origin_index'] + 1, receipt['maturity_index'] + 1
                truth = torch.as_tensor(signals['values'][start:end], device=device, dtype=m.center.dtype)[None]
                mask = torch.as_tensor(signals['available'][start:end], device=device)[None] & torch.isfinite(truth)
                truth = torch.where(mask, truth, m.center)
                qx, qa = intra._batch_windows(simple_data, signals, [q], m, device)
                result = measure_geometry(m, x, a, truth, mask, qx, qa, fast,
                    save_path=args.output / f'{sid}_{q}_matrices.npz')
                assert result['state_loss_before'] == receipt['loss_before']
                result.update(origin=receipt['origin_index'], query=q, lead_ms=float(lead),
                    current_fast_sha256=m.summarize_fast(fast)['sha256'],
                    accepted_fast_sha256=m.summarize_fast(updated)['sha256'])
                events.append(result)
            return updated, receipt

        with patch.object(intra, '_attempt', explained):
            scores, _ = intra.predict_shot(model, shot, intra=True, device='cuda')
        error = float(np.max(np.abs(scores - expected)))
        assert error == 0
        expected_events = [e for r in prior['rows'] if r['shot'] == sid for e in r['events']
                           if e['lead_ms'] is not None and 30 < e['lead_ms'] <= 125]
        assert [e['query'] for e in events] == [e['query'] for e in expected_events]
        assert [e['accepted_fast_sha256'] for e in events] == [e['candidate_sha256'] for e in expected_events]
        row = dict(shot=sid, replay_curve_max_error=error, events=events)
        rows.append(row)
        _durable_json(args.output / f'{sid}.json', row)
        print(json.dumps(dict(shot=sid, events=len(events), fractions=[e['rank_sensitivity'][1]['projected_gradient_fraction']
            for e in events], descent_cosines=[e['state_descent_to_logit_increase_cosine'] for e in events])), flush=True)
    assert _hash(DEFAULT_PARENT) == PARENT_SHA256
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v.detach().cpu(), initial[k], rtol=0, atol=0)
    _durable_json(args.output / 'completed.json', dict(protocol=protocol, rows=rows,
        all_curves_exact=True, registered_tensors_unchanged=True,
        interpretation='selected local geometry, not a deployable update or independent test'))


if __name__ == '__main__': main()
