"""Retrospective update-location diagnostic, NOT an online learning method.

True labels deliberately select/optimize seven already declared error endpoints.
An external parameter delta is discarded after each probe. Equal parameter-space
L2 radii across locations are NOT equal function-space budgets. Best visited
points are neither global capacity bounds nor recovered whole-shot predictions.
"""
from __future__ import annotations

import argparse
import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call

from .. import simple_data
from ..causal_input import apply_input_contract
from ..fast_update import normalized_step
from ..simple_online import _hash
from .fast_capacity_probe import (DEFAULT_SHOTS, _snapshot, _check_immutable,
    _threshold, _steps, load_completed_static, select_endpoint, write_report)

SITES = ('fast128', 'embedding', 'encoder_norms', 'last_ffn')
DIAGNOSTIC = 'POSTHOC_ORACLE_INTERNAL_LOCATION'


def parameter_spec(model, site):
    """Only existing input/temporal parameters; neither output head is selected."""
    if site not in SITES or model.cfg.fast_site != 'tokens':
        raise ValueError('a token-adaptation source and declared internal site are required')
    last = model.cfg.layers - 1
    names = []
    for name, _ in model.named_parameters():
        if (site == 'embedding' and name.startswith('embedding.')) or (
            site == 'encoder_norms' and name.startswith('encoder.layers.')
            and ('.norm1.' in name or '.norm2.' in name)) or (
            site == 'last_ffn' and name.startswith(f'encoder.layers.{last}.linear')):
            names.append(name)
    parameters = dict(model.named_parameters())
    return dict(site=site, names=names, count=(2*model.cfg.dim if site == 'fast128'
                 else sum(parameters[n].numel() for n in names)))


def forward_delta(model, spec, delta, values, available):
    if delta.shape != (spec['count'],):
        raise ValueError('flat delta does not match selected internal parameters')
    # Even a late FFN-only delta must use the sealed runner's differentiable
    # attention dispatch. Without this zero-gradient-path leaf, early frozen
    # blocks select fused inference kernels, which failed CUDA zero-delta
    # identity. The leaf is never stepped; only ``delta`` is optimized.
    fast = model.initial_fast(len(values)).detach().requires_grad_(True)
    if spec['site'] == 'fast128':
        return model(values, available, fast + delta[None])
    parameters, replacement, offset = dict(model.named_parameters()), {}, 0
    for name in spec['names']:
        original = parameters[name]
        replacement[name] = original + delta[offset:offset+original.numel()].view_as(original)
        offset += original.numel()
    return functional_call(model, replacement, (values, available, fast), strict=False)


def reference_change(before, after, labels, threshold):
    before, after, labels = [torch.as_tensor(x).detach().cpu().flatten()
                             for x in (before, after, labels)]
    if (not len(labels) or before.shape != labels.shape or after.shape != labels.shape
            or not torch.isfinite(before).all() or not torch.isfinite(after).all()
            or not torch.isin(labels, torch.tensor([0., 1.])).all()):
        raise ValueError('nonempty matching probabilities and binary reference labels required')
    old, new, positive = before >= threshold, after >= threshold, labels.bool()
    return dict(unit='endpoint_not_shot', n_endpoints=len(labels), n_positive=int(positive.sum()),
        n_negative=int((~positive).sum()),
        lost_correct_negative=int((~positive & ~old & new).sum()),
        lost_correct_positive=int((positive & old & ~new).sum()),
        new_correct_negative=int((~positive & old & ~new).sum()),
        new_correct_positive=int((positive & ~old & new).sum()),
        mean_absolute_probability_change=float((after-before).abs().mean()),
        max_absolute_probability_change=float((after-before).abs().max()))


def _reference_predictions(model, spec, delta, values, available):
    # Keep the same differentiable dispatch as endpoint prediction, then detach.
    # This avoids mixing PyTorch's fused inference and gradient encoder paths.
    logits = []
    with torch.enable_grad():
        for start in range(0, len(values), 32):
            d = delta.detach().requires_grad_(True)
            logits.append(forward_delta(model, spec, d, values[start:start+32],
                                        available[start:start+32])['logit'].detach().cpu())
    return torch.cat(logits)


def probe_location(model, values, available, label, threshold, site, *, reference,
                   steps=32, step=.10, radius=1.):
    threshold, steps = _threshold(threshold), _steps(steps)
    if isinstance(label, bool) or not isinstance(label, Integral) or label not in (0, 1):
        raise ValueError('retrospective binary integer label required')
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
           or v <= 0 for v in (step, radius)):
        raise ValueError('positive finite step and radius required')
    if (any(p.requires_grad or p.grad is not None for p in model.parameters())
            or any(m.training for m in model.modules())):
        raise ValueError('all registered parameters must remain frozen, gradient-free and eval')
    spec = parameter_spec(model, site)
    ref_values, ref_available, ref_labels = reference
    for x, a in ((values, available), (ref_values, ref_available)):
        if (x.ndim != 3 or x.shape[1:] != (model.cfg.window, 23) or not len(x)
                or x.dtype != model.center.dtype or x.device != model.center.device
                or a.shape != x.shape or a.dtype != torch.bool or a.device != x.device
                or not a.any() or not torch.isfinite(x[a]).all()):
            raise ValueError('finite observed causal windows and matching masks required')
    if len(values) != 1 or ref_labels.shape != (len(ref_values),):
        raise ValueError('one target endpoint and matching reference labels required')
    reference_change(torch.zeros(len(ref_values)), torch.zeros(len(ref_values)), ref_labels, threshold)
    before = _snapshot(model)
    delta = torch.zeros(spec['count'], device=values.device, dtype=values.dtype, requires_grad=True)
    initial = delta.detach().clone()
    ref_before = _reference_predictions(model, spec, delta, ref_values, ref_available)
    best_delta, best_index, trace = initial, 0, []
    first_crossing_step, first_crossing_delta, first_crossing_reference = None, None, None
    direction = 1 if label else -1
    try:
        with torch.enable_grad():
            for index in range(steps+1):
                logit = forward_delta(model, spec, delta, values.detach(), available)['logit'][0]
                if not torch.isfinite(logit):
                    raise FloatingPointError('nonfinite target risk logit')
                probability = float(logit.detach().sigmoid())
                trace.append(dict(step=index, logit=float(logit.detach()), probability=probability,
                    delta_norm=float(delta.detach().double().norm()),
                    threshold_reached=(probability >= threshold if label else probability < threshold)))
                if first_crossing_step is None and trace[-1]['threshold_reached']:
                    first_crossing_step, first_crossing_delta = index, delta.detach().clone()
                if direction*trace[-1]['logit'] > direction*trace[best_index]['logit']:
                    best_index, best_delta = index, delta.detach().clone()
                if index == steps:
                    break
                gradient, = torch.autograd.grad(-direction*logit, delta)
                trace[-1]['gradient_norm'] = float(gradient.detach().double().norm())
                delta = normalized_step(delta.detach()[None], initial[None], gradient.detach()[None],
                            step=step, radius=radius)[0].detach().requires_grad_(True)
        ref_after = _reference_predictions(model, spec, best_delta, ref_values, ref_available)
        if first_crossing_delta is not None:
            first_ref = _reference_predictions(model, spec, first_crossing_delta, ref_values, ref_available)
            first_crossing_reference = reference_change(ref_before.sigmoid(), first_ref.sigmoid(), ref_labels, threshold)
            first_crossing_reference['mean_absolute_logit_change'] = float((first_ref-ref_before).abs().mean())
    finally:
        immutable = _check_immutable(model, before)
    change = reference_change(ref_before.sigmoid(), ref_after.sigmoid(), ref_labels, threshold)
    change['mean_absolute_logit_change'] = float((ref_after-ref_before).abs().mean())
    change['max_absolute_logit_change'] = float((ref_after-ref_before).abs().max())
    parameters = dict(model.named_parameters())
    parent_norm = (float(torch.cat([parameters[n].detach().flatten() for n in spec['names']]).double().norm())
                   if spec['names'] else None)
    return dict(diagnostic=DIAGNOSTIC, online_performance=False, site=site, label=int(label),
        parameter_count=spec['count'], parameter_names=spec['names'], threshold=threshold,
        steps=steps, step_size=step, radius=radius, parent_selected_parameter_norm=parent_norm,
        initial_probability=trace[0]['probability'], best_probability=trace[best_index]['probability'],
        final_probability=trace[-1]['probability'], best_step=best_index,
        best_threshold_reached=trace[best_index]['threshold_reached'],
        first_crossing_step=first_crossing_step, first_crossing_reference=first_crossing_reference,
        best_reference=change, immutability=immutable, trace=trace)


def _window(shot, end, model):
    prefix = {key: shot[key][:end+1] for key in ('values', 'available')}
    x, a = simple_data.window(prefix, end, model.cfg.window)
    return (torch.as_tensor(x, device=model.center.device, dtype=model.center.dtype),
            torch.as_tensor(a, device=model.center.device, dtype=torch.bool))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'checkpoint', 'prior', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=32)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output exists; diagnostic only creates a new report')
    sealed = load_completed_static(args.run, list(DEFAULT_SHOTS))
    protocol, threshold = sealed['protocol'], sealed['threshold']
    hashes = dict(sealed['input_sha256'])
    for path, digest in ((args.checkpoint, protocol['source_sha256']), (args.prior, protocol['prior_sha256'])):
        if _hash(path) != digest:
            raise ValueError('source/prior differs from sealed parent')
        hashes[str(path)] = digest
    from .run import load_parent
    wrapper, state, gate, _ = load_parent(args.checkpoint, args.device, persistent_site='embedding', prior=args.prior)
    if state['model_config'] != protocol['model_config'] or state['threshold'] != threshold or not gate['qualified']:
        raise ValueError('source config, qualification or threshold changed')
    model = wrapper.base.eval().requires_grad_(False)
    rate_contract = protocol['args']['rate_contract']
    order = json.loads((args.run/'order.json').read_text())
    ref_x, ref_a, ref_y, ref_records = [], [], [], []
    # Fixed first thirty shots, not selected by performance or outcome. Uniform
    # endpoints plus four retrospective timely positive endpoints per D shot.
    # The label-dependent addition is ONLY for this offline collateral audit.
    for row in order[:30]:
        shot = apply_input_contract(simple_data.read_shot('EAST', row['shot'], rate_contract=rate_contract))
        out = simple_data.attach_outcome(shot, 'EAST', {})
        valid = np.flatnonzero(out['valid'])
        ends = set(valid[np.linspace(0, len(valid)-1, min(8, len(valid)), dtype=int)].tolist())
        if out['is_disrupt']:
            lead = np.round((out['down']-out['times'])*1000., 9)
            positive = np.flatnonzero(out['valid'] & (lead > 30.) & (lead <= 125.))
            ends.update(positive[np.linspace(0, len(positive)-1, min(4, len(positive)), dtype=int)].tolist())
        for end in sorted(ends):
            x, a = _window(shot, end, model)
            ref_x.append(x); ref_a.append(a); ref_y.append(out['y'][end])
            ref_records.append(dict(shot=shot['shot'], endpoint=end, label=int(out['y'][end])))
    reference = (torch.stack(ref_x), torch.stack(ref_a), torch.tensor(ref_y))
    points = []
    for record in sealed['records']:
        trace = record['trace']
        shot = apply_input_contract(simple_data.read_shot('EAST', trace['shot'], rate_contract=rate_contract))
        if not np.array_equal(shot['times'], trace['times']):
            raise ValueError('reconstructed timestamps differ from sealed stream')
        point = select_endpoint(trace, simple_data.attach_outcome(shot, 'EAST', {}))
        x, a = _window(shot, point['end_index'], model)
        for site in SITES:
            result = probe_location(model, x[None], a[None], point['label'], threshold, site,
                                    reference=reference, steps=args.steps)
            if abs(result['initial_probability'] - point['sealed_probability']) > 1e-5:
                raise ValueError('zero-delta prediction does not reproduce sealed risk')
            result.update(point)
            points.append(result)
            print(json.dumps(dict(shot=point['shot'], site=site,
                initial=result['initial_probability'], best=result['best_probability'],
                reached=result['best_threshold_reached'], first_step=result['first_crossing_step'],
                first_reference=result['first_crossing_reference'])), flush=True)
    if any(_hash(Path(path)) != digest for path, digest in hashes.items()):
        raise AssertionError('source/prior/sealed files changed')
    write_report(args.output, dict(diagnostic=DIAGNOSTIC, online_performance=False,
        description='best visited true-label-directed endpoint diagnostic, not global bound or shot recovery',
        scope='disposable internal deltas; no saved model, no risk head or alarm changes',
        comparability='equal parameter L2 radius is NOT equal functional capacity across locations',
        reference_selection='first30 locked shots; up to8 uniform valid endpoints plus4 retrospective 30<lead<=125 positives',
        reference_records=ref_records, reference_unit='endpoint_not_shot',
        retrospective_labels_used=True, input_sha256=hashes, points=points))


if __name__ == '__main__':
    main()
