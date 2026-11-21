"""Explicit prefix-only density calibration control, not a new online policy.

Use at most ten optimizer proposals on ONE already-arrived H25 episode, then
freeze the disposable bias for the rest of the shot. This differs in timing
and optimization budget from continuous adaptation and is not an equal-budget
mechanism attribution. Original source tensors, thresholds and risk head stay
unchanged; no target outcome is read by this predictor.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import torch

from . import simple_data
from .cross_density import CrossDensityModel, load_head, snapshot
from .redesign_20261004.intra import (
    _attempt, _batch_windows, _finite, _output, _record, _state_changes, _vector,
)
from .redesign_20261004.run import (
    DEFAULT_SOURCE, DEFAULT_ORDER, DEFAULT_ROSTER, load_parent, load_roster,
    validate_order, validate_roster_rate_contract, run_arm,
)
from .simple_online import _durable_json, _hash


def predict_prefix(model, shot, intra, device='cpu', stride=25, step=1.,
                   radius=12., objective='forward_holdout', data_api=simple_data):
    if (not isinstance(model, CrossDensityModel) or stride != 25 or step != 1.
            or radius != 12. or objective != 'forward_holdout'):
        raise ValueError('fixed cross-density prefix control contract required')
    if shot.get('device', 'EAST') != 'EAST':
        raise ValueError('online prefix control supports EAST only')
    signals = dict(values=np.asarray(shot['values']), available=np.asarray(shot['available']))
    times = np.asarray(shot['times'])
    size = len(signals['values'])
    if (signals['values'].shape != (size, 23)
            or signals['available'].shape != signals['values'].shape
            or signals['available'].dtype != bool or times.shape != (size,)
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError('flat23 signals, boolean availability and increasing times required')
    modes = [(m, m.training) for m in model.modules()]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    parameters = {n: p.detach().clone() for n, p in model.named_parameters()}
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    scores = np.empty(size, np.float32)
    audit = dict(intra=bool(intra), objective=objective, horizon=25, stride=25,
        step=step, radius=radius, score_count=size, attempts=[], pending=[],
        fast_parameter_count=1, fast_discarded=True, fast_frozen_after_prefix=True,
        updated_signal_origins=0, maximum_prefix_proposals=10,
        forecast_fit_steps=[1,12], forecast_holdout_steps=[13,25],
        holdout_used_for_gradient=False, holdout_is_independent_risk_validation=False,
        control='one matured prefix episode; up to ten proposals; then hold bias',
        equal_budget_to_continuous=False)
    try:
        model.eval()
        for p, _ in flags: p.requires_grad_(False)
        with torch.inference_mode(False), torch.enable_grad():
            fast = model.initial_fast(1).detach().clone().to(device).requires_grad_(True)
            # Emit indices 0..25 before seeing their signals as training targets.
            stop = min(size, 26)
            if stop:
                values, seen = _batch_windows(data_api, signals, range(stop), model, device)
                out = _output(model, values, seen, fast.expand(stop, -1))
                if not _finite(out['logit']): raise FloatingPointError('nonfinite prefix risk')
                scores[:stop] = out['logit'].detach().sigmoid().cpu().numpy()
            if intra and size > 25:
                pending = (_record(0, times, fast), values[:1].detach().clone(),
                           seen[:1].detach().clone(), out['forecast'][:1].detach().clone())
                for iteration in range(10):
                    # A new record for each proposal; the sealed origin forecast
                    # is never overwritten with a retrospectively fitted one.
                    proposal = (deepcopy(pending[0]), *pending[1:])
                    fast, record = _attempt(model, proposal, signals, times, fast,
                        step, radius, objective, device)
                    record['prefix_optimization_step'] = iteration + 1
                    audit['attempts'].append(record)
                    if not record['accepted']: break
                audit['updated_signal_origins'] = 1
            for start in range(stop, size, 256):
                end = min(size, start + 256)
                values, seen = _batch_windows(data_api, signals, range(start, end), model, device)
                out = _output(model, values, seen, fast.expand(end-start, -1))
                if not _finite(out['logit']): raise FloatingPointError('nonfinite postprefix risk')
                scores[start:end] = out['logit'].detach().sigmoid().cpu().numpy()
            audit['final_fast'] = _vector(fast)
        audit['attempted_updates'] = len(audit['attempts'])
        audit['accepted_updates'] = sum(r['accepted'] for r in audit['attempts'])
        (pn, changed_p), (bn, changed_b) = _state_changes(model, parameters, buffers)
        audit.update(model_parameter_change_norm=pn, model_buffer_change_norm=bn)
        if changed_p or changed_b: raise RuntimeError('registered model state changed during prefix control')
        return scores, audit
    finally:
        for p, enabled in flags: p.requires_grad_(enabled)
        for m, training in modes: m.training = training


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--head', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('new output required')
    if not torch.cuda.is_available(): raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(4); torch.manual_seed(131)
    parent, state, gate, _ = load_parent(DEFAULT_SOURCE, 'cuda', persistent_site='embedding')
    model = CrossDensityModel(parent).cuda().eval()
    receipt = load_head(args.head, model)
    screen = json.loads((args.head.parent/'validation.json').read_text())
    if screen.get('source_screen_passed') is not True:
        raise ValueError('source signal screen required before EAST diagnostic')
    original = {k: v.detach().clone() for k, v in parent.state_dict().items()}
    order = json.loads(DEFAULT_ORDER.read_text()); ids, roster = load_roster(DEFAULT_ROSTER)
    validate_order(order, ids); validate_roster_rate_contract(roster, 'nominal')
    args.output.mkdir(parents=True)
    code = snapshot(args.output)
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'a', zipfile.ZIP_DEFLATED) as z:
        z.write(Path(__file__), 'prefix_density_control.py')
    protocol = dict(scope='DEVELOPMENT PREFIX-ONLY NUMERICAL CALIBRATION CONTROL',
        parent_sha256=gate['checkpoint_sha256'], threshold=float(state['threshold']),
        source_head_sha256=_hash(args.head), source_receipt=receipt, selected_shots=120,
        temporary_parameters=1, stride=25, step=1., radius=12., objective='forward_holdout',
        maximum_proposals=10, updated_signal_origins=1, fast_frozen_after_prefix=True,
        equal_budget_to_continuous=False, source_training=False, jtext_online=False,
        target_scaler_fit=False, inter_updates=False, risk_correction_head=False,
        alarm_tuning=False, natural_outcomes=True, support110=False,
        order_sha256=_hash(DEFAULT_ORDER), roster_sha256=_hash(DEFAULT_ROSTER),
        control_code_sha256=_hash(Path(__file__)), code_sha256=code,
        code_snapshot_sha256=_hash(args.output/'code_snapshot.zip'))
    _durable_json(args.output/'protocol.json', protocol)
    _durable_json(args.output/'order.json', order); _durable_json(args.output/'roster.json', roster)
    provider = SimpleNamespace(window=simple_data.window, attach_outcome=simple_data.attach_outcome,
        read_shot=lambda device, sid: simple_data.read_shot(device, sid, rate_contract='nominal'))
    result = run_arm(model, order=order, arm='intra', threshold=float(state['threshold']),
        output_dir=args.output/'prefix_intra', limit=120, device='cuda', stride=25,
        step=1., radius=12., objective='forward_holdout', data_api=provider, predictor=predict_prefix)
    if (any(not torch.equal(parent.state_dict()[k], v) for k, v in original.items())
            or _hash(DEFAULT_SOURCE) != gate['checkpoint_sha256']):
        raise RuntimeError('qualified original source changed')
    _durable_json(args.output/'comparison.json', dict(prefix_intra=result))
    print(json.dumps({k: v for k, v in result['metrics'].items() if k != 'predictions'}), flush=True)


if __name__ == '__main__': main()
