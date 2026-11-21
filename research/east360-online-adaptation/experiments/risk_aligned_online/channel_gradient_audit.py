"""Post-hoc channel-gradient interventions on an archived EAST intra replay.

Labels enter ONLY this forensic tool. It neither trains a rule nor changes an
online acceptance decision. Local derivatives do not establish plasma causes.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from . import io23, simple_data
from .causal_input import apply_input_contract
from .ended_density_replay import EndedDensityReplay
from .fast_update import normalized_step
from .gradient_transport import TransportFfnModel
from .redesign_20261004 import intra
from .redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from .simple_online import _durable_json, _hash


def channel_losses(prediction, truth, available, center, scale):
    """23 additive contributions to EXACT original global masked SmoothL1.

    Divide by all valid time/channel pairs, not separately by each channel.
    That distinction matters when sensor availability differs.
    """
    if (prediction.ndim != 3 or prediction.shape[-1] != 23
            or truth.shape != prediction.shape or available.shape != prediction.shape
            or available.dtype != torch.bool or center.shape != (23,) or scale.shape != (23,)
            or not torch.isfinite(center).all() or not torch.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError('finite source coordinates and matching flat23 forecasts required')
    available = available & torch.isfinite(truth)
    if not torch.isfinite(prediction[available]).all():
        raise FloatingPointError('nonfinite forecast on an observed channel')
    target = torch.where(available, truth.detach(), center)
    target = torch.asinh((target-center)/scale)
    prediction = torch.where(available, prediction, 0.)
    loss = F.smooth_l1_loss(prediction, target, reduction='none')
    return loss.masked_fill(~available, 0.).sum((0, 1))/available.sum().clamp_min(1)


def component_gradients(pieces, fast):
    if pieces.shape != (23,) or not fast.requires_grad:
        raise ValueError('23 differentiable components and gradient-carrying state required')
    gradients = torch.stack([torch.autograd.grad(pieces[c], fast, retain_graph=True)[0].reshape(-1)
                             for c in range(23)])
    total, = torch.autograd.grad(pieces.sum(), fast)
    torch.testing.assert_close(gradients.sum(0), total.reshape(-1), rtol=3e-4, atol=3e-7)
    if not torch.isfinite(gradients).all():
        raise FloatingPointError('nonfinite channel gradient')
    return gradients.detach()


def measure_event(model, x, a, truth, observed, qx, qa, fast, label, *, step=.05, radius=1.):
    """Same-state one-step comparison, including explicitly label-aware oracles.

    The actual candidate ALWAYS uses the original signal-only update. All
    alternate candidates are discarded and never affect a replay trajectory.
    """
    if label not in (0, 1) or fast.shape != (1, model.temporary_dim):
        raise ValueError('binary diagnostic label and one-shot FFN state required')
    current = fast.detach().clone().requires_grad_(True)
    output = model(x, a, current)
    parts = channel_losses(output['forecast'], truth, observed, model.center, model.scale)
    signal_before = model.forecast_loss(output['forecast'], truth, observed)
    torch.testing.assert_close(parts.sum(), signal_before, rtol=1e-6, atol=1e-7)
    part_values = parts.detach()
    gradients = component_gradients(parts, current)
    total = gradients.sum(0)[None]
    # Recompute the original total gradient too: summing channel derivatives
    # changes float reduction order. Actual trajectory identity uses THIS one.
    original_loss = model.forecast_loss(model(x, a, current)['forecast'], truth, observed)
    original_gradient, = torch.autograd.grad(original_loss, current)
    torch.testing.assert_close(total, original_gradient, rtol=3e-4, atol=3e-7)
    logit = model(qx, qa, current)['logit']
    target = logit.new_full(logit.shape, float(label))
    risk_before = F.binary_cross_entropy_with_logits(logit, target)
    risk_gradient, = torch.autograd.grad(risk_before, current)
    risk_gradient = risk_gradient.detach()
    probability_before = float(logit.detach().sigmoid()[0])
    valid = observed & torch.isfinite(truth)

    def probe(gradient, *, use_transport=True):
        transform = getattr(model, 'intra_gradient_transform', None) if use_transport else None
        used = gradient if transform is None else transform(gradient)
        candidate = normalized_step(current.detach(), torch.zeros_like(current), used,
                                    step=step, radius=radius).detach()
        with torch.no_grad():
            future = model(qx, qa, candidate)['logit']
            risk_after = F.binary_cross_entropy_with_logits(future, target)
            forecast = model(x, a, candidate)['forecast']
            signal_after = model.forecast_loss(forecast, truth, observed)
        gain = float(signal_before.detach()-signal_after)
        tolerance = 8*torch.finfo(signal_before.dtype).eps*max(
            abs(float(signal_before.detach())), abs(float(signal_after)), 1e-12)
        difference = candidate-current.detach()
        length = difference.double().norm()
        rn = risk_gradient.double().norm()
        cosine = -(risk_gradient.double()*difference.double()).sum()/(rn*length).clamp_min(1e-20)
        return dict(probability_after=float(future.sigmoid()[0]),
                    probability_change=float(future.sigmoid()[0])-probability_before,
                    risk_after=float(risk_after), risk_change=float(risk_after-risk_before.detach()),
                    projected_first_order_risk_change=float((risk_gradient*difference).sum()),
                    descent_risk_cosine=float(cosine), signal_after=float(signal_after),
                    signal_gain=gain, would_pass_original_signal_acceptance=bool(gain > tolerance),
                    step_norm=float(length), candidate_sha256=model.summarize_fast(candidate)['sha256'])

    actual = probe(original_gradient.detach())
    channels = [dict(feature=name, observed_points=int(valid[:, :, c].sum()),
                     signal_loss=float(part_values[c]), gradient_norm=float(gradients[c].double().norm()),
                     raw_signal_risk_dot=float((gradients[c].double()*risk_gradient.reshape(-1).double()).sum()))
                for c, name in enumerate(io23.RAW23)]
    omissions = [dict(omitted=io23.RAW23[c], **probe(original_gradient.detach()-gradients[c][None]))
                 for c in range(23) if bool(valid[:, :, c].any())]
    # This answers only whether bounded channel emphasis CAN change this
    # local direction. It uses the hidden label and is NOT a usable policy.
    dots = gradients.double() @ risk_gradient.reshape(-1).double()
    weights = torch.where(dots > 0, 2., .25).to(gradients)
    weighted_gradient = (weights[:, None]*gradients).sum(0)[None]
    bounded_oracle = probe(weighted_gradient)
    return dict(offline_oracle=True, gradient_decomposition_verified=True,
                probability_before=probability_before, risk_before=float(risk_before.detach()),
                signal_before=float(signal_before.detach()), actual=actual, channels=channels,
                best_leave_one_out=min(omissions, key=lambda r: r['risk_after']),
                bounded_weight_oracle=dict(weights=weights.cpu().tolist(), **bounded_oracle),
                plain_signal_control=probe(original_gradient.detach(), use_transport=False),
                plain_bounded_weight_oracle=probe(weighted_gradient, use_transport=False),
                interpretation='single-update derivative/intervention, not an online policy or physical cause')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', type=int, nargs='+', default=[66437, 78923, 95165, 60557, 56606, 85711])
    args = p.parse_args()
    if args.output.exists():
        p.error('new diagnostic output required')
    torch.set_num_threads(4)
    protocol = json.loads((args.run/'protocol.json').read_text())
    if (protocol['source_sha256'] != PARENT_SHA256 or protocol['internal_site'] != 'gradient_transport'
            or protocol['intra']['objective'] != 'level' or not protocol['preserve_observed_channels']):
        raise ValueError('declared original gradient-transport, masked-density intra replay required')
    order = json.loads((args.run/'order.json').read_text())
    ids = [r['shot'] for r in order]
    if not set(args.cases) <= set(ids) or len(set(args.cases)) != len(args.cases):
        raise ValueError('distinct cases in the original locked order required')
    statistics_path = args.run/'static/alignment.json'
    statistics = json.loads(statistics_path.read_text())
    if statistics['calibration_ids'] != ids[:30]:
        raise ValueError('statistics must belong to the same ended calibration prefix')
    statistics = {k: np.asarray(v, bool if k == 'usable' else np.float32)
                  if k in ('center', 'scale', 'usable') else v for k, v in statistics.items()}
    operator_path = args.run/'source_gradient_operator.pt'
    if _hash(operator_path) != protocol['source_gradient_transport']['artifact_sha256']:
        raise ValueError('archived operator identity mismatch')
    asset = torch.load(operator_path, map_location='cpu', weights_only=True)
    base, state = load_parent(DEFAULT_PARENT)
    model = TransportFfnModel(base, asset['keys'], asset['coefficients']).to('cuda').eval()
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    provider = EndedDensityReplay(state['source_center'], state['source_scale'],
        output_dir=args.run/'intra', order=ids, preserve_observed_channels=True)
    args.output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), Path(__file__).parent/'gradient_transport.py',
             Path(__file__).parent/'persistent_ffn.py', Path(intra.__file__),
             Path(__file__).parent/'simple23.py', Path(__file__).parent/'fast_update.py']
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, file.relative_to(Path(__file__).parent))
    diagnostic_protocol = dict(scope='POSTHOC_SELECTED_ERROR_MECHANISM_DIAGNOSTIC',
        parent_sha256=PARENT_SHA256, original_run=str(args.run.resolve()), cases=args.cases,
        original_protocol_sha256=_hash(args.run/'protocol.json'),
        source_operator_sha256=_hash(operator_path), statistics_sha256=_hash(statistics_path),
        code_snapshot_sha256=_hash(args.output/'code_snapshot.zip'),
        threshold=protocol['threshold'], horizon=25, stride=100, step=.05, radius=1.,
        online_updates_changed=False, new_model_trained=False, alarm_changed=False,
        current_outcomes_used_only_posthoc=True, population_harm_rate=False,
        oracle_rule='label-aware bounded .25/2 weights and best single omitted channel; NOT deployed')
    _durable_json(args.output/'protocol.json', diagnostic_protocol)
    rows = []
    original_attempt = intra._attempt
    for sid in args.cases:
        index = ids.index(sid)+1
        seal_path = args.run/'intra'/f'{index:04d}_{sid}_seal.json'
        seal = json.loads(seal_path.read_text())
        trace_path = args.run/'intra'/seal['npz']
        if _hash(trace_path) != seal['sha256']:
            raise ValueError('archived whole-shot seal changed')
        with np.load(trace_path, allow_pickle=False) as trace:
            expected_times, expected_scores = trace['times'].copy(), trace['scores'].copy()
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        if not np.array_equal(raw['times'], expected_times):
            raise ValueError('diagnostic signal times differ from sealed trajectory')
        # Explicit forensic state reconstruction, not an online fitting call.
        provider.statistics = statistics if index > 30 else None
        shot = provider.reencode(raw)
        outcome = simple_data.attach_outcome(shot, 'EAST', {})
        events = []

        def explained_attempt(model, pending, signals, times, fast, step, radius, objective, device, previous=None):
            updated, receipt = original_attempt(model, pending, signals, times, fast,
                step, radius, objective, device, previous=previous)
            query = receipt['maturity_index']+1
            if receipt['accepted'] and query < len(times) and bool(outcome['valid'][query]):
                _, x, a, _ = pending
                origin, maturity = receipt['origin_index'], receipt['maturity_index']
                truth = torch.as_tensor(signals['values'][origin+1:maturity+1],
                    dtype=model.center.dtype, device=device)[None]
                observed = torch.as_tensor(signals['available'][origin+1:maturity+1], device=device)[None]
                qx, qa = intra._batch_windows(simple_data, signals, [query], model, device)
                explanation = measure_event(model, x, a, truth, observed, qx, qa, fast,
                                            int(outcome['y'][query]), step=step, radius=radius)
                if explanation['actual']['candidate_sha256'] != model.summarize_fast(updated)['sha256']:
                    raise ValueError('explanation candidate differs from actual accepted update')
                lead = None if not outcome['is_disrupt'] else round((outcome['down']-times[query])*1000., 9)
                explanation.update(origin=origin, query=query, lead_ms=lead,
                                   point_label=int(outcome['y'][query]), causal_candidate_parity=True)
                if outcome['is_disrupt'] and lead > 125:
                    # A negative point label wants risk DOWN here, although
                    # a timely first alarm on this D shot counts as detection.
                    explanation['early_D_objective_conflict'] = True
                events.append(explanation)
            return updated, receipt

        with patch.object(intra, '_attempt', explained_attempt):
            scores, audit = intra.predict_shot(model, shot, intra=True, device='cuda')
        error = float(np.max(np.abs(scores-expected_scores)))
        if error != 0.:
            raise ValueError(f'forensic replay differs from original sealed scores: {error}')
        row = dict(shot=sid, is_disrupt=bool(outcome['is_disrupt']), sealed_trace_sha256=seal['sha256'],
                   replay_curve_max_error=error, accepted_updates=audit['accepted_updates'], events=events)
        rows.append(row)
        _durable_json(args.output/f'{sid}_channels.json', row)
        print(json.dumps(dict(shot=sid, accepted=audit['accepted_updates'], measured=len(events),
            harmful=sum(e['actual']['risk_change'] > 0 for e in events), curve_error=error)), flush=True)
    if _hash(DEFAULT_PARENT) != PARENT_SHA256 or any(
            not torch.equal(v.detach().cpu(), frozen[k]) for k, v in model.state_dict().items()):
        raise ValueError('forensic replay changed source or persistent state')
    _durable_json(args.output/'summary.json', dict(protocol=diagnostic_protocol, rows=rows,
        classifier_unchanged=True, persistent_unchanged=True,
        interpretation='selected local counterexamples; not an EAST performance improvement'))


if __name__ == '__main__':
    main()
