"""Frozen-classifier conditional forecast feasibility on historical CV20 only."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import (shot_weights, misplaced_labels, ridge_fit, ridge_predict,
                       direct_targets, fit_metric, maturity_error)

READ = lambda p: json.loads(p.read_text())
REFERENCE = io23.ROOT/'experiments/risk_aligned_online/runs/increment_forecast_cv20_20261023'
REF_SHA = '3f5e0d0003104b1c54e413badfa5d85147a59b377b566d1d8a532bf20ff23e61'
PLAN = io23.ROOT/'docs/research/source_event_rebuild_20261022/条件演化反馈_可行性约定_20261023.md'
HORIZONS = (5, 10, 25)


def encode(model, signal, ends):
    if set(signal) != {'values', 'available'}:
        raise ValueError('signal-only context required')
    hidden, risk = [], []
    for start in range(0, len(ends), 64):
        windows = [simple_data.window(signal, int(e), 128) for e in ends[start:start+64]]
        x, mask = [torch.as_tensor(np.stack(v), device='cuda') for v in zip(*windows)]
        with torch.no_grad():
            h = model(x, mask, mode='risk_state')['risk_state']
            hidden.append(h.cpu().numpy())
            risk.append(model.base.risk_head(h).squeeze(-1).sigmoid().cpu().numpy())
    return np.concatenate(hidden).astype('float64'), np.concatenate(risk)


def fit_heads(features, target, valid, ids, y):
    weights = shot_weights(ids, y)
    center = np.einsum('n,nd->d', weights, features)
    scale = np.sqrt(np.einsum('n,nd->d', weights, (features-center)**2))
    keep = scale > 64*np.finfo(float).eps
    x = (features[:, keep]-center[keep])/scale[keep]
    covariance, active = fit_metric(target, valid, weights)
    flat = np.where(active[None, None], target, 0.).reshape(len(x), -1)
    coefficients = {'unconditional': ridge_fit(x, flat, weights, .1, device='cuda')}
    for name, labels in (('real', y), ('misplaced', misplaced_labels(ids, y))):
        w = shot_weights(ids, labels)
        for c in (0, 1):
            selected = labels == c
            coefficients[f'{name}{c}'] = ridge_fit(x[selected], flat[selected], w[selected], .1, device='cuda')
    return dict(center=center, scale=scale, keep=keep, covariance=covariance, active=active, **coefficients)


def forecast(features, heads):
    keep = heads['keep']
    x = (features[:, keep]-heads['center'][keep])/heads['scale'][keep]
    return {k: ridge_predict(x, heads[k]).reshape(-1, 25, 23)
            for k in ('unconditional', 'real0', 'real1', 'misplaced0', 'misplaced1')}


def describe(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return dict(n=len(x), mean=float(x.mean()) if len(x) else None,
                median=float(np.median(x)) if len(x) else None,
                positive_fraction=float((x > 0).mean()) if len(x) else None)


def evaluate(root, folds):
    rows = []; access = []
    for fold in folds:
        for sid in fold['held']:
            disruptive, down = io23.read_outcome('EAST', sid)
            access.append(dict(shot=sid, fold=fold['fold'], role='evaluation_after_all_seals'))
            with np.load(root/f'fold{fold["fold"]}'/f'{sid}.npz') as z:
                origin_lead = np.round((down-z['times'])*1000, 9) if disruptive else None
                for h in HORIZONS:
                    lead = np.round((down-z[f'times{h}'])*1000, 9) if disruptive else None
                    valid = np.isfinite(z[f'real_score{h}'])
                    critical = valid & (origin_lead <= 125) & (lead > 30) if disruptive else valid
                    # Later feedback is described, never counted as timely.
                    late = valid & (origin_lead <= 125) & (lead <= 30) & (lead >= 0) if disruptive else np.zeros(len(valid), bool)
                    for arm in ('real', 'misplaced'):
                        score = z[f'{arm}_score{h}']
                        rows.append(dict(shot=sid, fold=fold['fold'], disruptive=bool(disruptive),
                            horizon=h, arm=arm, critical=describe(score[critical]), late=describe(score[late]),
                            full=describe(score[valid]),
                            swapped_critical=describe(-score[critical]) if arm == 'real' else None,
                            unconditional=describe(z[f'unconditional_error{h}'][critical]),
                            persistence=describe(z[f'persistence_error{h}'][critical]),
                            safe_error=describe(z[f'{arm}0_error{h}'][critical]),
                            danger_error=describe(z[f'{arm}1_error{h}'][critical]),
                            first_timely_positive_lead_ms=float(lead[np.flatnonzero(critical & (score > 0))[0]])
                            if disruptive and (critical & (score > 0)).any() else None))
    summary = {}
    for h in HORIZONS:
        for arm in ('real', 'misplaced'):
            selected = [r for r in rows if r['horizon'] == h and r['arm'] == arm]
            ds = [r for r in selected if r['disruptive']]; ss = [r for r in selected if not r['disruptive']]
            summary[f'{arm}{h}'] = dict(
                danger_correct=sum(r['critical']['median'] is not None and r['critical']['median'] > 0 for r in ds),
                safe_correct=sum(r['critical']['median'] is not None and r['critical']['median'] < 0 for r in ss),
                danger_count=len(ds), safe_count=len(ss),
                miss70868_direction=next(r['critical']['median'] for r in ds if r['shot'] == 70868),
                danger_details=[dict(shot=r['shot'], **r['critical'], first_timely_positive_lead_ms=r['first_timely_positive_lead_ms']) for r in ds])
    real, wrong = summary['real5'], summary['misplaced5']
    gate = bool(real['danger_correct'] >= 3 and real['safe_correct'] >= 12
                and real['miss70868_direction'] is not None and real['miss70868_direction'] > 0
                and real['danger_correct']+real['safe_correct'] > wrong['danger_correct']+wrong['safe_correct'])
    _durable_json(root/'per_shot.json', rows); _durable_json(root/'evaluation_access.json', access)
    _durable_json(root/'summary.json', dict(direction_results=summary, neural_trial_gate=gate,
        warning='Direction diagnostic, NOT a classifier TPR/FPR or independent confirmation.'))
    return summary, gate


def main(root):
    if root.exists():
        raise FileExistsError(root)
    if not torch.cuda.is_available():
        raise RuntimeError('rl_env local CUDA required')
    assert sha(REFERENCE/'completed.json') == REF_SHA
    previous = READ(REFERENCE/'completed.json')
    for name, digest in previous['artifact_sha256'].items():
        assert sha(REFERENCE/name) == digest, name
    folds = READ(REFERENCE/'protocol.json')['folds']
    base, info = load_parent(DEFAULT_PARENT); source = state_cpu(base)
    center, scale = [v.cpu().numpy().astype('float64') for v in (base.center, base.scale)]
    torch.set_num_threads(2); torch.manual_seed(131); start = time.monotonic()
    root.mkdir(parents=True)
    paths = set(Path(__file__).parent.glob('*.py'))
    paths.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_conditional_forecast.py')
    for name, module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module, '__file__', None):
            paths.add(Path(module.__file__))
    with zipfile.ZipFile(root/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, str(path.relative_to(io23.ROOT)))
    _durable_json(root/'protocol.json', dict(folds=folds, source_sha256=PARENT_SHA256,
        reference=str(REFERENCE), reference_sha256=REF_SHA, plan_sha256=sha(PLAN),
        code_sha256={str(p.relative_to(io23.ROOT)): sha(p) for p in paths},
        threshold=info['threshold'], horizons=HORIZONS, primary_horizon=5, ridge=.1, covariance_shrinkage=.1,
        seed=131, test360_accessed=False, gpu=torch.cuda.get_device_name(), torch=torch.__version__))
    seals = []; accesses = []; prepare_counts = []
    original_read = simple_data.read_shot; original_outcome = io23.read_outcome
    for fold in folds:
        f = fold['fold']; directory = root/f'fold{f}'; directory.mkdir()
        checkpoint = REFERENCE/f'fold{f}'/'increment.pt'
        model = PairedFeedback(copy.deepcopy(base), np.ones(23)).cuda()
        model.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True)['model'])
        model.set_stage('frozen'); frozen = state_cpu(model)
        mapping = READ(REFERENCE/f'fold{f}'/'input_map.json')
        assert np.array_equal(mapping['gain'], np.ones(23)) and not np.any(mapping['offset'])
        xs, targets, masks, identities, labels, origins = [], [], [], [], [], []

        def read_guard(device, sid, **kwargs):
            if device != 'EAST' or sid not in fold['fit'] + fold['held']:
                raise AssertionError('outside declared CV20 data')
            return original_read(device, sid, **kwargs)

        def outcome_guard(device, sid):
            if device != 'EAST' or sid not in fold['fit']:
                raise AssertionError('held outcome before sealing')
            accesses.append(dict(fold=f, shot=sid, role='historical_fit'))
            return original_outcome(device, sid)

        with patch.object(simple_data, 'read_shot', side_effect=read_guard), patch.object(io23, 'read_outcome', side_effect=outcome_guard):
            for sid in fold['fit']:
                raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
                labeled = simple_data.attach_outcome(raw, 'EAST', {})
                signal = {k: raw[k] for k in ('values', 'available')}
                ends = np.arange(127, len(raw['times'])-25, 5)
                full_valid = np.lib.stride_tricks.sliding_window_view(labeled['valid'], 26).all(-1)
                ends = ends[full_valid[ends]]
                h, _ = encode(model, signal, ends)
                target, valid = direct_targets(signal, ends, center, scale, 25)
                valid[:, :, io23.RAW23.index('mode_number_n')] = False
                xs.append(h); targets.append(target); masks.append(valid)
                identities.append(np.full(len(ends), sid)); labels.append(labeled['y'][ends].astype(int)); origins.append(ends)
        features, target, valid, ids, y, ends = [np.concatenate(a) for a in (xs, targets, masks, identities, labels, origins)]
        heads = fit_heads(features, target, valid, ids, y)
        np.savez_compressed(directory/'heads.npz', **heads)
        np.savez_compressed(directory/'fit.npz', features=features, target=target, valid=valid, ids=ids, y=y, origins=ends)
        counts = dict(fold=f, rows=len(y), positive_rows=int(y.sum()),
            positive_shots=np.unique(ids[y == 1]).tolist(), negative_shots=np.unique(ids[y == 0]).tolist(),
            excluded_channels=[io23.RAW23[c] for c in np.flatnonzero(~heads['active'])],
            parent_sha256=sha(checkpoint))
        prepare_counts.append(counts); _durable_json(directory/'fit_receipt.json', counts)

        def forbidden_outcome(*args, **kwargs):
            raise AssertionError('no labels during held predictions')

        with patch.object(simple_data, 'read_shot', side_effect=read_guard), patch.object(io23, 'read_outcome', side_effect=forbidden_outcome):
            for sid in fold['held']:
                raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
                signal = {k: raw[k] for k in ('values', 'available')}
                # Include the tail where 5 points mature but 25 cannot. This
                # decision uses record length only, never disruption time.
                ends = np.arange(127, len(raw['times'])-5, 5)
                h, risk = encode(model, signal, ends); forecasts = forecast(h, heads)
                result = dict(origins=ends, times=raw['times'][ends], features=h, static=risk, **forecasts)
                for horizon in HORIZONS:
                    eligible = ends+horizon < len(raw['times']); mature = ends[eligible]
                    truth, mask = direct_targets(signal, mature, center, scale, horizon)
                    result[f'times{horizon}'] = np.full(len(ends), np.nan)
                    result[f'times{horizon}'][eligible] = raw['times'][mature+horizon]
                    for arm, predictions in dict(forecasts, persistence=np.zeros_like(forecasts['real0'])).items():
                        e = np.full(len(ends), np.nan)
                        e[eligible] = maturity_error(predictions[eligible], truth, mask, heads['covariance'], heads['active'])
                        result[f'{arm}_error{horizon}'] = e
                    for arm in ('real', 'misplaced'):
                        result[f'{arm}_score{horizon}'] = result[f'{arm}0_error{horizon}']-result[f'{arm}1_error{horizon}']
                path = directory/f'{sid}.npz'; np.savez_compressed(path, **result)
                seals.append(dict(fold=f, shot=sid, sha256=sha(path)))
        assert_unchanged(model, frozen)
        print(dict(stage='fold_complete', **counts, seconds=time.monotonic()-start), flush=True)
    _durable_json(root/'seals.json', seals); _durable_json(root/'fit_access.json', accesses)
    summary, gate = evaluate(root, folds)
    assert_unchanged(base, source); assert sha(DEFAULT_PARENT) == PARENT_SHA256
    done = dict(status='COMPLETED', seconds=time.monotonic()-start, direction_results=summary,
        neural_trial_gate=gate, source_unchanged=True, neural_updates=0, test360_accessed=False,
        artifacts={str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()})
    _durable_json(root/'completed.json', done)
    print({k: v for k, v in done.items() if k != 'artifacts'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
