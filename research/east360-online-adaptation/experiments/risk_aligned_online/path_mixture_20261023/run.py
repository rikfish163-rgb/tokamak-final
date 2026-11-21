"""2x2 context/aggregation feasibility; classifier and alarm remain unchanged."""
import argparse
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
from ..conditional_forecast_20261023.learning import direct_targets, shot_weights, misplaced_labels
from ..conditional_forecast_20261023.run import describe
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import signal_context, fit_context, whiten, neighbors, score_paths

READ = lambda p: json.loads(p.read_text())
REFERENCE = io23.ROOT/'experiments/risk_aligned_online/runs/conditional_forecast_cv20_20261023'
REF_SHA = '8e19be2b65eddbfdaae146d8d09acedfd7b60d6f1aed2907cba4b7a172f0adf5'
PLAN = io23.ROOT/'docs/research/source_event_rebuild_20261022/历史路径保留与上下文_配对约定_20261023.md'
VIEWS, LABELS, AGGREGATIONS, HORIZONS = ('cls', 'signal'), ('real', 'misplaced'), ('mean', 'mix'), (5, 10, 25)


def raw_context(ids, ends, active, center, scale, read):
    context = np.full((len(ids), 3*active.sum()), np.nan); valid = np.zeros(len(ids), bool)
    for sid in np.unique(ids):
        r = read(int(sid)); select = ids == sid
        h, mask = signal_context({k:r[k] for k in ('values', 'available')}, ends[select], center, scale, active)
        context[select] = h; valid[select] = mask
    return context, valid


def evaluate(root, folds):
    rows, access = [], []
    for fold in folds:
        for sid in fold['held']:
            disruptive, down = io23.read_outcome('EAST', sid)
            access.append(dict(fold=fold['fold'], shot=sid, phase='after_all_seals'))
            with np.load(root/f'fold{fold["fold"]}'/f'{sid}.npz') as z:
                origin_lead = np.round((down-z['times'])*1000, 9) if disruptive else None
                for horizon in HORIZONS:
                    lead = np.round((down-z[f'times{horizon}'])*1000, 9) if disruptive else None
                    for view in VIEWS:
                        for label in LABELS:
                            for aggregation in AGGREGATIONS:
                                score = z[f'{view}_{label}_{aggregation}_score{horizon}']
                                valid = np.isfinite(score)
                                critical = valid & (origin_lead <= 125) & (lead > 30) if disruptive else valid
                                rows.append(dict(shot=sid, fold=fold['fold'], disruptive=bool(disruptive),
                                    view=view, label=label, aggregation=aggregation, horizon=horizon,
                                    critical=describe(score[critical]), full=describe(score[valid]),
                                    first_positive_lead_ms=float(lead[np.flatnonzero(critical & (score > 0))[0]])
                                    if disruptive and (critical & (score > 0)).any() else None))
    summary = {}
    for view in VIEWS:
        for label in LABELS:
            for aggregation in AGGREGATIONS:
                for h in HORIZONS:
                    relevant = [r for r in rows if (r['view'], r['label'], r['aggregation'], r['horizon']) == (view, label, aggregation, h)]
                    ds = [r for r in relevant if r['disruptive']]; ss = [r for r in relevant if not r['disruptive']]
                    summary[f'{view}_{label}_{aggregation}_{h}'] = dict(
                        danger_correct=sum(r['critical']['median'] is not None and r['critical']['median'] > 0 for r in ds),
                        safe_correct=sum(r['critical']['median'] is not None and r['critical']['median'] < 0 for r in ss),
                        danger_details=[dict(shot=r['shot'], **r['critical']) for r in ds])
    real = summary['signal_real_mix_5']; wrong = summary['signal_misplaced_mix_5']
    miss = next(r['median'] for r in real['danger_details'] if r['shot'] == 70868)
    gate = bool(real['danger_correct'] >= 3 and real['safe_correct'] >= 12 and miss is not None and miss > 0
                and real['danger_correct']+real['safe_correct'] > wrong['danger_correct']+wrong['safe_correct'])
    _durable_json(root/'per_shot.json', rows); _durable_json(root/'evaluation_access.json', access)
    _durable_json(root/'summary.json', dict(results=summary, neural_trial_gate=gate,
        meaning='per-shot median feedback direction, NOT alarm TPR/FPR'))
    return summary, gate


def main(root):
    if root.exists(): raise FileExistsError(root)
    if not torch.cuda.is_available(): raise RuntimeError('rl_env local CUDA required')
    assert sha(REFERENCE/'completed.json') == REF_SHA
    old = READ(REFERENCE/'completed.json'); p = READ(REFERENCE/'protocol.json')
    for name, digest in old['artifacts'].items(): assert sha(REFERENCE/name) == digest, name
    assert READ(REFERENCE/'verification.json')['status'] == 'VERIFIED'
    base, _ = load_parent(DEFAULT_PARENT)
    center, scale = [t.cpu().numpy().astype(float) for t in (base.center, base.scale)]
    torch.set_num_threads(2); start = time.monotonic(); root.mkdir(parents=True)
    paths = set(Path(__file__).parent.glob('*.py'))
    paths.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_path_mixture.py')
    for name, module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module, '__file__', None): paths.add(Path(module.__file__))
    with zipfile.ZipFile(root/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths): archive.write(path, str(path.relative_to(io23.ROOT)))
    _durable_json(root/'protocol.json', dict(folds=p['folds'], reference=str(REFERENCE), reference_sha256=REF_SHA,
        source_sha256=PARENT_SHA256, plan_sha256=sha(PLAN), code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in paths},
        primary='signal_real_mix_5', mixture_temperature=1., context_shrinkage=.1, threshold=p['threshold'],
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), test360_accessed=False, neural_updates=0))
    seals = []; reads = []
    def forbidden(*args, **kwargs): raise AssertionError('held outcomes not allowed before all seals')
    with patch.object(io23, 'read_outcome', side_effect=forbidden):
        for plan in p['folds']:
            f = plan['fold']; out = root/f'fold{f}'; out.mkdir(); ref = REFERENCE/f'fold{f}'
            def read(sid):
                if sid not in plan['fit']+plan['held']: raise AssertionError('outside CV20 identities')
                reads.append(dict(fold=f, shot=sid, role='fit' if sid in plan['fit'] else 'held_signal'))
                return apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
            with np.load(ref/'fit.npz') as fit, np.load(ref/'heads.npz') as heads:
                active, cov = heads['active'], heads['covariance']
                context, usable = raw_context(fit['ids'], fit['origins'], active, center, scale, read)
                ids, y, targets, origins = [fit[k][usable] for k in ('ids', 'y', 'target', 'origins')]
                features = dict(cls=fit['features'][usable], signal=context[usable])
                labels = dict(real=y, misplaced=misplaced_labels(ids, y))
                weights = shot_weights(ids, y)
                maps = {view:fit_context(v, weights) for view,v in features.items()}
                history = {view:whiten(v, maps[view]) for view,v in features.items()}
                bank = dict(ids=ids, y=y, misplaced=labels['misplaced'], target=targets, origins=origins,
                    active=active, covariance=cov, original_usable=usable)
                for view in VIEWS:
                    bank[f'{view}_context'] = features[view]; bank[f'{view}_white'] = history[view]
                    bank.update({f'{view}_{k}':v for k,v in maps[view].items()})
                np.savez_compressed(out/'bank.npz', **bank)
                for sid in plan['held']:
                    raw = read(sid); signal = {k:raw[k] for k in ('values', 'available')}
                    with np.load(ref/f'{sid}.npz') as prior:
                        ends = prior['origins']; physical, valid = signal_context(signal, ends, center, scale, active)
                        if not valid.any(): raise ValueError('no observed context in held shot')
                        query = dict(cls=prior['features'], signal=physical)
                        result = dict(origins=ends, times=raw['times'][ends], context_valid=valid, static=prior['static'],
                            cls_context=query['cls'], signal_context=query['signal'])
                        for h in HORIZONS:
                            result[f'times{h}'] = prior[f'times{h}'].copy()
                        for view in VIEWS:
                            q = whiten(query[view][valid], maps[view])
                            for label in LABELS:
                                chosen = neighbors(history[view], q, ids, labels[label], device='cuda')
                                for c, (indices, distances, shotids) in chosen.items():
                                    key = f'{view}_{label}{c}'
                                    result[f'{key}_indices'] = indices; result[f'{key}_distance'] = distances
                                    result[f'{key}_shotids'] = shotids
                                    paths = targets[indices]
                                    for h in HORIZONS:
                                        mature = ends[valid]+h < len(raw['times'])
                                        truth, mask = direct_targets(signal, ends[valid][mature], center, scale, h)
                                        mean, mix, individual = score_paths(paths[mature], truth, mask, cov, active)
                                        for agg, scores in (('mean', mean), ('mix', mix)):
                                            value = np.full(len(ends), np.nan)
                                            value[np.flatnonzero(valid)[mature]] = scores
                                            result[f'{key}_{agg}_energy{h}'] = value
                                        result[f'{key}_individual{h}'] = individual
                                for agg in AGGREGATIONS:
                                    for h in HORIZONS:
                                        result[f'{view}_{label}_{agg}_score{h}'] = result[f'{view}_{label}0_{agg}_energy{h}']-result[f'{view}_{label}1_{agg}_energy{h}']
                        path = out/f'{sid}.npz'; np.savez_compressed(path, **result)
                        seals.append(dict(fold=f, shot=sid, sha256=sha(path)))
            print(dict(fold=f, stage='sealed', shots=len(seals), seconds=time.monotonic()-start), flush=True)
    _durable_json(root/'seals.json', seals); _durable_json(root/'signal_access.json', reads)
    summary, gate = evaluate(root, p['folds'])
    assert sha(DEFAULT_PARENT) == PARENT_SHA256
    done = dict(status='COMPLETED', seconds=time.monotonic()-start, results=summary, neural_trial_gate=gate,
        source_unchanged=True, neural_updates=0, test360_accessed=False,
        artifacts={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json', done)
    print({k:v for k,v in done.items() if k != 'artifacts'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
