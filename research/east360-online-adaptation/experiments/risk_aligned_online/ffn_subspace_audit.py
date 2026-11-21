"""Whole-SOURCE validation and outcome-stratified diagnosis of update geometry.

This retains failed fragment gates verbatim. It does not promote candidates,
touch EAST, train a classifier, tune thresholds or create a JTEXT online lane.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from . import io23, simple_data, simple_train
from .ffn_subspace import FfnSubspaceModel
from .ffn_subspace_train import FORMAT
from .ffn_subspace_stress import distort_source_record
from .redesign_20261004.intra import predict_source_diagnostic
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from .simple_online import _durable_json, _hash, seal_predictions

RUNS = Path(__file__).parent / 'runs'
NORMAL = RUNS / 'ffn_subspace_source256_20261009'
PROTECTED = RUNS / 'ffn_subspace_protected256_20261009'


def whole_source_screen(results):
    """Predeclared count dominance, not a new threshold or a significance test.

    The proposed rule must not lose static timely TPs or add static FPs. It
    must dominate both controls with at least one strictly improved count.
    This gate concerns SOURCE validation only; passing is not EAST success.
    """
    required = ('static', 'random', 'signal', 'risk')
    if not all(name in results for name in required):
        raise ValueError('static, random, signal and risk results required')
    cohort = {(results[n]['n_disrupt'], results[n]['n_safe']) for n in required}
    if len(cohort) != 1 or min(next(iter(cohort))) < 1:
        raise ValueError('the same nonempty two-class cohort is required')
    risk, baseline = results['risk'], results['static']; reasons = []
    if risk['tp'] < baseline['tp']: reasons.append('source_static_tp_lost')
    if risk['fp'] > baseline['fp']: reasons.append('source_static_fp_added')
    for name in ('random', 'signal'):
        control = results[name]
        if not (risk['tp'] >= control['tp'] and risk['fp'] <= control['fp']
                and (risk['tp'] > control['tp'] or risk['fp'] < control['fp'])):
            reasons.append(f'not_strictly_better_than_{name}')
    return dict(passed=not reasons, reasons=reasons,
                meaning='SOURCE operational-count diagnostic; not EAST gain or independent validation')


def partition_fragments(rows, outcomes):
    groups = {key: [] for key in ('safe', 'disrupt_prepositive', 'disrupt_positive')}
    for row in rows:
        sid, label = row['shot'], row['label']
        if sid not in outcomes or type(outcomes[sid]) is not bool:
            raise ValueError('actual source outcome required for every fragment')
        if label not in (0, 1): raise ValueError('binary endpoint label required')
        if not outcomes[sid] and label:
            raise ValueError('a safe source shot cannot have a positive label')
        key = ('disrupt_positive' if label else 'disrupt_prepositive') if outcomes[sid] else 'safe'
        groups[key].append(row)
    return {key: dict(n=len(selected),
        risk_before=float(np.mean([r['risk_before'] for r in selected])) if selected else None,
        risk_after=float(np.mean([r['risk_after'] for r in selected])) if selected else None,
        beneficial=sum(r['risk_after'] < r['risk_before'] for r in selected))
        for key, selected in groups.items()}


def load_candidate(base, directory, arm):
    protocol = json.loads((directory / 'protocol.json').read_text())
    path = directory / f'{arm}.pt'
    if (protocol['format'] != FORMAT or not protocol['source_only'] or protocol['east_access']
            or protocol['test_access'] or protocol['parent_sha256'] != _hash(DEFAULT_PARENT)
            or _hash(path) != protocol['basis_sha256'][arm]):
        raise ValueError('completed source-only basis provenance mismatch')
    saved = torch.load(path, map_location='cuda', weights_only=True)
    if (saved['format'] != FORMAT or saved['parent_sha256'] != protocol['parent_sha256']
            or saved['arm'] != arm or saved['basis'].shape != (16576, 8)):
        raise ValueError('update basis metadata/shape mismatch')
    model = FfnSubspaceModel(deepcopy(base), rank=8, seed=131).cuda().eval()
    with torch.no_grad(): model.basis.copy_(saved['basis'])
    error = float((model.basis.T @ model.basis - torch.eye(8, device='cuda')).detach().abs().max())
    if error > 2e-6 or not torch.isfinite(model.basis).all():
        raise ValueError('update basis is not finite orthonormal')
    return model, protocol


def verify_seals(directory, receipts, threshold):
    rows = []
    for receipt in receipts:
        path = directory / receipt['npz']
        if _hash(path) != receipt['sha256']: raise ValueError('source seal changed')
        with np.load(path) as sealed:
            sid = int(sealed['shot']); times = sealed['times'].copy(); scores = sealed['scores'].copy()
        if sid != receipt['shot']: raise ValueError('source seal shot mismatch')
        disruptive, down = io23.read_outcome('J-TEXT', sid)
        rows.append(dict(shot=sid, is_disrupt=disruptive, down=down, times=times, scores=scores))
    return simple_train.metrics(rows, threshold, lead=10.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--normal-run', type=Path, default=NORMAL)
    parser.add_argument('--protected-run', type=Path, default=PROTECTED)
    parser.add_argument('--no-protected', action='store_true',
                        help='diagnose a new same-budget random/signal/risk experiment without a fourth arm')
    parser.add_argument('--coordinate-stress', action='store_true',
                        help='whole SOURCE stress diagnostic; never an EAST or physical simulator')
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('new source diagnostic directory required')
    if not torch.cuda.is_available(): raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(4); torch.manual_seed(131)
    base, parent = load_parent(DEFAULT_PARENT)
    roles = simple_data.source_roles(); ids = roles['val']
    original = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    specs = dict(random=(args.normal_run, 'random'), signal=(args.normal_run, 'signal'),
                 risk=(args.normal_run, 'risk'))
    if not args.no_protected: specs['protected'] = (args.protected_run, 'risk')
    candidates, training = {}, {}
    for name, (directory, arm) in specs.items():
        candidates[name], training[name] = load_candidate(base, directory, arm)
        if training[name]['val_ids'] != ids: raise ValueError('source VAL roles changed')
    if len({p['episode_manifest_sha256'] for p in training.values()}) != 1:
        raise ValueError('candidate training samples differ')
    args.output.mkdir(parents=True)
    root = Path(__file__).parent
    files = [root / n for n in ('ffn_subspace.py', 'ffn_subspace_train.py', 'ffn_subspace_audit.py',
        'ffn_subspace_stress.py',
        'simple_data.py', 'simple_train.py', 'simple23.py', 'io23.py', 'fast_update.py')]
    files += [root / 'redesign_20261004' / n for n in ('intra.py', 'prepare_prior.py')]
    files += [root / 'ffn_intra_20261005/model.py']
    with zipfile.ZipFile(args.output / 'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.write(path, str(path.relative_to(root)))
    fragment_reports, failed_gates = {}, {}
    outcomes = {sid: bool(io23.read_outcome('J-TEXT', sid)[0]) for sid in ids}
    for name, (directory, arm) in specs.items():
        saved = json.loads((directory / 'validation.json').read_text())
        fragment_reports[name] = partition_fragments(saved['reports'][arm]['rows'], outcomes)
        failed_gates[name] = saved['screen']
    protocol = dict(scope='OFFLINE_SOURCE_DIAGNOSTIC; no JTEXT online deployment, no EAST, no TEST',
        source_sha256=_hash(DEFAULT_PARENT), validation_ids=ids, lead_rule='strict >10ms',
        threshold=float(parent['threshold']), threshold_tuning=False, source_training=False,
        basis_training_manifests={n: p['episode_manifest_sha256'] for n, p in training.items()},
        original_fragment_gates_retained=failed_gates,
        source_endpoint_groups='actual safe / disruptive before positive window / disruptive positive',
        selection='all canonical VAL160, no performance-based shot filtering',
        stride=100, inner_step=.05, radius=1., horizon=25,
        source_record='preterminal valid prefix, matching original offline source evaluation',
        code_sha256={str(path.relative_to(root)): _hash(path) for path in files},
        basis_sha256={n: _hash(d / f'{a}.pt') for n, (d, a) in specs.items()},
        completed_training_protocol_sha256={n: _hash(d / 'protocol.json') for n, (d, a) in specs.items()},
        status='diagnostic only; original failed fragment gates do not become passed')
    protocol['coordinate_stress'] = dict(enabled=args.coordinate_stress, seed=20261010,
        numerical_only=True, target_access=False, clean_probability=.3,
        gain_log_range=[-.35, .35], bias_range=[-1., 1.],
        comparison='all four arms use the same convention per full source record')
    _durable_json(args.output / 'protocol.json', protocol)
    _durable_json(args.output / 'fragment_groups.json', fragment_reports)
    # Offline source outcomes delimit preterminal records only; the updater
    # receives a whitelist of signals and never reads y, down or is_disrupt.
    shots = simple_train.load_shots(ids, simple_data.advanced_times())
    if args.coordinate_stress:
        moved, conventions = [], {}
        for shot in shots:
            altered, receipt = distort_source_record(candidates['random'], shot,
                np.random.default_rng(20261010 + shot['shot']))
            moved.append(altered); conventions[str(shot['shot'])] = receipt
        shots = moved
        _durable_json(args.output / 'source_stress_conventions.json', conventions)
        protocol['source_stress_conventions_sha256'] = _hash(args.output / 'source_stress_conventions.json')
        _durable_json(args.output / 'protocol.json', protocol)
    results, audits = {}, {}; started = time.monotonic()
    for name in ('static', *specs):
        model = candidates['random' if name == 'static' else name]
        directory = args.output / name; directory.mkdir()
        rows, receipts, updates = [], [], 0
        basis = model.basis.detach().clone()
        for index, shot in enumerate(shots, 1):
            valid = np.flatnonzero(shot['valid'])
            if not len(valid) or not np.array_equal(valid, np.arange(len(valid))):
                raise ValueError('source validation requires a nonempty valid prefix')
            signals = dict(device='J-TEXT', values=shot['values'][valid],
                           available=shot['available'][valid], times=shot['times'][valid])
            scores, audit = predict_source_diagnostic(model, signals, intra=name != 'static',
                                                     device='cuda')
            receipt = seal_predictions(directory, index, shot['shot'], signals['times'], scores,
                                       arm=name, threshold=parent['threshold'])
            receipts.append(receipt)
            rows.append(dict(shot=shot['shot'], is_disrupt=shot['is_disrupt'], down=shot['down'],
                             times=signals['times'], scores=scores))
            updates += audit['accepted_updates']
            _durable_json(directory / f'{index:04d}_{shot["shot"]}_audit.json', audit)
            if index % 20 == 0 or index == len(shots):
                m = simple_train.metrics(rows, parent['threshold'], lead=10.)
                print(json.dumps(dict(arm=name, shots=index, tp=m['tp'], fp=m['fp'],
                    accepted=updates, elapsed_s=time.monotonic() - started)), flush=True)
        result = simple_train.metrics(rows, parent['threshold'], lead=10.)
        if verify_seals(directory, receipts, parent['threshold']) != result:
            raise RuntimeError('source whole-shot sealed metric verification failed')
        if not torch.equal(model.basis, basis): raise RuntimeError('basis changed in validation')
        if any(not torch.equal(model.base.state_dict()[k].cpu(), v) for k, v in original.items()):
            raise RuntimeError('original source tensors changed')
        results[name] = result; audits[name] = dict(seals=len(receipts), accepted=updates)
        _durable_json(args.output / 'result.json', dict(metrics=results, audits=audits,
            elapsed_s=time.monotonic() - started, fragment_groups=fragment_reports))
    if _hash(DEFAULT_PARENT) != protocol['source_sha256']:
        raise RuntimeError('qualified source checkpoint changed')
    _durable_json(args.output / 'result.json', dict(metrics=results, audits=audits,
        elapsed_s=time.monotonic() - started, fragment_groups=fragment_reports,
        whole_source_screen=whole_source_screen(results)))
    print(json.dumps(dict(completed=True, summary={n: {k: m[k] for k in ('tp','fp','tpr','fpr')}
                                                  for n, m in results.items()},
                         whole_source_screen=whole_source_screen(results))), flush=True)


if __name__ == '__main__': main()
