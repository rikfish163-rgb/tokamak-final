"""Replay prior updates, seal future forecasts, then evaluate unseen targets."""
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
from ..forecast_token_scope_20261023.learning import ARMS, PatchForecast
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .diagnostic import replay, forecast_error, summarize

READ = lambda p: json.loads(p.read_text())
REFERENCE = io23.ROOT / 'experiments/risk_aligned_online/runs/forecast_token_scope_cv20_20261023'
REF_SHA = '7e4ba1b6a51e69eb1eccc74b32d6dd6c905eec568d03f9f2af390fdb46dcf89d'
PLAN = io23.ROOT / 'docs/research/source_event_rebuild_20261022/炮内更新后未来预测_诊断约定_20261023.md'


def score(root, plans):
    rows = []; evaluation = []
    for plan in plans:
        folder = root / f'fold{plan["fold"]}'
        mapping = READ(REFERENCE / f'fold{plan["fold"]}' / 'input_map.json')
        for sid in plan['held']:
            s = simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal')), 'EAST', {})
            signal = transform({k: s[k] for k in ('values', 'available')}, mapping)
            evaluation.append(dict(shot=sid, phase='all_future_forecasts_sealed'))
            with np.load(folder / f'{sid}.npz', allow_pickle=False) as pack:
                center, scale = pack['center'], pack['scale']
                for j, count in enumerate(pack['counts']):
                    stop = int(count) + 25
                    valid = signal['available'][count:stop] & np.isfinite(signal['values'][count:stop])
                    truth = np.arcsinh((np.where(valid, signal['values'][count:stop], center).astype('float64') - center) / scale)
                    last_valid = pack['last_available'][j] & np.isfinite(pack['last_value'][j])
                    common = valid & last_valid[None]
                    persistence = np.broadcast_to(np.arcsinh((np.where(last_valid, pack['last_value'][j], center).astype('float64') - center) / scale), truth.shape)
                    persist_error = forecast_error(persistence, truth, common)
                    for arm in ARMS:
                        heads = dict(before=pack[f'before_{arm}'][j], after=pack[f'after_{arm}'][j],
                                     static=pack[f'static_{arm.split("_")[0]}'][j])
                        errors = {k: forecast_error(v, truth, common) for k, v in heads.items()}
                        all_errors = {k: forecast_error(v, truth, valid)['mean'] for k, v in heads.items()}
                        rows.append(dict(shot=sid, fold=plan['fold'], arm=arm, origin=int(count)-1,
                            target_first=int(count), target_last=stop-1, target_end_time=float(s['times'][stop-1]),
                            is_disrupt=bool(s['is_disrupt']), future_all_valid=bool(s['valid'][count:stop].all()),
                            future_any_positive=bool((s['y'][count:stop] > .5).any()),
                            observed=int(common.sum()), excluded_for_persistence=int(valid.sum()-common.sum()),
                            persistence=persist_error['mean'], **{k: v['mean'] for k, v in errors.items()},
                            per_channel={k: v['per_channel'] for k, v in errors.items()}, all_observed=all_errors))
    _durable_json(root / 'windows.json', rows)
    _durable_json(root / 'evaluation_access.json', evaluation)
    groups = dict(all=rows, valid=[r for r in rows if r['future_all_valid']],
                  safe=[r for r in rows if not r['is_disrupt'] and r['future_all_valid']],
                  disruptive=[r for r in rows if r['is_disrupt'] and r['future_all_valid']],
                  positive=[r for r in rows if r['future_any_positive'] and r['future_all_valid']])
    result = {name: summarize(r) for name, r in groups.items()}
    _durable_json(root / 'summary.json', result)
    return result


def main(root):
    if root.exists():
        raise FileExistsError(root)
    if not torch.cuda.is_available():
        raise RuntimeError('rl_env local CUDA required')
    assert sha(REFERENCE / 'completed.json') == REF_SHA
    prior, p = READ(REFERENCE / 'completed.json'), READ(REFERENCE / 'protocol.json')
    for name, digest in prior['artifact_sha256'].items():
        assert sha(REFERENCE / name) == digest, name
    assert sha(DEFAULT_PARENT) == PARENT_SHA256
    torch.set_num_threads(2); torch.manual_seed(131); start = time.monotonic()
    base, _ = load_parent(DEFAULT_PARENT)
    parent = PairedFeedback(base, np.ones(23)).cuda(); parent.set_stage('frozen')
    initial = state_cpu(parent)
    root.mkdir(parents=True)
    paths = set(Path(__file__).parent.glob('*.py'))
    paths.add(io23.ROOT / 'experiments/risk_aligned_online/tests/test_future_forecast_diagnostic.py')
    for name, module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module, '__file__', None):
            paths.add(Path(module.__file__))
    with zipfile.ZipFile(root / 'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, str(path.relative_to(io23.ROOT)))
    _durable_json(root / 'protocol.json', dict(reference=str(REFERENCE), reference_sha256=REF_SHA,
        source_sha256=PARENT_SHA256, folds=p['folds'], plan_path=str(PLAN), plan_sha256=sha(PLAN),
        code_sha256={str(path.relative_to(io23.ROOT)): sha(path) for path in paths},
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        design='unchanged updates; direct unseen25 targets; equal-shot means; no deployment selection',
        test360_accessed=False, independent_confirmation=False))
    seals = []; steps = 0; read_signal = simple_data.read_shot; access = []
    def forbidden_outcome(device, sid):
        raise AssertionError(f'outcome before all future forecasts sealed: {device}/{sid}')
    with patch.object(io23, 'read_outcome', side_effect=forbidden_outcome):
        for plan in p['folds']:
            f = plan['fold']; folder = root / f'fold{f}'; folder.mkdir()
            ref = REFERENCE / f'fold{f}'; models = {}
            for head in ('full', 'patch'):
                model = copy.deepcopy(parent)
                if head == 'patch':
                    model.base.forecast_head = PatchForecast(model.base.forecast_head, model.cfg.dim)
                model.load_state_dict(torch.load(ref / f'{head}.pt', map_location='cuda', weights_only=True)['model'])
                model.set_stage('frozen'); models[head] = model
            frozen = {h: state_cpu(m) for h, m in models.items()}
            mapping = READ(ref / 'input_map.json')
            def signal_guard(device, sid, **kwargs):
                assert device == 'EAST' and sid in plan['held']
                access.append(dict(device=device, shot=sid, phase='issue'))
                return read_signal(device, sid, **kwargs)
            with patch.object(simple_data, 'read_shot', side_effect=signal_guard):
                for sid in plan['held']:
                    s = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
                    packets, updates = replay(models, transform({k: s[k] for k in ('values', 'available')}, mapping))
                    assert updates == READ(ref / 'shots' / f'{sid}_updates.json')
                    steps += sum(r['gradient_steps'] for a in updates.values() for r in a)
                    packets.update(center=parent.base.center.cpu().numpy(), scale=parent.base.scale.cpu().numpy())
                    path = folder / f'{sid}.npz'; np.savez_compressed(path, **packets)
                    seals.append(dict(fold=f, shot=sid, sha256=sha(path), forecasts=len(packets['origins'])))
            for h, m in models.items():
                assert_unchanged(m, frozen[h])
            print(dict(stage='sealed', fold=f, shots=len(seals), seconds=time.monotonic()-start), flush=True)
    _durable_json(root / 'prediction_receipts.json', seals)
    _durable_json(root / 'signal_access.json', access)
    assert steps == prior['neural_steps']
    summary = score(root, p['folds'])
    assert_unchanged(parent, initial); assert sha(DEFAULT_PARENT) == PARENT_SHA256
    done = dict(status='COMPLETED', seconds=time.monotonic()-start, updates_reproduced=steps,
        complete_future_origins=sum(r['forecasts'] for r in seals), source_unchanged=True,
        test360_accessed=False, independent_confirmation=False,
        artifact_sha256={str(path.relative_to(root)): sha(path) for path in root.rglob('*') if path.is_file()})
    _durable_json(root / 'completed.json', done)
    print(dict(status='COMPLETED', seconds=done['seconds'], summary={a: {k:v for k,v in r.items() if k!='per_shot'} for a,r in summary['all'].items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
