"""Fixed source-only pair: full diagnostic change versus unexpected change."""
import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..context_memory_20261028.checkpoint import read
from ..context_memory_20261028.conditioned_run import summarize
from ..context_memory_20261028.forecast_audit import persistence_error
from ..context_memory_20261028.source_controls import evaluate_modes
from ..direction_teaching_20261026.learning import loss_from_example
from ..redesign_20261004.prepare_prior import (
    load_parent, DEFAULT_PARENT, PARENT_SHA256, classifier_state, assert_classifier_unchanged,
)
from ..simple_online import _durable_json
from ..simple_train import sha
from ..source_task_memory_20261026.learning import to_device, validate_roles
from .forecast import DirectForecast
from .memory import ExpectedAdapter


def screen(candidate, control):
    checks = {}
    for metric in ('bag', 'local'):
        modes = candidate['summary']['all']
        for baseline in ('off', 'reverse'):
            checks[f'{metric}_better_than_{baseline}'] = modes['memory'][metric] < modes[baseline][metric] - 1e-6
        checks[f'{metric}_better_than_zero_expectation'] = modes['memory'][metric] < control['summary']['all']['memory'][metric] - 1e-6
        for group in ('D', 'S'):
            part = candidate['summary'][group]
            checks[f'{group}_{metric}_not_worse'] = part['memory'][metric] <= part['off'][metric] + 1e-6
    f = candidate['forecast']['all']
    checks.update(fast_improves_expectation=f['adapted'] < f['expectation'],
                  fast_majority=f['improved'] >= 81,
                  fast_safe_majority=candidate['forecast']['S']['improved'] >= 41)
    return dict(eligible=all(checks.values()), checks=checks,
                scope='SOURCE_MECHANISM_SCREEN_NOT_EAST_ACCEPTANCE')


def run(output, reference, forecast_root):
    if output.absolute().parent != io23.ROOT / 'probe_outputs' or output.exists():
        raise ValueError('new output in this worktree/probe_outputs required')
    if not torch.cuda.is_available():
        raise RuntimeError('rl_env local CUDA required')
    torch.set_num_threads(2); torch.manual_seed(131)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    roles = simple_data.source_roles(); validate_roles(roles)
    p, done = read(reference, 'protocol.json'), read(reference, 'complete.json')
    fp, fd = read(forecast_root, 'protocol.json'), read(forecast_root, 'complete.json')
    checkpoint = forecast_root / 'forecast_epoch10.pt'
    if (not done['complete'] or not done['source_unchanged'] or not fd['complete']
            or not fd['gate']['eligible'] or fd['final_epoch'] != 10
            or p['source_sha256'] != PARENT_SHA256 or fp['source_sha256'] != PARENT_SHA256
            or p['train_ids'] != roles['train'] or fp['canonical_train_ids'] != roles['train']
            or p['val_ids'] != roles['val'] or fp['val_ids'] != roles['val']
            or sha(checkpoint) != fd['checkpoint_sha256']):
        raise ValueError('verified source-only reference and qualified forecast required')
    cache = Path(p['cache_root'])
    targets_path = Path(p['reference_root']) / 'local_targets.pt'
    if (sha(cache / 'inventory.json') != p['cache_inventory_sha256']
            or sha(targets_path) != p['reference_hashes']['local_targets.pt']
            or sha(reference / 'forecast_weights.json') != p['forecast_weights_sha256']
            or fp['forecast_weights_sha256'] != p['forecast_weights_sha256']):
        raise ValueError('source cache/targets/scales changed')
    targets = torch.load(targets_path, map_location='cpu', weights_only=True)
    weights = torch.tensor(read(reference, 'forecast_weights.json')['weights'])
    examples = {}
    for r in read(cache, 'inventory.json'):
        sid = r['shot']; path = cache / 'cache' / f'{sid}.pt'
        if r['role'] not in ('train', 'val') or sid not in roles[r['role']] or sha(path) != r['cache_sha256']:
            raise ValueError('TRAIN/VAL cache only')
        e = torch.load(path, map_location='cpu', weights_only=True)
        e.update(targets[sid]); examples[sid] = e
    if set(examples) != set(roles['train'] + roles['val']):
        raise ValueError('complete TRAIN1353/VAL160 required')
    stored = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if stored['protocol_sha256'] != sha(forecast_root / 'protocol.json'):
        raise ValueError('forecast checkpoint protocol differs')
    predictor = DirectForecast(); predictor.load_state_dict(stored['model'])
    base, parent = load_parent(DEFAULT_PARENT); base.cuda()
    frozen = classifier_state(base)
    models = {k: ExpectedAdapter(base, weights, predictor, use_expected=k == 'expected').cuda().eval()
              for k in ('zero', 'expected')}
    models['expected'].load_state_dict(models['zero'].state_dict())
    for k, value in models['zero'].state_dict().items():
        torch.testing.assert_close(value, models['expected'].state_dict()[k], rtol=0, atol=0)
    allowed = {'memory.key.weight', 'memory.key.bias', 'memory.kernel', 'memory.rate', 'memory.output.weight'}
    for model in models.values():
        if {n for n, q in model.named_parameters() if q.requires_grad} != allowed:
            raise ValueError('only source-taught context generator/readout may train')
    output.mkdir(parents=True)
    plan = io23.ROOT / 'EXPECTED_RESIDUAL_PAIR_PLAN.md'
    imported = {Path(m.__file__) for name, m in list(sys.modules.items())
                if name.startswith('experiments.risk_aligned_online.') and getattr(m, '__file__', None)}
    protocol = dict(profile='expected_residual_pair_v1_NOT_DEPLOYABLE', source_sha256=PARENT_SHA256,
        threshold=float(parent['threshold']), train_ids=roles['train'], val_ids=roles['val'],
        reference=str(reference), cache=str(cache), targets=str(targets_path),
        cache_inventory_sha256=sha(cache / 'inventory.json'), targets_sha256=sha(targets_path),
        forecast_root=str(forecast_root), forecast_sha256=sha(checkpoint),
        forecast_weights_sha256=sha(reference / 'forecast_weights.json'),
        plan_sha256=sha(plan), code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in imported},
        seed=131, epochs=2, learning_rate=.001, accumulation_shots=8, gradient_clip=1.,
        forecast_weight=1., trainable_names=sorted(allowed),
        trainable_parameters=sum(q.numel() for q in models['zero'].parameters() if q.requires_grad),
        frozen_predictor_parameters=sum(q.numel() for q in predictor.parameters()),
        controls='reverse raw matured targets then subtract issued F; hold raw change zero means error=-F; off c=0',
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        east_records_read=0, test220_records_read=0, source_event_labels_read=0)
    _durable_json(output / 'protocol.json', protocol)
    with zipfile.ZipFile(output / 'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(imported):
            archive.write(f, str(f.relative_to(io23.ROOT)))
        archive.write(plan, plan.name)
    threshold = protocol['threshold']; start = time.monotonic()

    def example(sid):
        return to_device(examples[sid], 'cuda')

    @torch.no_grad()
    def evaluate(model):
        rows, forecasts = [], []
        for sid in roles['val']:
            e = example(sid)
            rows.append(dict(shot=sid, disruptive=e['flag'], modes=evaluate_modes(model, e, threshold)))
            trace = model.memory.trace(e['means'], e['complete'], mode='memory')
            off = model.memory.trace(e['means'], e['complete'], mode='off')
            forecasts.append(dict(shot=sid, disruptive=e['flag'], adapted=float(trace['forecast_loss']),
                                  expectation=float(off['forecast_loss']),
                                  persistence=persistence_error(e['means'], e['complete'], weights=weights.cuda())))
        summary = {}
        for group in ('all', 'D', 'S'):
            subset = [r for r in forecasts if group == 'all' or r['disruptive'] == (group == 'D')]
            summary[group] = {k: float(np.mean([r[k] for r in subset]))
                              for k in ('adapted', 'expectation', 'persistence')}
            summary[group].update(shots=len(subset), improved=sum(r['adapted'] < r['expectation'] for r in subset))
        return dict(summary=summarize(rows), forecast=summary, rows=rows, forecast_rows=forecasts,
                    scope='SAMPLED_SOURCE_VAL160_NOT_TPR_FPR')

    counts = {flag: sum(examples[sid]['flag'] == flag for sid in roles['train']) for flag in (False, True)}
    results = {}
    stage_a = {r['shot']: r for r in read(forecast_root, 'val_epoch10.json')['rows']}
    for kind, model in models.items():
        arm = output / kind; arm.mkdir()
        initial = evaluate(model)
        if kind == 'expected':
            for r in initial['forecast_rows']:
                if not np.isclose(r['expectation'], stage_a[r['shot']]['prediction'], atol=1e-7, rtol=1e-6):
                    raise ValueError('frozen forecast does not reproduce stage A')
        _durable_json(arm / 'val_epoch0.json', initial)
        parameters = [q for q in model.memory.parameters() if q.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=.001)
        history = []; rng = np.random.default_rng(131)
        for epoch in (1, 2):
            ids = rng.permutation(roles['train']).tolist()
            _durable_json(arm / f'order_epoch{epoch}.json', ids)
            for lo in range(0, len(ids), 8):
                group = ids[lo:lo+8]; optimizer.zero_grad(set_to_none=True); losses = []
                for sid in group:
                    e = example(sid)
                    loss, info = loss_from_example(model, e, threshold, 'detached', forecast_weight=1.)
                    (loss * (.5 * len(ids) / counts[e['flag']] / len(group))).backward()
                    losses.append(dict(shot=sid, loss=float(loss.detach()), forecast=info['forecast_loss']))
                gradient = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
                if any(q.grad is not None for q in model.memory.predictor.parameters()):
                    raise ValueError('frozen expectation received a gradient')
                optimizer.step()
                history.append(dict(epoch=epoch, ids=group, losses=losses, gradient_norm=gradient))
                if time.monotonic() - start > 3000:
                    raise TimeoutError('50-minute pair bound')
                if len(history) % 20 == 0 or lo+len(group) == len(ids):
                    progress = dict(arm=kind, epoch=epoch, shots=lo+len(group), seconds=time.monotonic()-start)
                    _durable_json(output / 'progress.json', progress); print(json.dumps(progress), flush=True)
            assert_classifier_unchanged(base, frozen)
        result = evaluate(model)
        _durable_json(arm / 'val_epoch2.json', result); _durable_json(arm / 'training.json', history)
        for epoch in (1, 2):
            seen = [sid for r in history if r['epoch'] == epoch for sid in r['ids']]
            if len(seen) != 1353 or set(seen) != set(roles['train']):
                raise ValueError('incomplete training epoch')
        if not any(r['gradient_norm'] > 0 for r in history):
            raise ValueError('no source learning')
        for name, value in model.memory.predictor.state_dict().items():
            torch.testing.assert_close(value.cpu(), stored['model'][name], rtol=0, atol=0)
        torch.save(dict(profile=protocol['profile'], kind=kind, protocol_sha256=sha(output/'protocol.json'),
                        memory={k: v.detach().cpu() for k, v in model.memory.state_dict().items()}), arm/'memory_epoch2.pt')
        results[kind] = dict(summary=result['summary'], forecast=result['forecast'])
    for epoch in (1, 2):
        if read(output/'zero', f'order_epoch{epoch}.json') != read(output/'expected', f'order_epoch{epoch}.json'):
            raise ValueError('paired orders differ')
    for f, value in protocol['code_sha256'].items():
        if sha(io23.ROOT / f) != value:
            raise ValueError('experiment code changed during run')
    if sha(DEFAULT_PARENT) != PARENT_SHA256 or sha(plan) != protocol['plan_sha256']:
        raise ValueError('parent or plan changed')
    completed = dict(complete=True, results=results, gate=screen(results['expected'], results['zero']),
        seconds=time.monotonic()-start, source_classifier_unchanged=True, forecast_unchanged=True,
        east_records_read=0, test220_records_read=0,
        checkpoints={k: sha(output/k/'memory_epoch2.pt') for k in models})
    _durable_json(output/'complete.json', completed)
    print(json.dumps(completed['gate'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--forecast-root', type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.reference, args.forecast_root)
