"""Paired source-only mechanism probe, not a deployable EAST checkpoint."""
import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23
from ..direction_teaching_20261026.learning import loss_from_example
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..source_task_memory_20261026.learning import to_device
from .checkpoint import read, load_context_memory
from .conditioned import ConditionedReadout
from .source_controls import evaluate_modes

PLAN = io23.ROOT/'docs/research/source_event_rebuild_20261023/状态条件内部读出_配对约定_20261028.md'


def gate(candidate, control):
    checks = {}
    for metric in ('bag', 'local'):
        all_modes = candidate['all']
        checks[f'{metric}_better_than_off'] = all_modes['memory'][metric] < all_modes['off'][metric]-1e-6
        checks[f'{metric}_better_than_reverse'] = all_modes['memory'][metric] < all_modes['reverse'][metric]-1e-6
        checks[f'{metric}_better_than_additive'] = all_modes['memory'][metric] < control['all']['memory'][metric]-1e-6
        for part in ('D', 'S'):
            checks[f'{part}_{metric}_not_worse'] = candidate[part]['memory'][metric] <= candidate[part]['off'][metric]+1e-6
    return dict(eligible_for_next_experiment=all(checks.values()), checks=checks,
                scope='SOURCE_MECHANISM_SCREEN_NOT_EAST_ACCEPTANCE')


def summarize(rows):
    return {name: {mode: {key: float(np.mean([r['modes'][mode][key] for r in selected]))
                         for key in selected[0]['modes'][mode]} for mode in selected[0]['modes']}
            for name, selected in (('all', rows), ('D', [r for r in rows if r['disruptive']]),
                                   ('S', [r for r in rows if not r['disruptive']]))}


def run(root, reference):
    if root.exists():
        raise FileExistsError(root)
    if not torch.cuda.is_available():
        raise RuntimeError('rl_env local CUDA required')
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    old = read(reference, 'protocol.json')
    cache = Path(old['cache_root'])
    targets_path = Path(old['reference_root'])/'local_targets.pt'
    if (sha(cache/'inventory.json') != old['cache_inventory_sha256']
            or sha(targets_path) != old['reference_hashes']['local_targets.pt']):
        raise ValueError('frozen source-only cache/targets changed')
    inventory = read(cache, 'inventory.json')
    for row in inventory:
        if sha(cache/'cache'/f"{row['shot']}.pt") != row['cache_sha256']:
            raise ValueError('source cache changed')
    targets = torch.load(targets_path, map_location='cpu', weights_only=True)
    base, parent = load_parent(DEFAULT_PARENT); base.cuda()
    models, receipts = {}, {}
    for kind in ('additive', 'state_conditioned'):
        model = ConditionedReadout(base, kind=kind).cuda().eval()
        receipts[kind] = load_context_memory(model, reference/'context8_epoch2.pt')
        with torch.no_grad():
            model.memory.output.weight.zero_()
        if [n for n, p in model.named_parameters() if p.requires_grad] != ['memory.output.weight']:
            raise ValueError('only internal diagnostic readout may train')
        models[kind] = model
    if receipts['additive'] != receipts['state_conditioned']:
        raise ValueError('same frozen feedback generator required')
    frozen = {k: v.clone() for k, v in models['additive'].state_dict().items()
              if k != 'memory.output.weight'}
    root.mkdir(parents=True)
    imported = {Path(m.__file__) for name, m in list(sys.modules.items())
                if name.startswith('experiments.risk_aligned_online.') and getattr(m, '__file__', None)}
    protocol = dict(profile='conditioned_readout_probe_v1_NOT_DEPLOYABLE',
        source_sha256=PARENT_SHA256, source_memory=receipts['additive'],
        train_ids=old['train_ids'], val_ids=old['val_ids'], threshold=float(parent['threshold']),
        cache_root=str(cache), cache_inventory_sha256=sha(cache/'inventory.json'),
        targets_sha256=sha(targets_path), epochs=2, seed=131, learning_rate=.001,
        accumulation_shots=8, gradient_clip=1., parameters_per_arm=5888,
        plan_sha256=sha(PLAN), code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in imported},
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), east_records_read=0, test220_records_read=0)
    _durable_json(root/'protocol.json', protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(imported):
            archive.write(f, str(f.relative_to(io23.ROOT)))
    threshold = protocol['threshold']
    start = time.monotonic()

    def example(sid):
        e = torch.load(cache/'cache'/f'{sid}.pt', map_location='cpu', weights_only=True)
        e.update(targets[sid])
        return to_device(e, 'cuda')

    def evaluate(model):
        rows = []
        for sid in old['val_ids']:
            e = example(sid)
            rows.append(dict(shot=sid, disruptive=e['flag'], modes=evaluate_modes(model, e, threshold)))
        return dict(summary=summarize(rows), rows=rows, scope='SAMPLED_SOURCE_VAL160_NOT_FULL_SHOT_METRICS')

    counts = {flag: sum(r['role'] == 'train' and r['disruptive'] == flag for r in inventory)
              for flag in (False, True)}
    finals = {}
    for kind, model in models.items():
        arm = root/kind; arm.mkdir()
        _durable_json(arm/'val_epoch0.json', evaluate(model))
        optimizer = torch.optim.Adam([model.memory.output.weight], lr=.001)
        rng = np.random.default_rng(131)
        history = []
        for epoch in (1, 2):
            ids = rng.permutation(old['train_ids']).tolist()
            _durable_json(arm/f'order_epoch{epoch}.json', ids)
            for lo in range(0, len(ids), 8):
                group = ids[lo:lo+8]; optimizer.zero_grad(set_to_none=True)
                losses = []
                for sid in group:
                    e = example(sid)
                    loss, _ = loss_from_example(model, e, threshold, 'detached')
                    (loss*(.5*len(ids)/counts[e['flag']]/len(group))).backward()
                    losses.append(float(loss.detach()))
                norm = float(torch.nn.utils.clip_grad_norm_([model.memory.output.weight], 1., error_if_nonfinite=True))
                optimizer.step()
                history.append(dict(epoch=epoch, ids=group, losses=losses, gradient_norm=norm))
                if time.monotonic()-start > 3600:
                    raise TimeoutError('one-hour source probe limit')
                if len(history) % 20 == 0 or lo+len(group) == len(ids):
                    progress = dict(arm=kind, epoch=epoch, shots=lo+len(group), seconds=time.monotonic()-start)
                    _durable_json(root/'progress.json', progress)
                    print(json.dumps(progress), flush=True)
            result = evaluate(model)
            _durable_json(arm/f'val_epoch{epoch}.json', result)
        for k, v in model.state_dict().items():
            if k != 'memory.output.weight':
                torch.testing.assert_close(v, frozen[k], atol=0, rtol=0)
        for epoch in (1, 2):
            seen = [sid for r in history if r['epoch'] == epoch for sid in r['ids']]
            if len(seen) != 1353 or set(seen) != set(old['train_ids']):
                raise ValueError('complete TRAIN1353 required in each epoch')
        if not any(r['gradient_norm'] > 0 for r in history):
            raise ValueError('readout received no learning signal')
        _durable_json(arm/'training.json', history)
        torch.save(dict(profile=protocol['profile'], kind=kind, protocol_sha256=sha(root/'protocol.json'),
                        output_weight=model.memory.output.weight.detach().cpu()), arm/'readout_epoch2.pt')
        finals[kind] = result['summary']
    for epoch in (1, 2):
        if read(root/'additive', f'order_epoch{epoch}.json') != read(root/'state_conditioned', f'order_epoch{epoch}.json'):
            raise ValueError('paired source orders differ')
    if sha(DEFAULT_PARENT) != PARENT_SHA256 or sha(PLAN) != protocol['plan_sha256']:
        raise ValueError('source model or preregistered plan changed')
    screen = gate(finals['state_conditioned'], finals['additive'])
    _durable_json(root/'complete.json', dict(complete=True, source_and_generator_unchanged=True,
        seconds=time.monotonic()-start, east_records_read=0, test220_records_read=0,
        summary=finals, gate=screen,
        checkpoint_sha256={kind: sha(root/kind/'readout_epoch2.pt') for kind in models}))
    print(json.dumps(screen, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.reference)
