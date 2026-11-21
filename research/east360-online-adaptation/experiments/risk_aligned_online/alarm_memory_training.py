"""Matched outer-objective control for latent_memory_source_20260929.

Reuses exactly the verified TRAIN1353/VAL160 feature cache and RNG schedule;
only endpoint-balanced BCE is replaced by a timely-shot alarm surrogate.
No EAST data, threshold search, or source classifier changes are permitted.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from . import simple_data, simple_train
from .alarm_objective import timely_shot_loss
from .latent_memory import LatentMemory, LatentMemoryConfig
from .latent_training import cache_role, metric_rows, pack, selection_rank
from .simple23 import Config, Simple23
from .source_gate import qualify_source
from .ttt_training import public


def timing_summary(result):
    leads = [row['lead_ms'] for row in result['predictions'] if row['timely']]
    return {'above_300ms': sum(x > 300. for x in leads),
            'above_1000ms': sum(x > 1000. for x in leads)}


def train_step(memory, optimizer, rows, threshold):
    memory.train()
    b = pack(rows, 'cuda')
    optimizer.zero_grad(set_to_none=True)
    output = memory(b['features'], b['base_logits'], observed=b['observed'], state=None, adapt=True)
    loss = timely_shot_loss(output['logits'], rows, threshold)
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite outer loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(memory.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(loss.detach()), grad_norm=float(norm),
                mean_abs_delta=float(output['delta'][b['valid']].detach().abs().mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new output directory required')
    config = json.loads((args.reference/'config.json').read_text())
    hashes = json.loads((args.reference/'hashes.json').read_text())
    cache_path = args.reference/'source_features.pt'
    if simple_train.sha(cache_path) != hashes['source_features_sha256']:
        raise ValueError('reference feature cache SHA mismatch')
    parent_path = Path(config['parent'])
    gate = qualify_source(parent_path)
    if not gate['qualified'] or gate['checkpoint_sha256'] != config['parent_sha256']:
        raise ValueError('qualified identical parent required')
    if not torch.cuda.is_available():
        raise RuntimeError('local CUDA required')
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    rng = np.random.default_rng(config['seed'])
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    # Preserve the reference RNG consumption even though cached features are reused.
    model = Simple23(Config(**parent['model_config']), parent['source_center'], parent['source_scale'])
    model.load_state_dict(parent['model'])
    model.cuda().eval().requires_grad_(False)
    memory = LatentMemory(LatentMemoryConfig(**config['model_config'])).cuda()
    optimizer = torch.optim.AdamW(memory.parameters(), lr=config['lr'], weight_decay=.01)
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    if cache['parent_sha256'] != gate['checkpoint_sha256']:
        raise ValueError('cache parent mismatch')
    roles = simple_data.source_roles()
    for name in ('train', 'val'):
        if [row['shot'] for row in cache[name]] != roles[name]:
            raise ValueError('cache source roster mismatch')
    threshold = config['threshold']
    args.output.mkdir(parents=True)
    config.update(format='qualified-source-alarm-memory-v1',
                  objective='BCE(max timely logit - logit fixed threshold, shot outcome)',
                  label_contract='source endpoint labels unchanged; new memory OUTER uses whole-shot outcome',
                  reference=str(args.reference.resolve()), cache_sha256=hashes['source_features_sha256'])
    config['code_sha256'] = {name: simple_train.sha(Path(__file__).with_name(name)) for name in
                            ('alarm_objective.py', 'alarm_memory_training.py', 'latent_memory.py',
                             'latent_training.py', 'latent_features.py', 'simple_train.py', 'simple_data.py')}
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    baseline = simple_train.metrics(metric_rows(None, cache['val'], 'cuda'), threshold)
    print('validation_base', json.dumps(public(baseline)), flush=True)
    (args.output/'validation_base.json').write_text(json.dumps(baseline, indent=2))
    groups = {c: [row for row in cache['train'] if row['is_disrupt'] == c] for c in (False, True)}
    best, history = None, []
    for epoch in range(1, config['epochs']+1):
        logs = []
        for _ in range(config['steps']):
            batch = []
            for _ in range(config['batch_size']):
                group = groups[bool(rng.integers(2))]
                batch.append(group[int(rng.integers(len(group)))])
            logs.append(train_step(memory, optimizer, batch, threshold))
        result = simple_train.metrics(metric_rows(memory, cache['val'], 'cuda'), threshold)
        row = dict(epoch=epoch, training={k: float(np.mean([log[k] for log in logs])) for k in logs[0]},
                   validation=public(result), timing=timing_summary(result),
                   paired_delta_J=result['J']-baseline['J'])
        history.append(row)
        (args.output/'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
        rank = selection_rank(result)
        if best is None or rank < best:
            best = rank
            torch.save(dict(format=config['format'], config=config, memory_config=asdict(memory.cfg),
                            memory={k: v.detach().cpu().clone() for k, v in memory.state_dict().items()},
                            threshold=threshold, validation=result, epoch=epoch), args.output/'source.pt')
    selected = torch.load(args.output/'source.pt', map_location='cuda', weights_only=False)
    memory.load_state_dict(selected['memory'])
    test = cache_role(model, roles['test'], simple_data.advanced_times(), 'cuda')
    base = simple_train.metrics(metric_rows(None, test, 'cuda'), threshold)
    adapted = simple_train.metrics(metric_rows(memory, test, 'cuda'), threshold)
    report = dict(jtext=base, jtext_intra=adapted, selected_epoch=selected['epoch'],
                  validation_gain=selected['validation']['J']-baseline['J'],
                  timing=dict(static=timing_summary(base), intra=timing_summary(adapted)),
                  evidence='DEVELOPMENT_ONLY; matched source outer-objective control; NOT EAST improvement')
    (args.output/'test220.json').write_text(json.dumps(report, indent=2))
    (args.output/'hashes.json').write_text(json.dumps({p.name: simple_train.sha(p) for p in
                                                    (args.output/'source.pt', args.output/'test220.json',
                                                     args.output/'config.json')}, indent=2))
    if simple_train.sha(parent_path) != gate['checkpoint_sha256']:
        raise AssertionError('parent changed')
    print(json.dumps(dict(completed=True, static=public(base), adapted=public(adapted),
                          validation_gain=report['validation_gain'], timing=report['timing'])), flush=True)


if __name__ == '__main__':
    main()
