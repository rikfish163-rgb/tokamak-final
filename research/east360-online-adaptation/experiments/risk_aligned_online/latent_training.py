"""Train temporary TTT memory on frozen qualified causal source features.

No source classifier weight, source threshold or EAST sample is changed. A
memory contribution is identically zero without within-shot updates. Its
learned views and readout receive supervision only through the outer loss.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from . import simple_data, simple_train
from .latent_features import extract_shot
from .simple23 import Config, Simple23
from .source_gate import qualify_source
from .ttt_training import balanced_risk_loss, public


def pack(rows, device):
    if not rows:
        raise ValueError('nonempty complete-shot rows required')
    n = max(len(row['y']) for row in rows)
    dim = rows[0]['features'].shape[-1]
    features = np.zeros((len(rows), n, dim), np.float32)
    logits, labels = np.zeros((len(rows), n), np.float32), np.zeros((len(rows), n), np.float32)
    valid, observed = np.zeros((len(rows), n), bool), np.zeros((len(rows), n), bool)
    for i, row in enumerate(rows):
        count = len(row['y'])
        features[i, :count] = row['features']
        logits[i, :count] = row['base_logits']
        labels[i, :count] = row['y']
        observed[i, :count] = row['observed']
        valid[i, :count] = True
    return {k: torch.as_tensor(v, device=device) for k, v in dict(
        features=features, base_logits=logits, labels=labels, valid=valid, observed=observed).items()}


def metric_rows(memory, rows, device):
    predictions = []
    if memory is not None:
        memory.eval()
    with torch.no_grad():
        for row in rows:
            b = pack([row], device)
            logits = b['base_logits'] if memory is None else memory(
                b['features'], b['base_logits'], observed=b['observed'], state=None, adapt=True)['logits']
            predictions.append(dict(shot=row['shot'], is_disrupt=row['is_disrupt'],
                                    down=row['down'], times=row['times'],
                                    scores=logits[0].sigmoid().cpu().numpy()))
    return predictions


def selection_rank(measured):
    return (measured['tpr'] < .9 or measured['fpr'] >= .2,
            -measured['J'], -measured['tpr'], measured['fpr'])


def cache_role(model, ids, advanced, device):
    rows = []
    for i, sid in enumerate(ids):
        shot = simple_data.attach_outcome(simple_data.read_shot('J-TEXT', sid), 'J-TEXT', advanced)
        ends = np.flatnonzero(shot['valid'])
        if not len(ends) or not np.array_equal(ends, np.arange(len(ends))):
            raise ValueError('source supervision must have a nonempty contiguous prefix')
        count = len(ends)
        signals = dict(values=shot['values'][:count], available=shot['available'][:count])
        features, logits = extract_shot(model, signals, device)
        rows.append(dict(shot=sid, features=features, base_logits=logits,
                         observed=signals['available'].any(-1), y=shot['y'][:count],
                         times=shot['times'][:count], is_disrupt=shot['is_disrupt'], down=shot['down']))
        if (i+1) % 200 == 0:
            print(f'causal source features {i+1}/{len(ids)}', flush=True)
    return rows


def train_step(memory, optimizer, rows, device):
    memory.train()
    batch = pack(rows, device)
    optimizer.zero_grad(set_to_none=True)
    out = memory(batch['features'], batch['base_logits'], observed=batch['observed'], state=None, adapt=True)
    loss = balanced_risk_loss(out['logits'], batch['labels'], batch['valid'])
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite memory training loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(memory.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(loss.detach()), grad_norm=float(norm),
                mean_abs_delta=float(out['delta'][batch['valid']].detach().abs().mean()))


def main():
    from .latent_memory import LatentMemoryConfig, LatentMemory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()
    if args.output.exists() or min(args.epochs, args.steps, args.batch_size) < 1:
        raise ValueError('new output and positive budget required')
    if not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('positive finite learning rate required')
    gate = qualify_source(args.parent)
    if not gate['qualified']:
        raise ValueError(f'qualified temporal source required: {gate}')
    if not torch.cuda.is_available():
        raise RuntimeError('local CUDA required')
    torch.set_num_threads(4)
    torch.manual_seed(131)
    torch.cuda.manual_seed_all(131)
    rng = np.random.default_rng(131)
    parent = torch.load(args.parent, map_location='cpu', weights_only=False)
    model = Simple23(Config(**parent['model_config']), parent['source_center'], parent['source_scale'])
    model.load_state_dict(parent['model'])
    model.cuda().eval().requires_grad_(False)
    threshold = float(parent['threshold'])
    cfg = LatentMemoryConfig(input_dim=model.cfg.dim)
    memory = LatentMemory(cfg).cuda()
    optimizer = torch.optim.AdamW(memory.parameters(), lr=args.lr, weight_decay=.01)
    roles, advanced = simple_data.source_roles(), simple_data.advanced_times()
    args.output.mkdir(parents=True)
    config = dict(format='qualified-source-ttt-memory-v1', parent=str(args.parent.resolve()),
                  parent_sha256=gate['checkpoint_sha256'], parent_gate=gate,
                  model_config=asdict(cfg), threshold=threshold, source_roles=roles,
                  labels='max(25ms,AdvancedTime)', evidence='DEVELOPMENT_ONLY',
                  epochs=args.epochs, steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                  seed=131, selection='VAL160 paired J at unchanged parent threshold; source gate first',
                  code_sha256={name: simple_train.sha(Path(__file__).with_name(name)) for name in
                               ('latent_memory.py', 'latent_training.py', 'latent_features.py',
                                'ttt_training.py', 'simple23.py', 'simple_data.py', 'io23.py')})
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    train = cache_role(model, roles['train'], advanced, 'cuda')
    val = cache_role(model, roles['val'], advanced, 'cuda')
    torch.save(dict(parent_sha256=gate['checkpoint_sha256'], train=train, val=val), args.output/'source_features.pt')
    baseline = simple_train.metrics(metric_rows(None, val, 'cuda'), threshold)
    (args.output/'validation_base.json').write_text(json.dumps(baseline, indent=2))
    print('validation_base', json.dumps(public(baseline)), flush=True)
    groups = {c: [s for s in train if s['is_disrupt'] == c] for c in (False, True)}
    best, history = None, []
    for epoch in range(1, args.epochs+1):
        logs = []
        for _ in range(args.steps):
            rows = []
            for _ in range(args.batch_size):
                group = groups[bool(rng.integers(2))]
                rows.append(group[int(rng.integers(len(group)))])
            logs.append(train_step(memory, optimizer, rows, 'cuda'))
        result = simple_train.metrics(metric_rows(memory, val, 'cuda'), threshold)
        row = dict(epoch=epoch, training={k: float(np.mean([log[k] for log in logs])) for k in logs[0]},
                   validation=public(result), paired_delta_J=result['J']-baseline['J'])
        history.append(row)
        (args.output/'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
        rank = selection_rank(result)
        if best is None or rank < best:
            best = rank
            torch.save(dict(format=config['format'], config=config, memory_config=asdict(cfg),
                            memory={k: v.detach().cpu().clone() for k, v in memory.state_dict().items()},
                            threshold=threshold, validation=result, epoch=epoch), args.output/'source.pt')
    selected = torch.load(args.output/'source.pt', map_location='cuda', weights_only=False)
    memory.load_state_dict(selected['memory'])
    # TEST is first encoded after the memory epoch is selected and saved.
    test = cache_role(model, roles['test'], advanced, 'cuda')
    base_result = simple_train.metrics(metric_rows(None, test, 'cuda'), threshold)
    adapt_result = simple_train.metrics(metric_rows(memory, test, 'cuda'), threshold)
    report = dict(jtext=base_result, jtext_intra=adapt_result, selected_epoch=selected['epoch'],
                  validation_gain=selected['validation']['J']-baseline['J'],
                  evidence='DEVELOPMENT_ONLY; base unchanged, temporary neural memory experimental',
                  provenance=dict(checkpoint_sha256=simple_train.sha(args.output/'source.pt'),
                                  parent_sha256=gate['checkpoint_sha256'], selected_test_count=len(test)))
    (args.output/'test220.json').write_text(json.dumps(report, indent=2))
    (args.output/'hashes.json').write_text(json.dumps(dict(
        source_pt_sha256=simple_train.sha(args.output/'source.pt'),
        test220_sha256=simple_train.sha(args.output/'test220.json'),
        source_features_sha256=simple_train.sha(args.output/'source_features.pt'),
        config_sha256=simple_train.sha(args.output/'config.json')), indent=2))
    if simple_train.sha(args.parent) != gate['checkpoint_sha256']:
        raise AssertionError('parent artifact was modified')
    print(json.dumps(dict(completed=True, static=public(base_result), adapted=public(adapt_result),
                          validation_gain=report['validation_gain'])), flush=True)


if __name__ == '__main__':
    main()
