"""J-TEXT-only whole-shot training for the experimental TTT risk state.

Both the no-update and adapted paths receive equal risk supervision. The
inner state never receives labels. The static VAL160 path alone selects the
epoch and threshold; the adapted path is a paired diagnostic, not a selector.
No EAST data or labels are read by this entry point.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import io23, simple_data, simple_train


def pack_shots(shots, device):
    """Keep every observed preterminal point, padding only after a shot ends."""
    prefixes = []
    for shot in shots:
        valid = np.asarray(shot['valid'], bool)
        ends = np.flatnonzero(valid)
        if not len(ends) or not np.array_equal(ends, np.arange(len(ends))):
            raise ValueError('whole-shot training requires a nonempty valid prefix')
        prefixes.append(len(ends))
    if not prefixes:
        raise ValueError('nonempty batch required')
    shape = (len(shots), max(prefixes))
    x = np.zeros((*shape, 23), np.float32)
    available = np.zeros_like(x, bool)
    labels = np.zeros(shape, np.float32)
    valid = np.zeros(shape, bool)
    for i, (shot, n) in enumerate(zip(shots, prefixes)):
        x[i, :n] = shot['values'][:n]
        available[i, :n] = shot['available'][:n]
        labels[i, :n] = shot['y'][:n]
        valid[i, :n] = True
    return {key: torch.as_tensor(value, device=device) for key, value in
            dict(values=x, available=available, labels=labels, valid=valid).items()}


def balanced_risk_loss(logits, labels, valid):
    """Equal observed class means; padded endpoints have no loss or gradient.

    This is a classification surrogate, not a differentiable guarantee of
    timely TPR or shot-level FPR. Those are always measured separately.
    """
    if logits.shape != labels.shape or valid.shape != logits.shape:
        raise ValueError('matching endpoint tensors required')
    if valid.dtype != torch.bool or not valid.any():
        raise ValueError('nonempty boolean endpoint mask required')
    errors = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    groups = [valid & (labels > .5), valid & (labels <= .5)]
    return torch.stack([errors[group].mean() for group in groups if group.any()]).mean()


def train_step(model, optimizer, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    frozen = model(batch['values'], batch['available'], adapt=False)['logits']
    adapted = model(batch['values'], batch['available'], adapt=True)['logits']
    base_loss = balanced_risk_loss(frozen, batch['labels'], batch['valid'])
    adapted_loss = balanced_risk_loss(adapted, batch['labels'], batch['valid'])
    loss = (base_loss + adapted_loss) / 2
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite outer risk loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(loss.detach()), static_loss=float(base_loss.detach()),
                adapted_loss=float(adapted_loss.detach()), grad_norm=float(norm))


def score(model, shots, device, *, adapt):
    model.eval()
    rows = []
    with torch.no_grad():
        for shot in shots:
            b = pack_shots([shot], device)
            # state=None resets all temporary memory at the start of EVERY shot.
            logits = model(b['values'], b['available'], state=None, adapt=adapt)['logits']
            scores = logits[0].sigmoid().cpu().numpy()
            rows.append(dict(shot=shot['shot'], is_disrupt=shot['is_disrupt'],
                             down=shot['down'], times=shot['times'][:len(scores)], scores=scores))
    return rows


def public(result):
    return {k: v for k, v in result.items() if k != 'predictions'}


def main():
    from .ttt_state import TTTConfig, TTTRiskState

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--steps', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=131)
    a = p.parse_args()
    if a.output.exists() or min(a.epochs, a.steps, a.batch_size) < 1:
        raise ValueError('new output and positive training budget required')
    if not np.isfinite(a.lr) or a.lr <= 0:
        raise ValueError('positive finite learning rate required')
    if not torch.cuda.is_available():
        raise RuntimeError('local CUDA required; CPU is for unit tests only')
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    rng = np.random.default_rng(a.seed)
    roles = simple_data.source_roles()
    advanced = simple_data.advanced_times()
    train = simple_train.load_shots(roles['train'], advanced)
    val = simple_train.load_shots(roles['val'], advanced)
    scaler = simple_data.fit_source_scaler(train)
    cfg = TTTConfig()
    model = TTTRiskState(cfg, scaler['center'], scaler['scale']).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    classes = {key: [s for s in train if s['is_disrupt'] == key] for key in (False, True)}
    if any(not value for value in classes.values()):
        raise ValueError('both TRAIN shot classes required')
    a.output.mkdir(parents=True)
    config = dict(format='ttt-risk-state-v1', source_role_ids=roles, model_config=asdict(cfg),
                  feature_names=list(io23.RAW23), labels='max(25ms,AdvancedTime)',
                  input_contract='simple_data: J-TEXT ip*1000; TRAIN-only asinh scaler',
                  objective='equal static and unrolled TTT risk BCE; inner labels prohibited',
                  state_reset='every full shot; never random middle-of-shot reset',
                  selection='VAL160 static path, FPR<0.10; paired adapt does not select',
                  epochs=a.epochs, steps=a.steps, batch_size=a.batch_size, lr=a.lr, seed=a.seed,
                  evidence='DEVELOPMENT_ONLY; experimental architecture screening',
                  code_sha256={name: simple_train.sha(Path(__file__).with_name(name)) for name in
                               ('ttt_training.py', 'ttt_state.py', 'io23.py', 'simple_data.py', 'simple_train.py')})
    (a.output/'config.json').write_text(json.dumps(config, indent=2))
    best, history = None, []
    for epoch in range(1, a.epochs+1):
        logs = []
        for _ in range(a.steps):
            sampled = []
            for _ in range(a.batch_size):
                group = classes[bool(rng.integers(2))]
                sampled.append(group[int(rng.integers(len(group)))])
            logs.append(train_step(model, opt, pack_shots(sampled, 'cuda')))
        static = simple_train.select_threshold(score(model, val, 'cuda', adapt=False), .1)
        adapted = simple_train.metrics(score(model, val, 'cuda', adapt=True), static['threshold'])
        row = dict(epoch=epoch, training={k: float(np.mean([x[k] for x in logs])) for k in logs[0]},
                   static=public(static), adapted=public(adapted))
        history.append(row)
        (a.output/'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
        rank = simple_train.validation_rank(static)
        if best is None or rank < best:
            best = rank
            torch.save(dict(format=config['format'], model={k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()}, model_config=asdict(cfg),
                       config=config, source_center=scaler['center'], source_scale=scaler['scale'],
                       threshold=static['threshold'], validation=static,
                       validation_adapted=adapted, epoch=epoch), a.output/'source.pt')
    selected = torch.load(a.output/'source.pt', map_location='cuda', weights_only=False)
    model.load_state_dict(selected['model'])
    test = simple_train.load_shots(roles['test'], advanced)
    static = simple_train.metrics(score(model, test, 'cuda', adapt=False), selected['threshold'])
    adapted = simple_train.metrics(score(model, test, 'cuda', adapt=True), selected['threshold'])
    report = dict(jtext=static, jtext_intra=adapted, epoch=selected['epoch'],
                  evidence='DEVELOPMENT_ONLY', threshold=selected['threshold'],
                  provenance=dict(checkpoint_sha256=simple_train.sha(a.output/'source.pt'),
                                  selected_test_count=len(test), selection_source='VAL160 static only'))
    (a.output/'test220.json').write_text(json.dumps(report, indent=2))
    (a.output/'hashes.json').write_text(json.dumps(dict(
        source_pt_sha256=simple_train.sha(a.output/'source.pt'),
        test220_sha256=simple_train.sha(a.output/'test220.json'),
        config_sha256=simple_train.sha(a.output/'config.json')), indent=2))
    print(json.dumps(dict(completed=True, selected_epoch=selected['epoch'],
                          static=public(static), adapted=public(adapted))), flush=True)


if __name__ == '__main__':
    main()
