"""J-TEXT TRAIN-only auxiliary masked decoders; qualified base is immutable.

Common reconstruction warmup, then equal-budget signal vs risk-directed heads.
Risk-directed learning differentiates through one disposable FFN step and uses
only the subsequent source label. This is a hypothesis probe, not a new source
classifier, an EAST support phase, or a claim of learning physical mechanisms.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from . import fast_meta
from .fast_update import normalized_step
from .ffn_intra_20261005.source import split_shots
from .masked_prefix_feedback import MaskedPrefixModel
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent, load_source_train
from .simple_online import _durable_json, _hash


def current_window(e):
    # Existing source episode: origin t, arrived truth t+1..t+25,
    # query t+26. No query diagnostic is included in the inner feedback.
    return (torch.cat((e['x'], e['truth']), 1)[:, -128:],
            torch.cat((e['a'], e['mask']), 1)[:, -128:])


def train_step(model, optimizer, e, *, phase, mode):
    if mode not in ('signal', 'risk'):
        raise ValueError('signal or risk mode required')
    expected = {id(p) for p in model.decoder.parameters()}
    actual = [id(p) for g in optimizer.param_groups for p in g['params']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('source optimizer may update only the masked decoder')
    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    x, a = current_window(e)
    sums = dict(signal_loss=0., risk_before=0., risk_after=0., objective=0.)
    # One FFN state per episode, not one shared delta across different shots.
    for index in range(len(x)):
        fast = model.initial_fast(1).detach().requires_grad_()
        loss = model.reconstruction_loss(x[index:index+1], a[index:index+1], fast, phase=phase)
        gradient, = torch.autograd.grad(loss, fast, create_graph=mode == 'risk', retain_graph=True)
        updated = normalized_step(fast, fast.detach(), gradient)
        qx, qa = e['qx'][index:index+1], e['qa'][index:index+1]
        after = F.binary_cross_entropy_with_logits(model(qx, qa, updated)['logit'], e['y'][index:index+1])
        with torch.no_grad():
            before = F.binary_cross_entropy_with_logits(model(qx, qa)['logit'], e['y'][index:index+1])
        objective = loss if mode == 'signal' else after + .20*loss
        if not torch.isfinite(objective):
            raise FloatingPointError('nonfinite source-only decoder objective')
        (objective/len(x)).backward()
        for key, value in dict(signal_loss=loss, risk_before=before, risk_after=after, objective=objective).items():
            sums[key] += float(value.detach())/len(x)
    norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(sums, head_grad_norm=float(norm))


def evaluate(model, shots, pools):
    rng = np.random.default_rng(912)
    rows = []
    for iteration in range(64):
        e = fast_meta.sample(shots, pools, model.cfg, rng, 1, model.center.device)
        x, a = current_window(e)
        fast = model.initial_fast(1).detach().requires_grad_()
        loss = model.reconstruction_loss(x, a, fast, phase=iteration % 4)
        gradient, = torch.autograd.grad(loss, fast)
        updated = normalized_step(fast.detach(), fast.detach(), gradient.detach())
        with torch.no_grad():
            before = F.binary_cross_entropy_with_logits(model(e['qx'], e['qa'])['logit'], e['y'])
            after = F.binary_cross_entropy_with_logits(model(e['qx'], e['qa'], updated)['logit'], e['y'])
        rows.append(dict(signal_loss=float(loss.detach()), risk_before=float(before), risk_after=float(after)))
    return dict(count=len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in rows[0]},
                episodes=rows, scope='TRAIN-held-out source episodes; not whole-shot performance')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('new auxiliary-training directory required')
    if not torch.cuda.is_available(): raise RuntimeError('rl_env local CUDA required')
    torch.set_num_threads(4); torch.manual_seed(131)
    args.output.mkdir(parents=True)
    base, parent = load_parent(DEFAULT_PARENT)
    model = MaskedPrefixModel(base).cuda().eval()
    initial = {k: v.detach().cpu().clone() for k, v in model.base.state_dict().items()}
    model.decoder.requires_grad_(True)
    shots, ids = load_source_train()
    train, holdout = split_shots(shots)
    pools = fast_meta.episode_pools(train, model.cfg)
    valid_pools = fast_meta.episode_pools(holdout, model.cfg)
    root = Path(__file__).parent
    files = ['masked_prefix_train.py', 'masked_prefix_feedback.py', 'persistent_ffn.py',
             'ffn_intra_20261005/model.py', 'ffn_intra_20261005/source.py', 'fast_update.py',
             'fast_meta.py', 'fast_direction.py', 'simple23.py', 'simple_data.py', 'io23.py',
             'redesign_20261004/intra.py', 'redesign_20261004/prepare_prior.py']
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in files: archive.write(root/name, name)
    protocol = dict(scope='SOURCE_ONLY_AUXILIARY_TRAINING', parent_sha256=PARENT_SHA256,
        train_roles='canonical TRAIN1353, split80/20 by shot', train_ids=[s['shot'] for s in train],
        holdout_ids=[s['shot'] for s in holdout], all_train_ids=ids, east_data_used=False,
        source_classifier_changed=False, source_scale_changed=False,
        label='original J-TEXT max(25ms,AdvancedTime) endpoint label, query after inner window',
        warmup_steps=256, warmup_batch=32, branch_steps=256, branch_batch=8,
        optimizer='AdamW lr1e-4 weight_decay0 gradient_clip1', seed=131,
        signal_objective='masked SmoothL1', risk_objective='source BCE after one FFN step +0.20 masked SmoothL1',
        inner_step=.05, inner_radius=1., inner_unroll_steps=1,
        acceptance_not_differentiated=True,
        mask='every fourth 8-point patch; phase cycles0..3; original missing values never targets',
        auxiliary_decoder='64 -> 8x23, shared across temporal tokens, read before final invalid-token zeroing',
        decoder_coordinates='source asinh; frozen original layernorm activation before source output affine',
        pretrained_for_reconstruction=False,
        limitation='classification-trained parent, unlike the jointly prepared MAE backbone in TTT-MAE',
        selection='fixed budget, last checkpoint; no EAST-label selection',
        code_snapshot_sha256=_hash(args.output/'code_snapshot.zip'))
    _durable_json(args.output/'protocol.json', protocol)
    started = time.monotonic()
    rng = np.random.default_rng(131)
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=1e-4, weight_decay=0.)
    rows = []
    for iteration in range(256):
        e = fast_meta.sample(train, pools, model.cfg, rng, 32, 'cuda')
        x, a = current_window(e)
        optimizer.zero_grad(set_to_none=True)
        loss = model.reconstruction_loss(x, a, model.initial_fast(len(x)), phase=iteration % 4)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        rows.append(dict(step=iteration+1, loss=float(loss.detach()), gradient_norm=float(norm)))
        if (iteration+1) % 32 == 0:
            print(json.dumps(dict(stage='warmup', **rows[-1], elapsed_seconds=time.monotonic()-started)), flush=True)
    _durable_json(args.output/'warmup.json', rows)
    common = deepcopy(model.decoder.state_dict())
    evaluations = {}
    for mode in ('signal', 'risk'):
        model.decoder.load_state_dict(common)
        rng = np.random.default_rng(3131)
        optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=1e-4, weight_decay=0.)
        history = []
        for iteration in range(256):
            e = fast_meta.sample(train, pools, model.cfg, rng, 8, 'cuda')
            row = train_step(model, optimizer, e, phase=iteration % 4, mode=mode)
            history.append(dict(step=iteration+1, **row))
            if (iteration+1) % 32 == 0:
                print(json.dumps(dict(stage=mode, **history[-1], elapsed_seconds=time.monotonic()-started)), flush=True)
        model.decoder.requires_grad_(False)
        evaluations[mode] = evaluate(model, holdout, valid_pools)
        _durable_json(args.output/f'{mode}_training.json', history)
        with (args.output/f'{mode}_head.pt').open('xb') as handle:
            torch.save(dict(decoder=model.decoder.state_dict(), mode=mode, parent_sha256=PARENT_SHA256,
                            protocol=protocol), handle)
        model.decoder.requires_grad_(True)
    for name, tensor in model.base.state_dict().items():
        torch.testing.assert_close(tensor.detach().cpu(), initial[name], atol=0, rtol=0)
    assert not model.memory.detach().count_nonzero() and _hash(DEFAULT_PARENT) == PARENT_SHA256
    _durable_json(args.output/'evaluation.json', evaluations)
    _durable_json(args.output/'completed.json', dict(base_bitwise_unchanged=True,
        elapsed_seconds=time.monotonic()-started, east_data_used=False,
        decoder_sha256={mode: _hash(args.output/f'{mode}_head.pt') for mode in ('signal', 'risk')}))
    print(json.dumps(dict(completed=True, evaluation={k: {n: v for n, v in r.items() if n != 'episodes'}
                                                     for k, r in evaluations.items()})), flush=True)


if __name__ == '__main__': main()
