"""Same-capacity CLS/data-token forecast heads; unchanged source risk network."""

import argparse
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import torch

import moving_peak_replay_20261105 as r
from experiments.risk_aligned_online.fast_meta import episode_pools, sample
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import (
    PARENT_SHA256, load_source_train,
)
from readout_forecast_20261107 import TokenForecast
from run_source_meta_stream_20261107 import replay, recount


FIT = r.ROOT/'probe_outputs/readout_forecast_source_20261107_v1'
SITES = ('cls', 'last_data')
STEPS, BATCH, LR, SEED = 512, 32, 3e-4, 131


def make_models(base):
    heads = {k: TokenForecast(base.cfg.dim, base.cfg.horizon, k).to(base.center.device) for k in SITES}
    heads['last_data'].load_state_dict(heads['cls'].state_dict())
    models = dict(original=base)
    for name, head in heads.items():
        model = deepcopy(base)
        model.forecast_head = head
        model.requires_grad_(False)
        model.forecast_head.requires_grad_(True)
        models[name] = model.eval()
    return models


def fit_step(models, optimizers, e):
    losses = {}
    for name in SITES:
        model = models[name]
        model.zero_grad(set_to_none=True)
        prediction = model(e['x'], e['a'])['forecast']
        loss = model.forecast_loss(prediction, e['truth'], e['mask'])
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite real-signal forecast loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.forecast_head.parameters(), 1., error_if_nonfinite=True)
        optimizers[name].step()
        losses[name] = float(loss.detach())
    return losses


def check_static(models):
    original = models['original'].state_dict()
    for name in SITES:
        for key, value in models[name].state_dict().items():
            if not key.startswith('forecast_head.') and not torch.equal(value, original[key]):
                raise ValueError('an auxiliary readout changed the qualified risk network')


def fit(out):
    start = time.monotonic()
    r.configure()
    torch.manual_seed(SEED)
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    models = make_models(base.cuda())
    out.mkdir(parents=True, exist_ok=False)
    train, ids = load_source_train()
    pools = episode_pools(train, base.cfg)
    rng = np.random.default_rng(SEED)
    protocol = dict(status='FIXED_CAPACITY_READOUT_SOURCE', source=str(r.DEFAULT_PARENT),
        parent_sha256=PARENT_SHA256, sites=SITES, seed=SEED, steps=STEPS, batch=BATCH, lr=LR,
        source_train_ids=ids, shared_batches=True, shared_head_initialization=True,
        head_parameters={k: sum(p.numel() for p in models[k].forecast_head.parameters()) for k in SITES},
        readout='CLS or last observed data token after the same original temporal encoder',
        objective='masked direct-H25 SmoothL1 in original source-asinh coordinates',
        source_label_use='TRAIN endpoint labels for the common balanced sampler only; no risk loss',
        source_val_or_test_use=False, east_use=False, checkpoint_selection=False,
        source_files={str(p): p.read_text() for p in (Path(__file__),
            r.ROOT/'readout_forecast_20261107.py',
            r.ROOT/'experiments/risk_aligned_online/simple23.py',
            r.ROOT/'experiments/risk_aligned_online/fast_meta.py')})
    r.write_new(out/'protocol.json', protocol)
    opts = {k: torch.optim.AdamW(models[k].forecast_head.parameters(), lr=LR, weight_decay=0.) for k in SITES}
    history = []
    for step in range(STEPS):
        e = sample(train, pools, base.cfg, rng, BATCH, 'cuda')
        history.append(fit_step(models, opts, e))
        if (step+1) % 64 == 0:
            print(dict(step=step+1, **history[-1]), flush=True)
    check_static(models)
    heads = {k: {n: v.detach().cpu() for n, v in models[k].forecast_head.state_dict().items()} for k in SITES}
    with (out/'heads.pt').open('xb') as handle:
        torch.save(heads, handle)
    r.write_new(out/'fit.json', dict(status='FIT_COMPLETE', history=history,
        qualified_risk_parameters_unchanged=True, seconds=time.monotonic()-start))
    print('completed two matched forecast readouts', flush=True)


def predict(out):
    r.configure()
    base, parent = r.load_parent(r.DEFAULT_PARENT)
    models = make_models(base.cuda())
    protocol, done = (r.read(FIT/f'{k}.json') for k in ('protocol', 'fit'))
    if (protocol['parent_sha256'] != PARENT_SHA256 or done['status'] != 'FIT_COMPLETE'
            or len(done['history']) != STEPS or protocol['sites'] != list(SITES)):
        raise ValueError('both completed same-source forecast readouts required')
    saved = torch.load(FIT/'heads.pt', map_location='cuda', weights_only=True)
    for name in SITES:
        models[name].forecast_head.load_state_dict(saved[name], strict=True)
        models[name].requires_grad_(False)
    check_static(models)
    # Only auxiliary heads differ. The verified original TEST220 qualification
    # belongs to the bitwise-identical static risk path, not to online behavior.
    original_report = r.read(r.ROOT/'probe_outputs/source_meta_native360_20261107_v1/source_test220.json')['original']
    report = dict(original=original_report)
    for name in SITES:
        report[name] = dict(static_risk_identical_to_qualified_original=True,
                           qualification_basis='same risk tensors; no new source TEST selection')
    replay(out, models, parent,
        dict(kind='same_capacity_readout_location', protocol=protocol,
             source_fit_seconds=done['seconds']), report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'predict', 'score'))
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    {'fit': fit, 'predict': predict, 'score': recount}[args.phase](args.out)
