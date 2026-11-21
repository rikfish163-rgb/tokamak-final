"""Pure EAST intra-shot adaptation using the SAME source inner-update operator.

Blocked execution is mathematically causal (no time-mixing normalization or
bidirectional layers). It is an offline replay optimization, not a claim of
measured per-point real-time latency. No outcome/event fields enter prediction.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .. import simple_data
from ..causal_input import apply_input_contract
from ..fixed_test_protocol import load_test_contract, score_fixed_test
from ..simple_online import seal_predictions, _durable_json
from ..simple_train import sha
from .data import feedback_sample
from .model import Config, EventForecastModel, update_fast, arrived_loss


def predict_causal_shot(model, shot, *, intra, stride, lr, radius, device):
    forbidden = {'down', 'is_disrupt', 'y', 'valid', 'event_target', 'event_intervals'}
    if forbidden & set(shot):
        raise ValueError('prediction must not receive outcome or offline event fields')
    if stride < 1 or len(shot['values']) < 1:
        raise ValueError('positive stride and nonempty shot required')
    model.eval()
    fast = model.initial_fast(1).detach().requires_grad_(True)
    origin_fast = fast.detach().clone()
    scores = np.empty(len(shot['values']), 'float32')
    updates, issued = [], {}
    for lo in range(0, len(scores), stride):
        hi = min(lo+stride, len(scores))
        start = max(0, lo-model.cfg.window+1)
        x = torch.as_tensor(shot['values'][start:hi], device=device)[None]
        mask = torch.as_tensor(shot['available'][start:hi], device=device)[None]
        with torch.no_grad():
            # Every output in this block depends only on its own prefix and
            # the fast state fixed BEFORE the block. Recompute past states
            # with current parameters; no stale hidden-state/weight mixing.
            output = model.sequence(x, mask, fast)
            scores[lo:hi] = output['logit'].sigmoid()[0, lo-start:].cpu().numpy()
            if intra:
                for endpoint in range(lo, hi):
                    # Immutable forecast as actually issued at this origin.
                    # Used for honest prediction-error diagnostics, never as
                    # a differentiable training output or an observed input.
                    issued[endpoint] = output['forecast'][:, endpoint-start].detach().clone()
        now = hi-1
        origin = now-model.cfg.horizon
        if not intra or origin < 0 or hi == len(scores):
            continue
        sample = feedback_sample(shot, origin=origin, now=now, cfg=model.cfg)
        values = [torch.as_tensor(sample[k], device=device)[None]
                  for k in ('x', 'mask', 'target', 'target_mask')]
        with torch.no_grad():
            observed_error, _ = arrived_loss(model, issued[origin], values[0][:, -1],
                                             values[1][:, -1], values[2], values[3])
        fast, info = update_fast(model, fast, *values, lr=lr, radius=radius)
        if info['updated']:
            updates.append(dict(now_index=now, origin_index=origin,
                last_target_index=origin+model.cfg.horizon,
                first_affected_prediction=hi, loss=info['loss'],
                issued_forecast_loss=float(observed_error.mean()),
                observations=info['observations'], gradient_norm=info['gradient_norm'],
                fast_distance=float((fast-origin_fast).norm().detach())))
        issued = {t: prediction for t, prediction in issued.items() if t > origin}
    if not np.isfinite(scores).all(): raise FloatingPointError('nonfinite issued risk trace')
    return scores, dict(accepted_updates=len(updates), updates=updates,
        parameter_scope='last temporal block internal low-rank matrix',
        temporary_state_discarded=True, supervised_updates=0,
        timing='predict current block first; arrived diagnostic feedback affects NEXT block',
        runtime='offline causal replay, not certified real-time execution')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=['static', 'intra'], required=True)
    p.add_argument('--stride', type=int, default=16)
    p.add_argument('--lr', type=float, default=.1)
    p.add_argument('--radius', type=float, default=1.)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError('do not overwrite a stream')
    if not torch.cuda.is_available(): raise RuntimeError('local CUDA required')
    source_sha = sha(args.source)
    qualification = json.loads((args.source.parent/'test220.json').read_text())
    q = qualification['metrics']
    if (qualification['checkpoint_sha256'] != source_sha or not qualification['qualified']
            or q['tpr'] < .9 or q['fpr'] >= .2 or q['n_disrupt'] != 110 or q['n_safe'] != 110):
        raise ValueError('source must first qualify on full strict-lead J-TEXT TEST220')
    torch.set_num_threads(4)
    ckpt = torch.load(args.source, weights_only=False, map_location='cuda')
    model = EventForecastModel(Config(**ckpt['model_config']), **ckpt['coordinates']).to('cuda')
    model.load_state_dict(ckpt['model'])
    original = {k: v.detach().clone() for k,v in model.state_dict().items()}
    contract = load_test_contract('EAST'); start = time.monotonic()
    args.output.mkdir(parents=True)
    _durable_json(args.output/'protocol.json', dict(source_sha256=source_sha, arm=args.arm,
        contract=contract, stride=args.stride, lr=args.lr, radius=args.radius,
        threshold=ckpt['threshold'], normalization='source TRAIN only; no EAST statistics fitted',
        input='flat23; J-TEXT ip conversion inherited; EAST MNM semantic mask and CIII sentinel retained',
        source_meta_prepared=ckpt.get('source_meta_prepared', False),
        label_access='outcome is read only AFTER whole-shot durable prediction seal',
        code_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')}))
    rows, updates = [], 0
    for index, sid in enumerate(contract['shot_ids'], 1):
        shot = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        scores, audit = predict_causal_shot(model, shot, intra=args.arm=='intra', stride=args.stride,
                                          lr=args.lr, radius=args.radius, device='cuda')
        receipt = seal_predictions(args.output, index, sid, shot['times'], scores,
                                   arm=args.arm, threshold=ckpt['threshold'])
        revealed = simple_data.attach_outcome(shot, 'EAST', {})
        valid = revealed['valid']
        rows.append(dict(shot=sid, is_disrupt=revealed['is_disrupt'], down=revealed['down'],
                         times=shot['times'][valid], scores=scores[valid]))
        audit['seal_sha256'] = receipt['sha256']
        _durable_json(args.output/f'{index:04d}_{sid}_audit.json', audit)
        updates += audit['accepted_updates']
        if index % 20 == 0:
            print(json.dumps(dict(arm=args.arm, shots=index, updates=updates, elapsed_seconds=time.monotonic()-start)), flush=True)
    for k,v in original.items():
        if not torch.equal(v, model.state_dict()[k]): raise AssertionError(f'source parameter changed: {k}')
    if sha(args.source) != source_sha: raise AssertionError('source file changed')
    metrics = score_fixed_test('EAST', rows, ckpt['threshold'])
    report = dict(status='COMPLETED', arm=args.arm, metrics=metrics, accepted_intra_updates=updates,
                  source_unchanged=True, inter_updates=0, independent_confirmation=False,
                  elapsed_seconds=time.monotonic()-start)
    _durable_json(args.output/'completed.json', report)
    print(json.dumps({**report, 'metrics': {k:v for k,v in metrics.items() if k!='predictions'}}), flush=True)


if __name__ == '__main__': main()
