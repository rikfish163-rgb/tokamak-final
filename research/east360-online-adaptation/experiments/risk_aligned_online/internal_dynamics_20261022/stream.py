"""Full fixed EAST TEST360; pure intra neural adaptation, no label feedback."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .. import simple_data
from ..causal_input import apply_input_contract
from ..event_adapt_20261022.data import feedback_sample
from ..fixed_test_protocol import load_test_contract, score_fixed_test
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import seal_predictions, _durable_json
from ..simple_train import sha
from .model import InternalDynamics, feedback_loss, temporary_step


def load_model(path, device):
    path = Path(path)
    receipt = json.loads((path.parent/'completed.json').read_text())
    if receipt['selected_sha256'] != sha(path) or not receipt['source_static_qualified']:
        raise ValueError('prepared model SHA or qualified source receipt mismatch')
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck['parent_sha256'] != PARENT_SHA256: raise ValueError('qualified parent mismatch')
    parent, old = load_parent(DEFAULT_PARENT)
    if ck['threshold'] != old['threshold']: raise ValueError('source threshold may not change')
    for k,v in parent.state_dict().items():
        if not torch.equal(v, ck['model']['parent.'+k]): raise ValueError(f'changed parent tensor: {k}')
    model = InternalDynamics(parent, ck['change_scale'], ck['active'], rank=ck['rank'])
    model.load_state_dict(ck['model']); model.to(device).eval()
    return model, ck


def predict(model, shot, *, intra, device, feedback_history='full'):
    forbidden = {'y', 'down', 'valid', 'is_disrupt', 'event_target', 'event_intervals'}
    if forbidden & set(shot): raise ValueError('outcome/event fields forbidden during prediction')
    if feedback_history not in ('full', 'recent3'): raise ValueError('unknown feedback history')
    n = len(shot['values'])
    if n < 1: raise ValueError('nonempty shot required')
    fast = model.initial_fast(1)
    scores = np.empty(n, 'float32'); updates = []; issued = {}; recent = []
    for lo in range(0, n, model.cfg.horizon):
        hi = min(lo+model.cfg.horizon, n)
        windows = [simple_data.window(shot, end, model.cfg.window) for end in range(lo, hi)]
        x, mask = [torch.as_tensor(np.stack(v), device=device) for v in zip(*windows)]
        out = model(x, mask, fast.expand(hi-lo, -1, -1))
        scores[lo:hi] = out['logit'].detach().sigmoid().cpu().numpy()
        issued[hi-1] = out['forecast'][-1:].detach()
        del out
        now = hi-1; origin = now-model.cfg.horizon
        if not intra or origin < 0 or hi == n: continue
        sample = feedback_sample(shot, origin=origin, now=now, cfg=model.cfg)
        x, mask, truth, tm = [torch.as_tensor(sample[k], device=device)[None]
                              for k in ('x', 'mask', 'target', 'target_mask')]
        old_error, _ = feedback_loss(model, issued.pop(origin), x[:, -1], mask[:, -1], truth, tm)
        if feedback_history == 'recent3':
            # Match the source's three-step preparation span. Recompute from
            # zero using only the last three fully arrived feedback windows;
            # no full-shot accumulated state or unavailable future is used.
            recent = (recent+[(x, mask, truth, tm)])[-3:]
            fast = model.initial_fast(1)
            for past in recent:
                fast, info = temporary_step(model, fast, *past)
        else:
            fast, info = temporary_step(model, fast, x, mask, truth, tm)
        updates.append(dict(now_index=now, origin_index=origin, last_target_index=now,
            first_affected_prediction=hi, gradient_steps=len(recent) if feedback_history == 'recent3' else 1,
            issued_forecast_loss=float(old_error.mean()), **info))
    if not np.isfinite(scores).all(): raise FloatingPointError('nonfinite issued scores')
    return scores, dict(updates=updates, accepted_updates=len(updates), inter_updates=0,
        gradient_steps=sum(u['gradient_steps'] for u in updates), feedback_history=feedback_history,
        temporary_state_discarded=True, final_fast_norm=float(fast.detach().norm()),
        feedback_timing='predict block; completed H25 arrives; update; affect next block only')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=['static', 'intra'], required=True)
    p.add_argument('--feedback-history', choices=['full', 'recent3'], default='full')
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('local CUDA required')
    torch.set_num_threads(4); start = time.monotonic()
    model, ck = load_model(args.source, 'cuda')
    frozen = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    contract = load_test_contract('EAST'); args.output.mkdir(parents=True)
    _durable_json(args.output/'protocol.json', dict(source=str(args.source.resolve()), source_sha256=sha(args.source),
        parent_sha256=PARENT_SHA256, threshold=ck['threshold'], arm=args.arm, test_contract=contract,
        preparation_mode=ck['config']['arguments']['mode'], support110_used=False, target_statistics_fit=False,
        feedback_history=args.feedback_history,
        input='flat23, source TRAIN coordinates; unchanged EAST MNM semantic mask and CIII sentinel mask',
        labels='read after durable whole-shot prediction seal; scoring only, zero supervised updates',
        update='internal patch matrix before final attention; same step=.1, radius=1 and H25 as source preparation',
        temporal_order='sorted fixed TEST360 shot IDs', independent_confirmation=False,
        runtime='offline causal replay, no measured real-time latency guarantee',
        code_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')}))
    rows, total = [], 0
    for index, sid in enumerate(contract['shot_ids'], 1):
        s = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        scores, audit = predict(model, s, intra=args.arm == 'intra', device='cuda', feedback_history=args.feedback_history)
        seal = seal_predictions(args.output, index, sid, s['times'], scores, arm=args.arm, threshold=ck['threshold'])
        labeled = simple_data.attach_outcome(s, 'EAST', {})
        valid = labeled['valid']
        rows.append(dict(shot=sid, is_disrupt=labeled['is_disrupt'], down=labeled['down'],
                         times=s['times'][valid], scores=scores[valid]))
        audit['seal_sha256'] = seal['sha256']; total += audit['accepted_updates']
        _durable_json(args.output/f'{index:04d}_{sid}_audit.json', audit)
        if index % 20 == 0:
            print(json.dumps(dict(stage='EAST_TEST', shots=index, updates=total,
                                 elapsed_seconds=time.monotonic()-start)), flush=True)
    for k,v in frozen.items():
        if not torch.equal(v, model.state_dict()[k].cpu()): raise AssertionError(f'persistent state changed: {k}')
    result = score_fixed_test('EAST', rows, ck['threshold'])
    np.savez_compressed(args.output/'test360_traces.npz', **{
        f'{r["shot"]}_{key}':r[key] for r in rows for key in ('times', 'scores')})
    report = dict(status='COMPLETED', metrics=result, accepted_intra_updates=total, inter_updates=0,
        source_unchanged=True, elapsed_seconds=time.monotonic()-start, independent_confirmation=False)
    _durable_json(args.output/'completed.json', report)
    print(json.dumps({**report, 'metrics':{k:v for k,v in result.items() if k != 'predictions'}}), flush=True)


if __name__ == '__main__': main()
