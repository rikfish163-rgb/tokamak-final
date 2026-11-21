"""Source-VAL mechanism probe: known observation changes, no EAST inference.

This is deliberately allowed to diagnose failed source candidates, but never
qualifies them for deployment. Labels only score already-completed updates.
The synthetic transform does not simulate plasma trajectories or mechanisms.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..event_adapt_20261022.train import load_source
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import pools, sha
from .model import MaskedState, adapt_observed, masked_views, reconstruction_loss


def distort(model, x, available, gain, bias):
    if gain.shape != (23,) or bias.shape != (23,) or not (gain > 0).all():
        raise ValueError('one finite positive gain and bias per channel required')
    if not torch.isfinite(gain).all() or not torch.isfinite(bias).all():
        raise ValueError('finite synthetic transform required')
    # A fixed transform, not a mean or scale estimated from the current/future
    # shot. FP64 round-trip consistency was required by the previous probe.
    c, q = model.base.center.double(), model.base.scale.double()
    valid = available & torch.isfinite(x)
    clean = torch.where(valid, x, model.base.center).double()
    z = torch.asinh((clean-c)/q)
    moved = (z*gain.double()+bias.double()).sinh()*q+c
    return torch.where(valid, moved.to(x.dtype), x).detach()


def load_source_candidate(path, device):
    path = Path(path)
    receipt = json.loads((path.parent/'completed.json').read_text())
    if sha(path) != receipt['source_sha256']: raise ValueError('source artifact changed')
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck['config']['parent_sha256'] != PARENT_SHA256 or ck['config']['format'] != 'masked-state-v1':
        raise ValueError('source provenance mismatch')
    base, _ = load_parent(DEFAULT_PARENT)
    model = MaskedState(base, ck['model']['level_scale'])
    model.load_state_dict(ck['model']); model.to(device).set_stage('frozen')
    return model, ck, receipt


def sample_windows(model, val, n):
    rng = np.random.default_rng(922)
    eligible = []
    for s in val:
        v = s['valid'].copy()
        if s['is_disrupt']:
            v &= np.round((s['down']-s['times'])*1000, 9) > 10.
        eligible.append(dict(s, valid=v))
    groups = pools(eligible); windows = []
    for k in range(n):
        group = groups[k % 2]; i, ends = group[int(rng.integers(len(group)))]
        end = int(rng.choice(ends)); s = val[i]
        x, m = simple_data.window(s, end, model.cfg.window)
        gain = np.exp(rng.uniform(-.35, .35, 23)).astype('float32')
        bias = rng.uniform(-1., 1., 23).astype('float32')
        windows.append(dict(shot=s['shot'], end=end, y=int(s['y'][end] > .5),
            x=x, mask=m, gain=gain, bias=bias))
    return windows


def summary(records):
    result = {}
    for condition in ('clean', 'shift'):
        rows = [r for r in records if r['condition'] == condition]
        result[condition] = dict(n=len(rows),
            before_bce=float(np.mean([r['before_bce'] for r in rows])),
            after_bce=float(np.mean([r['after_bce'] for r in rows])),
            risk_improved=sum(r['after_bce'] < r['before_bce'] for r in rows),
            reconstruction_improved=sum(r['after_completion'] < r['before_completion'] for r in rows),
            reconstruction_better_risk_worse=sum(r['after_completion'] < r['before_completion'] and r['after_bce'] > r['before_bce'] for r in rows),
            before_clean_logit_mse=float(np.mean([(r['before_logit']-r['clean_logit'])**2 for r in rows])),
            after_clean_logit_mse=float(np.mean([(r['after_logit']-r['clean_logit'])**2 for r in rows])),
            classes={str(y):dict(n=sum(r['y'] == y for r in rows),
                before_bce=float(np.mean([r['before_bce'] for r in rows if r['y'] == y])),
                after_bce=float(np.mean([r['after_bce'] for r in rows if r['y'] == y]))) for y in (0,1)})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--windows', type=int, default=128)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if args.windows < 2 or args.windows % 2: p.error('positive even window count required')
    if not torch.cuda.is_available(): raise RuntimeError('local CUDA required')
    torch.set_num_threads(2); start = time.monotonic()
    model, ck, receipt = load_source_candidate(args.source, 'cuda')
    old = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    roles = io23.source_roles(); val = load_source(roles['val'], io23.advanced_times())
    examples = sample_windows(model, val, args.windows)
    args.output.mkdir(parents=True)
    digest = hashlib.sha256()
    for e in examples:
        for k in ('x', 'mask', 'gain', 'bias'): digest.update(e[k].tobytes())
    protocol = dict(source=str(args.source.resolve()), source_sha256=sha(args.source),
        source_qualified=receipt['source_qualified'], split='JTEXT VAL160 only',
        learning_rates=[.001,.01,.1], steps=4, windows=args.windows,
        sequence_sha256=digest.hexdigest(), label_usage='score after signal-only updates, no label in update',
        synthetic='fixed per-window asinh gain exp(U[-.35,.35]) and bias U[-1,1]; not physical mechanism',
        qualification='failure analysis only if source gate failed; never bypass source gate',
        east_accessed=False, independent_confirmation=False,
        code_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')})
    _durable_json(args.output/'protocol.json', protocol)
    result = []
    for rate in protocol['learning_rates']:
        records = []
        for e in examples:
            x = torch.as_tensor(e['x'], device='cuda')[None]
            mask = torch.as_tensor(e['mask'], device='cuda')[None]
            gain = torch.as_tensor(e['gain'], device='cuda'); bias = torch.as_tensor(e['bias'], device='cuda')
            origin = {k:p.detach().clone().requires_grad_(True) for k,p in model.named_parameters() if k in set(model.encoder_names())}
            clean_logit = float(model.predict_fast(x, mask, origin)['logit'].detach())
            for condition in ('clean','shift'):
                moved = x if condition == 'clean' else distort(model, x, mask, gain, bias)
                before_logit = float(model.predict_fast(moved, mask, origin)['logit'].detach())
                fast, audit = adapt_observed(model, moved, mask, learning_rate=rate, steps=4)
                after_logit = float(model.predict_fast(moved, mask, fast)['logit'].detach())
                xx, mm, held = masked_views(moved, mask)
                pred = model.predict_fast(xx, mm, fast, mode='reconstruct')['reconstruction']
                after_completion = float(reconstruction_loss(model, pred, xx, held).detach())
                records.append(dict(shot=e['shot'], end=e['end'], y=e['y'], condition=condition,
                    before_logit=before_logit, after_logit=after_logit, clean_logit=clean_logit,
                    before_bce=float(np.logaddexp(0., before_logit)-e['y']*before_logit),
                    after_bce=float(np.logaddexp(0., after_logit)-e['y']*after_logit),
                    before_completion=audit['losses'][0], after_completion=after_completion,
                    parameter_delta_norm=audit['parameter_delta_norm']))
        measures = summary(records)
        gate = measures['clean']['after_bce'] <= measures['clean']['before_bce'] and measures['shift']['after_bce'] < measures['shift']['before_bce']
        _durable_json(args.output/f'rate_{rate}.json', dict(summary=measures, records=records, probe_gate=gate))
        row = dict(learning_rate=rate, summary=measures, probe_gate=gate)
        result.append(row); print(json.dumps(dict(stage='source_mechanism_probe', **row)), flush=True)
    if any(not torch.equal(v, model.state_dict()[k].cpu()) for k,v in old.items()):
        raise AssertionError('source mechanism probe modified persistent model')
    chosen = [r for r in result if r['probe_gate']]
    chosen_rate = min(chosen, key=lambda r:r['summary']['shift']['after_bce'])['learning_rate'] if chosen else None
    _durable_json(args.output/'completed.json', dict(status='COMPLETED', results=result,
        selected_rate=chosen_rate, source_qualified=receipt['source_qualified'],
        allow_east_candidate=bool(chosen and receipt['source_qualified']),
        persistent_unchanged=True, seconds=time.monotonic()-start,
        evidence='source window failure analysis; NOT shot-level online gains or physical recovery'))


if __name__ == '__main__': main()
