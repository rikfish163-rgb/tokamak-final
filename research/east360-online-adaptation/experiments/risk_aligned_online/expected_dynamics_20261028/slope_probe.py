"""Fit source dynamic priors; test matured-feedback forecasting on full EAST360."""
import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import read
from ..delayed_recurrent_memory_20261025.model import patch_signal
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import fit_prefix_map, transform
from .matured_slope import SlopeMemory


def fit_prior(examples):
    xx = torch.zeros(4, 23, dtype=torch.float64)
    xy = torch.zeros_like(xx); counts = torch.zeros_like(xx)
    for e in examples:
        a = e['complete'].clone(); a[:, 2] = False
        x = torch.where(a, e['means'].double(), 0.)
        d = torch.zeros_like(x); pair = torch.zeros_like(a)
        pair[1:] = a[1:] & a[:-1]; d[1:] = torch.where(pair[1:], x[1:]-x[:-1], 0.)
        for h in range(1, 5):
            if len(x) <= h:
                continue
            m = pair[:-h] & a[h:]; c = m.sum(0)
            xx[h-1] += (d[:-h].square()*m).sum(0)/c.clamp_min(1)
            xy[h-1] += (d[:-h]*(x[h:]-x[:-h])*m).sum(0)/c.clamp_min(1)
            counts[h-1] += c > 0
    moment = xx/counts.clamp_min(1)
    prior = torch.where(xx > 0, xy/xx.clamp_min(torch.finfo(xx.dtype).tiny), 0.)
    floor = moment.median(1, keepdim=True).values
    if (floor <= 0).any():
        raise ValueError('source excitation insufficient for declared ridge scale')
    ridge = torch.maximum(moment, floor)
    return prior.float(), ridge.float(), dict(moments=moment.tolist(), observed_shots=counts.tolist(),
        unexcited=(xx == 0).tolist(), method='equal_shot_matured_delta_regression_TRAIN_ONLY')


def summarize(rows):
    out = {}
    for group in ('all', 'D', 'S'):
        sub = [r for r in rows if group == 'all' or r['disruptive'] == (group == 'D')]
        if not sub:
            raise ValueError('empty scored group')
        out[group] = dict(shots=len(sub), errors={k:float(np.mean([r['errors'][k] for r in sub]))
            for k in sub[0]['errors']},
            improved_persistence=sum(r['errors']['memory']<r['errors']['persistence'] for r in sub),
            improved_frozen=sum(r['errors']['memory']<r['errors']['off'] for r in sub))
    return out


def errors(x, a, traces, weights, within=None):
    n = len(x)
    if n <= 4:
        raise ValueError('full four-horizon scoring requires five observed patches')
    a = a.clone(); a[:, 2] = False
    x = torch.where(a, x, 0.)
    y = torch.stack([x[h:n-4+h]-x[:-4] for h in range(1,5)], 1)
    valid = torch.stack([a[h:n-4+h]&a[:-4] for h in range(1,5)], 1)
    w = valid*weights
    if within is not None:
        w = w*torch.as_tensor(within)[..., None]
    if w.sum() <= 0:
        raise ValueError('no eligible target')
    result = {k:float(((t['forecast'][:-4]-y).square()*w).sum()/w.sum()) for k,t in traces.items()}
    result['persistence'] = float((y.square()*w).sum()/w.sum())
    if not all(np.isfinite(v) and v >= 0 for v in result.values()):
        raise ValueError('nonfinite or negative error')
    return result


@torch.no_grad()
def run(output, reference, stream):
    if output.absolute().parent != io23.ROOT/'probe_outputs' or output.exists():
        raise ValueError('new worktree output required')
    torch.set_num_threads(2); start = time.monotonic()
    p = read(reference, 'protocol.json'); done = read(reference, 'complete.json')
    roles = simple_data.source_roles(); cache = Path(p['cache'])
    if (not done['complete'] or roles['train'] != p['canonical_train_ids'] or roles['val'] != p['val_ids']
            or sha(cache/'inventory.json') != p['cache_inventory_sha256']):
        raise ValueError('frozen source TRAIN/VAL cache required')
    examples = {}
    for row in read(cache, 'inventory.json'):
        sid = row['shot']; path = cache/'cache'/f'{sid}.pt'
        if row['role'] not in ('train','val') or sid not in roles[row['role']] or sha(path) != row['cache_sha256']:
            raise ValueError('source cache identity mismatch')
        examples[sid] = torch.load(path, map_location='cpu', weights_only=True)
    prior, ridge, evidence = fit_prior([examples[s] for s in roles['train']])
    weights = torch.tensor(p['weights'])
    model = SlopeMemory(64, prior, ridge, weights).eval()
    base, _ = load_parent(DEFAULT_PARENT)
    previous = read(stream, 'protocol.json'); old_done = read(stream, 'completed.json')
    order = read(stream, 'order.json'); ids = [r['shot'] for r in order]
    mapping = read(stream, 'input_map.json'); hashes = read(stream, 'input_hashes.json')
    if (old_done['status'] != 'COMPLETED' or len(ids) != len(set(ids)) or len(ids) != 360
            or previous['source_sha256'] != PARENT_SHA256 or previous['warmup'] != 20
            or sha(stream/'order.json') != previous['order_sha256']
            or mapping['calibration_ids'] != ids[:20] or mapping['policy'] != 'density_gain'):
        raise ValueError('locked complete360 required')
    output.mkdir(parents=True); (output/'seals').mkdir()
    files = [Path(__file__), Path(__file__).parent/'matured_slope.py']
    protocol = dict(scope='MATURED_DIAGNOSTIC_RESPONSE_NOT_RISK', prior=prior.tolist(), ridge=ridge.tolist(),
        weights=weights.tolist(), train_ids=roles['train'], val_ids=roles['val'],
        source_classifier_sha256=PARENT_SHA256, cache=str(cache), cache_inventory_sha256=sha(cache/'inventory.json'),
        reference=str(reference), reference_protocol_sha256=sha(reference/'protocol.json'),
        stream=str(stream), stream_protocol_sha256=sha(stream/'protocol.json'), order=ids,
        window_patches=16, source_pseudo_observations=1, full360_includes_cold20=True,
        plan_sha256=sha(io23.ROOT/'MATURED_SLOPE_PLAN.md'),
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files})
    _durable_json(output/'protocol.json', protocol); _durable_json(output/'source_moments.json', evidence)
    val = []
    for sid in roles['val']:
        e = examples[sid]; x, a = e['means'], e['complete']
        traces = {m:model.trace(x,a,mode=m) for m in ('off','memory','reverse','hold')}
        val.append(dict(shot=sid, disruptive=e['flag'], errors=errors(x,a,traces,weights)))
    _durable_json(output/'source_val.json', dict(rows=val, summary=summarize(val)))
    print(dict(source_val=summarize(val)), flush=True)
    rows=[]; prefix=[]
    for i,sid in enumerate(ids):
        if sha(io23.path_for('EAST',sid)) != hashes[str(sid)]:
            raise ValueError('EAST record changed')
        raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        signal={k:raw[k] for k in ('values','available')}
        if i >= 20:
            signal=transform(signal,mapping)
        x,a=patch_signal(signal,base.center,base.scale)
        traces={m:model.trace(x,a,mode=m) for m in ('off','memory','reverse','hold')}
        _durable_json(output/'seals'/f'{i+1:04d}_{sid}.json',dict(shot=sid,index=i+1,outcome_used=False,
            sha256={m:hashlib.sha256(t['forecast'].numpy().tobytes()).hexdigest() for m,t in traces.items()}))
        flag, down=io23.read_outcome('EAST',sid)
        times=np.stack([raw['times'][(np.arange(len(x)-4)+h+1)*8-1] for h in range(1,5)],1)
        within=times <= down if flag else np.ones_like(times,dtype=bool)
        rows.append(dict(shot=sid,index=i+1,disruptive=flag,errors=errors(x,a,traces,weights,within),
                         updates=traces['memory']['updates']))
        if i < 20:
            prefix.append({k:raw[k] for k in ('shot','device','values','available')})
        if i == 19:
            rebuilt=fit_prefix_map(mapping['source_statistics'],prefix,order=ids,count=20,policy='density_gain')
            if rebuilt != mapping:
                raise ValueError('ended-prefix alignment differs')
            prefix=[]
        if (i+1)%60==0:
            print(dict(shots=i+1,seconds=time.monotonic()-start),flush=True)
    result={s:summarize(sub) for s,sub in (('full360',rows),('cold20',rows[:20]),('suffix340',rows[20:]))}
    checks={g+'_'+b:result['full360'][g]['errors']['memory'] < result['full360'][g]['errors'][b]
            for g in ('all','D','S') for b in ('persistence','off')}
    for path,h in protocol['code_sha256'].items():
        if sha(io23.ROOT/path) != h:
            raise ValueError('code changed during experiment')
    if sha(DEFAULT_PARENT) != PARENT_SHA256 or sha(io23.ROOT/'MATURED_SLOPE_PLAN.md') != protocol['plan_sha256']:
        raise ValueError('source/plan changed')
    _durable_json(output/'complete.json',dict(complete=True,results=result,rows=rows,
        gate=dict(eligible=all(checks.values()),checks=checks,scope='FORECAST_ONLY_NOT_INTRA_RISK'),
        seconds=time.monotonic()-start,source_classifier_unchanged=True,phase_labels_read=0))
    print(result['full360'],flush=True);print(checks,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('output','reference','stream'):
        parser.add_argument('--'+arg,type=Path,required=True)
    args=parser.parse_args();run(args.output,args.reference,args.stream)
