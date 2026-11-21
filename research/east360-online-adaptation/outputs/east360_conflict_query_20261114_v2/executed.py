"""Coverage of fixed-budget queries using already-landed diagnostic disagreements."""

import argparse
import csv
import os
from pathlib import Path
from shutil import copyfile
from time import monotonic_ns
from unittest.mock import patch

import numpy as np
from scipy.stats import rankdata

from joint_lora_hindsight_capacity_20261104 import write_new
from layer0_query_protocol_20261105 import reveal_block

from . import commissioning_attention_reweight_20261114 as attention
from . import io23
from .history_gradient_metric_20261102.run import read


REFERENCE = io23.ROOT/'outputs/east360_attention_reweight_20261114_v1'
FEEDBACK = io23.ROOT/'outputs/east360_local_dynamics_20261114_v2/forecast_sealed.json'
AGE, HOLD = 31, 32
METHODS = ('original','parent_near','surprise','high_unexplained','alternating','uniform')
SCORE_INDEX = ('parent_near','surprise','high_unexplained')
KINDS = ('timely','late','never','false','safe')


def latest_landed(issued, indices, *, offset=AGE):
    return np.searchsorted(issued+offset,indices,side='right')-1


def usable(at,supported,parent,tau,age,*,slot_end=None):
    hits = np.flatnonzero(parent>=tau)
    first = hits[0] if len(hits) else len(parent)
    known = at>=age
    if slot_end is not None:
        known &= (at>=slot_end) & (at<slot_end+HOLD)
    return supported & known & (at<first)


def unexplained_high(parent, signal):
    # Empirical percentiles use one completed shot's observable scores, never
    # their outcomes. Equal ranks produce no disagreement by construction.
    return np.maximum(0.,(rankdata(signal)-rankdata(parent))/len(parent))


def shot_values(parent,tau,payload):
    q = np.asarray(payload['q'],np.int64)
    # Respect one old issue slot and the recorded order, not the eventual end.
    t = np.arange(127,len(parent),4,dtype=np.int64)
    ix = latest_landed(q,t)
    valid = ix>=0
    ix = ix[valid]
    if not len(ix):
        return dict(windows=0,parent_near=0.,surprise=0.,high_unexplained=0.)
    t = t[valid]
    good = usable(t,np.ones(len(t),bool),parent,tau,age=127,slot_end=q[ix]+AGE)
    t,ix = t[good],ix[good]
    if not len(t):
        return dict(windows=0,parent_near=0.,surprise=0.,high_unexplained=0.)
    err = np.asarray(payload['error'],np.float64)
    # Actual change surprises both the frozen H25 forecast and holding the
    # last observation. Two hard checks, not large error on an easy variable.
    novelty = np.minimum(err[0,ix,1],err[1,ix,1])
    p = parent[t]
    return dict(windows=len(t),parent_near=float(p.max()),surprise=float(novelty.max()),
                high_unexplained=float(unexplained_high(p,novelty).max()))


def select(rows,method):
    # Prefer shots without a first alarm, where a wrong no-alarm decision can
    # be invisible to uncertainty selection. Tie order is the acquired block.
    return max(rows,key=lambda r:(not r['observed_alarm'],r[method],-r['rank']))['shot']


def alternate(original,rows,block):
    """Preserve original acquisition on half the predetermined blocks."""
    return select(rows,'parent_near') if block%2==0 else original


def run(out):
    p = read(REFERENCE/'protocol.json')
    s = read(REFERENCE/'seal.json')
    f = read(FEEDBACK)
    order = p['order']
    if (len(order)!=360 or s['order']!=f['order'] or s['order']!=order or p['startup']!=20
            or f['q_offset']!=AGE or tuple(f['names'][:2])!=('hold','source')):
        raise ValueError('the common feedback, read clock and fixed360 are required')
    out.mkdir()
    (out/'blocks').mkdir()
    copyfile(__file__,out/'executed.py')
    source_commit = read(attention.QUERY/'seal.json')['blocks']
    select_random = np.random.default_rng(41232)
    whole, histories = [], {k:[] for k in METHODS}
    tau = p['calibration']['0.2']
    oracle = io23.read_outcome
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('no labels before selection')):
        for b in range(1,19):
            original = attention.checked_selection(b,order,source_commit)
            rows = []
            for rank,sid in enumerate(order[(b-1)*20:b*20],(b-1)*20+1):
                file = REFERENCE/'shots'/f'{rank:04d}_{sid}.npz'
                with np.load(file,allow_pickle=False) as stored:
                    z = stored['bias_only']
                threshold = p['original'] if rank<=p['startup'] else tau
                row = dict(shot=sid,rank=rank,observed_alarm=bool((z>=threshold).any()))
                if rank>p['startup']:
                    row.update(shot_values(z,tau,f['shots'][str(sid)]))
                rows.append(row)
            ids = [r['shot'] for r in rows]
            if 1<b<18:
                choices = dict(original=original[0],alternating=alternate(original[0],rows,b),
                               uniform=int(select_random.choice(ids)),
                               **{k:select(rows,k) for k in SCORE_INDEX})
            else:
                choices = {}
            whole.append(dict(block=b,rank_end=b*20,rows=rows,choice=choices))
            # Commit all arms before reading any outcomes of this block. The
            # union is a simulation convenience; no arm consumes another's label.
            if choices:
                q = list(dict.fromkeys(choices.values()))
                item = out/'blocks'/f'{b:02d}_commit.json'
                write_new(item,dict(selected=q,methods=choices,committed_ns=monotonic_ns()))
                fd = os.open(item.parent,os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                found = reveal_block(item,q,oracle)
                write_new(out/'blocks'/f'{b:02d}_reveal.json',found)
                known = {x['shot']:x for x in found['outcomes']}
                for k,sid in choices.items():
                    histories[k].append(dict(**known[sid],rank=order.index(sid)+1,available_from=b*20+1))
    if any(len(v)!=16 for v in histories.values()):
        raise ValueError('all policies require the same 16-query label budget')
    write_new(out/'seal.json',dict(order=order,threshold=tau,choices=whole,histories=histories,
        inputs=[str(REFERENCE),str(FEEDBACK)],
        condition='coverage only; fixed shadow alarms, each arm sees only its own 16 queries'))


def category(row):
    if row['disruptive']:
        return 'timely' if row['timely'] else 'late' if row['alarm'] else 'never'
    return 'false' if row['alarm'] else 'safe'


def summarize(out):
    s = read(out/'seal.json')
    ref = read(REFERENCE/'per_shot.json')
    combined = []
    result = {}
    for arm in ('bias_only','current_5'):
        r = ref['0.2'][arm]
        if s['order']!=[x['shot'] for x in r]:
            raise ValueError('wrong frozen outcomes')
        key = {x['shot']:category(x) for x in r}
        lookup = {x['shot']:x for x in r}
        target = {k:[dict(shot=v['shot'],rank=i) for i,v in enumerate(r,1) if i>20
                     and key[v['shot']]==k] for k in KINDS}
        result[arm] = {}
        for k,history in s['histories'].items():
            subset = {x['shot'] for x in history}
            insert = {j:[h for h in history if key[h['shot']]==j] for j in KINDS}
            for h in history:
                a = lookup[h['shot']]
                if (h['disruptive'],h['down'])!=(a['disruptive'],a['down_s']):
                    raise AssertionError('reveal has a different original outcome')
                combined.append(dict(arm=arm,method=k,shot=h['shot'],rank=h['rank'],
                                     available_from=h['available_from'],category=key[h['shot']]))
            covered = {j:[t['shot'] for t in target[j] if t['shot'] not in subset
                         and any(h['available_from']<=t['rank'] for h in insert[j])] for j in KINDS}
            result[arm][k] = dict(selected={j:len(x) for j,x in insert.items()},
                coverage={j:dict(covered=len(v),denominator=sum(t['shot'] not in subset for t in target[j]),
                                 shots=v) for j,v in covered.items()})
    write_new(out/'result.json',result)
    with (out/'queries.csv').open('x',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    print({k:{m:v['selected'] for m,v in t.items()} for k,t in result.items()},flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output',type=Path)
    opts = cli.parse_args()
    run(opts.output.resolve())
    summarize(opts.output.resolve())
