"""Attribute the guard by replaying the same already-fitted global proposals as shadow states."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from run_layer0_direct_transfer_20261112 import (
    DIRECT, PARENT, block_files, direct_curves, warning_rows,
)
import moving_peak_replay_20261105 as r
import run_layer0_query_memory_20261105 as core
from experiments.risk_aligned_online.warning_lead_metrics import summarize
from verify_confirmation_raw_replay_20261105 import counts, pair, raw_truth, score


SOURCE = r.ROOT/'probe_outputs/layer0_direct_transfer360_20261112_v2'
SHADOW = ('shadow_direct','shadow_with_intra')
ARMS = (*PARENT,*SHADOW)


def load_released_proposal(path, receipt, *, completed_rank):
    if receipt['first_next_rank']!=completed_rank+1 or 'fit' not in receipt:
        raise ValueError('proposal cannot be used before its recorded release')
    if r.sha(path)!=receipt['proposal_sha256']:
        raise ValueError('frozen proposal changed')
    with np.load(path,allow_pickle=False) as z:
        return z['delta']


def run(out):
    r.configure()
    start = time.monotonic()
    p,s,result = (r.read(SOURCE/k) for k in ('protocol.json','seal.json','result.json'))
    order,cfg,_,base,digest,mapping,_,fit = core.parents()
    if (order!=p['order'] or digest!=p['source_state_sha256'] or len(s['shots'])!=360
            or r.sha(SOURCE/'protocol.json')!=s['protocol_sha256'] or not result['guard_recount']):
        raise ValueError('requires the completed independently scored direct-transfer run')
    (out/'shots').mkdir(parents=True)
    r.write_new(out/'protocol.json',dict(scope='EAST360_FROZEN_PROPOSAL_GUARD_SHADOW_DEVELOPMENT',
        parent=str(SOURCE), parent_seal=r.sha(SOURCE/'seal.json'), order=order, threshold=p['threshold'],
        policy='use latest already fitted proposal beginning on its original first_next_rank; '
               'ignore accept/reject only; never refit or read labels',
        arms=ARMS, code_sha256=r.sha(Path(__file__))))
    state = state_file = None
    rule = r.load_rule(fit,base.center.device)
    seals, events = [], []
    with patch.object(r.io23,'read_outcome',side_effect=RuntimeError('all outcomes forbidden in shadow')):
        for rank,sid in enumerate(order,1):
            filename = f'{rank:04d}_{sid}.npz'
            old = SOURCE/'shots'/filename
            if r.sha(old)!=s['shots'][rank-1]['curve_sha256']:
                raise ValueError('matched parent curve changed')
            with np.load(old,allow_pickle=False) as z:
                t = z['times']
                curves = {k:z[k] for k in PARENT}
            if state is None:
                curves.update({k:curves[a] for k,a in zip(SHADOW,PARENT,strict=True)})
            else:
                raw = core.raw_for(sid,cfg['raw_sha256'])
                np.testing.assert_array_equal(t,raw['times'])
                case = core.as_case(base,raw,rank,mapping,rule)
                a,b = direct_curves(base,case,state).values()
                curves.update(dict(zip(SHADOW,(a,b),strict=True)))
            for k,a in zip(SHADOW,PARENT,strict=True):
                np.testing.assert_array_equal(curves[k][:135],curves[a][:135])
            path = out/'shots'/filename
            r.save_arrays(path,**curves)
            seals.append(dict(rank=rank,shot=sid,curve_sha256=r.sha(path),proposal=state_file))
            if rank%p['block']==0:
                block = rank//p['block']
                _,_,update = block_files(SOURCE,block)
                if r.sha(update)!=s['blocks'][block-1]['update_sha256']:
                    raise ValueError('historical proposal receipt changed')
                receipt = r.read(update)
                if 'fit' in receipt:
                    state_file = f'{block:02d}_proposal.npz'
                    delta = load_released_proposal(SOURCE/'blocks'/state_file,receipt,completed_rank=rank)
                    state = torch.as_tensor(delta,device=base.center.device)
                    events.append(dict(proposal=state_file,rank_start=rank+1,decision=receipt['decision']))
                print(dict(rank=rank,frozen_proposals=len(events)),flush=True)
    if r.state_digest(base)!=digest:
        raise ValueError('frozen parent changed')
    r.write_new(out/'seal.json',dict(curves=seals,events=events,seconds=time.monotonic()-start))


def score_run(out):
    p,s = (r.read(out/k) for k in ('protocol.json','seal.json'))
    if (len(p['order'])!=360 or len(s['curves'])!=360 or p['parent_seal']!=r.sha(SOURCE/'seal.json')
            or p['code_sha256']!=r.sha(Path(__file__))):
        raise ValueError('missing shadow seal or changed input')
    prior = r.read(SOURCE/'per_shot.json')
    rows = {k:[] for k in ARMS}
    state = None
    schedule = {e['rank_start']:e['proposal'] for e in s['events']}
    for rank,sid in enumerate(p['order'],1):
        file = out/'shots'/f'{rank:04d}_{sid}.npz'
        state = schedule.get(rank,state)
        if s['curves'][rank-1]!=dict(rank=rank,shot=sid,curve_sha256=r.sha(file),proposal=state):
            raise ValueError('incomplete or early shadow state')
        y,t = raw_truth(r.io23.path_for('EAST',sid),sid)
        with np.load(file,allow_pickle=False) as z:
            for k in ARMS:
                row = score(t,z[k],y,p['threshold'])
                if k in PARENT and row!=prior[k][rank-1]:
                    raise ValueError('shadow scoring changed a fixed parent')
                rows[k].append(row)
    metrics = {k:counts(v) for k,v in rows.items()}
    paired = {f'{k}_vs_{a}':pair(rows[k],prior[a]) for k,a in zip(SHADOW,DIRECT,strict=True)}
    r.write_new(out/'per_shot.json',rows)
    r.write_new(out/'result.json',dict(status='COMPLETE_FROZEN_GUARD_SHADOW_DEVELOPMENT',
        metrics=metrics,paired=paired,raw_verified=len(ARMS)*360,
        leads={k:summarize(warning_rows(v)) for k,v in rows.items()},
        interpretation='fixed rejected proposal path; not an unguarded retrain or a deployable operating point'))
    print({k:(a['tp'],a['fp'],a['tp_30_to_3000']) for k,a in metrics.items()},flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('out',type=Path)
    args = cli.parse_args()
    run(args.out.resolve())
    score_run(args.out.resolve())
