"""Diagnose one-step training versus actual history; no model is optimized."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23
from ..contextual_feedback_20261023.learning import WeightPolicy, policy_loss
from ..contextual_feedback_20261023.run import read_signal
from ..continual_episode_20261023.learning import episode_indices
from ..continual_episode_20261023.run import labeled
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..diverse_support_20261023.run import READ, make_parent
from ..heldout_threshold_20261023.run import REFERENCE, models_for_calibration
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import query_loss

PREVIOUS = io23.ROOT/'experiments/risk_aligned_online/runs/contextual_feedback360_20261023'
PREVIOUS_SHA = 'd8bbfe4e8ca3ab62845d4cef4149b00b742d5476b2534e7a6e3bd9912dcdebc1'


def gradient(loss,policy):
    return torch.cat([g.detach().flatten().double() for g in torch.autograd.grad(loss,tuple(policy.parameters()))])


def similarity(a,b):
    denominator = float(a.norm()*b.norm())
    return None if denominator == 0 else float(a@b)/denominator


def main(output):
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(2);torch.manual_seed(131);started=time.monotonic()
    assert sha(PREVIOUS/'completed.json') == PREVIOUS_SHA
    old=READ(PREVIOUS/'protocol.json')
    ckpath=PREVIOUS/'checkpoints/before000.pt'
    assert sha(ckpath)==READ(PREVIOUS/'completed.json')['artifact_sha256']['checkpoints/before000.pt']
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];saved=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json')
    policy=WeightPolicy(old['normalization']['center'],old['normalization']['scale']).cuda()
    policy.load_state_dict(torch.load(ckpath,map_location='cuda',weights_only=True)['fixed']);psaved=state_cpu(policy)
    allowed=set(old['preparation_ids']+old['calibration_ids']);reader=io23.read_outcome
    def guarded(device,sid):
        assert device=='EAST' and sid in allowed
        return reader(device,sid)
    rows=[]
    with patch.object(io23,'read_outcome',side_effect=guarded):
        for group,ids in (('prepare20',old['preparation_ids']),('calibration10_diagnostic_only',old['calibration_ids'])):
            for sid in ids:
                if time.monotonic()-started>900:raise TimeoutError('15min diagnostic limit')
                s=labeled(read_signal(sid),io23.read_outcome('EAST',sid),mapping)
                all_es=episode_indices(s);chosen=[]
                for label in (0,1):
                    es=[e for e in all_es if e['label']==label]
                    if es:
                        for index in sorted({len(es)//2,len(es)-1}):chosen.append(es[index])
                for e in chosen:
                    old_loss,old_row=policy_loss(model,policy,[(s,e)],device='cuda');g0=gradient(old_loss,policy)
                    new1,r1=query_loss(model,policy,s,e,device='cuda',tail_steps=1);g1=gradient(new1,policy)
                    new2,r2=query_loss(model,policy,s,e,device='cuda',tail_steps=2);g2=gradient(new2,policy)
                    assert r1['after_bce']==r2['after_bce']
                    rows.append(dict(group=group,shot=sid,**e,updates=len(r2['all_counts']),
                        before_bce=old_row[0]['before_bce'],one_step_bce=old_row[0]['after_bce'],
                        history_bce=r2['after_bce'],cosine_one_vs_history1=similarity(g0,g1),
                        cosine_one_vs_history2=similarity(g0,g2),cosine_history1_vs_history2=similarity(g1,g2),
                        norm_one=float(g0.norm()),norm_history1=float(g1.norm()),norm_history2=float(g2.norm())))
                print(dict(shot=sid,episodes=len(rows),seconds=time.monotonic()-started),flush=True)
    assert_unchanged(model,saved);assert_unchanged(policy,psaved);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    summary={}
    for group in sorted({r['group'] for r in rows}):
        rs=[r for r in rows if r['group']==group]
        summary[group]=dict(episodes=len(rs),shots=len({r['shot'] for r in rs}),
            max_cumulative_updates=max(r['updates'] for r in rs))
        for key in ('cosine_one_vs_history1','cosine_one_vs_history2','cosine_history1_vs_history2'):
            a=[r[key] for r in rs if r[key] is not None]
            summary[group][key]=dict(n=len(a),negative=sum(x<0 for x in a),median=float(np.median(a)),minimum=min(a))
    _durable_json(output,dict(status='COMPLETED_POSTHOC_TRAIN_DEPLOYMENT_PROBE',rows=rows,summary=summary,
        policy_unchanged=True,source_unchanged=True,test360_labels_used=False,seconds=time.monotonic()-started,
        previous_sha256=PREVIOUS_SHA,checkpoint_sha256=sha(ckpath),
        code_sha256={str(p.relative_to(io23.ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},
        limitation='No training or performance claim. Calibration data used only to diagnose; '
                   'their labels never enter the policy. Last-step gradients are truncated and context is stopped.'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);main(p.parse_args().output)
