"""POSTHOC ORACLE: ignore early disruptive labels; never an online policy."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .. import io23
from ..contextual_feedback_20261023.run import read_signal
from ..continual_episode_20261023.audit import recount
from ..continual_episode_20261023.run import labeled
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..diverse_support_20261023.run import READ,make_parent
from ..heldout_threshold_20261023.run import REFERENCE,models_for_calibration
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha,metrics
from ..warmstart_alignment_20261022.run import public
from .learning import predict

PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/正确监督不压早期破裂片段_诊断追加约定_20261023.md'


def teaching_mask(labels,valid,disruptive):
    labels,valid=np.asarray(labels),np.asarray(valid)
    if labels.shape!=valid.shape or valid.dtype!=bool or not np.isin(labels[valid],[0,1]).all():
        raise ValueError('valid binary oracle supervision required')
    if not disruptive and labels[valid].any():raise ValueError('safe labels must be zero')
    return valid & (labels==1) if disruptive else valid.copy()


def main(parent,root):
    if root.exists():raise FileExistsError(root)
    torch.set_num_threads(2);torch.manual_seed(131);started=time.monotonic()
    old=READ(parent/'protocol.json');done=READ(parent/'completed.json')
    for f,h in done['artifact_sha256'].items():assert sha(parent/f)==h,f
    assert READ(parent/'verification.json')['completed_sha256']==sha(parent/'completed.json')
    base,_=make_parent();model=models_for_calibration(base)['mapped_fixed'];saved=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');order=old['shot_ids']
    protocol=dict(old,parent_completed_sha256=sha(parent/'completed.json'),
        selective_mask=True,plan_sha256=sha(PLAN),code_sha256={str(f.relative_to(io23.ROOT)):sha(f)
            for f in (Path(__file__),Path(__file__).with_name('learning.py'))},
        scope='ADDED_POSTHOC_LABEL_ORACLE; NOT DEPLOYABLE; EARLY DISRUPTIVE WINDOWS EXCLUDED')
    root.mkdir(parents=True);(root/'shots').mkdir();_durable_json(root/'protocol.json',protocol)
    rows={a:[] for a in old['arms']};updates=[];outcomes=[];checks=0
    for number,sid in enumerate(order,1):
        if time.monotonic()-started>1200:raise TimeoutError('20min diagnostic budget')
        flag,down=io23.read_outcome('EAST',sid);outcomes.append(dict(shot=sid,is_disrupt=flag,down=down))
        with np.load(parent/'shots'/f'{sid}.npz') as previous:
            times=previous['times'].copy();valid=previous['valid'].copy();static=previous['static'].copy()
            if not flag:
                curves={a:previous[a].copy() for a in ('oracle1','oracle4')}
                receipt=dict(shot=sid,reused_safe_parent=True,parent_curve_sha256=sha(parent/'shots'/f'{sid}.npz'))
            else:
                s=labeled(read_signal(sid),(flag,down),mapping)
                np.testing.assert_array_equal(times,s['times']);np.testing.assert_array_equal(valid,s['valid'])
                curves,receipt=predict(model,{k:s[k] for k in ('values','available')},s['y'],teaching_mask(s['y'],s['valid'],True))
                receipt=dict(shot=sid,reused_safe_parent=False,updates=receipt)
        curves['static']=static;updates.append(receipt)
        for a in curves:
            np.testing.assert_allclose(curves[a][:256],static[:256],atol=2e-7,rtol=1e-6)
            rows[a].append(dict(shot=sid,times=times[valid],scores=curves[a][valid],is_disrupt=flag,down=down))
        np.savez_compressed(root/'shots'/f'{sid}.npz',times=times,valid=valid,**curves)
        if number%40==0:print(dict(stage='SELECTIVE_LABEL_ORACLE',shots=number,seconds=time.monotonic()-started),flush=True)
    results={a:metrics(rs,old['threshold'],lead=30.) for a,rs in rows.items()}
    for i,sid in enumerate(order):
        o=outcomes[i]
        with np.load(root/'shots'/f'{sid}.npz') as z:
            for a in results:
                r=dict(shot=sid,**recount(z['times'],z[a],o['is_disrupt'],o['down'],old['threshold']))
                assert r==results[a]['predictions'][i];checks+=1
    assert_unchanged(model,saved);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(root/'updates.json',updates);_durable_json(root/'per_shot.json',results)
    artifacts=[root/'protocol.json',root/'updates.json',root/'per_shot.json',*sorted((root/'shots').glob('*.npz'))]
    final=dict(status='COMPLETED_NONDEPLOYABLE_SELECTIVE_ORACLE',recounted=checks,source_unchanged=True,
        persistent_risk_model_unchanged=True,seconds=time.monotonic()-started,
        newly_inferred_disruptive_shots=sum(r['is_disrupt'] for r in outcomes),
        reused_safe_shots=sum(not r['is_disrupt'] for r in outcomes),metrics={a:public(v) for a,v in results.items()},
        artifact_sha256={str(f.relative_to(root)):sha(f) for f in artifacts})
    _durable_json(root/'completed.json',final);print(final['metrics'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();main(args.parent,args.output)
