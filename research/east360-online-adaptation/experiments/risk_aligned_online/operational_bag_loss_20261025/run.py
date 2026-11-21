"""Fit on20 authorized past shots; seal seven complete10-shot predictions."""
import argparse
import gc
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile
import numpy as np
import torch

from .. import io23
from ..feedback_cone_feasibility_20261024.run import (
    READ,make_parent,models_for_calibration,state_cpu,assert_unchanged,
    signal,transform,REFERENCE,DEFAULT_PARENT,PARENT_SHA256,
)
from ..feedback_cone_feasibility_20261024.stream import signal_parts
from ..paired_feedback_20261022.model import initial_encoder
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.run import predict_many
from ..signal_driven_span_20261024 import run as evaluator
from .learning import (ARMS,feedback_context,train_policy,predict,eligible,scan_bag,candidate_ends,operational_losses)

PREVIOUS=io23.ROOT/'experiments/risk_aligned_online/runs/finite_conditional_loss_cal10_20261025'
GEOMETRY=io23.ROOT/'experiments/risk_aligned_online/runs/clock_channel_loss_cal10_20261025'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/整炮及时报警外层目标_事前约定_20261025.md'


def build_history(model,history,mapping,threshold,*,progress=None):
    shots={};receipts=[];bytes_used=0
    for sid in history:
        raw=signal(sid);s=transform({k:raw[k] for k in ('values','available')},mapping)
        old=READ(GEOMETRY/'geometry'/f'{sid}.json')
        shot=dict(signal=s,times=raw['times'],flag=old['is_disrupt'],down=old['down'],cache={})
        ends=candidate_ends(raw['times'],shot['flag'],shot['down'])
        counts=[count for count in range(256,len(raw['times']),128) if ((ends>=count)&(ends<count+128)).any()]
        assert counts==sorted({r['count'] for r in old['rows']})
        for count in counts:
            fast=initial_encoder(model);parts,r=signal_parts(model,s,count,fast,hold=False,device='cuda')
            if parts is None:raise AssertionError('historical archived block must have observations')
            scale=float(parts.double().norm(dim=1).max())
            if scale<=0:raise AssertionError('historical archived block requires nonzero direction')
            c,mask=feedback_context(model,s,count,fast,hold=False,device='cuda')
            cpu=parts.cpu();bytes_used+=cpu.numel()*cpu.element_size()
            if bytes_used>5*1024**3:raise MemoryError('declared historical cache exceeds5GiB')
            shot['cache'][count]=dict(parts=cpu,context=c.cpu(),observed=mask.cpu(),scale=scale)
            receipts.append(dict(shot=sid,count=count,scale=scale,context=c.cpu().tolist(),observed=mask.cpu().tolist(),
                channel_gram=(parts.double()@parts.double().T).cpu().tolist(),observations=r['observed']))
        shot['baseline'],_=scan_bag(model,None,shot,threshold,with_gradient=False)
        shots[sid]=shot
        if progress:progress(sid,len(receipts),bytes_used)
    return shots,receipts,bytes_used


def evaluate(root,protocol):
    with patch.object(evaluator,'ARMS',ARMS):results=evaluator.evaluate(root,protocol)
    outcomes={r['shot']:r for r in READ(root/'outcomes.json')};details={}
    for arm in ('off',)+ARMS:
        rows=[]
        for sid in protocol['shot_ids']:
            o=outcomes[sid]
            with np.load(root/'shots'/f'{sid}.npz') as z:
                _,valid=io23.labels_at(z['times'],o['is_disrupt'],o['down'],125.)
                rows.append(dict(shot=sid,times=z['times'][valid],scores=z[arm][valid].astype(np.float64),
                    is_disrupt=o['is_disrupt'],down=o['down']))
        loss,detail=operational_losses(rows,protocol['threshold']);results[arm].update(loss);details[arm]=detail
    _durable_json(root/'bag_losses.json',details)
    return results


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    done=READ(PREVIOUS/'completed.json');assert READ(PREVIOUS/'audit.json')['status']=='passed'
    for name,digest in done['artifact_sha256'].items():assert sha(PREVIOUS/name)==digest,name
    old=READ(PREVIOUS/'protocol.json');history=old['preparation_ids'];ids=old['shot_ids']
    assert not set(history)&set(ids) and not set(history+ids)&set(old['excluded360_ids'])
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    assert sha(REFERENCE/'fold0/mapped_fixed/prepared.pt')==old['parent_sha256']
    torch.set_num_threads(2);torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True);torch.backends.cuda.matmul.allow_tf32=False
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];original=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');start=time.monotonic()
    root.mkdir(parents=True)
    for sub in ('shots','updates','training'):(root/sub).mkdir()
    files=set(Path(__file__).parent.glob('*.py'));files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_operational_bag_loss.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    p=dict(shot_ids=ids,preparation_ids=history,excluded360_ids=old['excluded360_ids'],threshold=old['threshold'],
        arms=('off',)+ARMS,parent_sha256=old['parent_sha256'],source_sha256=PARENT_SHA256,
        mapping_sha256=sha(REFERENCE/'input_map.json'),previous_completed_sha256=sha(PREVIOUS/'completed.json'),
        plan_sha256=sha(PLAN),epochs=10,optimizer='Adam',outer_lr=.05,step_bound=.01,
        global_trainable_parameters=24,context_trainable_parameters=47,update_every=128,first_update=256,
        parameter_reset='parent every update',inter_steps=0,alarm_updates=0,
        outer_objective='whole-shot max logit relative fixed threshold; disruptive eligible strict lead>30ms',
        labels='archived20 historical outcomes for meta-training; all7/all10 predictions sealed before new outcome reads',
        scope='DEVELOPMENT_ONLY_NOT_COMPLETE_CONTINUAL_SYSTEM',independent_confirmation=False,
        event_labels_added=False,code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
        input_sha256={str(io23.path_for('EAST',sid)):sha(io23.path_for('EAST',sid)) for sid in history+ids},
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        training_backend='grad-enabled math; exact32-chunk replay',deployment_backend='same no_grad fused backend as prior arms')
    _durable_json(root/'protocol.json',p)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
    def deny(*args,**kwargs):raise AssertionError('unsealed/new outcomes forbidden')
    def check_time():
        if time.monotonic()-start>1800:raise TimeoutError('declared30minute experiment limit')
    with patch.object(io23,'read_outcome',side_effect=deny):
        def cache_progress(sid,blocks,size):
            print(dict(stage='cache',shot=sid,blocks=blocks,GiB=round(size/1024**3,3),seconds=round(time.monotonic()-start,2)),flush=True)
            check_time()
        shots,receipts,bytes_used=build_history(model,history,mapping,p['threshold'],progress=cache_progress)
        _durable_json(root/'cache_receipts.json',receipts)
        _durable_json(root/'historical_baselines.json',{str(sid):s['baseline'] for sid,s in shots.items()})
        records={};selections={}
        def progress(kind,record):
            _durable_json(root/'training'/f'{kind}_{record["epoch"]:02d}.json',record)
            print(dict(stage='training',kind=kind,epoch=record['epoch'],loss=record['objective'],
                       history_gate=record['history_gate']['passed'],seconds=round(time.monotonic()-start,2)),flush=True)
            check_time()
        for kind in ('global','context'):
            best,logs=train_policy(model,kind,shots,p['threshold'],progress=progress)
            records[kind]=best['policy'];selections[kind]={k:best[k] for k in ('epoch','objective','history_gate')}
            _durable_json(root/f'{kind}_selection.json',dict(selections[kind],selection='minimum20-shot training objective only'))
        _durable_json(root/'policies.json',records)
        _durable_json(root/'policy_seal.json',dict(sha256=sha(root/'policies.json'),new_outcomes_accessed=0))
        del shots;gc.collect()
        stage_record=READ(PREVIOUS/'policies.json')['context'];seals=[]
        for sid in ids:
            raw=signal(sid);s=transform({k:raw[k] for k in ('values','available')},mapping)
            curves,updates=predict(model,s,records,stage_record)
            with torch.no_grad():baseline=predict_many(model,dict(parent=s),device='cuda')['parent']
            np.testing.assert_array_equal(curves['off'],baseline)
            with np.load(PREVIOUS/'shots'/f'{sid}.npz') as z:np.testing.assert_array_equal(curves['stage_old'],z['finite_context'])
            for arm in ARMS:np.testing.assert_array_equal(curves[arm][:256],baseline[:256])
            out=root/'shots'/f'{sid}.npz';np.savez_compressed(out,times=raw['times'],**curves)
            up=root/'updates'/f'{sid}.json';_durable_json(up,updates)
            seals.append(dict(shot=sid,prediction_sha256=sha(out),updates_sha256=sha(up)))
            print(dict(stage='prediction',shot=sid,seconds=round(time.monotonic()-start,2)),flush=True);check_time()
    _durable_json(root/'prediction_seals.json',seals)
    results=evaluate(root,p)
    assert_unchanged(model,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='completed',seconds=time.monotonic()-start,cache_bytes=bytes_used,results=results,
        eligible_for360=eligible(results,selections['context']['history_gate']['passed']),source_and_parent_unchanged=True,
        artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',result);print({k:v for k,v in result.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
