"""Complete360 gradient-path test; old two-path curves are sealed comparators."""
import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import read
from ..delayed_recurrent_memory_20261025.learning import original_predictions
from ..fixed_test_protocol import score_fixed_test
from ..latent_completion_20261028.model import Imputer
from ..latent_completion_20261028.stream import compact
from ..lifecycle_stream_20261025.protocol import AccessLedger
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256,classifier_state,assert_classifier_unchanged
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform,fit_prefix_map
from .model import SPECS,predict

ARMS=('cold','both_paths',*SPECS)


def run(output,previous):
    if output.exists() or output.absolute().parent!=io23.ROOT/'probe_outputs':raise ValueError('new worktree result required')
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,parent=load_parent(DEFAULT_PARENT);base.cuda();original=classifier_state(base);threshold=float(parent['threshold'])
    oldp=read(previous,'protocol.json');oldd=read(previous,'complete.json');oldcheck=read(previous,'audit.json')
    if not oldd['complete'] or not oldcheck['complete'] or oldcheck['complete_sha256']!=sha(previous/'complete.json'):
        raise ValueError('audited previous360 required')
    reference=Path(oldp['reference']);source_root=Path(oldp['source_root']);ids=oldp['order']
    prior=read(reference,'protocol.json');mapping=read(reference,'input_map.json');hashes=read(reference,'input_hashes.json')
    if len(ids)!=360 or len(set(ids))!=360 or prior['warmup']!=20 or oldp['threshold']!=threshold or mapping['calibration_ids']!=ids[:20]:
        raise ValueError('same complete stream and warmup required')
    for path,h in oldp['reference_hashes'].items():
        if sha(reference/path)!=h:raise ValueError('cold parent provenance changed')
    source=read(source_root,'complete.json');source_path=source_root/'task.pt'
    if sha(source_path)!=source['checkpoints']['task'] or not source['source_unchanged']:raise ValueError('source imputer changed')
    net=Imputer().cuda().eval().requires_grad_(False)
    net.load_state_dict(torch.load(source_path,map_location='cpu',weights_only=True)['model'])
    slow_state=torch.load(reference/'models/cold.pt',map_location='cpu',weights_only=True)['slow'].cuda()
    ledger=AccessLedger(ids,20);prefix=[];rows={k:[] for k in ARMS};receipts=[]
    output.mkdir();(output/'shots').mkdir();start=time.monotonic()
    files={Path(m.__file__) for n,m in list(sys.modules.items())
           if n.startswith('experiments.risk_aligned_online.') and getattr(m,'__file__',None)}
    protocol=dict(scope='DEVELOPMENT_ONLY_FULL360_GRADIENT_PATH_ABLATION',order=ids,arms=ARMS,
        previous=str(previous),previous_complete_sha256=sha(previous/'complete.json'),
        reference=str(reference),reference_hashes=oldp['reference_hashes'],source_root=str(source_root),task_sha256=sha(source_path),
        source_sha256=PARENT_SHA256,threshold=threshold,warmup=20,step=.05,radius=1.,update_interval_samples=128,
        reset_each_shot=True,inter_updates=0,alarm_updates=0,independent_confirmation=False,
        both_paths='archived task curves, not used by candidate update',
        stop_target='detach current full-view target each step; recompute next step',
        plan_sha256=sha(io23.ROOT/'GRADIENT_SPLIT_STREAM_PLAN.md'),
        code_sha256={str(p.relative_to(io23.ROOT)):sha(p) for p in files},gpu=torch.cuda.get_device_name(),torch=torch.__version__)
    _durable_json(output/'protocol.json',protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
        z.write(io23.ROOT/'GRADIENT_SPLIT_STREAM_PLAN.md','GRADIENT_SPLIT_STREAM_PLAN.md')
    def deny(*args,**kwargs):raise AssertionError('outcome accessed before sealing current prediction')
    for index,sid in enumerate(ids):
        if sha(io23.path_for('EAST',sid))!=hashes[str(sid)]:raise ValueError('input changed')
        with patch.object(io23,'read_outcome',side_effect=deny):
            raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            sig={k:raw[k] for k in ('values','available')}
            if ledger.ready:scores,info=predict(base,net,transform(sig,mapping),slow_state)
            else:
                original_curve=original_predictions(base,sig)
                scores={k:original_curve.copy() for k in ('cold',*SPECS)};info={}
        old_path=previous/'shots'/f'{index+1:04d}_{sid}.npz'
        oldseal=read(previous/'shots',f'{index+1:04d}_{sid}_seal.json')
        if sha(old_path)!=oldseal['sha256'] or oldseal['outcome_used']:raise ValueError('old curve seal changed')
        with np.load(old_path) as z:
            if not np.array_equal(raw['times'],z['times']):raise ValueError('timestamps changed')
            difference=float(np.max(np.abs(z['cold']-scores['cold'])))
            if difference>2e-6 or not np.array_equal(z['cold']>=threshold,scores['cold']>=threshold):raise ValueError('cold comparator mismatch')
            scores['both_paths']=z['task'].copy()
        path=output/'shots'/f'{index+1:04d}_{sid}.npz'
        with path.open('xb') as f:np.savez_compressed(f,times=raw['times'],**scores)
        digest=sha(path);ledger.seal(sid,digest)
        receipt=dict(shot=sid,index=index+1,sha256=digest,outcome_used=False,updates=info,cold_max_abs_error=difference)
        _durable_json(output/'shots'/f'{index+1:04d}_{sid}_seal.json',receipt);receipts.append(receipt)
        ledger.reveal(sid);labeled=simple_data.attach_outcome(raw,'EAST',{});valid=labeled['valid']
        for k in ARMS:rows[k].append(dict(shot=sid,times=raw['times'][valid],scores=scores[k][valid],
                                         is_disrupt=labeled['is_disrupt'],down=labeled['down']))
        if index<20:prefix.append({k:raw[k] for k in ('shot','device','values','available')})
        if index==19:
            if fit_prefix_map(mapping['source_statistics'],prefix,order=ids,count=20,policy='density_gain')!=mapping:
                raise ValueError('first20 mapping does not reproduce')
            prefix=[]
        if (index+1)%20==0:
            progress=dict(shots=index+1,seconds=time.monotonic()-start)
            _durable_json(output/'progress.json',progress);print(progress,flush=True)
    predictions={};results={};comparisons={}
    for name,r in rows.items():
        predictions[name]=score_fixed_test('EAST',r,threshold)['predictions']
        results[name]={k:compact(sub,threshold) for k,sub in (('full360',r),('cold20',r[:20]),('suffix340',r[20:]))}
    for name in ARMS[1:]:
        rescued=[];harmed=[]
        for old,new in zip(predictions['cold'],predictions[name]):
            a=old['timely'] if old['is_disrupt'] else not old['alarm']
            b=new['timely'] if new['is_disrupt'] else not new['alarm']
            if b and not a:rescued.append(old['shot'])
            if a and not b:harmed.append(old['shot'])
        comparisons[name]=dict(rescued=rescued,harmed=harmed)
    assert_classifier_unchanged(base,original)
    if len(ledger.history)!=360 or sha(DEFAULT_PARENT)!=PARENT_SHA256:raise ValueError('stream incomplete or source changed')
    for path,h in protocol['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('code changed during run')
    c=results['stop_target']['full360'];b=results['cold']['full360'];r=results['stop_target_reverse']['full360']
    _durable_json(output/'per_shot.json',predictions);_durable_json(output/'receipts.json',receipts)
    done=dict(complete=True,results=results,comparisons=comparisons,seconds=time.monotonic()-start,source_unchanged=True,
        update_counts={k:sum(r['updates'].get(k,{}).get('updates',0) for r in receipts) for k in SPECS},
        necessary_intra_screen=bool(c['J']>b['J'] and c['J']>r['J'] and c['fpr']<=.20 and c['tp']>=b['tp']),
        scientific_goal_achieved=False,independent_confirmation=False)
    _durable_json(output/'complete.json',done)
    print({k:v['full360'] for k,v in results.items()},flush=True);print(comparisons,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    a=p.parse_args();run(a.output,a.previous)
