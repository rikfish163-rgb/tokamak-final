"""Complete causal360 paired temporal-task experiment, including cold20."""
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
from ..fixed_test_protocol import score_fixed_test,validate_test_ids
from ..latent_completion_20261028.stream import compact
from ..lifecycle_stream_20261025.protocol import AccessLedger
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256,classifier_state,assert_classifier_unchanged
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform,fit_prefix_map
from .model import OrderJudge
from .online import ARMS,predict


def run(output,source_root,previous):
    if output.exists() or output.absolute().parent!=io23.ROOT/'probe_outputs':raise ValueError('new worktree output required')
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,parent=load_parent(DEFAULT_PARENT);base.cuda();original=classifier_state(base);threshold=float(parent['threshold'])
    oldp=read(previous,'protocol.json');oldcheck=read(previous,'audit.json')
    if not oldcheck['complete'] or oldcheck['complete_sha256']!=sha(previous/'complete.json'):
        raise ValueError('audited previous stream required')
    reference=Path(oldp['reference']);ids=oldp['order'];test_contract=validate_test_ids('EAST',ids)
    mapping=read(reference,'input_map.json');hashes=read(reference,'input_hashes.json')
    if oldp['warmup']!=20 or oldp['threshold']!=threshold or mapping['calibration_ids']!=ids[:20]:
        raise ValueError('same parent and full stream required')
    for path,h in oldp['reference_hashes'].items():
        if sha(reference/path)!=h:raise ValueError('cold parent provenance changed')
    done=read(source_root,'complete.json');source_protocol=read(source_root,'protocol.json');source_path=source_root/'judge.pt'
    if not done['complete'] or not done['source_unchanged'] or not done['source_only_gate'] or sha(source_path)!=done['checkpoint_sha256']:
        raise ValueError('source temporal gate and checkpoint required')
    if source_protocol['source_sha256']!=PARENT_SHA256:raise ValueError('source classifier mismatch')
    for path,h in source_protocol['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('source auxiliary implementation changed')
    state=torch.load(source_path,map_location='cpu',weights_only=True)
    if state['protocol_sha256']!=sha(source_root/'protocol.json'):raise ValueError('source protocol changed')
    net=OrderJudge().cuda().eval().requires_grad_(False);net.load_state_dict(state['model'])
    original_net={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    slow=torch.load(reference/'models/cold.pt',map_location='cpu',weights_only=True)['slow'].cuda()
    ledger=AccessLedger(ids,20);prefix=[];rows={k:[] for k in ARMS};receipts=[]
    output.mkdir();(output/'shots').mkdir();start=time.monotonic()
    files={Path(m.__file__) for n,m in list(sys.modules.items())
           if n.startswith('experiments.risk_aligned_online.') and getattr(m,'__file__',None)}
    protocol=dict(scope='DEVELOPMENT_ONLY_FULL360_TEMPORAL_RELATION',order=ids,arms=ARMS,test_contract=test_contract,
        previous=str(previous),previous_complete_sha256=sha(previous/'complete.json'),
        reference=str(reference),reference_hashes=oldp['reference_hashes'],source_root=str(source_root),judge_sha256=sha(source_path),
        source_complete_sha256=sha(source_root/'complete.json'),source_sha256=PARENT_SHA256,threshold=threshold,
        warmup=20,step=.05,radius=1.,update_interval_samples=128,per_shot_permutation_seed=131,
        reset_each_shot=True,inter_updates=0,alarm_updates=0,independent_confirmation=False,
        source_and_judge_frozen=True,plan_sha256=sha(io23.ROOT/'TEMPORAL_RELATION_STREAM_PLAN.md'),
        code_sha256={str(p.relative_to(io23.ROOT)):sha(p) for p in files},gpu=torch.cuda.get_device_name(),torch=torch.__version__)
    _durable_json(output/'protocol.json',protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
        z.write(io23.ROOT/'TEMPORAL_RELATION_STREAM_PLAN.md','TEMPORAL_RELATION_STREAM_PLAN.md')
    def deny(*args,**kwargs):raise AssertionError('outcome accessed before sealing current prediction')
    for index,sid in enumerate(ids):
        if sha(io23.path_for('EAST',sid))!=hashes[str(sid)]:raise ValueError('raw input changed')
        with patch.object(io23,'read_outcome',side_effect=deny):
            raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            sig={k:raw[k] for k in ('values','available')}
            if ledger.ready:scores,info=predict(base,net,transform(sig,mapping),slow)
            else:
                curve=original_predictions(base,sig);scores={k:curve.copy() for k in ARMS};info={}
        old_path=previous/'shots'/f'{index+1:04d}_{sid}.npz';seal=read(previous/'shots',f'{index+1:04d}_{sid}_seal.json')
        if sha(old_path)!=seal['sha256'] or seal['outcome_used']:raise ValueError('previous curve seal changed')
        with np.load(old_path) as z:
            if not np.array_equal(raw['times'],z['times']):raise ValueError('timestamps changed')
            difference=float(np.max(np.abs(z['cold']-scores['cold'])))
            if difference>2e-6 or not np.array_equal(z['cold']>=threshold,scores['cold']>=threshold):raise ValueError('cold comparator mismatch')
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
            a=old['timely'] if old['is_disrupt'] else not old['alarm'];b=new['timely'] if new['is_disrupt'] else not new['alarm']
            if b and not a:rescued.append(old['shot'])
            if a and not b:harmed.append(old['shot'])
        comparisons[name]=dict(rescued=rescued,harmed=harmed)
    assert_classifier_unchanged(base,original)
    if not all(torch.equal(v.detach().cpu(),original_net[k]) for k,v in net.state_dict().items()):raise ValueError('auxiliary head changed')
    if len(ledger.history)!=360 or sha(DEFAULT_PARENT)!=PARENT_SHA256:raise ValueError('incomplete stream or changed source')
    for path,h in protocol['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('code changed during stream')
    c=results['relation']['full360'];b=results['cold']['full360']
    _durable_json(output/'per_shot.json',predictions);_durable_json(output/'receipts.json',receipts)
    result=dict(complete=True,results=results,comparisons=comparisons,seconds=time.monotonic()-start,source_unchanged=True,
        auxiliary_head_unchanged=True,update_counts={k:sum(r['updates'].get(k,{}).get('updates',0) for r in receipts) for k in ARMS[1:]},
        necessary_intra_screen=bool(c['J']>max(results[k]['full360']['J'] for k in ('cold','wrong_label','smooth')) and c['fpr']<=.20 and c['tp']>=b['tp']),
        scientific_goal_achieved=False,independent_confirmation=False)
    _durable_json(output/'complete.json',result)
    print({k:v['full360'] for k,v in results.items()},flush=True);print(comparisons,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    a=p.parse_args();run(a.output,a.source_root,a.previous)
