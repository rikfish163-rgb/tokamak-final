"""Full360 protected Q/K contraction with spectrum-matched controls."""
import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23,simple_data
from ..attention_weights_20261029.learning import validate_fit_ids
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import read
from ..delayed_recurrent_memory_20261025.learning import original_predictions
from ..fixed_test_protocol import score_fixed_test,validate_test_ids
from ..latent_completion_20261028.stream import compact
from ..lifecycle_stream_20261025.protocol import AccessLedger
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform,fit_prefix_map
from ..attention_basis_20261029.learning import MatrixAttention
from .learning import ARMS,predict


def run(output,previous,policy_dir):
    if output.exists() or output.absolute().parent!=io23.ROOT/'probe_outputs':raise ValueError('new worktree output required')
    oldp=read(previous,'protocol.json');olda=read(previous,'audit.json');tp=read(policy_dir,'protocol.json');fit=read(policy_dir,'complete.json')
    if (not olda['complete'] or olda['complete_sha256']!=sha(previous/'complete.json') or not fit['complete']
            or fit['protocol_sha256']!=sha(policy_dir/'protocol.json') or fit['policy_sha256']!=sha(policy_dir/'policy.json')):
        raise ValueError('audited previous stream and completed protected fitting required')
    if fit['step']<=0:raise ValueError('zero-step diagnostic must not launch a redundant stream')
    if fit['pairs_sha256']!=sha(policy_dir/'historical_pairs.npz'):raise ValueError('historical pairs changed')
    ids=oldp['order'];ref=Path(oldp['reference']);validate_test_ids('EAST',ids);validate_fit_ids(tp['fit_ids'],ids)
    if tp['fit_ids']!=ids[:20] or tp['order']!=ids or tp['reference_hashes']!=oldp['reference_hashes']:raise ValueError('same ended cold20 and parent required')
    if tp['plan_sha256']!=sha(io23.ROOT/'ATTENTION_PROTECTED_STREAM_PLAN.md'):raise ValueError('plan changed')
    for path,h in tp['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('fit implementation changed')
    for path,h in oldp['reference_hashes'].items():
        if sha(ref/path)!=h:raise ValueError('cold reference changed')
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,parent=load_parent(DEFAULT_PARENT);base.cuda();net=MatrixAttention(base);frozen={k:v.cpu().clone() for k,v in net.state_dict().items()}
    slow=torch.load(ref/'models/cold.pt',map_location='cpu',weights_only=True)['slow'].cuda();slow_before=slow.clone()
    mapping=read(ref,'input_map.json');hashes=read(ref,'input_hashes.json');threshold=float(parent['threshold'])
    if threshold!=oldp['threshold'] or threshold!=tp['threshold'] or mapping['calibration_ids']!=ids[:20]:raise ValueError('threshold or prefix changed')
    ledger=AccessLedger(ids,20);rows={k:[] for k in ARMS};receipts=[];prefix=[];policy=None
    output.mkdir();(output/'shots').mkdir();started=time.monotonic()
    files={Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('experiments.risk_aligned_online.') and getattr(m,'__file__',None)}
    protocol=dict(scope='DEVELOPMENT_ONLY_FULL360_COLD20_QK_PROTECTED',order=ids,arms=ARMS,previous=str(previous),previous_complete_sha256=sha(previous/'complete.json'),
        policy_dir=str(policy_dir),policy_complete_sha256=sha(policy_dir/'complete.json'),policy_sha256=fit['policy_sha256'],reference=str(ref),
        reference_hashes=oldp['reference_hashes'],source_sha256=PARENT_SHA256,threshold=threshold,warmup=20,step=fit['step'],
        reset_each_feedback=True,reset_each_shot=True,inter_updates=0,alarm_updates=0,plan_sha256=sha(io23.ROOT/'ATTENTION_PROTECTED_STREAM_PLAN.md'),
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},gpu=torch.cuda.get_device_name(),torch=torch.__version__)
    _durable_json(output/'protocol.json',protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
        z.write(io23.ROOT/'ATTENTION_PROTECTED_STREAM_PLAN.md','ATTENTION_PROTECTED_STREAM_PLAN.md')
    def deny(*args,**kwargs):raise AssertionError('outcome accessed before current-shot seal')
    for i,sid in enumerate(ids):
        if sha(io23.path_for('EAST',sid))!=hashes[str(sid)]:raise ValueError('raw changed')
        with patch.object(io23,'read_outcome',side_effect=deny):
            raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'));sig={k:raw[k] for k in ('values','available')}
            if ledger.ready:
                if policy is None:raise ValueError('contraction policy missing after cold20')
                scores,info=predict(net,transform(sig,mapping),slow,policy)
            else:
                curve=original_predictions(base,sig);scores={k:curve.copy() for k in ARMS};info={}
        name=f'{i+1:04d}_{sid}';oldpath=previous/'shots'/f'{name}.npz';seal=read(previous/'shots',f'{name}_seal.json')
        if sha(oldpath)!=seal['sha256'] or seal['outcome_used']:raise ValueError('prior seal invalid')
        with np.load(oldpath) as old:
            error=float(np.abs(old['cold']-scores['cold']).max())
            if error>2e-6 or not np.array_equal(old['times'],raw['times']) or not np.array_equal(old['cold']>=threshold,scores['cold']>=threshold):raise ValueError('cold curve changed')
        path=output/'shots'/f'{name}.npz'
        with path.open('xb') as f:np.savez_compressed(f,times=raw['times'],**scores)
        digest=sha(path);ledger.seal(sid,digest)
        receipt=dict(shot=sid,index=i+1,sha256=digest,outcome_used=False,updates=info,cold_max_abs_error=error,policy_active=ledger.ready)
        _durable_json(output/'shots'/f'{name}_seal.json',receipt);receipts.append(receipt)
        ledger.reveal(sid);labeled=simple_data.attach_outcome(raw,'EAST',{});valid=labeled['valid']
        for k in ARMS:rows[k].append(dict(shot=sid,times=raw['times'][valid],scores=scores[k][valid],is_disrupt=labeled['is_disrupt'],down=labeled['down']))
        if i<20:prefix.append({k:raw[k] for k in ('shot','device','values','available')})
        if i==19:
            if fit_prefix_map(mapping['source_statistics'],prefix,order=ids,count=20,policy='density_gain')!=mapping:raise ValueError('prefix map changed')
            policy=read(policy_dir,'policy.json');prefix=[]
        if (i+1)%20==0:
            progress=dict(shots=i+1,seconds=time.monotonic()-started);_durable_json(output/'progress.json',progress);print(progress,flush=True)
    predictions={k:score_fixed_test('EAST',v,threshold)['predictions'] for k,v in rows.items()}
    results={k:{s:compact(vv,threshold) for s,vv in (('full360',v),('cold20',v[:20]),('suffix340',v[20:]))} for k,v in rows.items()};comparisons={}
    for k in ARMS[1:]:
        rescued=[];harmed=[]
        for old,new in zip(predictions['cold'],predictions[k]):
            before=old['timely'] if old['is_disrupt'] else not old['alarm'];after=new['timely'] if new['is_disrupt'] else not new['alarm']
            if after and not before:rescued.append(old['shot'])
            if before and not after:harmed.append(old['shot'])
        comparisons[k]=dict(rescued=rescued,harmed=harmed)
    if not all(torch.equal(v.cpu(),frozen[k]) for k,v in net.state_dict().items()) or not torch.equal(slow,slow_before):raise ValueError('persistent network changed')
    for path,h in protocol['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('implementation changed during stream')
    if len(ledger.history)!=360 or sha(DEFAULT_PARENT)!=PARENT_SHA256 or sha(policy_dir/'policy.json')!=fit['policy_sha256']:raise ValueError('incomplete or changed artifacts')
    _durable_json(output/'per_shot.json',predictions);_durable_json(output/'receipts.json',receipts)
    c=results['protected']['full360'];b=results['cold']['full360'];r=results['random_contraction']['full360']
    result=dict(complete=True,results=results,comparisons=comparisons,seconds=time.monotonic()-started,
        update_counts={k:sum(v['updates'].get(k,{}).get('updates',0) for v in receipts) for k in ARMS[1:]},
        necessary_intra_screen=bool(c['tp']>=b['tp'] and c['fpr']<=.2 and c['J']>max(b['J'],r['J'])),
        scientific_goal_achieved=False,independent_confirmation=False,source_unchanged=True)
    _durable_json(output/'complete.json',result);print({k:v['full360'] for k,v in results.items()},flush=True);print(comparisons,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--policy',type=Path,required=True)
    a=p.parse_args();run(a.output,a.previous,a.policy)
