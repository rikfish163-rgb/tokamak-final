"""Full360 matched-parent intra experiment, including all20 startup shots."""
import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import read
from ..delayed_recurrent_memory_20261025.learning import original_predictions
from ..fixed_test_protocol import score_fixed_test
from ..lifecycle_stream_20261025.protocol import AccessLedger
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha,metrics
from ..warmstart_alignment_20261022.alignment import fit_prefix_map,transform
from .model import Imputer,predict

ARMS=('cold','plain','task','task_reverse')


def compact(rows,threshold):
    m=metrics(rows,threshold,lead=30.)
    m['early_over3000']=sum(p['timely'] and p['lead_ms']>3000 for p in m['predictions'])
    del m['predictions'];return m


def run(output,source_root,reference):
    if output.exists() or output.absolute().parent!=io23.ROOT/'probe_outputs':raise ValueError('new worktree output required')
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,parent=load_parent(DEFAULT_PARENT);base.cuda();threshold=float(parent['threshold'])
    prior=read(reference,'protocol.json');done=read(reference,'completed.json')
    order=read(reference,'order.json');ids=[r['shot'] for r in order]
    mapping=read(reference,'input_map.json');hashes=read(reference,'input_hashes.json')
    if (done['status']!='COMPLETED' or len(ids)!=360 or len(set(ids))!=360 or prior['warmup']!=20
            or prior['source_sha256']!=PARENT_SHA256 or prior['threshold']!=threshold
            or mapping['calibration_ids']!=ids[:20] or mapping['policy']!='density_gain'
            or sha(reference/'order.json')!=prior['order_sha256']):raise ValueError('fixed360 parent contract required')
    training=read(source_root,'complete.json');tp=read(source_root,'protocol.json')
    if not training['complete'] or not training['source_unchanged'] or tp['source_sha256']!=PARENT_SHA256:
        raise ValueError('completed source-only imputer training required')
    nets={}
    for name in ('plain','task'):
        path=source_root/f'{name}.pt'
        if sha(path)!=training['checkpoints'][name]:raise ValueError('imputer changed')
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state['protocol_sha256']!=sha(source_root/'protocol.json'):raise ValueError('source protocol changed')
        net=Imputer().cuda().eval().requires_grad_(False);net.load_state_dict(state['model']);nets[name]=net
    cold_path=reference/'models/cold.pt'
    cold=torch.load(cold_path,map_location='cpu',weights_only=True)
    if set(cold)!={'slow','fast_output'} or cold['slow'].shape!=(64,64):raise ValueError('archived cold internal operator required')
    # The archived fast_output is NOT reused: zero fast memory is the cold
    # comparator. Only its persistent first20-trained64x64 internal operator.
    slow=None;ledger=AccessLedger(ids,20);prefix=[];rows={k:[] for k in ARMS};receipts=[]
    output.mkdir(parents=True);(output/'shots').mkdir();start=time.monotonic()
    files={Path(m.__file__) for n,m in list(sys.modules.items())
           if n.startswith('experiments.risk_aligned_online.') and getattr(m,'__file__',None)}
    protocol=dict(scope='DEVELOPMENT_ONLY_COMPLETE360_INTRA_LATENT_COMPLETION',arms=ARMS,
        order=ids,source_classifier_sha256=PARENT_SHA256,threshold=threshold,warmup=20,
        source_root=str(source_root),source_complete_sha256=sha(source_root/'complete.json'),
        reference=str(reference),reference_hashes={p:sha(reference/p) for p in
            ('protocol.json','completed.json','order.json','input_hashes.json','input_map.json','models/cold.pt')},
        plan_sha256=sha(io23.ROOT/'LATENT_IMPUTATION_PLAN.md'),step=.05,radius=1.,update_interval_samples=128,
        inter_updates=0,alarm_updates=0,reset_each_shot=True,source_and_imputer_frozen=True,
        source_kl_is_not_online_risk_pseudolabel=True,independent_confirmation=False,
        code_sha256={str(p.relative_to(io23.ROOT)):sha(p) for p in files})
    _durable_json(output/'protocol.json',protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
    def deny(*args,**kwargs):raise AssertionError('EAST outcome accessed during prediction')
    for index,sid in enumerate(ids):
        if sha(io23.path_for('EAST',sid))!=hashes[str(sid)]:raise ValueError('EAST input changed')
        with patch.object(io23,'read_outcome',side_effect=deny):
            raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal={k:raw[k] for k in ('values','available')}
            if ledger.ready:
                signal=transform(signal,mapping)
                scores,info=predict(base,nets,signal,slow,reversed_arm=True)
            else:
                original=original_predictions(base,signal)
                scores={k:original.copy() for k in ARMS};info={}
        archived=reference/'shots'/f'{index+1:04d}_{sid}.npz'
        archived_seal=read(reference/'shots',f'{index+1:04d}_{sid}_seal.json')
        if sha(archived)!=archived_seal['sha256']:raise ValueError('archived reference curve changed')
        with np.load(archived) as old:
            if not np.array_equal(raw['times'],old['times']):raise ValueError('timestamp contract changed')
            difference=float(np.max(np.abs(scores['cold']-old['cold'])))
            if difference>2e-6 or not np.array_equal(scores['cold']>=threshold,old['cold']>=threshold):
                raise ValueError(f'cold parent numerical/alarm mismatch for {sid}: {difference}')
        path=output/'shots'/f'{index+1:04d}_{sid}.npz'
        with path.open('xb') as f:np.savez_compressed(f,times=raw['times'],**scores)
        digest=sha(path);ledger.seal(sid,digest)
        _durable_json(output/'shots'/f'{index+1:04d}_{sid}_seal.json',dict(shot=sid,index=index+1,
            sha256=digest,outcome_used=False,updates=info,cold_max_abs_error=difference,
            cold_alarm_sequence_identical=True))
        ledger.reveal(sid);outcome=simple_data.attach_outcome(raw,'EAST',{})
        valid=outcome['valid']
        for name in ARMS:rows[name].append(dict(shot=sid,times=raw['times'][valid],scores=scores[name][valid],
                                               is_disrupt=outcome['is_disrupt'],down=outcome['down']))
        receipts.append(dict(shot=sid,index=index+1,updates=info,cold_max_abs_error=difference,seal_sha256=digest))
        if index<20:prefix.append({k:raw[k] for k in ('shot','device','values','available')})
        if index==19:
            rebuilt=fit_prefix_map(mapping['source_statistics'],prefix,order=ids,count=20,policy='density_gain')
            if rebuilt!=mapping:raise ValueError('first20 map does not reproduce')
            slow=cold['slow'].cuda();prefix=[]
        if (index+1)%10==0:
            progress=dict(shots=index+1,seconds=time.monotonic()-start)
            _durable_json(output/'progress.json',progress);print(progress,flush=True)
    results={};predictions={}
    for name,r in rows.items():
        score=score_fixed_test('EAST',r,threshold);predictions[name]=score['predictions']
        results[name]={section:compact(sub,threshold) for section,sub in
            (('full360',r),('cold20',r[:20]),('suffix340',r[20:]),('high_proxy90',r[-90:]))}
    if len(ledger.history)!=360 or sha(DEFAULT_PARENT)!=PARENT_SHA256:raise ValueError('stream incomplete or source changed')
    for path,h in protocol['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('code changed during experiment')
    baseline=results['cold']['full360'];candidate=results['task']['full360'];control=results['task_reverse']['full360']
    comparisons={}
    for name in ARMS[1:]:
        rescues=[];harms=[]
        for old,new in zip(predictions['cold'],predictions[name]):
            good_old=old['timely'] if old['is_disrupt'] else not old['alarm']
            good_new=new['timely'] if new['is_disrupt'] else not new['alarm']
            if good_new and not good_old:rescues.append(old['shot'])
            if good_old and not good_new:harms.append(old['shot'])
        comparisons[name]=dict(rescued=rescues,harmed=harms)
    _durable_json(output/'per_shot.json',predictions);_durable_json(output/'receipts.json',receipts)
    completed=dict(complete=True,results=results,comparisons=comparisons,seconds=time.monotonic()-start,
        necessary_intra_screen=bool(candidate['J']>baseline['J'] and candidate['J']>control['J'] and candidate['fpr']<=.20),
        scientific_goal_achieved=False,independent_confirmation=False,source_unchanged=True,
        max_cold_error=max(r['cold_max_abs_error'] for r in receipts),
        update_counts={k:sum(r['updates'].get(k,{}).get('updates',0) for r in receipts) for k in ARMS[1:]})
    _durable_json(output/'complete.json',completed)
    print({k:v['full360'] for k,v in results.items()},flush=True);print(comparisons,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    a=p.parse_args();run(a.output,a.source_root,a.reference)
