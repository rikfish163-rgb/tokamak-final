"""Source-only preparation followed by locked source-test qualification/EAST.

One fixed candidate, no target threshold or hyperparameter scan. Test sets
have participated in development; this run cannot establish generalization.
"""
import argparse
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import torch

from . import simple_data,simple_train
from .aligned_encoder import encoder_parameters,source_step,predict_shot
from .cross_channel_shot_screen import evaluate_shot
from .cross_channel_stable import HEAD,FourViewModel
from .ended_context_experiment import side_metrics
from .fixed_test_protocol import load_test_contract,score_fixed_test
from .operating_curve_audit import load_sealed
from .redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_parent,load_source_train
from .redesign_20261004.run import run_arm
from .simple_online import _durable_json,_hash


def source_test(model,folder,threshold):
    folder.mkdir();contract=load_test_contract('J-TEXT');ids=contract['shot_ids']
    advanced=simple_data.advanced_times();advanced={sid:advanced[sid] for sid in ids if sid in advanced}
    rows=[];audits=[]
    for index,sid in enumerate(ids,1):
        row,audit=evaluate_shot(model,sid,folder,advanced,intra=False,predictor=predict_shot)
        rows.append(row);audits.append(dict(shot=sid,**audit))
        if index%55==0:print(json.dumps(dict(stage='source_test',shots=index)),flush=True)
    result=score_fixed_test('J-TEXT',rows,threshold)
    # Re-read all complete seals rather than trusting in-memory summaries.
    recount=[]
    for row,audit in zip(rows,audits):
        path=folder/f"{row['shot']}.npz"
        if _hash(path)!=audit['sealed_sha256']:raise AssertionError('source seal changed')
        s=simple_data.attach_outcome(simple_data.read_shot('J-TEXT',row['shot']),'J-TEXT',advanced)
        with np.load(path,allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['times'],s['times'])
            recount.append(dict(row,times=saved['times'][s['valid']],scores=saved['scores'][s['valid']]))
    if score_fixed_test('J-TEXT',recount,threshold)!=result:
        raise AssertionError('source test recount differs')
    _durable_json(folder/'metrics.json',result)
    _durable_json(folder/'audits.json',audits)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('new output path required')
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    torch.set_num_threads(4);torch.manual_seed(131);rng=np.random.default_rng(131)
    args.output.mkdir(parents=True);started=time.monotonic()
    base,parent=load_parent(DEFAULT_PARENT)
    teacher=FourViewModel(base,site='embedding').cuda().eval().requires_grad_(False)
    head=torch.load(HEAD,map_location='cpu',weights_only=True)
    if head['parent_sha256']!=PARENT_SHA256 or head['branch']!='signal':
        raise ValueError('source-only diagnostic head required')
    teacher.decoder.load_state_dict(head['decoder'])
    original={k:v.detach().clone() for k,v in teacher.state_dict().items()}
    student=deepcopy(teacher);params=encoder_parameters(student)
    for p in params.values():p.requires_grad_(True)
    student.decoder.requires_grad_(True)
    optimizer=torch.optim.AdamW([
        dict(params=list(params.values()),lr=1e-5),
        dict(params=list(student.decoder.parameters()),lr=1e-4)],weight_decay=0.)
    threshold=float(parent['threshold']);root=Path(__file__).parent
    with zipfile.ZipFile(args.output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in ('aligned_encoder.py','aligned_encoder_experiment.py','fixed_test_protocol.py',
            'cross_channel_prefix.py','cross_channel_stable.py','cross_channel_shot_screen.py',
            'simple23.py','simple_data.py','io23.py','simple_train.py','simple_online.py',
            'causal_input.py','redesign_20261004/prepare_prior.py','redesign_20261004/intra.py',
            'redesign_20261004/run.py','ended_context_experiment.py','operating_curve_audit.py'):
            z.write(root/name,name)
    roles=simple_data.source_roles()
    protocol=dict(scope='SOURCE PREPARED COPY + FIXED TEST EVALUATION; DEVELOPMENT-EXPOSED',
        parent_sha256=PARENT_SHA256,head_sha256=_hash(HEAD),source_train_ids=roles['train'],
        threshold=threshold,threshold_reselection=False,target_pretraining=False,
        source_steps=512,source_batch=32,source_encoder_lr=1e-5,source_decoder_lr=1e-4,
        source_loss='source BCE + mean64D teacher feature squared error +0.20 masked SmoothL1',
        source_window_lengths=[8,16,32,64,128,128,128,128],source_head_frozen=True,
        source_copy_changes=list(params),original_parent_overwritten=False,
        source_gate='full TEST220 strict lead>10ms: TPR>=0.90 and FPR<0.20',
        source_gate_is_development_acceptance=True,
        target_arms=['prepared_static','prepared_intra'],target_test=load_test_contract('EAST'),
        target_rule='predict t, reconstruct observed channels, update original encoder copy for t+1',
        target_loss='mean of4 channel-masked SmoothL1 losses, no risk pseudo-label or teacher risk anchor',
        target_update=dict(first_index=7,interval=32,sgd_lr=.001,clip_norm=1.,total_radius=.5),
        intra_state_reset_every_shot=True,inter_shot_update=False,
        acceptance='TPR30 improves over prepared_static AND FPR<=.20 AND J improves',
        selection='one fixed configuration, last training step; no target-based selection',
        literature_inspiration=['TTT-MAE NeurIPS2022','CTA arXiv2507.05221v1'],
        is_exact_literature_reproduction=False,code_snapshot_sha256=_hash(args.output/'code_snapshot.zip'))
    _durable_json(args.output/'protocol.json',protocol)
    print(json.dumps(dict(stage='source_loading',shots=len(roles['train']))),flush=True)
    shots,ids=load_source_train();pools=simple_train.pools(shots)
    if ids!=roles['train']:raise AssertionError('source TRAIN identity mismatch')
    history=[]
    for iteration in range(512):
        x,a,y,_,_=simple_train.batch(shots,pools,rng,32,student.cfg,'cuda')
        length=protocol['source_window_lengths'][iteration%8]
        a=a.clone();a[:,:128-length]=False
        row=source_step(student,teacher,optimizer,x,a,y.float(),phase=iteration%4)
        history.append(dict(step=iteration+1,length=length,**row))
        if (iteration+1)%64==0:
            _durable_json(args.output/'training.json',history)
            print(json.dumps(dict(stage='source_prepare',**history[-1])),flush=True)
    student.requires_grad_(False)
    for k,v in teacher.state_dict().items():torch.testing.assert_close(v,original[k],rtol=0,atol=0)
    changed=[k for k,v in student.state_dict().items() if not torch.equal(v,original[k])]
    if not changed or any(not k.startswith(('base.embedding.','base.encoder.','base.norm.','decoder.')) for k in changed):
        raise AssertionError('candidate changed an undeclared module')
    candidate={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
    with (args.output/'prepared.pt').open('xb') as handle:
        torch.save(dict(model=candidate,parent_sha256=PARENT_SHA256,protocol=protocol),handle)
    result=source_test(student,args.output/'jtext_test220',threshold)
    qualified=result['tpr']>=.90 and result['fpr']<.20
    source_summary={k:v for k,v in result.items() if k!='predictions'}
    _durable_json(args.output/'source_gate.json',dict(qualified=qualified,metrics=source_summary,
        candidate_sha256=_hash(args.output/'prepared.pt'),changed_tensor_names=changed))
    print(json.dumps(dict(stage='source_gate',qualified=qualified,metrics=source_summary)),flush=True)
    if not qualified:
        _durable_json(args.output/'completed.json',dict(status='SOURCE_CANDIDATE_FAILED',
            qualified=False,source=source_summary,east_run=False,
            elapsed_seconds=time.monotonic()-started,parent_unchanged=_hash(DEFAULT_PARENT)==PARENT_SHA256))
        return
    order=[dict(shot=sid) for sid in load_test_contract('EAST')['shot_ids']]
    _durable_json(args.output/'order.json',order)
    provider=SimpleNamespace(read_shot=partial(simple_data.read_shot,rate_contract='nominal'),
        attach_outcome=simple_data.attach_outcome,window=simple_data.window)
    results={};signals=None;decisions={}
    for name,arm in [('prepared_static','static'),('prepared_intra','intra')]:
        student.load_state_dict(candidate);student.requires_grad_(False)
        folder=args.output/name;folder.mkdir()
        emitted=run_arm(student,order=order,arm=arm,threshold=threshold,output_dir=folder/'stream',
            stride=32,data_api=provider,predictor=predict_shot)
        if signals is None:
            signals=[simple_data.attach_outcome(provider.read_shot('EAST',r['shot']),'EAST',{}) for r in order]
        sealed,seals=load_sealed(folder/'stream',order,signals,threshold)
        rows=[]
        for r in sealed:
            valid=r['valid'].copy()
            if r['is_disrupt']:valid &= r['times']<=r['down']
            rows.append(dict(r,times=r['times'][valid],scores=r['scores'][valid]))
        exact=score_fixed_test('EAST',rows,threshold)
        if exact!=emitted['metrics']:raise AssertionError('sealed TEST360 recount differs')
        side=side_metrics(folder/'stream',order,threshold,0,360)
        if any(exact[k]!=side[k] for k in exact):raise AssertionError('independent side recount differs')
        for k,v in student.state_dict().items():torch.testing.assert_close(v.cpu(),candidate[k],rtol=0,atol=0)
        results[name]={k:v for k,v in side.items() if k!='predictions'}
        results[name]['accepted_updates']=emitted['accepted_intra_updates']
        decisions[name]=exact['predictions']
        _durable_json(folder/'verification.json',dict(seals=seals,candidate_restored=True,sealed_recount=True))
        _durable_json(args.output/'progress.json',results)
    baseline,new=results['prepared_static'],results['prepared_intra']
    passed=new['tpr']>baseline['tpr'] and new['fpr']<=.20 and new['J']>baseline['J']
    flips=[]
    for old,newrow in zip(decisions['prepared_static'],decisions['prepared_intra']):
        if (old['alarm'],old['timely'])!=(newrow['alarm'],newrow['timely']):
            flips.append(dict(shot=old['shot'],before=old,after=newrow))
    if _hash(DEFAULT_PARENT)!=PARENT_SHA256:raise AssertionError('qualified parent changed')
    _durable_json(args.output/'completed.json',dict(status='COMPLETED',results=results,
        source=source_summary,source_qualified=True,meets_intra_goal=bool(passed),flips=flips,
        parent_unchanged=True,candidate_restored_after_every_shot=True,
        elapsed_seconds=time.monotonic()-started,independent_confirmation=False))
    print(json.dumps(dict(stage='completed',meets_intra_goal=bool(passed),results=results)),flush=True)


if __name__=='__main__':main()
