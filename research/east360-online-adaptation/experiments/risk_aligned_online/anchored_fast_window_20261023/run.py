"""Fixed paired experiment: only current-window parameter change enters features."""
import argparse
import copy
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ..internal_fast_window_20261023.run import (
    io23,READ,make_parent,models_for_calibration,REFERENCE,read_signal,labeled,
    transform,state_cpu,assert_unchanged,_durable_json,sha,DEFAULT_PARENT,PARENT_SHA256,
    training_pools,sample_batch,risk_step,outcome_record,metrics,public)
from .model import AnchoredAdapter
from .learning import predict,ARMS

PREVIOUS=io23.ROOT/'experiments/risk_aligned_online/runs/centered_fast_window360_20261023'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/只写临时变化_事前约定.md'


def ranking(rows):
    """Post-hoc diagnostic ONLY: not a calibration rule or deployable result."""
    label=np.array([r['is_disrupt'] for r in rows],bool); peaks=[]
    for r in rows:
        use=np.round((r['down']-r['times'])*1000,9)>30 if r['is_disrupt'] else np.ones(len(r['times']),bool)
        # Explicit sentinel below all probabilities: such a shot has no
        # eligible timely prediction, not a fabricated zero-time observation.
        peaks.append(float(r['scores'][use].max()) if use.any() else -1.)
    s=np.array(peaks); d=s[label]; safe=s[~label]
    if not len(d) or not len(safe): return dict(status='single_class_smoke_only')
    auc=float(((d[:,None]>safe).sum()+.5*(d[:,None]==safe).sum())/(len(d)*len(safe)))
    np.testing.assert_allclose(auc,roc_auc_score(label,s),atol=1e-14,rtol=0)
    candidates=np.r_[np.unique(s[s>=0]),np.nextafter(max(float(s.max()),0.),np.inf)]
    frontier={}
    for cap in (.1,.15,.2):
        limit=int(np.floor(cap*len(safe)+1e-9))
        frontier[str(cap)]=max(int((d>=t).sum()) for t in candidates if int((safe>=t).sum())<=limit)
    return dict(status='TEST_LABEL_DIAGNOSTIC_NOT_DEPLOYABLE',timely_peak_auc=auc,oracle_tp_at_fpr_cap=frontier)


def evaluate(root,protocol,outcomes):
    rows={a:[] for a in ARMS}; old_rows={a:[] for a in ('parent','capacity')}
    for r in outcomes:
        with np.load(root/'shots'/f'{r["shot"]}.npz') as z,np.load(PREVIOUS/'shots'/f'{r["shot"]}.npz') as old:
            _,valid=io23.labels_at(z['times'],r['is_disrupt'],r['down'],125.)
            for collection,archive in ((rows,z),(old_rows,old)):
                for a in collection: collection[a].append(dict(shot=r['shot'],times=z['times'][valid],
                    scores=archive[a][valid],is_disrupt=r['is_disrupt'],down=r['down']))
    result={a:metrics(v,protocol['threshold'],lead=30.) for a,v in rows.items()}
    for a,v in result.items(): v['very_early_over3000']=sum(r['timely'] and r['lead_ms']>3000 for r in v['predictions'])
    _durable_json(root/'per_shot.json',result)
    _durable_json(root/'diagnostic_ranking.json',dict(current={a:ranking(v) for a,v in rows.items()},
        preceding_static_capacity={a:ranking(v) for a,v in old_rows.items()}))
    blocks=[dict(end=n,cumulative={a:public(metrics(v[:n],protocol['threshold'],lead=30.)) for a,v in rows.items()})
            for n in range(20,len(outcomes)+1,20)]
    _durable_json(root/'blocks.json',blocks)
    return {a:public(v) for a,v in result.items()}


def main(root,smoke=False):
    if root.exists(): raise FileExistsError(root)
    if not torch.cuda.is_available(): raise RuntimeError('local rl_env CUDA required')
    prior=READ(PREVIOUS/'protocol.json'); done=READ(PREVIOUS/'completed.json')
    assert READ(PREVIOUS/'verification.json')['completed_sha256']==sha(PREVIOUS/'completed.json')
    for f,h in done['artifact_sha256'].items(): assert sha(PREVIOUS/f)==h,f
    torch.set_num_threads(2); torch.manual_seed(131); start=time.monotonic()
    base,_=make_parent(); parent=models_for_calibration(base)['mapped_fixed']
    fixed=AnchoredAdapter(copy.deepcopy(parent.base),rule='fixed').cuda().eval()
    normalized=AnchoredAdapter(copy.deepcopy(parent.base),rule='normalized').cuda().eval()
    initial=torch.load(PREVIOUS/'layers.pt',map_location='cpu',weights_only=True)['initial']
    for m in (fixed,normalized): m.layer.load_state_dict(initial)
    original=state_cpu(parent.base); root.mkdir(parents=True); (root/'shots').mkdir()
    mapping=READ(REFERENCE/'input_map.json'); ids=prior['shot_ids'][:2] if smoke else prior['shot_ids']; steps=4 if smoke else 512
    files=set(Path(__file__).parent.glob('*.py')); files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_anchored_fast_window.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None): files.add(Path(module.__file__))
    protocol=dict(prior,shot_ids=ids,additional_preparation_steps=steps,arms=ARMS,variant='delta_only_write_fixed_vs_trace_normalized',
        scope='SMOKE_ONLY' if smoke else 'DEVELOPMENT_ONLY; exact parent when updates off',
        previous_completed_sha256=sha(PREVIOUS/'completed.json'),plan_sha256=sha(PLAN),
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files})
    _durable_json(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(files): archive.write(f,str(f.relative_to(io23.ROOT)))
    reader=io23.read_outcome; records=[]; access=[]
    def prep_reader(device,sid):
        assert device=='EAST' and sid in prior['preparation_ids']
        access.append(dict(shot=sid,phase='historical_preparation')); return reader(device,sid)
    with patch.object(io23,'read_outcome',side_effect=prep_reader):
        for sid in prior['preparation_ids']: records.append(labeled(read_signal(sid),io23.read_outcome('EAST',sid),mapping))
    _durable_json(root/'preparation_outcomes.json',[dict(shot=s['shot'],is_disrupt=s['is_disrupt'],down=s['down']) for s in records])
    pools=training_pools(records); rng=np.random.default_rng(131); models=dict(fixed=fixed,normalized=normalized)
    opts={name:torch.optim.Adam(m.layer.parameters(),lr=.001) for name,m in models.items()}
    train=[]; old_train=READ(PREVIOUS/'training.json')
    def deny(*args,**kwargs): raise AssertionError('no stream outcomes before all predictions are sealed')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for step in range(steps):
            x,m,y,samples=sample_batch(records,pools,rng,'cuda'); assert samples==old_train[step]['samples']
            with torch.no_grad(): h,valid=normalized.tokens(x,m)
            row=dict(step=step,samples=samples)
            for name,net in models.items(): row[name]=risk_step(net,opts[name],h,valid,y,mode='on')
            train.append(row)
            if (step+1)%128==0: print(dict(stage='prepare',step=step+1,seconds=time.monotonic()-start),flush=True)
    _durable_json(root/'training.json',train)
    torch.save(dict(initial=initial,**{k:state_cpu(m.layer) for k,m in models.items()}),root/'layers.pt')
    states={name:state_cpu(m) for name,m in models.items()}; seals=[]
    with patch.object(io23,'read_outcome',side_effect=deny):
        for index,sid in enumerate(ids):
            s=read_signal(sid); curves,traces=predict(fixed,normalized,transform({k:s[k] for k in ('values','available')},mapping))
            with np.load(PREVIOUS/'shots'/f'{sid}.npz') as z:
                np.testing.assert_array_equal(s['times'],z['times']); np.testing.assert_array_equal(curves['parent'],z['parent'])
            trace_hashes={}
            for name,t in traces.items():
                assert all(np.isfinite(v).all() for v in t.values()),(sid,name)
                assert np.all(t['loss_after']<=t['loss_before']+1e-6+1e-5*t['loss_before']),(sid,name)
                f=root/'shots'/f'{sid}_{name}_inner.npz'; np.savez_compressed(f,**t); trace_hashes[name]=sha(f)
            f=root/'shots'/f'{sid}.npz'; np.savez_compressed(f,times=s['times'],**curves)
            seals.append(dict(shot=sid,prediction_sha256=sha(f),inner_sha256=trace_hashes))
            if (index+1)%20==0 or index+1==len(ids):
                progress=dict(shots=index+1,seconds=time.monotonic()-start)
                _durable_json(root/'progress.json',progress); print(progress,flush=True)
    _durable_json(root/'prediction_seals.json',seals); outcomes=[]
    for index,r in enumerate(seals):
        sid=r['shot']; assert sha(root/'shots'/f'{sid}.npz')==r['prediction_sha256']
        outcomes.append(outcome_record(sid,reader('EAST',sid),r['prediction_sha256'],index)); access.append(dict(shot=sid,phase='after_all_prediction_seals'))
    _durable_json(root/'outcomes.json',outcomes); _durable_json(root/'label_access.json',access)
    for name,m in models.items(): assert_unchanged(m,states[name]); assert_unchanged(m.base,original)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=evaluate(root,protocol,outcomes)
    completed=dict(status='COMPLETED_SMOKE' if smoke else 'COMPLETED_ANCHORED_WINDOW_EXPERIMENT',metrics=result,
        source_unchanged=True,parent_unchanged=True,seconds=time.monotonic()-start,
        artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',completed); print({k:v for k,v in completed.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('root',type=Path); parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); main(args.root,args.smoke)
