"""Three full matched-direction controls; labels only after all curves are sealed."""
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
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..diverse_support_20261023.run import READ,make_parent
from ..event_prefix_probe import read_events,DEFAULT_LABELS,LABEL_SHA256
from ..feedback_gradient_20261023.geometry import lead_ms
from ..heldout_threshold_20261023.run import models_for_calibration
from ..matured_negative_20261023.historical_control import PRIMARY,PRIMARY_SHA,REFERENCE,negative_replay
from ..matured_negative_20261023.run import historical_replay
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics,sha
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .learning import ARMS,predict

PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/到期反馈相对风险_实验约定_20261023.md'
ALL_ARMS=('static','previous_bce')+ARMS


def evaluate(root,p):
    records={a:[] for a in ALL_ARMS};horizon={a:dict(has_30_125=0,early_only=0) for a in ALL_ARMS};cases=[]
    previous=READ(PRIMARY/'per_shot.json')
    a={r['shot']:r for r in previous['static/calibrated']['predictions']}
    b={r['shot']:r for r in previous['matured_plus_replay/calibrated']['predictions']}
    lost=[sid for sid in a if a[sid]['timely'] and not b[sid]['timely']]
    checked=0
    for sid in p['shot_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        assert not s['is_disrupt'] or (s['times']<s['down']).all()
        with np.load(root/'shots'/f'{sid}.npz') as z:
            np.testing.assert_array_equal(z['times'],s['times'])
            lead=lead_ms(s['times'],s['down']) if s['is_disrupt'] else None
            near=(lead>30)&(lead<=125)&s['valid'] if lead is not None else None
            peaks={}
            for arm in ALL_ARMS:
                records[arm].append(dict(shot=sid,times=s['times'][s['valid']],scores=z[arm][s['valid']],is_disrupt=s['is_disrupt'],down=s['down']))
                if s['is_disrupt']:
                    mask=(z[arm]>=p['threshold'])&s['valid']
                    horizon[arm]['has_30_125']+=int((mask&near).any())
                    horizon[arm]['early_only']+=int((mask&(lead>125)).any() and not (mask&near).any())
                    if sid in lost:peaks[arm]=float(z[arm][near].max()) if near.any() else None
            if sid in lost:cases.append(dict(shot=sid,static_lead=a[sid]['lead_ms'],previous_lead=b[sid]['lead_ms'],peaks_30_125=peaks))
        updates=READ(root/'shots'/f'{sid}_updates.json')
        for arm,rs in updates.items():
            for r in rs:
                count=r['first_affected_prediction'];ends=r['negative_indices'];now=float(s['times'][count-1])
                assert r['certified_event_free_through']==now
                assert all(e>=127 and e<count and s['valid'][e] and s['y'][e]==0 for e in ends)
                assert all(125<=age<253 for age in lead_ms(s['times'][ends],now))
                checked+=len(ends)
    result={}
    for level,tau in [('calibrated',p['threshold']),('source_threshold',p['source_threshold'])]:
        for arm in ALL_ARMS:
            r=metrics(records[arm],tau,lead=30.)
            r['very_early_over3000']=sum(v['timely'] and v['lead_ms']>3000 for v in r['predictions'])
            result[arm+'/'+level]=r
    assert sha(DEFAULT_LABELS)==LABEL_SHA256
    events,metadata=read_events(DEFAULT_LABELS)
    diagnostic=dict(scope='POSTHOC_NOT_TRAINING',threshold=p['threshold'],horizon_counts=horizon,lost_cases=cases,
                    matured_negative_uses_checked=checked,event_file=str(DEFAULT_LABELS),event_sha256=LABEL_SHA256,
                    event_metadata_overlap=sorted(set(lost)&set(metadata)),event_interval_overlap=sorted(set(lost)&set(events)))
    _durable_json(root/'horizon_diagnosis.json',diagnostic)
    _durable_json(root/'per_shot.json',result)
    for level in ('calibrated','source_threshold'):
        assert result['static/'+level]==previous['static/'+level]
        assert result['previous_bce/'+level]==previous['matured_plus_replay/'+level]
    return {k:public(v) for k,v in result.items()}


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    assert sha(PRIMARY/'completed.json')==PRIMARY_SHA
    previous=READ(PRIMARY/'completed.json');p=READ(PRIMARY/'protocol.json')
    for name,digest in previous['artifact_sha256'].items():assert sha(PRIMARY/name)==digest,name
    for name,digest in p['code_sha256'].items():assert sha(io23.ROOT/name)==digest,name
    for name,digest in p['input_sha256'].items():assert sha(name)==digest,name
    torch.set_num_threads(2);torch.manual_seed(131);started=time.monotonic()
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];before=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');positive,pm=historical_replay(p['preparation_ids'],mapping)
    assert pm==p['replay_manifest'];negative,nm=negative_replay(pm,mapping)
    root.mkdir(parents=True);(root/'shots').mkdir()
    files=set(Path(__file__).parent.glob('*.py'));files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_matured_rank.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    protocol=dict(shot_ids=p['shot_ids'],preparation_ids=p['preparation_ids'],positive_manifest=pm,negative_manifest=nm,
        threshold=p['threshold'],source_threshold=p['source_threshold'],source_sha256=PARENT_SHA256,primary_sha256=PRIMARY_SHA,
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},plan_sha256=sha(PLAN),input_sha256=p['input_sha256'],
        event_labels_used_for_adaptation=False,additional_target_label_identities=0,inter_updates=0,alarm_updates=0,
        arms=list(ALL_ARMS),learning_rate=.001,clip_norm=1.,independent_confirmation=False,
        scope='CONDITIONAL_OFFLINE: pre-event status assumed; dose controls depend on current-rank step length',
        gpu=torch.cuda.get_device_name(),torch=torch.__version__)
    _durable_json(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
    def deny(*args,**kwargs):raise AssertionError('current/future outcomes forbidden before sealed predictions')
    receipts=[];steps={a:0 for a in ARMS}
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in p['shot_ids']:
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=dict(transform({k:s[k] for k in ('values','available')},mapping),times=s['times'])
            curves,rows=predict(model,signal,negative,positive)
            with np.load(PRIMARY/'shots'/f'{sid}.npz') as z:
                np.testing.assert_array_equal(curves['static'],z['static']);curves['previous_bce']=z['matured_plus_replay'].copy()
            path=root/'shots'/f'{sid}.npz';up=root/'shots'/f'{sid}_updates.json'
            np.savez_compressed(path,times=s['times'],**curves);_durable_json(up,rows)
            for arm,rs in rows.items():steps[arm]+=sum(r['gradient_steps'] for r in rs)
            receipts.append(dict(shot=sid,prediction_sha256=sha(path),update_sha256=sha(up)))
            if len(receipts)%20==0:print(dict(shots=len(receipts),seconds=time.monotonic()-started),flush=True)
    _durable_json(root/'prediction_receipts.json',receipts)
    assert_unchanged(model,before);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    metrics_=evaluate(root,protocol)
    done=dict(status='COMPLETED_CONDITIONAL_RANK_COMPARISON',metrics=metrics_,steps=steps,seconds=time.monotonic()-started,
              source_unchanged=True,independent_confirmation=False,
              artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',done)
    print({k:v for k,v in done.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
