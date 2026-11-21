"""Conditional full-stream experiment, not deployment without event-status input."""
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
from ..feedback_gradient_20261023.geometry import lead_ms
from ..heldout_threshold_20261023.run import REFERENCE,REFERENCE_SHA,models_for_calibration
from ..persistence_control_20261023.run import CALIBRATION,CALIBRATION_SHA
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics,sha
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .learning import MODES,predict_conditional

PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/已度过窗口反馈_条件性实验约定_20261023.md'


def historical_replay(ids,mapping):
    windows=[];manifest=[]
    reader=io23.read_outcome
    def guard(device,sid):
        assert device=='EAST' and sid in ids
        return reader(device,sid)
    with patch.object(io23,'read_outcome',side_effect=guard):
        for sid in ids:
            s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
            if not s['is_disrupt']:continue
            signal=transform({k:s[k] for k in ('values','available')},mapping)
            leads=lead_ms(s['times'],s['down'])
            eligible=np.flatnonzero(s['valid'] & (s['y']>0) & (leads>30) & (np.arange(len(leads))>=127))
            if not len(eligible):raise ValueError(f'no eligible historical positive: {sid}')
            for desired in (100.,60.):
                end=int(eligible[np.argmin(abs(leads[eligible]-desired))])
                windows.append(simple_data.window(signal,end,128))
                manifest.append(dict(shot=sid,index=end,desired_lead_ms=desired,actual_lead_ms=float(leads[end])))
    assert len(windows)==8 and len({r['shot'] for r in manifest})==4
    return tuple(torch.as_tensor(np.stack(v),device='cuda') for v in zip(*windows)),manifest


def evaluate(root,protocol):
    arms=('static','real_forecast')+MODES;rows={a:[] for a in arms};consistency=[]
    for sid in protocol['shot_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        # Post-hoc validation only. No outcome-dependent update scheduling.
        preterminal=not s['is_disrupt'] or bool(np.all(s['times']<s['down']))
        if not preterminal:raise AssertionError('conditional event-free packet assumption is false')
        updates=READ(root/'shots'/f'{sid}_updates.json')
        checked=0
        for arm in MODES:
            for r in updates[arm]:
                ends=r['negative_indices']; count=r['first_affected_prediction']
                assert r['certified_event_free_through']==float(s['times'][count-1])
                assert all(e<count and s['valid'][e] and s['y'][e]==0 for e in ends)
                assert all(v>=125. for v in lead_ms(s['times'][ends],s['times'][count-1]))
                checked+=len(ends)
        consistency.append(dict(shot=sid,preterminal_record=preterminal,checked_negative_uses=checked))
        with np.load(root/'shots'/f'{sid}.npz') as z:
            np.testing.assert_array_equal(z['times'],s['times'])
            for arm in arms:
                rows[arm].append(dict(shot=sid,times=s['times'][s['valid']],scores=z[arm][s['valid']],is_disrupt=s['is_disrupt'],down=s['down']))
    results={}
    for label,tau in (('calibrated',protocol['threshold']),('source_threshold',protocol['source_threshold'])):
        for arm in arms:
            r=metrics(rows[arm],tau,lead=30.)
            r['very_early_over3000']=sum(p['timely'] and p['lead_ms']>3000 for p in r['predictions'])
            results[arm+'/'+label]=r
    _durable_json(root/'per_shot.json',results)
    _durable_json(root/'maturity_consistency.json',consistency)
    return {k:public(v) for k,v in results.items()}


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    assert sha(REFERENCE/'completed.json')==REFERENCE_SHA
    assert sha(CALIBRATION/'completed.json')==CALIBRATION_SHA
    for base in (REFERENCE,CALIBRATION):
        for name,digest in READ(base/'completed.json')['artifact_sha256'].items():assert sha(base/name)==digest,name
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic()
    parent,source_threshold=make_parent();model=models_for_calibration(parent)['mapped_fixed'];before=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');rp=READ(REFERENCE/'protocol.json')
    replay,manifest=historical_replay(rp['selected'],mapping)
    root.mkdir(parents=True);(root/'shots').mkdir()
    files=set(Path(__file__).parent.glob('*.py'));files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_matured_negative.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    protocol=dict(shot_ids=rp['shot_ids'],preparation_ids=rp['selected'],replay_manifest=manifest,
        threshold=READ(CALIBRATION/'thresholds.json')['mapped_fixed']['0.2']['threshold'],source_threshold=source_threshold,
        source_sha256=PARENT_SHA256,reference_sha256=REFERENCE_SHA,calibration_sha256=CALIBRATION_SHA,
        input_sha256=rp['input_sha256'],plan_sha256=sha(PLAN),code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
        scope='CONDITIONAL_OFFLINE: reliable live non-event status assumed; not certified available in deployment',
        independent_confirmation=False,neural_preparation_steps=0,inter_updates=0,alarm_updates=0,
        additional_target_label_identities=0,arms=list(('static','real_forecast')+MODES),learning_rate=.001,clip_norm=1.,
        loss_coefficients=dict(negative=.5,replay=.5),
        gpu=torch.cuda.get_device_name(),torch=torch.__version__)
    _durable_json(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
    def deny(*args,**kwargs):raise AssertionError('current/future outcomes unavailable during conditional predictions')
    steps={a:0 for a in MODES};receipts=[]
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in protocol['shot_ids']:
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=dict(transform({k:s[k] for k in ('values','available')},mapping),times=s['times'])
            curves,updates=predict_conditional(model,signal,replay)
            with np.load(REFERENCE/'fold0/shots'/f'{sid}.npz') as z:
                np.testing.assert_array_equal(curves['static'],z['mapped_fixed_static'])
                curves['real_forecast']=z['mapped_fixed_intra'].copy()
            for arm in MODES:steps[arm]+=sum(r['gradient_steps'] for r in updates[arm])
            path=root/'shots'/f'{sid}.npz';up=root/'shots'/f'{sid}_updates.json'
            np.savez_compressed(path,times=s['times'],**curves);_durable_json(up,updates)
            receipts.append(dict(shot=sid,prediction_sha256=sha(path),update_sha256=sha(up)))
            if len(receipts)%20==0:print(dict(stage='conditional_predict',shots=len(receipts),seconds=time.monotonic()-start),flush=True)
    _durable_json(root/'prediction_receipts.json',receipts)
    assert_unchanged(model,before);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    results=evaluate(root,protocol)
    reference=READ(CALIBRATION/'completed.json')['metrics']
    assert results['static/calibrated']==reference['mapped_fixed_static/0.2']
    assert results['real_forecast/calibrated']==reference['mapped_fixed_intra/0.2']
    done=dict(status='COMPLETED_CONDITIONAL_OFFLINE',metrics=results,online_steps=steps,
              seconds=time.monotonic()-start,source_unchanged=True,independent_confirmation=False,
              artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',done)
    print({k:v for k,v in done.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
