"""Replay real states, then diagnose real/flat/change gradients with offline labels."""
import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..diverse_support_20261023.run import READ, make_parent
from ..heldout_threshold_20261023.run import REFERENCE, REFERENCE_SHA, models_for_calibration
from ..history_tangent_20261023.learning import logit
from ..increment_forecast_20261023.learning import forecast_examples, target_mask, update
from ..paired_feedback_20261022.model import initial_encoder
from ..persistence_control_20261023.run import CALIBRATION, CALIBRATION_SHA
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .geometry import losses, trial_step, vector_geometry, lead_ms

PLAN = io23.ROOT/'docs/research/source_event_rebuild_20261022/真实反馈梯度拆分_事前诊断约定_20261023.md'


def same_state(model, signal, outcome, count, fast, real_after, old_scores, threshold):
    header = dict(shot=outcome['shot'], count=count, is_disrupt=outcome['is_disrupt'],
                  alarm_already_emitted=bool(np.any(old_scores[:count][outcome['valid'][:count]] >= threshold)))
    indices = np.arange(count, min(count+128, len(signal['values'])))
    keep = outcome['valid'][indices].copy()
    if outcome['is_disrupt']:
        keep &= lead_ms(outcome['times'][indices],outcome['down']) > 30.
    if not keep.any():
        return dict(header, status='NO_LATER_TIMELY_ENDPOINT')
    x, m, truth, tm, ends = forecast_examples({k:v[:count] for k,v in signal.items()}, count, model.cfg, 'cuda')
    mask = target_mask(x, m, truth, tm)
    if not mask.any():
        return dict(header, status='NO_OBSERVED_TARGET')
    prediction = model.predict_fast(x, m, fast)['forecast']
    parts = losses(model, prediction, x, m, truth, tm)
    gradients = {k:torch.autograd.grad(v, tuple(fast.values()), retain_graph=True)
                 for k,v in parts.items()}
    for a,b,c in zip(gradients['real'],gradients['flat'],gradients['change']):
        torch.testing.assert_close(c, a-b, atol=2e-8, rtol=2e-4)
    true_step = trial_step(fast, gradients['real'])
    for key in fast:
        torch.testing.assert_close(true_step[key], real_after[key], rtol=0, atol=0)
    output_grad = {k:torch.autograd.grad(v, prediction, retain_graph=True)[0].detach()
                   for k,v in parts.items()}
    channels = []
    target = torch.asinh((torch.where(mask,truth,model.base.center)-model.base.center)/model.base.scale)
    origin = torch.asinh((x[:, -1:]-model.base.center)/model.base.scale).expand_as(target)
    for c,name in enumerate(io23.RAW23):
        measured = mask[:,:,c]
        channels.append(dict(feature=name, count=int(measured.sum()),
            real_output_gradient_norm=float(output_grad['real'][:,:,c].norm()),
            flat_output_gradient_norm=float(output_grad['flat'][:,:,c].norm()),
            change_output_gradient_norm=float(output_grad['change'][:,:,c].norm()),
            arrived_change_rms=float((target[:,:,c]-origin[:,:,c])[measured].square().mean().sqrt())
                if measured.any() else None))
    windows = [simple_data.window(signal,int(e),128) for e in indices]
    xx, mm = [torch.as_tensor(np.stack(v),device='cuda') for v in zip(*windows)]
    def future_logits(state):
        return torch.cat([logit(model,xx[lo:lo+32],mm[lo:lo+32],state) for lo in range(0,len(xx),32)])
    z = future_logits(fast)
    valid = torch.as_tensor(keep,device='cuda')
    peak = z[valid].max()
    selected = int(indices[keep][int(z[valid].detach().argmax())])
    label = peak.new_tensor(float(outcome['is_disrupt']))
    risk = F.binary_cross_entropy_with_logits(peak,label)
    risk_gradient = torch.autograd.grad(risk,tuple(fast.values()))
    candidates = {}
    for name,g in gradients.items():
        trial = trial_step(fast,g)
        with torch.no_grad():
            next_peak = future_logits(trial)[valid].max()
            actual = F.binary_cross_entropy_with_logits(next_peak,label)-risk.detach()
        geom = vector_geometry(g,risk_gradient)
        norm = float(sum(v.square().sum() for v in g).sqrt())
        candidates[name] = dict(**geom, loss=float(parts[name].detach()),
            first_order_change=-.001/max(norm,1.)*geom['dot'],
            actual_risk_loss_change=float(actual), after_peak_probability=float(next_peak.sigmoid()))
    lead = float(lead_ms([outcome['times'][selected]],outcome['down'])[0]) if outcome['is_disrupt'] else None
    return dict(header,status='MEASURED',forecast_origins=ends,query_index=selected,query_lead_ms=lead,
        phase='safe' if lead is None else ('early_disruptive_gt125' if lead>125 else 'actionable_30_125'),
        before_peak_probability=float(peak.sigmoid().detach()),
        point_label125=float(outcome['y'][selected]), operational_label=float(label),
        real_vs_flat=vector_geometry(gradients['real'],gradients['flat']),
        change_vs_real=vector_geometry(gradients['change'],gradients['real']),
        candidates=candidates,channels=channels)


def replay_shot(model, sid, mapping, threshold):
    raw = apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
    signal = transform({k:raw[k] for k in ('values','available')},mapping)
    original = READ(REFERENCE/'fold0/shots'/f'{sid}_updates.json')['mapped_fixed']
    selected = {0,len(original)//2,len(original)-1} if original else set()
    snapshots = []; fast = initial_encoder(model); receipts = []
    def deny(*args,**kwargs): raise AssertionError('no outcome access during update replay')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for j,expected in enumerate(original):
            count=expected['first_affected_prediction']; before=fast
            fast,r=update(model,{k:v[:count] for k,v in signal.items()},count,fast,device='cuda')
            assert r==expected, (sid,j)
            receipts.append(r)
            if j in selected: snapshots.append((count,before,fast))
    # Actual trajectory is now sealed and checked; outcomes affect diagnostics only.
    outcome = simple_data.attach_outcome(raw,'EAST',{})
    with np.load(REFERENCE/'fold0/shots'/f'{sid}.npz') as z:
        old=z['mapped_fixed_intra'].copy()
    cases=[same_state(model,signal,outcome,count,before,after,old,threshold)
           for count,before,after in snapshots]
    return cases, len(receipts)


def summarize(cases):
    groups={}
    for opened in (True,False):
        for phase in ('safe','early_disruptive_gt125','actionable_30_125'):
            rs=[r for r in cases if r['status']=='MEASURED' and r['phase']==phase
                and (not r['alarm_already_emitted'])==opened]
            if not rs: continue
            key=f'{phase}/decision_{"open" if opened else "already_alarm"}'
            cs=[r['real_vs_flat']['cosine'] for r in rs if r['real_vs_flat']['cosine'] is not None]
            groups[key]=dict(records=len(rs),shots=len({r['shot'] for r in rs}),
                median_real_flat_cosine=float(np.median(cs)) if cs else None,
                median_change_real_norm_ratio=float(np.median([r['change_vs_real']['norm']/r['change_vs_real']['reference_norm']
                    for r in rs if r['change_vs_real']['reference_norm']>0])), candidates={})
            for arm in ('real','flat','change'):
                values=np.array([r['candidates'][arm]['actual_risk_loss_change'] for r in rs])
                groups[key]['candidates'][arm]=dict(better=int((values< -1e-7).sum()),
                    worse=int((values>1e-7).sum()),negligible=int((abs(values)<=1e-7).sum()),
                    median_change=float(np.median(values)),mean_change=float(values.mean()))
    return groups


def main(root):
    if root.exists(): raise FileExistsError(root)
    assert sha(REFERENCE/'completed.json')==REFERENCE_SHA
    assert sha(CALIBRATION/'completed.json')==CALIBRATION_SHA
    for folder in (REFERENCE,CALIBRATION):
        for name,digest in READ(folder/'completed.json')['artifact_sha256'].items(): assert sha(folder/name)==digest,name
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic()
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];frozen=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');rp=READ(REFERENCE/'protocol.json')
    threshold=READ(CALIBRATION/'thresholds.json')['mapped_fixed']['0.2']['threshold']
    root.mkdir(parents=True);(root/'shots').mkdir()
    files=set(Path(__file__).parent.glob('*.py'))
    files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_feedback_gradient_decomposition.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None): files.add(Path(module.__file__))
    protocol=dict(shot_ids=rp['shot_ids'],threshold=threshold,source_sha256=PARENT_SHA256,
        reference_sha256=REFERENCE_SHA,calibration_sha256=CALIBRATION_SHA,plan_sha256=sha(PLAN),
        scope='POSTHOC DIAGNOSTIC ONLY; real trajectory replay, future outcomes used only in shadow diagnostics',
        no_new_deployed_policy=True,independent_confirmation=False,neural_preparation_steps=0,
        input_sha256=rp['input_sha256'],code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files})
    _durable_json(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files): z.write(f,str(f.relative_to(io23.ROOT)))
    cases=[];updates=0
    for i,sid in enumerate(protocol['shot_ids'],1):
        local,n=replay_shot(model,sid,mapping,threshold);cases.extend(local);updates+=n
        _durable_json(root/'shots'/f'{sid}.json',local)
        if i%20==0: print(dict(stage='diagnose',shots=i,replayed_updates=updates,seconds=time.monotonic()-start),flush=True)
    assert_unchanged(model,frozen);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    summary=summarize(cases);_durable_json(root/'summary.json',summary)
    done=dict(status='COMPLETED_DIAGNOSTIC',replayed_updates=updates,cases=len(cases),
        measured=sum(r['status']=='MEASURED' for r in cases),source_unchanged=True,seconds=time.monotonic()-start,
        artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',done)
    print({k:v for k,v in done.items() if k!='artifact_sha256'},flush=True);print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
