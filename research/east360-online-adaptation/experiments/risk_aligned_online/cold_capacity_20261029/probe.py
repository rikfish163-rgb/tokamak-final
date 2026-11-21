"""Label-informed point reachability, NOT a deployable or causal online policy."""
import argparse
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import read
from ..delayed_recurrent_memory_20261025.learning import batch
from ..fixed_test_protocol import validate_test_ids
from ..latent_completion_20261028.model import tokens, readout
from ..redesign_20261004.prepare_prior import (
    load_parent, DEFAULT_PARENT, PARENT_SHA256, classifier_state, assert_classifier_unchanged,
)
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform


def choose_ends(times, scores, flag, down):
    times, scores = np.asarray(times), np.asarray(scores)
    if times.shape != scores.shape or times.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError('matching finite full-shot curve required')
    eligible = np.arange(len(times)) >= 128
    if flag:
        lead = np.round((float(down)-times)*1000, 9)
        selections = [('timely_peak', eligible & (lead>30)),
                      ('positive_window_peak', eligible & (lead>30) & (lead<=125))]
    else:
        selections = [('safe_peak', eligible)]
    rows = {}
    for name, mask in selections:
        ids = np.flatnonzero(mask)
        if len(ids):
            end = int(ids[np.argmax(scores[ids])])
            rows.setdefault(end, []).append(name)
    return [dict(end=e, reasons=reasons) for e, reasons in sorted(rows.items())]


def logits(base, x, available, state, slow):
    h, valid = tokens(base, x, available, state)
    return readout(base, h, valid, slow)


def optimize(forward, initial, signs, *, radius, steps, step):
    """Independent normalized logit ascent; retain best VISITED point, incl zero.

    Signs are supplied from terminal truth. A failure to cross threshold here
    is not a mathematical impossibility proof. No model parameters are owned.
    """
    if (initial.ndim!=2 or signs.shape!=(len(initial),) or not ((signs==1)|(signs==-1)).all()
            or not np.isfinite([radius, step]).all() or min(radius, step)<=0 or steps<0
            or not torch.isfinite(initial).all() or not (initial.norm(dim=1)<=radius).all()):
        raise ValueError('finite bounded independent states and +/-1 directions required')
    state = initial.detach().clone()
    best_state = state.clone(); best_value = torch.full_like(signs, -torch.inf)
    best_step = torch.zeros(len(signs), dtype=torch.long, device=signs.device)
    history, norms = [], []
    gradient_norm = torch.zeros_like(signs)
    for k in range(steps+1):
        state = state.detach().requires_grad_(True)
        z = forward(state)
        if z.shape!=signs.shape or not torch.isfinite(z).all():
            raise FloatingPointError('invalid probe output')
        value = signs*z
        better = value.detach()>best_value
        best_value = torch.maximum(best_value, value.detach())
        best_state[better] = state.detach()[better]; best_step[better] = k
        history.append(z.detach()); norms.append(state.detach().norm(dim=1))
        if k==steps:
            break
        g, = torch.autograd.grad(value.sum(), state)
        norm = g.norm(dim=1, keepdim=True)
        if not torch.isfinite(g).all():
            raise FloatingPointError('nonfinite oracle gradient')
        if k==0:
            gradient_norm = norm[:,0].detach()
        state = state.detach()+step*g/norm.clamp_min(1e-12)
        state = state/(state.norm(dim=1, keepdim=True)/radius).clamp_min(1.)
    return dict(state=best_state.detach(), best_step=best_step,
                logits=torch.stack(history), norms=torch.stack(norms), gradient_norm=gradient_norm)


def summarize(records, threshold, radius):
    key = f'radius{radius}'
    missed = [r for r in records if r['flag'] and not r['cold_timely']]
    safe_fp = [r for r in records if not r['flag'] and r['cold_alarm']]
    return dict(
        points=len(records), missed_shots_probed=len({r['shot'] for r in missed}),
        missed_shots_with_timely_point_crossing=sorted({r['shot'] for r in missed if r[key]['best_probability']>=threshold}),
        missed_shots_with_30_125_point_crossing=sorted({r['shot'] for r in missed if 'positive_window_peak' in r['reasons'] and r[key]['best_probability']>=threshold}),
        safe_fp_peak_below_threshold=[r['shot'] for r in safe_fp if r[key]['best_probability']<threshold],
        safe_fp_peak_below_without_early_alarm=[r['shot'] for r in safe_fp if r[key]['best_probability']<threshold and not r['early_alarm']],
        median_directed_logit_change=float(np.median([(1 if r['flag'] else -1)*(r[key]['best_logit']-r['initial_logit']) for r in records])),
        warning='point crossing is not an online result or a whole-shot FP rescue; no global upper bound')


def run(output, previous):
    if output.exists() or output.absolute().parent!=io23.ROOT/'probe_outputs':
        raise ValueError('new worktree-local diagnostic output required')
    torch.set_num_threads(2); torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    oldp=read(previous,'protocol.json'); audit=read(previous,'audit.json')
    if not audit['complete'] or audit['complete_sha256']!=sha(previous/'complete.json'):
        raise ValueError('audited full360 reference required')
    base,parent=load_parent(DEFAULT_PARENT); base.cuda(); original=classifier_state(base)
    threshold=float(parent['threshold']); ids=oldp['order']; validate_test_ids('EAST',ids)
    reference=Path(oldp['reference']); mapping=read(reference,'input_map.json'); hashes=read(reference,'input_hashes.json')
    if oldp['warmup']!=20 or oldp['threshold']!=threshold or mapping['calibration_ids']!=ids[:20]:
        raise ValueError('unchanged cold20 parent required')
    for path,h in oldp['reference_hashes'].items():
        if sha(reference/path)!=h:raise ValueError('cold reference changed')
    slow=torch.load(reference/'models/cold.pt',map_location='cpu',weights_only=True)['slow'].cuda()
    slow_before=slow.clone()
    cold={r['shot']:r for r in read(previous,'per_shot.json')['cold']}
    output.mkdir(); start=time.monotonic()
    files={Path(m.__file__) for n,m in list(sys.modules.items())
           if n.startswith('experiments.risk_aligned_online.') and getattr(m,'__file__',None)}
    protocol=dict(scope='POSTHOC_LABEL_INFORMED_POINT_DIAGNOSTIC', previous=str(previous),
        previous_complete_sha256=sha(previous/'complete.json'), previous_audit_sha256=sha(previous/'audit.json'),
        previous_per_shot_sha256=sha(previous/'per_shot.json'), order=ids, source_sha256=PARENT_SHA256,
        reference=str(reference), reference_hashes=oldp['reference_hashes'], threshold=threshold,
        warmup=20, radii=[1,4], steps=64, step=.1, state_dimensions=128,
        plan_sha256=sha(io23.ROOT/'COLD_CAPACITY_PLAN.md'), code_sha256={str(p.relative_to(io23.ROOT)):sha(p) for p in files},
        terminal_labels_used=True, online_policy_changed=False, global_upper_bound=False,
        gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    _durable_json(output/'protocol.json',protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
        z.write(io23.ROOT/'COLD_CAPACITY_PLAN.md','COLD_CAPACITY_PLAN.md')
    xs=[]; masks=[]; records=[]; coverage=[]
    for index,sid in enumerate(ids[20:],20):
        if sha(io23.path_for('EAST',sid))!=hashes[str(sid)]:raise ValueError('raw input changed')
        raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        flag,down=io23.read_outcome('EAST',sid)
        if bool(cold[sid]['is_disrupt'])!=flag:raise ValueError('outcome mismatch')
        path=previous/'shots'/f'{index+1:04d}_{sid}.npz'
        seal=read(previous/'shots',f'{index+1:04d}_{sid}_seal.json')
        if sha(path)!=seal['sha256'] or seal['outcome_used']:raise ValueError('curve seal changed')
        with np.load(path) as z:
            scores=z['cold'].copy()
            if not np.array_equal(raw['times'],z['times']):raise ValueError('times changed')
        ends=choose_ends(raw['times'],scores,flag,down)
        early=bool(not flag and (scores[:128]>=threshold).any())
        coverage.append(dict(shot=sid,flag=flag,early_alarm=early,points=len(ends),cold_alarm=cold[sid]['alarm'],cold_timely=cold[sid]['timely']))
        sig=transform({k:raw[k] for k in ('values','available')},mapping)
        if ends:
            x,a=batch(sig,[r['end'] for r in ends],'cpu'); xs.append(x); masks.append(a)
        for r in ends:
            records.append(dict(shot=sid,flag=flag,down=down,early_alarm=early,**r,
                time=float(raw['times'][r['end']]),lead_ms=float(round((down-raw['times'][r['end']])*1000,9)) if flag else None,
                cold_alarm=cold[sid]['alarm'],cold_timely=cold[sid]['timely'],sealed_probability=float(scores[r['end']]),
                reference_curve_sha256=seal['sha256']))
    if len(coverage)!=340 or len({r['shot'] for r in coverage})!=340 or not xs:
        raise ValueError('incomplete diagnostic cohort')
    x=torch.cat(xs); a=torch.cat(masks); witnesses={}
    for radius in protocol['radii']:
        key=f'radius{radius}'; all_states=[]; all_logits=[]; all_norms=[]; all_steps=[]
        for lo in range(0,len(x),32):
            xx=x[lo:lo+32].cuda(); aa=a[lo:lo+32].cuda(); rows=records[lo:lo+32]
            signs=xx.new_tensor([1 if r['flag'] else -1 for r in rows])
            result=optimize(lambda s:logits(base,xx,aa,s,slow),xx.new_zeros(len(xx),128),signs,
                            radius=radius,steps=protocol['steps'],step=protocol['step'])
            with torch.no_grad():best=logits(base,xx,aa,result['state'],slow)
            for i,r in enumerate(rows):
                initial=float(result['logits'][0,i]); p0=float(torch.sigmoid(result['logits'][0,i]))
                if abs(p0-r['sealed_probability'])>2e-6 or (p0>=threshold)!=(r['sealed_probability']>=threshold):
                    raise ValueError('cold zero-state comparator mismatch')
                k=int(result['best_step'][i]); z=float(best[i])
                if abs(z-float(result['logits'][k,i]))>3e-6:raise ValueError('best-state replay mismatch')
                r['initial_logit']=initial
                r[key]=dict(best_logit=z,best_probability=float(best[i].sigmoid()),best_step=k,
                    state_norm=float(result['state'][i].norm()),initial_gradient_norm=float(result['gradient_norm'][i]))
            all_states.append(result['state'].cpu()); all_logits.append(result['logits'].cpu())
            all_norms.append(result['norms'].cpu()); all_steps.append(result['best_step'].cpu())
        witnesses[key+'_state']=torch.cat(all_states).numpy(); witnesses[key+'_logits']=torch.cat(all_logits,1).numpy()
        witnesses[key+'_norms']=torch.cat(all_norms,1).numpy(); witnesses[key+'_best_step']=torch.cat(all_steps).numpy()
        print(dict(radius=radius,points=len(records),seconds=time.monotonic()-start),flush=True)
    path=output/'witnesses.npz'
    with path.open('xb') as f:np.savez_compressed(f,**witnesses)
    assert_classifier_unchanged(base,original)
    if not torch.equal(slow,slow_before) or sha(DEFAULT_PARENT)!=PARENT_SHA256:raise ValueError('parent changed')
    for path,h in protocol['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('code changed during diagnostic')
    _durable_json(output/'records.json',dict(coverage=coverage,records=records))
    groups={f'radius{r}':summarize(records,threshold,r) for r in protocol['radii']}
    result=dict(complete=True,scope=protocol['scope'],groups=groups,points=len(records),shots=340,
        early_fp_shots=[r['shot'] for r in coverage if r['early_alarm']],
        no_eligible_points=[r['shot'] for r in coverage if not r['points']],
        frozen_model_unchanged=True,scientific_goal_achieved=False,seconds=time.monotonic()-start,
        records_sha256=sha(output/'records.json'),witnesses_sha256=sha(output/'witnesses.npz'),
        protocol_sha256=sha(output/'protocol.json'))
    _durable_json(output/'complete.json',result); print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True);args=p.parse_args();run(args.output,args.previous)
