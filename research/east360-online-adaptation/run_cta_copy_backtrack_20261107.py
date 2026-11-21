"""Diagnose causally selected forecast descent in the already-trained copies."""

import argparse
from collections import Counter
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

import cta_copy_kernel_20261107 as kernel
import cta_copy_backtrack_20261107 as backtrack
import run_cta_copy_pilot_20261107 as p
from motion_metric_model_20261107 import scramble


FIT=p.ROOT/'probe_outputs/cta_copy_pilot_20261107_v1'
PLAN=p.ROOT/'CTA_COPY_BACKTRACK_PROTOCOL_20261107.md'


def evaluate(base,models,calibrations,data,rows,out,role,tau):
    teacher=p.teacher_outputs(base,data)['q']['logit'].cpu().numpy().astype(np.float64)
    values={'teacher':teacher}
    summaries={}
    for kind,model in models.items():
        before=p.snapshot(model)
        with torch.no_grad():
            static=kernel.forward_features(model,data['xq'],data['aq'])['logit']
            release=kernel.forward_features(model,data['xr'],data['ar'])['logit']
        curves={arm:[] for arm in p.ARMS}
        factors,proxy_before,proxy_after,norms,unmatched=[],[],[],[],[]
        for i in range(len(rows)):
            d=p.part(data,slice(i,i+1))
            calibration={k:calibrations[kind][k] for k in ('reference_norm','budget')}
            proposal=kernel.inner_step(model,d['x0'],d['a0'],d['truth'],d['mask'],**calibration)
            selection=backtrack.select_scale(model,d['x0'],d['a0'],d['truth'],d['mask'],proposal)
            factor=selection['factor']
            parameters=backtrack.scaled_parameters(model,proposal,factor)
            factors.append(factor)
            proxy_before.append(selection['loss_before'])
            proxy_after.append(selection['loss_after'])
            norms.append(factor*float(proposal['update_norm']))
            with torch.no_grad():
                query=kernel.forward_features(model,d['xq'],d['aq'],parameters=parameters)['logit'][0]
                anchor=kernel.forward_features(model,d['xr'],d['ar'],parameters=parameters)['logit'][0]
                curves['static'].append(float(static[i]))
                curves['real'].append(float(query))
                curves['scalar'].append(float(static[i]+anchor-release[i]))
            target=torch.asinh((torch.where(d['mask'],d['truth'],model.center)-model.center)/model.scale)
            for mode in ('time','channel'):
                wrong_truth=scramble(target,d['mask'],axis=mode).sinh()*model.scale+model.center
                pair=kernel.paired_steps(model,d['x0'],d['a0'],d['truth'],wrong_truth,d['mask'],**calibration)
                if abs(float(pair['real']['update_norm'])-float(proposal['update_norm']))>1e-6:
                    raise ValueError('primary real update changed with control')
                if not pair['unmatched_control'] and abs(float(pair['real']['update_norm'])-float(pair['wrong']['update_norm']))>1e-6:
                    raise ValueError('control parameter norm mismatch')
                unmatched.append(bool(pair['unmatched_control']) and factor>0)
                wrong=backtrack.scaled_parameters(model,pair['wrong'],factor)
                with torch.no_grad():
                    z=kernel.forward_features(model,d['xq'],d['aq'],parameters=wrong)['logit'][0]
                curves[mode].append(float(z))
        if not p.unchanged(model,before):
            raise ValueError('functional feedback changed parent')
        values.update({kind+'_'+arm:np.asarray(v,np.float64) for arm,v in curves.items()})
        values[kind+'_factor']=np.asarray(factors)
        values[kind+'_proxy_before']=np.asarray(proxy_before)
        values[kind+'_proxy_after']=np.asarray(proxy_after)
        summaries[kind]=dict(scale_counts={str(k):v for k,v in sorted(Counter(factors).items())},
            rejected_updates=sum(q==0 for q in factors),unmatched_controls=sum(unmatched),
            auxiliary_before=float(np.mean(proxy_before)),auxiliary_after=float(np.mean(proxy_after)),
            auxiliary_increases=sum(a>b+1e-7 for a,b in zip(proxy_after,proxy_before)),
            actual_parameter_norm_median=float(np.median(norms)),
            actual_logit_change_median=float(np.median(np.abs(np.asarray(curves['real'])-curves['static']))))
        print('evaluated',role,kind,'scales',summaries[kind]['scale_counts'],flush=True)
    with (out/(role+'_predictions.npz')).open('xb') as handle:
        np.savez_compressed(handle,**values)
    p.write(out/(role+'_sealed.json'),dict(status='SEALED_LOCAL_BACKTRACK_DIAGNOSTIC',rows=rows,fit=str(FIT)))
    with np.load(FIT/(role+'_predictions.npz'),allow_pickle=False) as original:
        parity=max(float(np.max(np.abs(values[k]-original[k]))) for k in ('teacher','forecast_only_static','aligned_static'))
    if parity>5e-6:
        raise ValueError('parent static predictions changed')
    labels=np.asarray([q['label'] for q in rows])
    losses={k:np.logaddexp(0.,(1-2*labels)*(v-tau)) for k,v in values.items()
            if k=='teacher' or any(k.endswith('_'+arm) for arm in p.ARMS)}
    table={}
    for kind in p.KINDS:
        table[kind]={}
        for group in p.GROUPS:
            teacher_loss=p.group_mean(losses['teacher'],rows,group)
            static_loss=p.group_mean(losses[kind+'_static'],rows,group)
            selected=np.array([row['group']==group for row in rows])
            delta=losses[kind+'_real']-losses[kind+'_static']
            table[kind][group]=dict(teacher_bce=teacher_loss,static_bce=static_loss,
                static_minus_teacher=static_loss-teacher_loss,
                adapted_minus_same_parent={arm:p.group_mean(losses[kind+'_'+arm]-losses[kind+'_static'],rows,group)
                                          for arm in p.ARMS if arm!='static'},
                windows=int(selected.sum()),risk_worse_windows=int((delta[selected]>1e-6).sum()))
    p.write(out/(role+'_result.json'),dict(status='DESCENT_POINT_DIAGNOSTIC_NOT_STREAM',table=table,
                                          summaries=summaries,static_parity_error=parity))
    return table


def main(out):
    if out.exists():raise FileExistsError(out)
    if shutil.disk_usage(p.ROOT).free<1024**3:raise RuntimeError('less than 1GiB reserve')
    start=time.monotonic()
    p.r.configure()
    torch.cuda.reset_peak_memory_stats()
    base,checkpoint=p.r.load_parent(p.r.DEFAULT_PARENT)
    base.cuda()
    source=p.snapshot(base)
    if p.read(FIT/'completed.json')['status']!='COMPLETE_LOCAL_COPY_PILOT':raise ValueError('completed fit required')
    models={kind:kernel.copy_for_training(base) for kind in p.KINDS}
    for kind,model in models.items():
        model.load_state_dict(torch.load(FIT/(kind+'.pt'),map_location=base.center.device,weights_only=True))
    calibrations=p.read(FIT/'fit.json')['calibrations']
    out.mkdir()
    p.write(out/'protocol.json',dict(status='FROZEN_MATURED_FEEDBACK_BACKTRACK',fit=str(FIT),
        maximum_halvings=8,armijo_coefficient=1e-4,source_training=False,target_training=False,
        code={f.name:f.read_text() for f in (Path(__file__),Path(backtrack.__file__),Path(kernel.__file__),PLAN)}))
    tau=math.log(checkpoint['threshold']/(1-checkpoint['threshold']))
    tables={}
    for role,path,load_role in (('source_val',p.REFERENCE/'val_sealed.json','val'),('east20',p.EAST/'sealed.json','east20')):
        rows=p.read(path)['rows']
        data=p.load_windows(rows,load_role,base.center.device)
        tables[role]=evaluate(base,models,calibrations,data,rows,out,role,tau)
    if not p.unchanged(base,source):raise ValueError('qualified source changed')
    screens={}
    for role,table in tables.items():
        c=table['aligned']
        def average(arm):return .5*sum(c[g]['adapted_minus_same_parent'][arm] for g in ('safe','timely'))
        screens[role]=(all(c[g]['adapted_minus_same_parent']['real'] < -1e-6 for g in ('safe','timely'))
            and c['earlier']['adapted_minus_same_parent']['real']<=1e-6
            and all(average('real')<average(k)-1e-6 for k in ('time','channel','scalar')))
    p.write(out/'completed.json',dict(status='COMPLETE_COPY_BACKTRACK_DIAGNOSTIC',seconds=time.monotonic()-start,
        window_screens=screens,source_unchanged=True,new_checkpoints=0,full_stream_executed=False,
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
        output_bytes=sum(f.stat().st_size for f in out.iterdir() if f.is_file())))
    print('complete',screens,'seconds',time.monotonic()-start,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    main(parser.parse_args().out)
