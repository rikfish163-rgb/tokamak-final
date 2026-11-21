"""Posthoc causal-window feature sensitivity, not a physics or policy claim.

Two numerical interventions at a sealed operational maximum: erase one
observed continuous feature to its source center, or remove only its window
location in source-asinh coordinates. Other features and availability stay
unchanged. A threshold crossing here is NOT a whole-shot TP/FP flip.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import io23, simple_data
from .alignment import fit_ended_shot_statistics
from .causal_input import apply_input_contract
from .density_mapping_audit import RAW, MAPPED, read_trace, counts
from .ended_density_replay import EndedDensityReplay
from .operational_memory import operational_state
from .persistent_ffn import PersistentFfnModel
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from .simple_online import _durable_json, _hash
from .verify_ended_decoder import classify

METHODS=('source_center','centered_shape')


def intervene_window(values,available,center,scale,channel,method):
    x,a,c,s=map(np.asarray,(values,available,center,scale))
    if (x.shape!=(128,23) or a.shape!=x.shape or a.dtype!=bool
            or not np.issubdtype(x.dtype,np.floating) or c.shape!=(23,) or s.shape!=(23,)
            or not np.isfinite(c).all() or not np.isfinite(s).all() or np.any(s<=0)
            or type(channel) is not int or not 0<=channel<23 or method not in METHODS
            or not np.isfinite(x[a]).all()):
        raise ValueError('finite observed W128 flat23, source coordinates and one continuous intervention required')
    out=x.copy();observed=a[:,channel]
    receipt=dict(channel=io23.RAW23[channel],method=method,observed_points=int(observed.sum()))
    if io23.RAW23[channel]=='mode_number_n':
        # Preserve the archived predeclared exclusion, not a categorical
        # claim: Shen's average toroidal mode and local HDF5 can be fractional.
        # Legacy receipt status is retained for numeric replay compatibility;
        # report73 corrects its interpretation. No intervention was measured.
        return out,dict(receipt,status='not_continuous_feature')
    if not observed.any():
        # Missingness is kept; no fabricated source-center measurement enters
        # the model. This is an explicit zero-information identity condition.
        return out,dict(receipt,status='no_observed_values')
    if method=='source_center':
        out[observed,channel]=c[channel]
    else:
        z=np.arcsinh((x[observed,channel].astype(np.float64)-float(c[channel]))/float(s[channel]))
        location=float(np.median(z))
        with np.errstate(over='raise',invalid='raise'):
            out[observed,channel]=float(c[channel])+float(s[channel])*np.sinh(z-location)
        if not np.isfinite(out[observed,channel]).all():
            raise FloatingPointError('nonfinite diagnostic intervention; not silently omitting a feature')
        receipt['normalized_location_removed']=location
    return out,dict(receipt,status='applied')


def summarize(records,threshold):
    collected={}
    for row in records:
        for intervention in row['interventions']:
            key=(row['coordinates'],row['category'],intervention['channel'],intervention['method'])
            collected.setdefault(key,[]).append((row['baseline_probability'],intervention))
    summary={}
    for (coordinates,category,channel,method),rows in collected.items():
        delta=[r['probability']-old for old,r in rows]
        result=dict(n_windows=len(rows),n_active=sum(r['status']=='applied' for _,r in rows),
            mean_probability_change=float(np.mean(delta)),median_probability_change=float(np.median(delta)),
            original_peak_crossing_down=sum(old>=threshold and r['probability']<threshold for old,r in rows),
            original_peak_crossing_up=sum(old<threshold and r['probability']>=threshold for old,r in rows))
        summary.setdefault(coordinates,{}).setdefault(category,{}).setdefault(channel,{})[method]=result
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('new diagnostic output required')
    order=json.loads((MAPPED.parent/'order.json').read_text())
    if json.loads((RAW.parent/'order.json').read_text())[:len(order)]!=order:
        raise AssertionError('reference orders differ')
    torch.set_num_threads(4);torch.manual_seed(131)
    base,state=load_parent(DEFAULT_PARENT)
    model=PersistentFfnModel(base).to('cuda').eval().requires_grad_(False)
    frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    threshold=float(state['threshold'])
    center=np.asarray(state['source_center']);scale=np.asarray(state['source_scale'])
    prefix=[apply_input_contract(simple_data.read_shot('EAST',r['shot'],rate_contract='nominal'))
            for r in order[:30]]
    statistics=fit_ended_shot_statistics(prefix,ended_ids=[r['shot'] for r in prefix])
    provider=EndedDensityReplay(center,scale,output_dir=MAPPED,order=[r['shot'] for r in order],
                                preserve_observed_channels=True)
    # Reconstruction of completed runs ONLY. No learner or live reveal gate
    # consumes these retrospective labels or this direct statistics assignment.
    provider.statistics=statistics
    records=[];reference=dict(raw=[],mapped=[]);max_error=0.;calls=0
    for index,item in enumerate(order,1):
        sid=item['shot']
        raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        ended=simple_data.attach_outcome(raw,'EAST',{})
        for coordinates,folder in (('raw',RAW),('mapped',MAPPED)):
            times,scores=read_trace(folder,index,sid,raw,threshold)
            decision=classify(times,scores,ended['valid'],threshold=threshold,
                              disruptive=ended['is_disrupt'],down=ended['down'])
            reference[coordinates].append(dict(shot=sid,**decision))
            peak=operational_state(dict(ended,_sealed_scores=scores),scores,threshold)
            if peak['query'] is None:
                raise ValueError('reference has no eligible operational maximum; do not delete this shot')
            q=peak['query']
            signals=ended if coordinates=='raw' or index<=30 else provider.reencode(ended)
            x,a=simple_data.window(signals,q,128)
            windows=[x];receipts=[]
            for channel in range(23):
                for method in METHODS:
                    changed,receipt=intervene_window(x,a,center,scale,channel,method)
                    other=np.arange(23)!=channel
                    if (not np.array_equal(changed[:,other],x[:,other],equal_nan=True)
                            or not np.array_equal(changed[~a[:,channel],channel],x[~a[:,channel],channel],equal_nan=True)):
                        raise AssertionError('intervention changed another feature or missing slot')
                    windows.append(changed);receipts.append(receipt)
            values=np.stack(windows)
            with torch.no_grad():
                probability=model(torch.as_tensor(values,device='cuda'),torch.as_tensor(
                    np.broadcast_to(a,values.shape).copy(),device='cuda'))['logit'].sigmoid().cpu().numpy()
            error=abs(float(probability[0])-float(scores[q]));max_error=max(max_error,error)
            if error>2e-6:raise AssertionError(f'baseline peak replay differs for {sid}: {error}')
            calls+=len(windows)
            if ended['is_disrupt']:category='disruptive_timely' if decision['timely'] else 'disruptive_miss'
            else:category='safe_fp' if decision['alarm'] else 'safe_tn'
            records.append(dict(shot=sid,index=index,coordinates=coordinates,category=category,
                query=q,time=float(times[q]),lead_ms=(round(float((ended['down']-times[q])*1000.),9)
                    if ended['is_disrupt'] else None),baseline_probability=float(probability[0]),
                interventions=[dict(r,probability=float(p)) for r,p in zip(receipts,probability[1:])]))
        if index%20==0:print(json.dumps(dict(checked_shots=index,peak_windows=len(records))),flush=True)
    if _hash(DEFAULT_PARENT)!=PARENT_SHA256 or model.memory.any():
        raise AssertionError('source or persistent state changed')
    for k,v in model.state_dict().items():torch.testing.assert_close(v.detach().cpu(),frozen[k],rtol=0,atol=0)
    result=dict(scope='POSTHOC DEVELOPMENT MODEL SENSITIVITY, NOT PHYSICAL CAUSALITY',
        parent_sha256=PARENT_SHA256,code_sha256=_hash(Path(__file__)),threshold=threshold,
        order_sha256=_hash(MAPPED.parent/'order.json'),calibration_ids=[r['shot'] for r in order[:30]],
        selected_shots=len(order),baseline_seals_verified=2*len(order),actual_window_forwards=calls,
        baseline_peak_max_error=max_error,source_and_model_tensors_unchanged=True,
        centered_shape_is_translation_only_in_source_asinh_coordinates=True,
        source_center_erases_selected_feature_waveform=True,no_training_no_threshold_selection=True,
        no_physical_causality_claim=True,whole_shot_intervention_metrics_not_computed=True,
        discrete_mode_continuous_interventions=False,methods=list(METHODS),
        reference_metrics={name:counts(rows) for name,rows in reference.items()},
        summary=summarize(records,threshold),records=records)
    _durable_json(args.output,result)
    print(json.dumps({k:result[k] for k in ('baseline_seals_verified','actual_window_forwards',
        'baseline_peak_max_error','reference_metrics')}),flush=True)


if __name__=='__main__':main()
