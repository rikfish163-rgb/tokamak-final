"""Fixed local-statistic x reference-age experiment, no slow parameter training."""
import argparse
import copy
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..fixed_test_protocol import score_fixed_test,load_test_contract
from ..local_transition_20261023.model import LocalTransition,fit_local
from ..local_transition_20261023.run import READ,load_model
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .model import parts,mean_shift,forward

REFERENCE=io23.ROOT/'experiments/risk_aligned_online/runs/local_transition20_test360_20261023'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/局部统计与较早参考_事前约定_20261023.md'
ARMS={'recent_mean':(0,'mean'),'lagged_ridge':(128,'ridge'),'lagged_mean':(128,'mean')}
NAMES=('prior','reference_fit','recent_ridge',*ARMS)


def load_candidate(device='cuda'):
    p=READ(REFERENCE/'protocol.json');done=READ(REFERENCE/'completed.json')
    path=REFERENCE/'static_trained.pt';assert sha(path)==done['checkpoint_sha256']['static_trained']
    parent,threshold=load_model(device=device)
    model=LocalTransition(copy.deepcopy(parent.base)).to(device)
    model.load_state_dict(torch.load(path,map_location=device,weights_only=True)['model'])
    model.eval().requires_grad_(False)
    mapping=READ(Path(p['reference'])/'input_map.json')
    return model,threshold,mapping


def probe(model,mapping):
    selection=READ(REFERENCE/'mechanism_controls/scale_fit.json')['selection']
    stats={'saturation':[],'key_std':[],'centered_energy_fraction':[],
           'largest_centered_sv_fraction':[],'full_vs_mean_change_abs':[]}
    for row in selection:
        s=apply_input_contract(simple_data.read_shot('EAST',row['shot'],rate_contract='nominal'))
        signal=transform({k:s[k] for k in ('values','available')},mapping)
        batch=[simple_data.window(signal,t,128) for t in row['endpoints']]
        x,m=[torch.as_tensor(np.stack(v),device=model.prior.device) for v in zip(*batch)]
        with torch.no_grad():
            r=parts(model,x,m);k=r['key'][:,:14];features=k[:,:,:8]
            centered=features-features.mean(1,keepdim=True)
            energy=centered.square().sum((1,2));sv=torch.linalg.svdvals(centered).square()
            fitted=fit_local(k,r['changes'][:,:14],r['valid'][:,:14],model.prior)
            full=torch.einsum('br,brc->bc',r['key'][:,14],fitted)
            mean=r['key'][:,14]@model.prior+mean_shift(k,r['changes'][:,:14],r['valid'][:,:14],model.prior)
            stats['saturation'].extend((features.abs()>.99).float().mean((1,2)).cpu().tolist())
            stats['key_std'].extend(features.std(1,correction=0).mean(-1).cpu().tolist())
            stats['centered_energy_fraction'].extend((energy/features.square().sum((1,2)).clamp_min(1e-20)).cpu().tolist())
            stats['largest_centered_sv_fraction'].extend((sv[:,0]/sv.sum(-1).clamp_min(1e-20)).cpu().tolist())
            stats['full_vs_mean_change_abs'].extend((full-mean).abs().mean(-1).cpu().tolist())
    def summary(values):
        return dict(n=len(values),mean=float(np.mean(values)),q10=float(np.quantile(values,.1)),
                    median=float(np.median(values)),q90=float(np.quantile(values,.9)),maximum=float(max(values)))
    return dict(scope='640 historical20 endpoints only; no TEST inference/outcomes; existing authorized selection',
                selection=selection,summary={k:summary(v) for k,v in stats.items()},values=stats)


def predict(model,signal,*,device='cuda',replay_recent=False):
    if set(signal)!={'values','available'}:raise ValueError('signals only')
    arms=dict(ARMS)
    if replay_recent:arms['recent_ridge']=(0,'ridge')
    pieces={k:[] for k in arms};stats={k:dict(points=0,active=0,fitted=0,error_sum=0.,targets=0,correction_sum=0.) for k in arms}
    for lo in range(0,len(signal['values']),32):
        ends=list(range(lo,min(lo+32,len(signal['values']))))
        batch=[simple_data.window(signal,t,128) for t in ends]
        x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*batch)]
        if lo<256:
            prior=model(x,m,mode='prior')['logit'].sigmoid().detach().cpu().numpy()
            for arm in arms:pieces[arm].append(prior.copy());stats[arm]['points']+=len(ends)
            continue
        old=[simple_data.window(signal,t-128,128) for t in ends]
        xx,mm=[torch.as_tensor(np.stack(v),device=device) for v in zip(*old)]
        for arm,(lag,kind) in arms.items():
            r=forward(model,x,m,xx if lag else x,mm if lag else m,kind=kind)
            pieces[arm].append(r['logit'].sigmoid().detach().cpu().numpy())
            v=stats[arm];v['points']+=len(ends);v['active']+=len(ends);v['fitted']+=int(r['fitted_windows'])
            v['error_sum']+=float(r['forecast_error_sum'].detach());v['targets']+=int(r['target_count'])
            v['correction_sum']+=float(r['correction'].detach().norm(dim=-1).sum())
    return {k:np.concatenate(v) for k,v in pieces.items()},stats


def main(output):
    if output.exists():raise FileExistsError(output)
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);started=time.monotonic()
    p,verified=READ(REFERENCE/'protocol.json'),READ(REFERENCE/'verification.json')
    assert verified['status']=='VERIFIED' and verified['completed_sha256']==sha(REFERENCE/'completed.json')
    for f,h in p['code_sha256'].items():assert sha(io23.ROOT/f)==h,f
    contract=load_test_contract('EAST');assert contract==p['contract']
    model,threshold,mapping=load_candidate();initial=state_cpu(model)
    output.mkdir(parents=True);(output/'shots').mkdir()
    paths={io23.ROOT/f for f in p['code_sha256']};paths.update(Path(__file__).parent.glob('*.py'))
    paths.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_context_separation.py')
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(paths):z.write(path,str(path.relative_to(io23.ROOT)))
    protocol=dict(reference=str(REFERENCE),checkpoint_sha256=sha(REFERENCE/'static_trained.pt'),
        reference_hashes={f:sha(REFERENCE/f) for f in ('protocol.json','verification.json','completed.json',
            'prediction_receipts.json','mechanism_controls/scale_fit.json')},
        plan_path=str(PLAN),plan_sha256=sha(PLAN),contract=contract,threshold=threshold,
        selected_ids=p['selected_ids'],code_sha256={str(path.relative_to(io23.ROOT)):sha(path) for path in paths},
        arms=ARMS,names=NAMES,first_affected_index=256,no_training=True,no_inter=True,no_alarm_change=True,
        prior_exposure='chosen to investigate previous positive development arm; not independent',
        independent_confirmation=False,fit_limit='context transitions0..13, last context observation is not used in fitting')
    _durable_json(output/'protocol.json',protocol)
    def denied(*args):raise AssertionError('no HDF outcomes before ALL curves sealed')
    with patch.object(io23,'read_outcome',side_effect=denied):
        detail=probe(model,mapping);_durable_json(output/'preparation_probe.json',detail)
        print(dict(stage='probe',summary=detail['summary']),flush=True)
        reference_receipts={r['shot']:r for r in READ(REFERENCE/'prediction_receipts.json')};seals=[]
        for index,sid in enumerate(contract['shot_ids']):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            predictions,stats=predict(model,transform({k:s[k] for k in ('values','available')},mapping))
            old=REFERENCE/'shots'/f'{sid}.npz';assert sha(old)==reference_receipts[sid]['prediction_sha256']
            with np.load(old,allow_pickle=False) as z:
                np.testing.assert_array_equal(s['times'],z['times'])
                predictions['prior']=z['static_prior'].copy();predictions['reference_fit']=z['static_fit'].copy()
                recent=z['static_fit'].copy();recent[:256]=z['static_prior'][:256];predictions['recent_ridge']=recent
            for arm in (*ARMS,'recent_ridge'):np.testing.assert_array_equal(predictions[arm][:256],predictions['prior'][:256])
            path=output/'shots'/f'{sid}.npz';sp=output/'shots'/f'{sid}_stats.json'
            np.savez_compressed(path,times=s['times'],**predictions);_durable_json(sp,stats)
            seals.append(dict(shot=sid,prediction_sha256=sha(path),stats_sha256=sha(sp),reference_sha256=sha(old)))
            if (index+1)%60==0:
                progress=dict(stage='stream',shots=index+1,seconds=time.monotonic()-started)
                _durable_json(output/'progress.json',progress);print(progress,flush=True)
        _durable_json(output/'prediction_receipts.json',seals)
    results={k:[] for k in NAMES}
    for sid in contract['shot_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(output/'shots'/f'{sid}.npz',allow_pickle=False) as z:
            for arm in NAMES:results[arm].append(dict(shot=sid,times=s['times'][s['valid']],scores=z[arm][s['valid']],is_disrupt=s['is_disrupt'],down=s['down']))
    scored={k:score_fixed_test('EAST',v,threshold) for k,v in results.items()}
    assert_unchanged(model,initial)
    _durable_json(output/'per_shot.json',scored)
    done=dict(status='COMPLETED',seconds=time.monotonic()-started,all_parameters_unchanged=True,
        protocol_sha256=sha(output/'protocol.json'),probe_sha256=sha(output/'preparation_probe.json'),
        metrics={k:dict(**public(v),very_early_over3000=sum(r['timely'] and r['lead_ms']>3000 for r in v['predictions'])) for k,v in scored.items()})
    _durable_json(output/'completed.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);main(p.parse_args().output)
