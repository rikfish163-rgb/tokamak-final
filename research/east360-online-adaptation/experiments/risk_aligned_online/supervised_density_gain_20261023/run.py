"""Fixed-budget label-fitted input scale and a paired temporary-neural control."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.learning import SIGNAL_KEYS, eligible_records, natural_batch
from ..deepcoral_prefix_20261022.run import assert_unchanged, predict_pairs, state_cpu
from ..fixed_test_protocol import load_test_contract, score_fixed_test
from ..labeled_alignment_20261023.learning import balanced_batch
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent, load_source_train
from ..simple23 import fit_forecaster
from ..simple_online import _durable_json
from ..simple_train import metrics, pools, sha
from ..temporal_dependence_20261023.run import REFERENCE, load_model
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import predict_many, public
from .learning import apply_gain, export_map

PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/标签约束单通道对齐_实验约定_20261023.md'
ARMS=('original_static','original_intra','learned_static','learned_intra')
STEPS=256


def prepare(model, source, target, old_map, output):
    # Match the original arm's forecast-head initialization, not a second fit
    # of its already-warmed head. All risk-network weights remain untouched.
    source_base,_=load_parent(DEFAULT_PARENT)
    model.base.forecast_head.load_state_dict(source_base.forecast_head.state_dict())
    del source_base
    model.set_stage('frozen');initial=state_cpu(model)
    log_gain=torch.tensor(np.log(old_map['gain'][12]),device='cuda',dtype=torch.float64,requires_grad=True)
    opt=torch.optim.Adam([log_gain],lr=.03)
    groups=pools(target);rng=np.random.default_rng(2131);training=[]
    reference_manifest=[json.loads(line) for line in (REFERENCE/'risk_training_manifest.jsonl').read_text().splitlines()]
    with (output/'gain_manifest.jsonl').open('x') as manifest:
        for step in range(STEPS):
            x,m,y,receipt=balanced_batch(target,groups,rng,model.cfg)
            assert receipt==reference_manifest[step]['target']
            manifest.write(json.dumps(dict(step=step,target=receipt))+'\n')
            opt.zero_grad(set_to_none=True)
            h=model(apply_gain(x,m,log_gain),m,mode='risk_state')['risk_state']
            loss=F.binary_cross_entropy_with_logits(model.base.risk_head(h).squeeze(-1),y)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite supervised scale loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([log_gain],1.,error_if_nonfinite=True)
            opt.step()
            training.append(dict(step=step,bce=float(loss.detach()),gradient_norm=float(norm),gain=float(log_gain.exp().detach())))
            if (step+1)%64==0:print(json.dumps(dict(stage='input_gain',**training[-1])),flush=True)
    assert_unchanged(model,initial)
    assert all(p.grad is None for p in model.parameters())
    mapping=export_map(old_map,float(log_gain.exp().detach()))
    _durable_json(output/'input_map.json',mapping);_durable_json(output/'gain_training.json',training)
    mapped=[dict(s,**transform({k:s[k] for k in ('values','available')},mapping)) for s in target]
    sf,se=eligible_records([{k:s[k] for k in SIGNAL_KEYS} for s in source],model.cfg,future=True)
    tf,te=eligible_records([{k:s[k] for k in SIGNAL_KEYS} for s in mapped],model.cfg,future=True)
    _durable_json(output/'forecast_exclusions.json',dict(source=se,target=te))
    optimizer=torch.optim.Adam(model.base.forecast_head.parameters(),lr=3e-4)
    rs,rt=np.random.default_rng(4131),np.random.default_rng(5131);history=[]
    reference_manifest=[json.loads(line) for line in (REFERENCE/'forecast_manifest.jsonl').read_text().splitlines()]
    with (output/'forecast_manifest.jsonl').open('x') as manifest:
        for step in range(STEPS):
            a=natural_batch(sf,rs,32,model.cfg,'cuda',future=True)
            b=natural_batch(tf,rt,32,model.cfg,'cuda',future=True)
            receipt=dict(step=step,source=a[-1],target=b[-1])
            assert receipt==reference_manifest[step]
            manifest.write(json.dumps(receipt)+'\n')
            x,m,t,tm=[torch.cat((v,w)) for v,w in zip(a[:-1],b[:-1])]
            history.append(dict(step=step,loss=fit_forecaster(model.base,optimizer,x,m,t,tm)))
    assert_unchanged(model,initial,('base.forecast_head.weight','base.forecast_head.bias'))
    model.set_stage('frozen')
    _durable_json(output/'forecast_training.json',history)
    torch.save(dict(model=state_cpu(model),config=model.cfg.to_dict()),output/'learned.pt')
    _durable_json(output/'preparation_receipts.json',dict(checkpoint_sha256=sha(output/'learned.pt'),
        changed_network_parameters=[k for k,v in model.state_dict().items() if not torch.equal(v.cpu(),initial[k])],
        input_parameters_fitted=1,risk_network_unchanged=True,forecast_sampling_identical=True,
        forecast_initialization='qualified original source head, not previous warm head'))
    return mapping


def main(output):
    if output.exists():raise FileExistsError(output)
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic()
    read=lambda p:json.loads(p.read_text())
    rv=read(REFERENCE/'verification.json');rp=read(REFERENCE/'protocol.json')
    assert rv['status']=='VERIFIED' and sha(REFERENCE/'completed.json')==rv['completed_sha256']
    contract=load_test_contract('EAST');assert contract==rp['contract']
    old_map=read(REFERENCE/'input_map.json');ids=old_map['calibration_ids']
    assert ids==rp['selected_ids'] and len(ids)==20 and not set(ids)&set(contract['shot_ids'])
    assert all(g==1 for c,g in enumerate(old_map['gain']) if c!=12) and not any(old_map['offset'])
    model,threshold=load_model();reference_state=state_cpu(model)
    output.mkdir(parents=True);(output/'shots').mkdir()
    files=list(Path(__file__).parent.glob('*.py'))+[io23.ROOT/'experiments/risk_aligned_online'/n for n in (
        'simple23.py','simple_data.py','io23.py','causal_input.py','simple_online.py','simple_train.py',
        'fixed_test_protocol.py','paired_feedback_20261022/model.py','masked_state_20261022/model.py',
        'redesign_20261004/prepare_prior.py','warmstart_alignment_20261022/alignment.py',
        'warmstart_alignment_20261022/run.py','deepcoral_prefix_20261022/run.py',
        'deepcoral_prefix_20261022/learning.py','labeled_alignment_20261023/learning.py',
        'temporal_dependence_20261023/run.py','tests/test_supervised_density_gain.py')]
    hashes={str(f.relative_to(io23.ROOT)):sha(f) for f in files}
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,str(f.relative_to(io23.ROOT)))
    protocol=dict(source_sha256=PARENT_SHA256,threshold=threshold,contract=contract,arms=ARMS,
        selected_ids=ids,reference=str(REFERENCE),reference_hashes={n:sha(REFERENCE/n) for n in (
            'protocol.json','completed.json','verification.json','input_only.pt','input_map.json',
            'prediction_receipts.json','risk_training_manifest.jsonl','forecast_manifest.jsonl')},
        plan_path=str(PLAN),plan_sha256=sha(PLAN),code_sha256=hashes,steps=STEPS,
        trained_input='positive scalar multiplying observed ne_nG; no offsets or other columns',
        risk_network_frozen=True,alarm_frozen=True,forecast_only_refit_steps=STEPS,
        preparation='20 historical TRAIN shots,4D16N; not chronological first20, not blind query',
        calibration_metadata_previously_examined=88,target_labels='fixed125ms',
        intra='temporary first encoder,SGD.001 cap1,H25,4arrived targets,every128,first256; reset eachshot',
        independent_confirmation=False,no_test_driven_checkpoint_selection=True,
        label_gate='TEST HDF outcome and DownTime inaccessible before all prediction seals; class split manifest is known')
    _durable_json(output/'protocol.json',protocol)
    read_outcome=io23.read_outcome;source_ids=set(io23.source_roles()['train']);accesses=[]
    def guarded(device,sid):
        allowed=source_ids if device=='J-TEXT' else set(ids) if device=='EAST' else set()
        if sid not in allowed:raise AssertionError('nontraining outcome access before all prediction seals')
        accesses.append(dict(device=device,shot=int(sid)))
        return read_outcome(device,sid)
    with patch.object(io23,'read_outcome',side_effect=guarded):
        source,_=load_source_train()
        target=[simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{}) for sid in ids]
        assert sum(s['is_disrupt'] for s in target)==4
        mapping=prepare(model,source,target,old_map,output);del source
        # All20 diagnostic is in-sample, never a validation/selection criterion.
        diagnostic={k:[] for k in ('original','learned')}
        for s in target:
            signal={k:s[k] for k in ('values','available')}
            for name,probs in predict_many(model,dict(original=transform(signal,old_map),learned=transform(signal,mapping)),device='cuda').items():
                diagnostic[name].append(dict(shot=s['shot'],times=s['times'][s['valid']],scores=probs[s['valid']],is_disrupt=s['is_disrupt'],down=s['down']))
        _durable_json(output/'preparation_in_sample.json',{k:public(metrics(v,threshold,lead=30.)) for k,v in diagnostic.items()})
        assert_unchanged(model,reference_state,('base.forecast_head.weight','base.forecast_head.bias'))
        frozen=state_cpu(model);reference_seals={r['shot']:r for r in read(REFERENCE/'prediction_receipts.json')};seals=[]
        for index,sid in enumerate(contract['shot_ids']):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            curves,updates=predict_pairs(dict(learned=model),transform({k:s[k] for k in ('values','available')},mapping))
            old_path=REFERENCE/'shots'/f'{sid}.npz';old_updates=REFERENCE/'shots'/f'{sid}_updates.json'
            assert sha(old_path)==reference_seals[sid]['prediction_sha256']
            assert sha(old_updates)==reference_seals[sid]['updates_sha256']
            with np.load(old_path,allow_pickle=False) as z:
                np.testing.assert_array_equal(s['times'],z['times'])
                curves.update(original_static=z['input_only_static'].copy(),original_intra=z['input_only_intra'].copy())
            for key in ('learned','original'):
                np.testing.assert_array_equal(curves[key+'_static'][:256],curves[key+'_intra'][:256])
            path=output/'shots'/f'{sid}.npz';up=output/'shots'/f'{sid}_updates.json'
            np.savez_compressed(path,times=s['times'],**curves);_durable_json(up,updates)
            seals.append(dict(shot=sid,prediction_sha256=sha(path),updates_sha256=sha(up),reference_prediction_sha256=sha(old_path)))
            if (index+1)%30==0:
                progress=dict(stage='test_stream',shots=index+1,seconds=time.monotonic()-start)
                _durable_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
        _durable_json(output/'prediction_receipts.json',seals)
        _durable_json(output/'pre_eval_label_access.json',accesses)
    rows={a:[] for a in ARMS}
    for sid in contract['shot_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(output/'shots'/f'{sid}.npz',allow_pickle=False) as z:
            for a in ARMS:
                rows[a].append(dict(shot=sid,times=s['times'][s['valid']],scores=z[a][s['valid']],is_disrupt=s['is_disrupt'],down=s['down']))
    results={a:score_fixed_test('EAST',r,threshold) for a,r in rows.items()}
    assert_unchanged(model,frozen);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(output/'per_shot.json',results)
    done=dict(status='COMPLETED',seconds=time.monotonic()-start,source_unchanged=True,
        gain_before=old_map['gain'][12],gain_after=mapping['gain'][12],
        metrics={a:dict(**public(r),very_early_over3000=sum(p['timely'] and p['lead_ms']>3000 for p in r['predictions'])) for a,r in results.items()})
    _durable_json(output/'completed.json',done);print(json.dumps(done),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();main(a.output)
