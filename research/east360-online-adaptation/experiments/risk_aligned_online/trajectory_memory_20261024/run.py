"""Fixed-parent five-arm full360 experiment; reveal only after all seals."""
import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..continual_episode_20261023.run import labeled
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..diverse_support_20261023.run import make_parent, READ
from ..heldout_threshold_20261023.run import models_for_calibration, REFERENCE
from ..history_tangent_20261023.learning import flat, logit
from ..paired_feedback_20261022.model import initial_encoder
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics, sha
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .learning import ARMS, make_bank, predict

PREVIOUS=io23.ROOT/'experiments/risk_aligned_online/runs/contextual_feedback360_20261023'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/原信号轨迹检索微调_事前约定_20261024.md'


def signal(sid):
    return apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))


def gradient_check(model,bank):
    indices=np.array([0,32,64,96,128,160,192,224,256,288])
    weights=torch.arange(1,len(indices)+1,device='cuda',dtype=torch.float32)
    weights/=weights.sum()
    fast=initial_encoder(model)
    z=logit(model,torch.as_tensor(bank['x'][indices],device='cuda'),
            torch.as_tensor(bank['mask'][indices],device='cuda'),fast)
    y=torch.as_tensor(bank['labels'][indices],device='cuda',dtype=z.dtype)
    loss=(weights*torch.nn.functional.binary_cross_entropy_with_logits(z,y,reduction='none')).sum()
    direct=flat(torch.autograd.grad(loss,tuple(fast.values()))).detach()
    cached=weights@bank['gradients'][indices]
    error=float((direct-cached).abs().max())
    torch.testing.assert_close(direct,cached,atol=2e-6,rtol=3e-4)
    return dict(indices=indices.tolist(),max_absolute_error=error,gradient_norm=float(direct.norm()))


def evaluate(root,protocol):
    rows={a:[] for a in ('off',)+ARMS};outcomes=[]
    for sid in protocol['shot_ids']:
        flag,down=io23.read_outcome('EAST',sid)
        outcomes.append(dict(shot=sid,is_disrupt=bool(flag),down=None if down is None else float(down)))
        with np.load(root/'shots'/f'{sid}.npz') as z:
            _,valid=io23.labels_at(z['times'],flag,down,125.)
            for a in rows:rows[a].append(dict(shot=sid,times=z['times'][valid],scores=z[a][valid],is_disrupt=flag,down=down))
    results={a:metrics(v,protocol['threshold'],lead=30.) for a,v in rows.items()}
    summary={a:dict(public(v),very_early_over3000=sum(r['timely'] and r['lead_ms']>3000 for r in v['predictions']))
             for a,v in results.items()}
    changes={}
    for arm in ARMS:
        changed=[]
        for before,after in zip(results['off']['predictions'],results[arm]['predictions']):
            good=lambda r:bool(r['timely']) if r['is_disrupt'] else not r['alarm']
            if good(before)!=good(after):changed.append(dict(shot=before['shot'],beneficial=good(after),before=before,after=after))
        changes[arm]=changed
    _durable_json(root/'outcomes.json',outcomes);_durable_json(root/'per_shot.json',results)
    _durable_json(root/'flips.json',changes)
    _durable_json(root/'blocks.json',[dict(end=end,metrics={a:public(metrics(v[:end],protocol['threshold'],lead=30.)) for a,v in rows.items()})
                                    for end in range(20,len(outcomes)+1,20)])
    return summary


def eligible(results):
    if any(r['n_disrupt']!=180 or r['n_safe']!=180 for r in results.values()):return False
    off=results['off'];real=results['trajectory']
    return (real['fpr']<=.20 and real['tpr']>=off['tpr'] and real['J']>off['J']
            and all(real['J']>results[a]['J'] for a in ('level','reverse','uniform')))


def main(root,limit=None):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    # The reference static curves use eval/no_grad with MHA fastpath enabled.
    # Shot60999 differs by3.94e-5 under the unfused path; use the SAME inference
    # backend, not a relaxed tolerance. Historical backward remains grad-enabled.
    torch.set_num_threads(2);torch.manual_seed(131);torch.backends.mha.set_fastpath_enabled(True)
    prior=READ(PREVIOUS/'protocol.json');order=prior['shot_ids']
    assert len(order)==360 and set(order)==set(io23.cohort_ids('EAST'))
    if limit is not None:order=order[:limit]
    assert not set(order)&set(prior['preparation_ids']+prior['calibration_ids'])
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed']
    original=state_cpu(model);mapping=READ(REFERENCE/'input_map.json');start=time.monotonic()
    root.mkdir(parents=True);(root/'shots').mkdir();(root/'receipts').mkdir()
    reader=io23.read_outcome
    def prep_guard(device,sid):
        assert device=='EAST' and sid in prior['preparation_ids']
        return reader(device,sid)
    with patch.object(io23,'read_outcome',side_effect=prep_guard):
        historical=[labeled(signal(sid),io23.read_outcome('EAST',sid),mapping) for sid in prior['preparation_ids']]
    files=set(Path(__file__).parent.glob('*.py'))
    files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_trajectory_memory.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    protocol=dict(shot_ids=order,preparation_ids=prior['preparation_ids'],calibration_ids=prior['calibration_ids'],
        threshold=prior['threshold'],source_threshold=prior['source_threshold'],source_sha256=PARENT_SHA256,
        parent_sha256=sha(REFERENCE/'fold0/mapped_fixed/prepared.pt'),mapping_sha256=sha(REFERENCE/'input_map.json'),
        previous_completed_sha256=sha(PREVIOUS/'completed.json'),plan_sha256=sha(PLAN),
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
        input_sha256={str(io23.path_for('EAST',s)):sha(io23.path_for('EAST',s)) for s in order+prior['preparation_ids']},
        arms=('off',)+ARMS,update_every=128,step_learning_rate=.01,step_reset='parent at every update',
        encoder_names=model.encoder_names(),inter_steps=0,alarm_updates=0,event_labels_used=False,
        label_access='preparation labels only until all five arms for every test shot are sealed',
        scope='DEVELOPMENT_ONLY; not chronological commissioning; not pure self-supervision',
        inference_backend='eval/no_grad,MHA_fastpath=True; reference-identical batch32',
        independent_confirmation=False,torch=torch.__version__,gpu=torch.cuda.get_device_name())
    _durable_json(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
    def deny(*args,**kwargs):raise AssertionError('no test outcome before all predictions sealed')
    with patch.object(io23,'read_outcome',side_effect=deny):
        bank=make_bank(model,historical)
        check=gradient_check(model,bank)
        np.savez_compressed(root/'bank.npz',**{k:v for k,v in bank.items() if isinstance(v,np.ndarray)})
        torch.save(dict(gradients=bank['gradients'].cpu()),root/'gradients.pt')
        _durable_json(root/'bank.json',dict(rows=bank['rows'],temperature=bank['temperature'],gradient_check=check,
             historical_backward_passes=len(bank['rows']),parameter_count=bank['gradients'].shape[1]))
        print(dict(stage='bank_ready',windows=len(bank['rows']),temperature=bank['temperature'],check=check),flush=True)
        total={a:0 for a in ARMS};max_error=0.
        for i,sid in enumerate(order):
            if time.monotonic()-start>1800:raise TimeoutError('declared30min exceeded')
            raw=signal(sid);mapped=transform({k:raw[k] for k in ('values','available')},mapping)
            curves,receipts=predict(model,mapped,bank)
            # Independently recomputed off arm must match the stronger parent,
            # not the failed aligned-state experiment's scores.
            with np.load(PREVIOUS/'shots'/f'{sid}.npz') as z:
                np.testing.assert_array_equal(raw['times'],z['times'])
                err=float(np.max(np.abs(curves['off']-z['static'])))
                np.testing.assert_array_equal(curves['off'],z['static'],err_msg=f'parent prediction mismatch {sid}')
                max_error=max(max_error,err)
            np.savez_compressed(root/'shots'/f'{sid}.npz',times=raw['times'],**curves)
            _durable_json(root/'receipts'/f'{sid}.json',receipts)
            for a in ARMS:total[a]+=sum(r[a]['step_norm']>0 for r in receipts)
            if (i+1)%10==0:print(dict(stage='predict',shots=i+1,seconds=round(time.monotonic()-start,1)),flush=True)
    assert_unchanged(model,original)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(root/'prediction_seals.json',{str(f.relative_to(root)):sha(f) for f in (root/'shots').glob('*.npz')})
    results=evaluate(root,protocol)
    # This is an eligibility decision for further development, not scientific
    # confirmation. Never auto-promote from a single development stream.
    passes=eligible(results)
    artifacts={str(f.relative_to(root)):sha(f) for f in root.iterdir() if f.is_file()}
    done=dict(status='completed',metrics=results,temporary_updates=total,static_max_error=max_error,
        candidate_promoted=False,eligible_for_inter_study=passes,seconds=time.monotonic()-start,
        artifact_sha256=artifacts,scientific_goal_achieved=False)
    _durable_json(root/'completed.json',done)
    print(done,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    try:main(args.out,args.limit)
    except Exception as exc:
        if args.out.exists():_durable_json(args.out/'failed.json',dict(type=type(exc).__name__,error=str(exc)))
        raise
