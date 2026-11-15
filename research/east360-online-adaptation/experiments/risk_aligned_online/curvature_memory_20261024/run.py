"""Eight-arm development experiment; one parent, fixed threshold, all360 seals."""
import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..diverse_support_20261023.run import make_parent, READ
from ..heldout_threshold_20261023.run import models_for_calibration, REFERENCE
from ..history_tangent_20261023.learning import flat, logit
from ..paired_feedback_20261022.model import initial_encoder
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics, sha
from ..trajectory_memory_20261024.run import signal
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .learning import ARMS, bernoulli_rows, solve_directions, predict

PREVIOUS=io23.ROOT/'experiments/risk_aligned_online/runs/trajectory_memory360_v2_20261024'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/历史预测敏感度解耦_事前约定_20261024.md'


def load_bank(root=PREVIOUS,device='cuda'):
    with np.load(root/'bank.npz') as z:bank={k:z[k].copy() for k in z.files}
    info=READ(root/'bank.json')
    bank.update(rows=info['rows'],temperature=info['temperature'])
    bank['gradients']=torch.load(root/'gradients.pt',map_location=device,weights_only=True)['gradients']
    return bank


def make_geometry(model,bank):
    fast=initial_encoder(model);jac=[];prob=[];max_error=0.
    for i,(x,m) in enumerate(zip(bank['x'],bank['mask'])):
        z=logit(model,torch.as_tensor(x[None],device='cuda'),torch.as_tensor(m[None],device='cuda'),fast)
        p=z.sigmoid().detach()[0]
        a=flat(torch.autograd.grad(z.sum(),tuple(fast.values()))).detach()
        recomputed=(p-float(bank['labels'][i]))*a
        torch.testing.assert_close(recomputed,bank['gradients'][i],atol=3e-6,rtol=5e-4)
        max_error=max(max_error,float((recomputed-bank['gradients'][i]).abs().max()))
        jac.append(a.cpu());prob.append(float(p))
    j=torch.stack(jac).double();p=torch.tensor(prob,dtype=torch.float64)
    q=torch.tensor([.5/np.sum(bank['labels']==y) for y in bank['labels']],dtype=torch.float64)
    u=bernoulli_rows(j,p,q)
    g=bank['gradients'].detach().cpu().double()
    full,diagonal,info=solve_directions(g,u)
    info.update(jacobian_backward_passes=len(jac),gradient_reconstruction_max_error=max_error,
        parameter_count=j.shape[1],history_count=len(jac),class_mass=[float(q[bank['labels']==y].sum()) for y in (0,1)],
        curvature_definition='historical-input Bernoulli model Fisher/GGN; not empirical label-gradient Fisher',
        solve_dtype='CPUfloat64',runtime_dtype='CUDAfloat32')
    # Check the actual cast directions against the float64 system, not just
    # the ideal mathematical solve. The bound is numerical, not an efficacy gate.
    cast=full.float().double();r=cast@u.T@u+info['damping']*cast-g
    info['float32_direction_relative_residual']=float(r.norm()/g.norm())
    if info['float32_direction_relative_residual']>1e-4:raise ArithmeticError('runtime direction cast inaccurate')
    tensors=dict(u=u.float(),full=full.float(),diagonal=diagonal.float(),jacobian=j.float(),p=p,q=q)
    for k in ('u','full','diagonal'):bank[k]=tensors[k].cuda()
    return tensors,info


def evaluate(root,protocol):
    rows={a:[] for a in ('off',)+ARMS};outcomes=[]
    for sid in protocol['shot_ids']:
        flag,down=io23.read_outcome('EAST',sid)
        outcomes.append(dict(shot=sid,is_disrupt=bool(flag),down=None if down is None else float(down)))
        with np.load(root/'shots'/f'{sid}.npz') as z:
            _,valid=io23.labels_at(z['times'],flag,down,125.)
            for a in rows:rows[a].append(dict(shot=sid,times=z['times'][valid],scores=z[a][valid],is_disrupt=flag,down=down))
    results={a:metrics(v,protocol['threshold'],lead=30.) for a,v in rows.items()}
    summary={a:dict(public(v),very_early_over3000=sum(r['timely'] and r['lead_ms']>3000 for r in v['predictions'])) for a,v in results.items()}
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
    off,full=results['off'],results['full']
    # All arms have exactly180D/180S. Compare integer net counts so a one-ULP
    # floating J difference cannot turn an identical result into a benefit.
    net=lambda r:r['tp']-r['fp']
    return (full['fp']<=36 and full['tp']>=off['tp'] and net(full)>net(off)
            and all(net(full)>net(results[a]) for a in ('level','reverse','uniform','raw_kl')))


def main(root,limit=None):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);torch.backends.mha.set_fastpath_enabled(True)
    prior=READ(PREVIOUS/'protocol.json');order=prior['shot_ids']
    assert len(order)==360 and set(order)==set(io23.cohort_ids('EAST'))
    if limit is not None:order=order[:limit]
    assert not set(order)&set(prior['preparation_ids']+prior['calibration_ids'])
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];original=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');start=time.monotonic()
    root.mkdir(parents=True);(root/'shots').mkdir();(root/'receipts').mkdir()
    files=set(Path(__file__).parent.glob('*.py'));files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_curvature_memory.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    protocol=dict(shot_ids=order,preparation_ids=prior['preparation_ids'],calibration_ids=prior['calibration_ids'],
        threshold=prior['threshold'],source_threshold=prior['source_threshold'],source_sha256=PARENT_SHA256,
        parent_sha256=sha(REFERENCE/'fold0/mapped_fixed/prepared.pt'),mapping_sha256=sha(REFERENCE/'input_map.json'),
        previous_completed_sha256=sha(PREVIOUS/'completed.json'),plan_sha256=sha(PLAN),
        historical_bank_sha256={f:sha(PREVIOUS/f) for f in ('bank.npz','bank.json','gradients.pt')},
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
        input_sha256={str(io23.path_for('EAST',s)):sha(io23.path_for('EAST',s)) for s in order+prior['preparation_ids']},
        arms=('off',)+ARMS,update_every=128,step_learning_rate=.01,damping_fraction=.01,
        step_reset='parent at every update',encoder_names=model.encoder_names(),inter_steps=0,alarm_updates=0,
        event_labels_used=False,label_access='cached preparation labels only until all eight arms all shots sealed',
        scope='DEVELOPMENT_ONLY; not chronological commissioning; not pure self-supervision',
        inference_backend='eval/no_grad,MHA_fastpath=True; reference-identical batch32',
        independent_confirmation=False,torch=torch.__version__,gpu=torch.cuda.get_device_name())
    _durable_json(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
    def deny(*args,**kwargs):raise AssertionError('no test outcome before all predictions sealed')
    with patch.object(io23,'read_outcome',side_effect=deny):
        bank=load_bank();assert set(bank['shots'])==set(prior['preparation_ids'])
        tensors,info=make_geometry(model,bank)
        torch.save(tensors,root/'geometry.pt');_durable_json(root/'geometry.json',info)
        print(dict(stage='geometry_ready',info=info,seconds=time.monotonic()-start),flush=True)
        total={a:0 for a in ARMS}
        for i,sid in enumerate(order):
            if time.monotonic()-start>1800:raise TimeoutError('declared30min exceeded')
            raw=signal(sid);mapped=transform({k:raw[k] for k in ('values','available')},mapping)
            curves,receipts=predict(model,mapped,bank)
            with np.load(PREVIOUS/'shots'/f'{sid}.npz') as z:
                np.testing.assert_array_equal(raw['times'],z['times'])
                np.testing.assert_array_equal(curves['off'],z['off'],err_msg=f'parent mismatch {sid}')
                np.testing.assert_array_equal(curves['raw'],z['trajectory'],err_msg=f'raw direction mismatch {sid}')
            np.savez_compressed(root/'shots'/f'{sid}.npz',times=raw['times'],**curves)
            _durable_json(root/'receipts'/f'{sid}.json',receipts)
            for a in ARMS:total[a]+=sum(r[a]['step_norm']>0 for r in receipts)
            if (i+1)%10==0:print(dict(stage='predict',shots=i+1,seconds=round(time.monotonic()-start,1)),flush=True)
    assert_unchanged(model,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(root/'prediction_seals.json',{str(f.relative_to(root)):sha(f) for f in (root/'shots').glob('*.npz')})
    results=evaluate(root,protocol)
    artifacts={str(f.relative_to(root)):sha(f) for f in root.iterdir() if f.is_file()}
    done=dict(status='completed',metrics=results,temporary_updates=total,static_and_raw_max_error=0.,
        candidate_promoted=False,eligible_for_inter_study=eligible(results),seconds=time.monotonic()-start,
        artifact_sha256=artifacts,scientific_goal_achieved=False)
    _durable_json(root/'completed.json',done);print(done,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    try:main(args.out,args.limit)
    except Exception as exc:
        if args.out.exists():_durable_json(args.out/'failed.json',dict(type=type(exc).__name__,error=str(exc)))
        raise
