"""Fixed seven-arm EAST360 intra-only run; no outcome until all seals."""
import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23
from ..curvature_memory_20261024.run import (
    READ,PREVIOUS,REFERENCE,load_bank,make_parent,models_for_calibration,signal,transform,
    DEFAULT_PARENT,PARENT_SHA256,state_cpu,assert_unchanged,
)
from ..simple_online import _durable_json
from ..simple_train import metrics,sha
from ..warmstart_alignment_20261022.run import public
from .learning import ARMS,prepare_bank,predict

SCREEN=io23.ROOT/'experiments/risk_aligned_online/runs/event_pairing_screen_20261024'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/事件危险配对引导炮内_360事前约定_20261024.md'


def load_metric_bank(device='cuda'):
    done=READ(SCREEN/'completed.json');recheck=READ(SCREEN/'same_history_recheck/completed.json')
    assert recheck['eligible_for_online_trial'] and not recheck['candidate_promoted']
    for f,h in done['artifact_sha256'].items():assert sha(SCREEN/f)==h,f
    for f,h in recheck['artifact_sha256'].items():assert sha(SCREEN/'same_history_recheck'/f)==h,f
    bank=load_bank(device=device)
    heads=torch.load(SCREEN/'encoders.pt',map_location='cpu',weights_only=True)
    with np.load(SCREEN/'standardizer.npz') as z:norm={k:z[k].copy() for k in z.files}
    prepare_bank(bank,heads,norm,device)
    return bank


def eligible(results):
    if any(r['n_disrupt']!=180 or r['n_safe']!=180 for r in results.values()):return False
    risk=results['risk'];net=lambda r:r['tp']-r['fp']
    return (risk['fp']<=36 and risk['tp']>=results['off']['tp']
            and all(net(risk)>net(results[a]) for a in ('off','mean','shuffle','family','reverse','uniform')))


def evaluate(root,protocol,*,arms=ARMS):
    rows={a:[] for a in ('off',)+arms};outcomes=[]
    for sid in protocol['shot_ids']:
        flag,down=io23.read_outcome('EAST',sid)
        outcomes.append(dict(shot=sid,is_disrupt=bool(flag),down=None if down is None else float(down)))
        with np.load(root/'shots'/f'{sid}.npz') as z:
            _,valid=io23.labels_at(z['times'],flag,down,125.)
            for a in rows:rows[a].append(dict(shot=sid,times=z['times'][valid],scores=z[a][valid],is_disrupt=flag,down=down))
    results={a:metrics(v,protocol['threshold'],lead=30.) for a,v in rows.items()}
    summary={a:dict(public(v),very_early_over3000=sum(r['timely'] and r['lead_ms']>3000 for r in v['predictions'])) for a,v in results.items()}
    changes={}
    for arm in arms:
        changed=[]
        for before,after in zip(results['off']['predictions'],results[arm]['predictions']):
            good=lambda r:bool(r['timely']) if r['is_disrupt'] else not r['alarm']
            if good(before)!=good(after):changed.append(dict(shot=before['shot'],beneficial=good(after),before=before,after=after))
        changes[arm]=changed
    _durable_json(root/'outcomes.json',outcomes);_durable_json(root/'per_shot.json',results);_durable_json(root/'flips.json',changes)
    _durable_json(root/'blocks.json',[dict(end=end,metrics={a:public(metrics(v[:end],protocol['threshold'],lead=30.)) for a,v in rows.items()})
                                    for end in range(20,len(outcomes)+1,20)])
    return summary


def main(root,limit=None):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);torch.backends.mha.set_fastpath_enabled(True)
    torch.backends.cuda.matmul.allow_tf32=False
    prior=READ(PREVIOUS/'protocol.json');order=prior['shot_ids']
    assert len(order)==360 and set(order)==set(io23.cohort_ids('EAST'))
    if limit is not None:order=order[:limit]
    assert not set(order)&set(prior['preparation_ids']+prior['calibration_ids'])
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];original=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');start=time.monotonic()
    root.mkdir(parents=True);(root/'shots').mkdir();(root/'receipts').mkdir()
    files=set(Path(__file__).parent.glob('*.py'));files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_event_routed.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    protocol=dict(shot_ids=order,preparation_ids=prior['preparation_ids'],calibration_ids=prior['calibration_ids'],
        threshold=prior['threshold'],source_sha256=PARENT_SHA256,parent_sha256=sha(REFERENCE/'fold0/mapped_fixed/prepared.pt'),
        mapping_sha256=sha(REFERENCE/'input_map.json'),previous_completed_sha256=sha(PREVIOUS/'completed.json'),
        plan_sha256=sha(PLAN),screen_completed_sha256=sha(SCREEN/'completed.json'),
        screen_recheck_sha256=sha(SCREEN/'same_history_recheck/completed.json'),
        metric_sha256={f:sha(SCREEN/f) for f in ('encoders.pt','standardizer.npz')},
        historical_bank_sha256={f:sha(PREVIOUS/f) for f in ('bank.npz','bank.json','gradients.pt')},
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
        input_sha256={str(io23.path_for('EAST',s)):sha(io23.path_for('EAST',s)) for s in order},
        arms=('off',)+ARMS,update_every=128,step_learning_rate=.01,step_reset='parent every128points',
        ensemble='all3 source seeds, concatenate/sqrt3; no selected seed',encoder_names=model.encoder_names(),
        inter_steps=0,alarm_updates=0,event_labels_used='source TRAIN offline only',
        label_access='cached historical labels only until all7 arms all shots sealed',
        scope='DEVELOPMENT_ONLY; not chronological commissioning; not pure self-supervision',
        inference_backend='eval/no_grad,MHA_fastpath=True; batch32',independent_confirmation=False,
        torch=torch.__version__,gpu=torch.cuda.get_device_name())
    _durable_json(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
    def deny(*args,**kwargs):raise AssertionError('no stream outcomes until every prediction sealed')
    with patch.object(io23,'read_outcome',side_effect=deny):
        bank=load_metric_bank();assert set(bank['shots'])==set(prior['preparation_ids'])
        np.savez_compressed(root/'metric_bank.npz',**bank['metric_bank'])
        _durable_json(root/'metric_bank.json',dict(temperature=bank['metric_temperature']))
        total={a:0 for a in ARMS}
        for i,sid in enumerate(order):
            if time.monotonic()-start>7200:raise TimeoutError('declared120minutes exceeded')
            raw=signal(sid);mapped=transform({k:raw[k] for k in ('values','available')},mapping)
            curves,receipts=predict(model,mapped,bank)
            with np.load(PREVIOUS/'shots'/f'{sid}.npz') as z:
                np.testing.assert_array_equal(raw['times'],z['times'])
                np.testing.assert_array_equal(curves['off'],z['off'],err_msg=f'parent mismatch{sid}')
            np.savez_compressed(root/'shots'/f'{sid}.npz',times=raw['times'],**curves)
            _durable_json(root/'receipts'/f'{sid}.json',receipts)
            for a in ARMS:total[a]+=sum(r[a]['step_norm']>0 for r in receipts)
            if (i+1)%10==0:print(dict(stage='predict',shots=i+1,seconds=round(time.monotonic()-start,1)),flush=True)
    assert_unchanged(model,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(root/'prediction_seals.json',{str(f.relative_to(root)):sha(f) for f in (root/'shots').glob('*.npz')})
    results=evaluate(root,protocol)
    done=dict(status='completed',metrics=results,temporary_updates=total,static_max_error=0.,
        candidate_promoted=False,eligible_for_inter_study=eligible(results),seconds=time.monotonic()-start,
        artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.iterdir() if f.is_file()},scientific_goal_achieved=False)
    _durable_json(root/'completed.json',done);print(done,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    try:main(args.out,args.limit)
    except Exception as exc:
        if args.out.exists():_durable_json(args.out/'failed.json',dict(type=type(exc).__name__,error=str(exc)))
        raise
