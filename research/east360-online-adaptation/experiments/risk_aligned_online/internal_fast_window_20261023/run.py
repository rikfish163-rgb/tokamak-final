"""Five-arm development experiment: no current-shot labels or alarm changes."""
import argparse
import copy
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23
from ..contextual_feedback_20261023.run import read_signal
from ..continual_episode_20261023.run import labeled,outcome_record
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..diverse_support_20261023.run import READ,make_parent
from ..heldout_threshold_20261023.run import REFERENCE,REFERENCE_SHA,models_for_calibration
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics,sha
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .model import WindowAdapter
from .learning import ARMS,training_pools,sample_batch,risk_step,predict

PRIOR=io23.ROOT/'experiments/risk_aligned_online/runs/contextual_feedback360_20261023'
PRIOR_SHA='d8bbfe4e8ca3ab62845d4cef4149b00b742d5476b2534e7a6e3bd9912dcdebc1'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/层内临时学习_首轮实验约定.md'


def evaluate(root,protocol,outcomes):
    rows={a:[] for a in ARMS}
    for r in outcomes:
        with np.load(root/'shots'/f'{r["shot"]}.npz') as z:
            _,valid=io23.labels_at(z['times'],r['is_disrupt'],r['down'],125.)
            for a in ARMS:
                rows[a].append(dict(shot=r['shot'],times=z['times'][valid],scores=z[a][valid],
                    is_disrupt=r['is_disrupt'],down=r['down']))
    result={a:metrics(v,protocol['threshold'],lead=30.) for a,v in rows.items()}
    for a,v in result.items():v['very_early_over3000']=sum(r['timely'] and r['lead_ms']>3000 for r in v['predictions'])
    _durable_json(root/'per_shot.json',result)
    blocks=[]
    for end in range(20,len(outcomes)+20,20):
        end=min(end,len(outcomes))
        blocks.append(dict(end=end,cumulative={a:public(metrics(v[:end],protocol['threshold'],lead=30.)) for a,v in rows.items()}))
    _durable_json(root/'blocks.json',blocks)
    return {a:public(v) for a,v in result.items()}


def main(root,smoke=False):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    assert sha(PRIOR/'completed.json')==PRIOR_SHA
    assert sha(REFERENCE/'completed.json')==REFERENCE_SHA
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    prior=READ(PRIOR/'protocol.json');mapping=READ(REFERENCE/'input_map.json')
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic()
    source,_=make_parent();parent=models_for_calibration(source)['mapped_fixed']
    adaptive=WindowAdapter(copy.deepcopy(parent.base)).cuda().eval();capacity=copy.deepcopy(adaptive)
    base_state=state_cpu(parent.base);initial_layer=state_cpu(adaptive.layer)
    root.mkdir(parents=True);(root/'shots').mkdir()
    ids=prior['shot_ids'][:2] if smoke else prior['shot_ids'];steps=4 if smoke else 512
    assert not set(ids)&(set(prior['preparation_ids'])|set(prior['calibration_ids']))
    files=set(Path(__file__).parent.glob('*.py'))
    files.update((io23.ROOT/'experiments/risk_aligned_online/tests').glob('test_internal_fast*.py'))
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    protocol=dict(shot_ids=ids,preparation_ids=prior['preparation_ids'],calibration_ids=prior['calibration_ids'],
        arms=ARMS,threshold=prior['threshold'],source_threshold=prior['source_threshold'],seed=131,
        additional_preparation_steps=steps,batch=32,prediction_batch=32,learning_rate=.001,gradient_clip=1.,rank=16,
        label_budget=30,inter_steps=0,eventlabels_used=False,alarm_updates=0,
        extra_parameter_count=sum(p.numel() for p in adaptive.layer.parameters()),
        trained_names=['layer.'+k for k,p in adaptive.layer.named_parameters()],
        scope='SMOKE_ONLY' if smoke else 'DEVELOPMENT_ONLY; historical20 preparation+10 calibration; window-reset internal adaptation',
        independent_confirmation=False,online_outcome_access='after_all360_prediction_seals; evaluation only',
        fast_reset='every endpoint to learned W0; no persistent current-shot or cross-shot state',
        source_sha256=PARENT_SHA256,parent_prepared_sha256=sha(REFERENCE/'fold0/mapped_fixed/prepared.pt'),
        prior_sha256=PRIOR_SHA,reference_sha256=REFERENCE_SHA,plan_sha256=sha(PLAN),mapping_sha256=sha(REFERENCE/'input_map.json'),
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
        input_sha256={str(io23.path_for('EAST',sid)):sha(io23.path_for('EAST',sid)) for sid in ids+prior['preparation_ids']},
        torch=torch.__version__,gpu=torch.cuda.get_device_name())
    _durable_json(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(files):archive.write(f,str(f.relative_to(io23.ROOT)))
    reader=io23.read_outcome;records=[];access=[]
    def prep_reader(device,sid):
        assert device=='EAST' and sid in prior['preparation_ids']
        access.append(dict(shot=sid,phase='historical_preparation'))
        return reader(device,sid)
    with patch.object(io23,'read_outcome',side_effect=prep_reader):
        for sid in prior['preparation_ids']:
            records.append(labeled(read_signal(sid),io23.read_outcome('EAST',sid),mapping))
    _durable_json(root/'preparation_outcomes.json',[dict(shot=s['shot'],is_disrupt=s['is_disrupt'],down=s['down']) for s in records])
    pools=training_pools(records);rng=np.random.default_rng(131)
    models=dict(capacity=capacity,adaptive=adaptive)
    opts={name:torch.optim.Adam(model.layer.parameters(),lr=.001) for name,model in models.items()}
    training=[]
    def deny(*args,**kwargs):raise AssertionError('no current or future outcome in preparation or prediction')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for step in range(steps):
            x,m,y,samples=sample_batch(records,pools,rng,'cuda')
            with torch.no_grad():h,valid=adaptive.tokens(x,m)
            r=dict(step=step,samples=samples)
            for name,model in models.items():
                r[name]=risk_step(model,opts[name],h,valid,y,mode='on' if name=='adaptive' else 'off')
            training.append(r)
            if (step+1)%128==0:print(dict(stage='preparation',step=step+1,capacity_loss=r['capacity']['loss'],adaptive_loss=r['adaptive']['loss'],seconds=time.monotonic()-start),flush=True)
    _durable_json(root/'training.json',training)
    torch.save(dict(initial=initial_layer,capacity=state_cpu(capacity.layer),adaptive=state_cpu(adaptive.layer)),root/'layers.pt')
    states={name:state_cpu(model) for name,model in models.items()};seals=[]
    with patch.object(io23,'read_outcome',side_effect=deny):
        for index,sid in enumerate(ids):
            raw=read_signal(sid);signal=transform({k:raw[k] for k in ('values','available')},mapping)
            curves,traces=predict(adaptive,capacity,signal)
            with np.load(PRIOR/'shots'/f'{sid}.npz') as previous:
                np.testing.assert_array_equal(raw['times'],previous['times'])
                np.testing.assert_array_equal(curves['parent'],previous['static'])
                assert np.array_equal(curves['parent']>=protocol['threshold'],previous['static']>=protocol['threshold']), 'parent alarm equivalence'
            path=root/'shots'/f'{sid}.npz'
            np.savez_compressed(path,times=raw['times'],**curves)
            tracepath=root/'shots'/f'{sid}_inner.npz';np.savez_compressed(tracepath,**traces)
            seals.append(dict(shot=sid,prediction_sha256=sha(path),inner_sha256=sha(tracepath)))
            if (index+1)%20==0 or index+1==len(ids):
                progress=dict(shots=index+1,last_shot=sid,seconds=time.monotonic()-start)
                _durable_json(root/'progress.json',progress);print(progress,flush=True)
    _durable_json(root/'prediction_seals.json',seals)
    outcomes=[]
    for index,r in enumerate(seals):
        sid=r['shot'];assert sha(root/'shots'/f'{sid}.npz')==r['prediction_sha256']
        outcomes.append(outcome_record(sid,reader('EAST',sid),r['prediction_sha256'],index))
        access.append(dict(shot=sid,phase='after_all_prediction_seals'))
    _durable_json(root/'outcomes.json',outcomes);_durable_json(root/'label_access.json',access)
    for name,model in models.items():
        assert_unchanged(model,states[name]);assert_unchanged(model.base,base_state)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=evaluate(root,protocol,outcomes)
    done=dict(status='COMPLETED_SMOKE' if smoke else 'COMPLETED_INTERNAL_WINDOW_EXPERIMENT',metrics=result,
        source_unchanged=True,parent_backbone_unchanged=True,seconds=time.monotonic()-start,
        artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',done);print({k:v for k,v in done.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();main(args.root,args.smoke)
