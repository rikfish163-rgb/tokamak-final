"""Same acquired20, three frozen representations, original linear head refit."""
import argparse
import copy
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..diverse_support_20261023.run import make_parent, READ
from ..increment_forecast_20261023.learning import predict
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics, sha
from ..warmstart_alignment_20261022.run import public
from ..temporal_scope_20261023.run import REFERENCE, REF_SHA
from .fit import endpoint_manifest, fit_readout

BOTH = io23.ROOT/'experiments/risk_aligned_online/runs/temporal_scope85_20261023'
BOTH_SHA = 'cf2ec4c8adb0f100c9bd5c33701d3facdaeffdc69e43e390a4341dc05310b0c1'
PLAN = io23.ROOT/'docs/research/source_event_rebuild_20261022/固定表示风险读出_事前约定_20261023.md'
KINDS = ('source','first','both')
NEW = ('source_readout','first_readout_static','first_readout_intra','both_readout_static','both_readout_intra')
OLD = ('source','first_static','first_intra','both_static','both_intra')
ARMS = OLD + NEW
HEAD = ('base.risk_head.weight','base.risk_head.bias')


def models_for(fold):
    parent, threshold = make_parent()
    models = {'source':parent}
    for kind,path in (('first',REFERENCE/f'fold{fold}'/'diverse'/'prepared.pt'),
                      ('both',BOTH/f'fold{fold}'/'prepared.pt')):
        model = copy.deepcopy(parent)
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
        model.set_stage('frozen');models[kind] = model
    return models, threshold


def features(model, signal, ends):
    rows = []
    for start in range(0,len(ends),32):
        windows = [simple_data.window(signal,int(e),128) for e in ends[start:start+32]]
        x,m = [torch.as_tensor(np.stack(v),device='cuda') for v in zip(*windows)]
        with torch.no_grad(): h = model(x,m,mode='risk_state')['risk_state']
        rows.append(h.cpu().numpy())
    out = np.concatenate(rows)
    if not np.isfinite(out).all():raise FloatingPointError('nonfinite frozen features')
    return out


def training_features(model, shots, manifest):
    by_id = {int(s['shot']):s for s in shots};result = np.empty((len(manifest),64),np.float32)
    for sid,s in by_id.items():
        idx = [i for i,r in enumerate(manifest) if r['shot']==sid]
        result[idx] = features(model,{k:s[k] for k in ('values','available')},[manifest[i]['end'] for i in idx])
    return result


def head_vector(model):
    return np.r_[model.base.risk_head.weight.detach().cpu().numpy().ravel(),
                 model.base.risk_head.bias.detach().cpu().numpy()].astype(np.float64)


def install_head(model, theta):
    with torch.no_grad():
        model.base.risk_head.weight.copy_(torch.as_tensor(theta[:-1].reshape(1,-1),device='cuda',dtype=torch.float32))
        model.base.risk_head.bias.copy_(torch.as_tensor(theta[-1:],device='cuda',dtype=torch.float32))


def static_scores(model, signal):
    h = features(model,signal,np.arange(len(signal['values'])))
    # Match the deployment's float32 linear head and sigmoid, not a float64 surrogate.
    with torch.no_grad():
        scores = model.base.risk_head(torch.as_tensor(h,device='cuda')).squeeze(-1).sigmoid().cpu().numpy()
    return scores


def score(root,plans,threshold):
    rows = {a:[] for a in ARMS};access=[];folds={}
    for plan in plans:
        local = {a:[] for a in ARMS}
        for sid in plan['held']:
            s = simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
            access.append(dict(fold=plan['fold'],shot=sid,phase='after_all_prediction_seals'))
            with np.load(root/f'fold{plan["fold"]}'/'shots'/f'{sid}.npz') as z:
                np.testing.assert_array_equal(z['times'],s['times'])
                for a in ARMS:
                    r=dict(shot=sid,times=s['times'][s['valid']],scores=z[a][s['valid']],is_disrupt=s['is_disrupt'],down=s['down'])
                    rows[a].append(r);local[a].append(r)
        folds[str(plan['fold'])]={a:public(metrics(v,threshold,lead=30.)) for a,v in local.items()}
    result={a:metrics(v,threshold,lead=30.) for a,v in rows.items()}
    _durable_json(root/'per_shot.json',result);_durable_json(root/'evaluation_access.json',access)
    return {a:dict(**public(v),very_early_over3000=sum(r['timely'] and r['lead_ms']>3000 for r in v['predictions'])) for a,v in result.items()},folds


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    refs = [(REFERENCE,REF_SHA),(BOTH,BOTH_SHA)]
    for folder,expected in refs:
        assert sha(folder/'completed.json')==expected
        for file,h in READ(folder/'completed.json')['artifact_sha256'].items():assert sha(folder/file)==h
    protocol=READ(REFERENCE/'protocol.json');plans=protocol['folds']
    assert not set(protocol['shot_ids'])&set(io23.cohort_ids('EAST'))
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic();root.mkdir(parents=True)
    files=set(Path(__file__).parent.glob('*.py'));files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_readout_probe.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):files.add(Path(module.__file__))
    p=dict(folds=plans,shot_ids=protocol['shot_ids'],threshold=protocol['threshold'],source_sha256=PARENT_SHA256,
           reference_completed_sha256={str(folder):expected for folder,expected in refs},
           input_sha256=protocol['input_sha256'],plan_sha256=sha(PLAN),arms=ARMS,
           code_sha256={str(v.relative_to(io23.ROOT)):sha(v) for v in files},
           label_budget_per_fold=20,independent_confirmation=False,test360_accessed=False,
           seed=131,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
           intervention='existing head only, frozen representations; historical development diagnostic')
    _durable_json(root/'protocol.json',p)
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for v in sorted(files):z.write(v,str(v.relative_to(io23.ROOT)))
    reader=io23.read_outcome;access=[];receipts=[];steps=0
    for plan in plans:
        f=plan['fold'];out=root/f'fold{f}';out.mkdir();(out/'shots').mkdir()
        selected=sorted(READ(REFERENCE/f'fold{f}'/'selection.json')['selected']['diverse'])
        _durable_json(out/'selection.json',dict(selected=selected))
        def training_guard(device,sid):
            assert device=='EAST' and sid in selected and sid not in plan['held']
            access.append(dict(fold=f,shot=sid,phase='selected_preparation'))
            return reader(device,sid)
        with patch.object(io23,'read_outcome',side_effect=training_guard):
            target=[simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{}) for sid in selected]
        manifest=endpoint_manifest(target);_durable_json(out/'endpoint_manifest.json',manifest)
        y=np.asarray([r['label'] for r in manifest]);weight=np.asarray([r['weight'] for r in manifest])
        models,threshold=models_for(f);assert threshold==p['threshold'];states={}
        def deny(*args,**kwargs):raise AssertionError('no outcome access during feature extraction or prediction')
        with patch.object(io23,'read_outcome',side_effect=deny):
            for kind,model in models.items():
                before=state_cpu(model);x=training_features(model,target,manifest);origin=head_vector(model)
                theta,info=fit_readout(x,y,weight,origin)
                np.savez_compressed(out/f'{kind}_fit.npz',x=x,y=y,weights=weight,origin=origin,theta=theta)
                install_head(model,theta);assert_unchanged(model,before,HEAD);states[kind]=state_cpu(model)
                _durable_json(out/f'{kind}_fit.json',info)
                print(dict(stage='head_fit',fold=f,kind=kind,**info),flush=True)
            for sid in plan['held']:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal={k:s[k] for k in ('values','available')};updates={}
                curves={'source_readout':static_scores(models['source'],signal)}
                for kind in ('first','both'):
                    scores,u,_=predict(models[kind],signal);updates[kind]=u
                    np.testing.assert_array_equal(scores['static'][:256],scores['intra'][:256])
                    curves.update({kind+'_readout_'+a:v for a,v in scores.items()})
                    steps+=sum(r['gradient_steps'] for r in u)
                with np.load(BOTH/f'fold{f}'/'shots'/f'{sid}.npz') as z:
                    np.testing.assert_array_equal(z['times'],s['times'])
                    curves.update({a:z[a].copy() for a in OLD})
                path=out/'shots'/f'{sid}.npz';up=out/'shots'/f'{sid}_updates.json'
                np.savez_compressed(path,times=s['times'],**curves);_durable_json(up,updates)
                receipts.append(dict(fold=f,shot=sid,prediction_sha256=sha(path),updates_sha256=sha(up)))
                if len(receipts)%5==0:print(dict(stage='predict',shots=len(receipts),seconds=time.monotonic()-start),flush=True)
        for kind,model in models.items():assert_unchanged(model,states[kind])
    _durable_json(root/'training_label_access.json',access);_durable_json(root/'prediction_receipts.json',receipts)
    results,folds=score(root,plans,p['threshold'])
    for a in OLD:assert results[a]==READ(BOTH/'completed.json')['metrics'][a]
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    done=dict(status='COMPLETED',metrics=results,fold_metrics=folds,seconds=time.monotonic()-start,
              head_fits=12,online_steps=steps,source_unchanged=True,test360_accessed=False,independent_confirmation=False,
              artifact_sha256={str(v.relative_to(root)):sha(v) for v in root.rglob('*') if v.is_file()})
    _durable_json(root/'completed.json',done);print({k:v for k,v in done.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
