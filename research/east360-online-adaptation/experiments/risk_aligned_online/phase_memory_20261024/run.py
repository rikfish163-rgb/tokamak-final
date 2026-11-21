"""No new EAST labels, no stream evaluation, no change to original risk model."""
import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23
from ..event_pairing_20261024.learning import features,mean_features,neighbour_score
from ..event_routed_20261024.learning import embed
from ..event_routed_20261024.run import SCREEN,PREVIOUS,READ,load_metric_bank,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import phase_score,make_fold,auc,summarize,eligible

WITHIN=io23.ROOT/'experiments/risk_aligned_online/runs/within_shot_pair360_20261024'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/历史阶段识别_留炮筛查事前约定_20261024.md'
MODES=('risk','mean','reverse','shuffle','family','raw','raw_mean','risk_131','risk_132','risk_133')
METHODS=('late_only','augmented','phase')


def representations(bank,early_x,early_mask,device='cuda'):
    late=np.stack([features(x,m,bank['center'],bank['scale']) for x,m in zip(bank['x'],bank['mask'])])
    early=np.stack([features(x,m,bank['center'],bank['scale']) for x,m in zip(early_x,early_mask)])
    normalizer=bank['normalizer'];outputs={}
    for mode in MODES:
        memory=[];query=[]
        for x in (late,early):
            if mode in ('raw','raw_mean'):
                v=mean_features(x) if mode=='raw_mean' else x
                z=((v-normalizer['mean'])/normalizer['scale']).astype('float64')
                q=z
            elif mode.startswith('risk_'):
                model=bank['metric_models']['risk'][int(mode.split('_')[1])-131]
                xx=torch.as_tensor(((x-normalizer['mean'])/normalizer['scale']).astype('float32'),device=device)
                with torch.no_grad():
                    z=model(xx).cpu().numpy().astype('float64')
                q=z
            else:
                key='risk' if mode=='reverse' else mode
                args=(bank['metric_models'][key],normalizer['mean'],normalizer['scale'])
                z=embed(x,*args,key,device)
                q=embed(x,*args,'reverse',device) if mode=='reverse' else z
            memory.append(z);query.append(q)
        outputs[mode]=dict(late=memory[0],early=memory[1],query_late=query[0],query_early=query[1])
    return outputs


def evaluate(view,shots,labels,early_ends):
    details={method:[] for method in METHODS}
    for dsid in np.unique(shots[labels==1]):
        for ssid in np.unique(shots[labels==0]):
            f=make_fold(shots,labels,early_ends,dsid,ssid);keep=f['history'];positives=keep[labels[keep]==1]
            q=np.concatenate([view['query_late'][f['late']],view['query_early'][f['early_unique']],view['query_late'][f['safe']]])
            nlate=len(f['late']);nearly=len(f['early_unique'])
            scores={};extra={}
            scores['late_only']=neighbour_score(q,view['late'][keep],shots[keep],labels[keep],exclude=np.full(len(q),-1))
            memory=np.concatenate([view['late'][keep],view['early'][positives]])
            ids=np.r_[shots[keep],shots[positives]];yy=np.r_[labels[keep],np.zeros(len(positives),int)]
            scores['augmented']=neighbour_score(q,memory,ids,yy,exclude=np.full(len(q),-1))
            scores['phase'],r=phase_score(q,view['early'][positives],view['late'][positives],shots[positives])
            extra['phase']=dict(temperature=r['temperature'],selected_original_rows=positives[r['selected']].tolist(),weights=r['weights'].tolist())
            for method,values in scores.items():
                l,e,s=np.split(values,[nlate,nlate+nearly]);diff=l-e[f['pair_to_unique']]
                details[method].append(dict(held_shots=[int(dsid),int(ssid)],history_shots=sorted(map(int,np.unique(shots[keep]))),
                    late_rows=f['late'].tolist(),early_unique_rows=f['early_unique'].tolist(),safe_rows=f['safe'].tolist(),
                    pair_to_unique=f['pair_to_unique'].tolist(),late_scores=l.tolist(),early_scores=e.tolist(),safe_scores=s.tolist(),
                    phase_auc=auc(l,e),late_safe_auc=auc(l,s),paired_order=float(np.mean((diff>0)+.5*(diff==0))),
                    **extra.get(method,{})))
    return {k:summarize(v) for k,v in details.items()},details


def main(root):
    if root.exists():
        raise FileExistsError(root)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():
        raise RuntimeError('local rl_env CUDA required')
    prior=READ(WITHIN/'completed.json');verified=READ(WITHIN/'verification.json')
    assert verified['completed_sha256']==sha(WITHIN/'completed.json')
    for f in ('pairs.npz','pairs.json'):
        assert sha(WITHIN/f)==prior['artifact_sha256'][f]
    start=time.monotonic();root.mkdir(parents=True)
    files=set(Path(__file__).parent.glob('*.py'));files.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_phase_memory.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):
            files.add(Path(module.__file__))
    _durable_json(root/'protocol.json',dict(scope='DEVELOPMENT_RETRIEVAL_ONLY; input mapping not refit per fold',
        source_sha256=PARENT_SHA256,plan_sha256=sha(PLAN),previous_completed_sha256=sha(WITHIN/'completed.json'),
        input_sha256={str(p):sha(p) for p in [WITHIN/'pairs.npz',WITHIN/'pairs.json',PREVIOUS/'bank.npz',PREVIOUS/'bank.json',SCREEN/'encoders.pt',SCREEN/'standardizer.npz']},
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},modes=MODES,methods=METHODS,
        training_steps=0,stream360_read=False,source_model_updated=False,
        comparison='hold1D+1S together; same3D15S history; unique early query endpoints;4 independent D identities',
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),timeout_seconds=600))
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f,str(f.relative_to(io23.ROOT)))
    def deny(*args,**kwargs):
        raise AssertionError('phase screen must use only archived historical inputs; no new shot/outcome access')
    with patch.object(io23,'read_outcome',side_effect=deny),patch.object(io23,'read_signals',side_effect=deny):
        bank=load_metric_bank()
        with np.load(WITHIN/'pairs.npz') as z:
            early_x=z['early_x'];early_mask=z['early_mask']
        early_ends=np.full(len(bank['shots']),-1,int)
        for r in READ(WITHIN/'pairs.json'):
            early_ends[r['row']]=r['early_end']
        views=representations(bank,early_x,early_mask)
        np.savez_compressed(root/'embeddings.npz',**{f'{a}_{k}':v for a,b in views.items() for k,v in b.items()},
                            shots=bank['shots'],labels=bank['labels'],early_ends=early_ends)
        results={};details={}
        for mode,view in views.items():
            if time.monotonic()-start>600:
                raise TimeoutError('declared10minutes exceeded')
            results[mode],details[mode]=evaluate(view,bank['shots'],bank['labels'],early_ends)
            print(dict(mode=mode,summary={k:{m:v[m] for m in ('phase_auc','late_safe_auc','paired_order')} for k,v in results[mode].items()}),flush=True)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(root/'results.json',results);_durable_json(root/'folds.json',details)
    done=dict(status='completed',eligible_for_neural_trial=eligible(results),candidate_promoted=False,scientific_goal_achieved=False,
        seconds=time.monotonic()-start,independent_disruptive_shots=4,unique_early_windows=18,
        source_risk_unchanged=True,stream360_read=False,
        artifact_sha256={f.name:sha(f) for f in root.iterdir() if f.is_file()})
    _durable_json(root/'completed.json',done);print(done,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    try:
        main(args.out)
    except Exception as exc:
        if args.out.exists():
            _durable_json(args.out/'failed.json',dict(type=type(exc).__name__,error=str(exc)))
        raise
